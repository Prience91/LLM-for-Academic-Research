from __future__ import annotations

import json
import os
import re
import sqlite3
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

from .config import Settings
from .indexing import GraniteEmbeddingFunction
from .models import Evidence, RetrievalResult


def _fts_query(query: str) -> str:
    tokens = re.findall(r"[\w-]+", query, flags=re.UNICODE)
    return " OR ".join(f'"{token.replace(chr(34), "")}"' for token in tokens[:32])


class QueryRewriter:
    def __init__(self, settings: Settings):
        from openai import OpenAI

        api_key = os.getenv("VLLM_API_KEY", "local-only")
        port = int(settings.get("serving.vllm_port"))
        self.client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key=api_key)
        self.model = str(settings.get("models.served"))
        serving = settings.section("serving")
        if "enable_thinking" not in serving or not isinstance(
            serving["enable_thinking"], bool
        ):
            raise ValueError("serving.enable_thinking must be explicitly configured as a boolean")
        self.enable_thinking = serving["enable_thinking"]

    def _complete(self, prompt: str, max_tokens: int, timeout: float = 30) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
            timeout=timeout,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
            },
        )
        return (response.choices[0].message.content or "").strip()

    def rewrite(self, query: str, history: list[dict[str, str]] | None) -> str:
        if not history:
            return query
        compact_history = history[-6:]
        prompt = (
            "Rewrite the latest user question as one standalone corpus-search query. "
            "Resolve pronouns and omitted subjects using the conversation. Do not answer it. "
            "Return only the rewritten search query.\n\n"
            f"Conversation: {json.dumps(compact_history, ensure_ascii=False)}\n"
            f"Latest question: {query}"
        )
        try:
            rewritten = self._complete(prompt, 96)
            return rewritten or query
        except Exception:
            return query

    @staticmethod
    def _json_payload(text: str) -> Any:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I)
        return json.loads(cleaned)

    @staticmethod
    def _fallback_obligations(query: str) -> list[str]:
        compact = re.sub(r"\s+", " ", query).strip().rstrip("?")
        parts = re.split(
            r"\s*(?:;|,\s+(?:and\s+)?what\b|,\s+why\b|\band\s+why\b|\band\s+what\b)\s*",
            compact,
            flags=re.I,
        )
        parts = [part.strip(" ,") for part in parts if len(part.split()) >= 4]
        return parts[:6] or [compact]

    @staticmethod
    def _explicit_obligations(query: str) -> list[str]:
        """Preserve every interrogative clause even when the semantic planner misses one."""
        compact = re.sub(r"\s+", " ", query).strip().rstrip("?")
        interrogative = r"(?:what|why|how|which|where|when)"
        context = ""
        body = compact
        if "," in compact:
            prefix, remainder = compact.split(",", 1)
            if not re.match(rf"(?i)^{interrogative}\b", prefix.strip()):
                context = prefix.strip()
                body = remainder.strip()
        parts = re.split(
            rf"(?:,\s*(?:and\s+)?|\s+and\s+)(?={interrogative}\b)",
            body,
            flags=re.I,
        )
        obligations = []
        for part in parts:
            cleaned = part.strip(" ,")
            if not cleaned:
                continue
            if context:
                cleaned = f"{context}, {cleaned}"
            obligations.append(cleaned)
        return obligations[:6] or [compact]

    def plan_obligations(self, query: str) -> list[str]:
        # Explicit clause preservation is deterministic. Semantic expansion happens only after
        # preliminary corpus evidence is available, preventing a query-only planner from inventing
        # requirements or collapsing several evidence-backed components into one broad item.
        return self._explicit_obligations(query)

    def expand_obligations(
        self, query: str, explicit: list[str], evidence: list[Evidence]
    ) -> tuple[list[str], bool]:
        if not evidence:
            return explicit, False
        passages = "\n\n".join(
            f"[{item.evidence_id}] {item.text}" for item in evidence
        )
        prompt = (
            "Using only the research question and preliminary corpus passages, identify 3-10 atomic, "
            "evidence-backed answer obligations. Split broad methods, findings, mechanisms, factors, "
            "barriers, limitations, comparisons, and implications into distinct substantive requirements. "
            "Cover every explicit question part; when a part contains several concrete components, create "
            "at least two obligations for that part. Prefer concrete requirements over generic statements "
            "such as 'the authors describe approaches.' Do not repeat, paraphrase, or nest the same point in "
            "multiple obligations. Every obligation must be directly relevant to the question and searchable "
            "on its own. Do not add outside facts or requirements unsupported by these passages. Return only "
            "a JSON array of strings.\n\n"
            f"Question: {query}\n\nExplicit question parts:\n"
            + "\n".join(f"- {item}" for item in explicit)
            + f"\n\nPreliminary passages:\n{passages}"
        )
        try:
            payload = self._json_payload(self._complete(prompt, 384, timeout=120))
            planned = [
                re.sub(r"\s+", " ", str(item)).strip()
                for item in payload
                if isinstance(item, str) and len(item.split()) >= 4
            ]
            obligations: list[str] = []
            for item in planned:
                item_tokens = set(re.findall(r"[a-z0-9]+", item.casefold()))
                if any(
                    len(item_tokens & set(re.findall(r"[a-z0-9]+", prior.casefold())))
                    / max(len(item_tokens | set(re.findall(r"[a-z0-9]+", prior.casefold()))), 1)
                    >= 0.55
                    for prior in obligations
                ):
                    continue
                obligations.append(item)
            # A semantic expansion may still miss an explicit clause. Add only clauses that have
            # no meaningful lexical representative among the atomic requirements.
            for item in explicit:
                item_tokens = set(re.findall(r"[a-z0-9]+", item.casefold()))
                if not any(
                    len(item_tokens & set(re.findall(r"[a-z0-9]+", prior.casefold())))
                    / max(len(item_tokens), 1)
                    >= 0.35
                    for prior in obligations
                ):
                    obligations.append(item)
            expanded = obligations[:10]
            return (expanded, True) if expanded else (explicit, False)
        except Exception:
            return explicit, False

    def validate_coverage(
        self, query: str, obligations: list[str], evidence: list[Evidence]
    ) -> list[dict[str, Any]]:
        if not obligations or not evidence:
            return []
        evidence_text = "\n\n".join(
            f"[{item.evidence_id}] {item.text}" for item in evidence
        )
        prompt = (
            "Map each evidence requirement to the supplied passages. A requirement is covered only when "
            "the passages contain enough explicit information to answer it without outside knowledge. "
            "Partial hints are not covered. Return only JSON in this form: "
            '{"coverage":[{"obligation_index":1,"covered":true,"evidence_ids":["E1"]}]}. '
            "Include exactly one entry for every numbered requirement.\n\n"
            f"Question: {query}\n\nRequirements:\n"
            + "\n".join(f"{index}. {item}" for index, item in enumerate(obligations, 1))
            + f"\n\nPassages:\n{evidence_text}"
        )
        try:
            payload = self._json_payload(self._complete(prompt, 640, timeout=120))
            rows = payload.get("coverage", []) if isinstance(payload, dict) else []
            parsed: list[dict[str, Any]] = []
            valid_ids = {item.evidence_id for item in evidence}
            for expected_index, obligation in enumerate(obligations, 1):
                row = next(
                    (
                        candidate
                        for candidate in rows
                        if int(candidate.get("obligation_index", 0)) == expected_index
                    ),
                    {},
                )
                ids = [
                    str(value)
                    for value in row.get("evidence_ids", [])
                    if str(value) in valid_ids
                ]
                parsed.append(
                    {
                        "obligation_index": expected_index,
                        "obligation": obligation,
                        "covered": bool(row.get("covered", False)) and bool(ids),
                        "evidence_ids": ids,
                        "validation": "semantic",
                    }
                )
            return parsed
        except Exception:
            return []


class HybridRetriever:
    def __init__(self, settings: Settings):
        import chromadb

        self.settings = settings
        self.configuration = settings.section("retrieval")
        manifest_path = settings.path("paths.active_index")
        if not manifest_path.exists():
            raise FileNotFoundError(f"Active index manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        embedding = GraniteEmbeddingFunction(self.manifest["embedding_model"])
        client = chromadb.PersistentClient(path=self.manifest["chroma_dir"])
        self.collection = client.get_collection(
            name=self.manifest["collection_name"], embedding_function=embedding
        )
        self.lexical_db = Path(self.manifest["lexical_db"])
        self.rewriter = QueryRewriter(settings)
        self._reranker = None
        self._cache: OrderedDict[str, RetrievalResult] = OrderedDict()

    @staticmethod
    def _text_similarity(first: str, second: str) -> float:
        left = set(re.findall(r"[a-z0-9]+", first.casefold()))
        right = set(re.findall(r"[a-z0-9]+", second.casefold()))
        return len(left & right) / len(left | right) if left and right else 0.0

    def _cache_get(self, query: str, history: list[dict[str, str]] | None) -> RetrievalResult | None:
        if history or not bool(self.configuration.get("enable_retrieval_cache", False)):
            return None
        key = re.sub(r"\s+", " ", query.casefold()).strip()
        result = self._cache.get(key)
        if result is not None:
            self._cache.move_to_end(key)
        return result

    def _cache_put(
        self, query: str, history: list[dict[str, str]] | None, result: RetrievalResult
    ) -> RetrievalResult:
        if history or not bool(self.configuration.get("enable_retrieval_cache", False)):
            return result
        key = re.sub(r"\s+", " ", query.casefold()).strip()
        self._cache[key] = result
        self._cache.move_to_end(key)
        maximum = int(self.configuration.get("retrieval_cache_size", 256))
        while len(self._cache) > maximum:
            self._cache.popitem(last=False)
        return result

    def _dense(self, query: str, limit: int | None = None) -> list[dict[str, Any]]:
        result = self.collection.query(
            query_texts=[query],
            n_results=limit or int(self.configuration["dense_candidates"]),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[dict[str, Any]] = []
        for chunk_id, text, metadata, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            similarity = 1.0 - float(distance)
            if similarity < float(self.configuration["min_dense_similarity"]):
                continue
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "metadata": metadata,
                    "dense_similarity": similarity,
                }
            )
        return hits

    def _lexical(self, query: str, limit: int | None = None) -> list[dict[str, Any]]:
        fts_query = _fts_query(query)
        if not fts_query:
            return []
        connection = sqlite3.connect(self.lexical_db)
        try:
            rows = connection.execute(
                """
                SELECT c.payload_json, bm25(chunks_fts) AS rank
                FROM chunks_fts
                JOIN chunks c USING (chunk_id)
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, limit or int(self.configuration["lexical_candidates"])),
            ).fetchall()
        finally:
            connection.close()
        return [
            {
                "chunk_id": (payload := json.loads(payload_json))["chunk_id"],
                "text": payload["text"],
                "payload": payload,
                "lexical_score": 1.0 / (1.0 + abs(float(rank))),
            }
            for payload_json, rank in rows
        ]

    def _rerank(self, query: str, candidates: list[dict[str, Any]]) -> None:
        if not candidates:
            return
        try:
            if self._reranker is None:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(str(self.settings.get("models.reranker")))
            scores = self._reranker.predict([(query, item["text"]) for item in candidates])
            for item, score in zip(candidates, scores):
                item["reranker_score"] = float(score)
        except Exception:
            for item in candidates:
                item["reranker_score"] = None

    @staticmethod
    def _candidate_score(item: dict[str, Any]) -> float:
        score = item.get("reranker_score")
        return float(score) if score is not None else float(item.get("fused_score", 0.0))

    def _rank_candidates(self, query: str, expanded: bool = False) -> list[dict[str, Any]]:
        multiplier = int(self.configuration.get("obligation_candidate_multiplier", 2)) if expanded else 1
        dense = self._dense(query, int(self.configuration["dense_candidates"]) * multiplier)
        lexical = self._lexical(query, int(self.configuration["lexical_candidates"]) * multiplier)
        if not dense:
            return []
        combined: dict[str, dict[str, Any]] = {}
        rrf_k = int(self.configuration["rrf_k"])
        for rank, item in enumerate(dense, start=1):
            candidate = combined.setdefault(item["chunk_id"], dict(item))
            candidate["fused_score"] = candidate.get("fused_score", 0.0) + 1.0 / (rrf_k + rank)
        for rank, item in enumerate(lexical, start=1):
            candidate = combined.setdefault(item["chunk_id"], dict(item))
            candidate["lexical_score"] = item["lexical_score"]
            candidate["fused_score"] = candidate.get("fused_score", 0.0) + 1.0 / (rrf_k + rank)
        limit = int(self.configuration["rerank_candidates"]) * multiplier
        candidates = sorted(
            combined.values(), key=lambda item: item["fused_score"], reverse=True
        )[:limit]
        self._rerank(query, candidates)
        return sorted(candidates, key=self._candidate_score, reverse=True)

    def _as_evidence(self, selected: list[dict[str, Any]]) -> list[Evidence]:
        evidence: list[Evidence] = []
        for index, item in enumerate(selected, start=1):
            metadata = item.get("metadata") or item.get("payload") or {}
            authors_raw = metadata.get("authors_json") or metadata.get("authors") or []
            authors = json.loads(authors_raw) if isinstance(authors_raw, str) else authors_raw
            year = int(metadata.get("year") or 0) or None
            evidence.append(
                Evidence(
                    evidence_id=f"E{index}",
                    chunk_id=item["chunk_id"],
                    document_id=str(metadata["document_id"]),
                    text=item["text"],
                    title=str(metadata.get("title") or "Untitled"),
                    authors=list(authors),
                    year=year,
                    journal=str(metadata.get("journal") or "") or None,
                    page_start=int(metadata.get("page_start") or 1),
                    page_end=int(metadata.get("page_end") or metadata.get("page_start") or 1),
                    dense_similarity=item.get("dense_similarity"),
                    lexical_score=item.get("lexical_score"),
                    fused_score=float(item.get("fused_score", 0.0)),
                    reranker_score=item.get("reranker_score"),
                    doi=str(metadata.get("doi") or "") or None,
                )
            )
        return evidence

    def retrieve(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
    ) -> RetrievalResult:
        cached = self._cache_get(query, history)
        if cached is not None:
            return cached
        rewritten = self.rewriter.rewrite(query, history)
        baseline = self._rank_candidates(rewritten)
        if not baseline:
            return self._cache_put(query, history, RetrievalResult(
                query=query,
                rewritten_query=rewritten,
                evidence=[],
                abstained=True,
                reason="no_dense_candidate_above_calibrated_threshold",
            ))

        obligation_enabled = bool(
            self.configuration.get("enable_obligation_aware_retrieval", False)
        )
        explicit_obligations = (
            self.rewriter.plan_obligations(rewritten) if obligation_enabled else []
        )
        preliminary_ranked = [
            self._rank_candidates(item) for item in explicit_obligations
        ]
        planning_candidates: list[dict[str, Any]] = []
        planning_ids: set[str] = set()
        planning_pool = baseline[:6] + [
            item for ranked in preliminary_ranked for item in ranked[:2]
        ]
        for item in planning_pool:
            if len(planning_candidates) >= 8:
                break
            if item["chunk_id"] not in planning_ids:
                planning_candidates.append(item)
                planning_ids.add(item["chunk_id"])
        planning_evidence = self._as_evidence(planning_candidates)
        if obligation_enabled:
            obligations, atomic_planning_succeeded = self.rewriter.expand_obligations(
                rewritten, explicit_obligations, planning_evidence
            )
        else:
            obligations, atomic_planning_succeeded = [], None
        ranked_by_obligation = [self._rank_candidates(item) for item in obligations]
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        per_document: dict[str, int] = defaultdict(int)
        context_tokens = 0
        maximum_context_tokens = int(self.configuration.get("maximum_context_tokens", 0))
        deduplicate = bool(self.configuration.get("enable_context_deduplication", False))
        deduplication_threshold = float(
            self.configuration.get("context_deduplication_similarity", 0.85)
        )
        maximum_per_document = int(
            self.configuration.get(
                "obligation_max_chunks_per_document",
                self.configuration["max_chunks_per_document"],
            )
            if obligations
            else self.configuration["max_chunks_per_document"]
        )
        final_chunks = int(self.configuration["final_chunks"])

        def try_add(candidate: dict[str, Any]) -> bool:
            nonlocal context_tokens
            if candidate["chunk_id"] in selected_ids or len(selected) >= final_chunks:
                return False
            if candidate.get("reranker_score") is not None and candidate[
                "reranker_score"
            ] < float(self.configuration["min_reranker_score"]):
                return False
            metadata = candidate.get("metadata") or candidate.get("payload") or {}
            document_id = str(metadata["document_id"])
            if per_document[document_id] >= maximum_per_document:
                return False
            if deduplicate and any(
                self._text_similarity(candidate["text"], prior["text"])
                >= deduplication_threshold
                for prior in selected
            ):
                return False
            candidate_tokens = int(metadata.get("token_count") or len(candidate["text"].split()))
            if maximum_context_tokens and context_tokens + candidate_tokens > maximum_context_tokens:
                return False
            selected.append(candidate)
            selected_ids.add(candidate["chunk_id"])
            per_document[document_id] += 1
            context_tokens += candidate_tokens
            return True

        # Reserve an evidence slot for every question-derived obligation before allowing
        # globally high-scoring or source-diverse passages to consume the context budget.
        provisional: list[dict[str, Any]] = []
        for obligation_index, ranked in enumerate(ranked_by_obligation, start=1):
            chosen = next((item for item in ranked if try_add(item)), None)
            provisional.append(
                {
                    "obligation_index": obligation_index,
                    "obligation": obligations[obligation_index - 1],
                    "covered": chosen is not None,
                    "evidence_ids": [],
                    "validation": "targeted_retrieval",
                }
            )

        preliminary_evidence = self._as_evidence(selected)
        coverage = (
            self.rewriter.validate_coverage(rewritten, obligations, preliminary_evidence)
            if obligations
            else []
        )
        if not coverage and obligations:
            id_by_chunk = {item.chunk_id: item.evidence_id for item in preliminary_evidence}
            for row, ranked in zip(provisional, ranked_by_obligation):
                row["evidence_ids"] = [
                    id_by_chunk[item["chunk_id"]]
                    for item in ranked
                    if item["chunk_id"] in id_by_chunk
                ][:2]
            coverage = provisional
        retrieval_passes = 1 + (1 if obligations else 0)

        # Run the bounded rescue while context positions are still available, before global fill.
        uncovered = [row for row in coverage if not row["covered"]]
        rescue_performed = False
        if uncovered and len(selected) < final_chunks:
            for row in uncovered:
                rescue_query = f"{rewritten}\nSpecific evidence requirement: {row['obligation']}"
                for candidate in self._rank_candidates(rescue_query, expanded=True):
                    if try_add(candidate):
                        rescue_performed = True
                        break
            retrieval_passes += 1

        # Fill remaining capacity from the union of baseline and obligation-specific ranks.
        union: dict[str, dict[str, Any]] = {}
        for rank, item in enumerate(baseline, start=1):
            merged = dict(item)
            merged["aggregate_score"] = self._candidate_score(item) + 1.0 / (60 + rank)
            union[item["chunk_id"]] = merged
        for ranked in ranked_by_obligation:
            for rank, item in enumerate(ranked, start=1):
                candidate = union.setdefault(item["chunk_id"], dict(item))
                candidate["aggregate_score"] = max(
                    float(candidate.get("aggregate_score", float("-inf"))),
                    self._candidate_score(item) + 1.0 / (60 + rank),
                )
        diversity_enabled = bool(self.configuration.get("enable_source_diversity", False))
        diversity_penalty = float(self.configuration.get("source_diversity_penalty", 0.0))
        remaining = sorted(
            union.values(), key=lambda item: float(item.get("aggregate_score", 0.0)), reverse=True
        )
        while remaining and len(selected) < final_chunks:
            remaining.sort(
                key=lambda item: float(item.get("aggregate_score", 0.0))
                - (
                    diversity_penalty
                    * per_document[str((item.get("metadata") or item.get("payload") or {})["document_id"])]
                    if diversity_enabled
                    else 0.0
                ),
                reverse=True,
            )
            try_add(remaining.pop(0))

        evidence = self._as_evidence(selected)
        if obligations and rescue_performed:
            validated = self.rewriter.validate_coverage(rewritten, obligations, evidence)
            if validated:
                coverage = validated

        coverage_complete = (
            bool(coverage)
            and all(row["covered"] for row in coverage)
            and atomic_planning_succeeded is not False
        )
        return self._cache_put(query, history, RetrievalResult(
            query=query,
            rewritten_query=rewritten,
            evidence=evidence,
            abstained=not evidence,
            reason=None if evidence else "no_candidate_survived_reranking_and_diversity",
            obligations=obligations,
            obligation_coverage=coverage,
            coverage_complete=coverage_complete if obligations else None,
            retrieval_passes=retrieval_passes,
            atomic_planning_succeeded=atomic_planning_succeeded,
        ))

    def document_evidence(self, document_id: str, max_chunks: int = 40) -> list[Evidence]:
        result = self.collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"],
        )
        rows = sorted(
            zip(result["ids"], result["documents"], result["metadatas"]),
            key=lambda row: int(row[2].get("ordinal") or 0),
        )
        if len(rows) > max_chunks:
            positions = {
                round(index * (len(rows) - 1) / (max_chunks - 1)) for index in range(max_chunks)
            }
            rows = [rows[position] for position in sorted(positions)]
        evidence: list[Evidence] = []
        for index, (chunk_id, text, metadata) in enumerate(rows, start=1):
            authors_raw = metadata.get("authors_json") or "[]"
            evidence.append(
                Evidence(
                    evidence_id=f"E{index}",
                    chunk_id=chunk_id,
                    document_id=document_id,
                    text=text,
                    title=str(metadata.get("title") or "Untitled"),
                    authors=json.loads(authors_raw),
                    year=int(metadata.get("year") or 0) or None,
                    journal=str(metadata.get("journal") or "") or None,
                    page_start=int(metadata.get("page_start") or 1),
                    page_end=int(metadata.get("page_end") or metadata.get("page_start") or 1),
                    doi=str(metadata.get("doi") or "") or None,
                )
            )
        return evidence
