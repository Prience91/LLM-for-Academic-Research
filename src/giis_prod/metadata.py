from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from difflib import SequenceMatcher
from typing import Any

from .io import stable_digest, write_json_atomic

_DOI = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_NON_WORD = re.compile(r"[^a-z0-9]+")


def normalize_doi(value: Any) -> str | None:
    text = str(value or "").strip()
    match = _DOI.search(urllib.parse.unquote(text))
    return match.group().rstrip(".,;)").lower() if match else None


def normalized_title(value: Any) -> str:
    return " ".join(_NON_WORD.sub(" ", str(value or "").casefold()).split())


def title_similarity(first: Any, second: Any) -> float:
    left = normalized_title(first)
    right = normalized_title(second)
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def title_relation(
    first: Any,
    second: Any,
    similarity_threshold: float = 0.93,
    minimum_fragment_tokens: int = 6,
    minimum_fragment_characters: int = 40,
) -> str:
    left = normalized_title(first)
    right = normalized_title(second)
    if not left or not right:
        return "conflict"
    if title_similarity(left, right) >= similarity_threshold:
        return "compatible"
    shorter, longer = sorted((left, right), key=len)
    if (
        len(shorter) >= minimum_fragment_characters
        and len(shorter.split()) >= minimum_fragment_tokens
        and shorter in longer
    ):
        return "verified_truncation"
    return "conflict"


def _first_year(value: Any) -> int | None:
    if isinstance(value, list):
        for item in value:
            result = _first_year(item)
            if result:
                return result
        return None
    match = _YEAR.search(str(value or ""))
    return int(match.group()) if match else None


def _crossref_authors(message: dict[str, Any]) -> list[str]:
    authors: list[str] = []
    for author in message.get("author") or []:
        name = " ".join(
            part.strip()
            for part in (str(author.get("given") or ""), str(author.get("family") or ""))
            if part.strip()
        )
        if name:
            authors.append(name)
    return authors


def _openalex_authors(work: dict[str, Any]) -> list[str]:
    return [
        str(item.get("author", {}).get("display_name") or "").strip()
        for item in work.get("authorships") or []
        if str(item.get("author", {}).get("display_name") or "").strip()
    ]


def _crossref_record(message: dict[str, Any]) -> dict[str, Any]:
    container = message.get("container-title") or []
    publisher = str(message.get("publisher") or "").strip() or None
    return {
        "title": str((message.get("title") or [""])[0]).strip(),
        "authors": _crossref_authors(message),
        "year": _first_year(
            message.get("published-print")
            or message.get("published-online")
            or message.get("issued")
        ),
        "journal": str(container[0]).strip() if container else None,
        "publisher": publisher,
        "volume": str(message.get("volume") or "").strip() or None,
        "issue": str(message.get("issue") or "").strip() or None,
        "doi": normalize_doi(message.get("DOI")),
        "document_type": str(message.get("type") or "unknown"),
    }


def _openalex_record(work: dict[str, Any]) -> dict[str, Any]:
    location = work.get("primary_location") or {}
    source = location.get("source") or {}
    return {
        "title": str(work.get("display_name") or work.get("title") or "").strip(),
        "authors": _openalex_authors(work),
        "year": _first_year(work.get("publication_year")),
        "journal": str(source.get("display_name") or "").strip() or None,
        "publisher": str(source.get("host_organization_name") or "").strip() or None,
        "volume": str((work.get("biblio") or {}).get("volume") or "").strip() or None,
        "issue": str((work.get("biblio") or {}).get("issue") or "").strip() or None,
        "doi": normalize_doi(work.get("doi")),
        "document_type": str(work.get("type") or "unknown"),
    }


@dataclass(slots=True)
class EnrichmentResult:
    values: dict[str, Any]
    verified: bool
    confidence: float
    match_source: str
    provenance: dict[str, str]
    warnings: list[str] = field(default_factory=list)


class MetadataResolver:
    """Resolve bibliographic metadata with strict, cached, auditable matching."""

    def __init__(self, settings: Any):
        configuration = settings.section("metadata")
        self.enabled = bool(configuration["enabled"])
        self.allow_network = bool(configuration["allow_network"])
        self.crossref_enabled = bool(configuration["crossref_enabled"])
        self.openalex_enabled = bool(configuration["openalex_enabled"])
        self.openalex_api_key = str(configuration.get("openalex_api_key") or "").strip()
        self._provider_blocked_until: dict[str, float] = {}
        self.timeout = float(configuration["request_timeout_seconds"])
        self.retries = int(configuration["maximum_retries"])
        self.delay = float(configuration["request_delay_seconds"])
        self.threshold = float(configuration["minimum_title_similarity"])
        self.ambiguity_margin = float(configuration["minimum_ambiguity_margin"])
        self.minimum_fragment_tokens = int(
            configuration.get("minimum_title_fragment_tokens", 6)
        )
        self.minimum_fragment_characters = int(
            configuration.get("minimum_title_fragment_characters", 40)
        )
        self.user_agent = str(configuration["user_agent"])
        self.contact_email = str(configuration.get("contact_email") or "").strip()
        self.filename_is_curated_title = bool(
            configuration.get("filename_is_curated_title", False)
        )
        self.cache_dir = settings.path("paths.metadata_cache")

    def _base_provenance(self, values: dict[str, Any]) -> dict[str, str]:
        return {
            key: (
                "curated_filename"
                if key == "title" and self.filename_is_curated_title
                else "pdf_native_or_title_page"
            )
            for key, value in values.items()
            if value
        }

    def _cached_json(self, provider: str, key: str, url: str) -> dict[str, Any] | None:
        cache_path = self.cache_dir / provider / f"{stable_digest(key, length=40)}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return payload.get("response")
        if not self.allow_network:
            return None

        state_path = self.cache_dir / provider / "rate_limit_state.json"
        billable_openalex_search = provider == "openalex" and key.startswith("title:")
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            blocked_until = float(state.get("blocked_until_epoch") or 0.0)
            if blocked_until > time.time():
                if state.get("reason") != "daily_budget_floor" or billable_openalex_search:
                    return None
        if self._provider_blocked_until.get(provider, 0.0) > time.time():
            return None

        headers = {"Accept": "application/json", "User-Agent": self.user_agent}
        for attempt in range(self.retries + 1):
            try:
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
                    response_headers = response.headers
                if provider == "openalex":
                    def number(name: str) -> float | None:
                        try:
                            value = response_headers.get(name)
                            return float(value) if value is not None else None
                        except (TypeError, ValueError):
                            return None
                    remaining = number("X-RateLimit-Remaining")
                    limit = number("X-RateLimit-Limit")
                    used = number("X-RateLimit-Credits-Used")
                    reset_seconds = number("X-RateLimit-Reset")
                    state = {
                        "provider": provider,
                        "observed_at": datetime.now(UTC).isoformat(),
                        "authenticated": bool(self.openalex_api_key),
                        "daily_limit": limit,
                        "daily_remaining": remaining,
                        "credits_used": used,
                        "reset_after_seconds": reset_seconds,
                    }
                    if (
                        billable_openalex_search
                        and remaining is not None
                        and remaining <= max(used or 0.0, 10.0)
                    ):
                        blocked_until = time.time() + max(reset_seconds or 60.0, 1.0)
                        state.update(
                            reason="daily_budget_floor",
                            blocked_until_epoch=blocked_until,
                        )
                    write_json_atomic(state_path, state)
                write_json_atomic(
                    cache_path,
                    {
                        "provider": provider,
                        "lookup_key": key,
                        "retrieved_at": datetime.now(UTC).isoformat(),
                        "response": result,
                    },
                )
                if self.delay:
                    time.sleep(self.delay)
                return result
            except urllib.error.HTTPError as exc:
                if provider == "openalex" and self.openalex_api_key and exc.code in {401, 403}:
                    write_json_atomic(
                        state_path,
                        {
                            "provider": provider,
                            "observed_at": datetime.now(UTC).isoformat(),
                            "authenticated": False,
                            "reason": "project_key_rejected",
                            "http_status": exc.code,
                        },
                    )
                    return None
                if exc.code == 429:
                    try:
                        retry_after = max(float(exc.headers.get("Retry-After", "60")), 1.0)
                    except (TypeError, ValueError):
                        retry_after = 60.0
                    blocked_until = time.time() + retry_after
                    self._provider_blocked_until[provider] = blocked_until
                    write_json_atomic(
                        state_path,
                        {
                            "provider": provider,
                            "observed_at": datetime.now(UTC).isoformat(),
                            "retry_after_seconds": retry_after,
                            "blocked_until_epoch": blocked_until,
                            "reason": "provider_429",
                            "authenticated": bool(
                                provider == "openalex" and self.openalex_api_key
                            ),
                        },
                    )
                    return None
                if attempt < self.retries:
                    time.sleep(min(2**attempt, 4))
            except (OSError, ValueError, urllib.error.URLError):
                if attempt < self.retries:
                    time.sleep(min(2**attempt, 4))
        return None

    def _crossref_doi(self, doi: str) -> dict[str, Any] | None:
        if not self.crossref_enabled:
            return None
        encoded = urllib.parse.quote(doi, safe="")
        payload = self._cached_json(
            "crossref", f"doi:{doi}", f"https://api.crossref.org/works/{encoded}"
        )
        message = (payload or {}).get("message")
        return _crossref_record(message) if isinstance(message, dict) else None

    def _openalex_doi(self, doi: str) -> dict[str, Any] | None:
        if not self.openalex_enabled:
            return None
        identifier = urllib.parse.quote(doi, safe="")
        query = f"https://api.openalex.org/works/https://doi.org/{identifier}"
        params: dict[str, str] = {}
        if self.contact_email:
            params["mailto"] = self.contact_email
        if self.openalex_api_key:
            params["api_key"] = self.openalex_api_key
        if params:
            query += "?" + urllib.parse.urlencode(params)
        payload = self._cached_json("openalex", f"doi:{doi}", query)
        return _openalex_record(payload) if isinstance(payload, dict) else None

    def _crossref_title(self, title: str) -> list[dict[str, Any]]:
        if not self.crossref_enabled:
            return []
        params = urllib.parse.urlencode({"query.title": title, "rows": 3})
        payload = self._cached_json(
            "crossref", f"title:{normalized_title(title)}", f"https://api.crossref.org/works?{params}"
        )
        items = ((payload or {}).get("message") or {}).get("items") or []
        return [_crossref_record(item) for item in items if isinstance(item, dict)]

    def _openalex_title(self, title: str) -> list[dict[str, Any]]:
        if not self.openalex_enabled:
            return []
        params: dict[str, str | int] = {"search": title, "per-page": 3}
        if self.contact_email:
            params["mailto"] = self.contact_email
        if self.openalex_api_key:
            params["api_key"] = self.openalex_api_key
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
        payload = self._cached_json("openalex", f"title:{normalized_title(title)}", url)
        return [
            _openalex_record(item)
            for item in (payload or {}).get("results") or []
            if isinstance(item, dict)
        ]

    def _strict_title_match(
        self, title: str, year: int | None, candidates: list[tuple[str, dict[str, Any]]]
    ) -> tuple[str, dict[str, Any], float, str] | None:
        ranked: list[tuple[float, str, dict[str, Any], str]] = []
        for provider, record in candidates:
            score = title_similarity(title, record.get("title"))
            relation = title_relation(
                title,
                record.get("title"),
                self.threshold,
                self.minimum_fragment_tokens,
                self.minimum_fragment_characters,
            )
            if relation == "conflict":
                continue
            candidate_year = record.get("year")
            if year and candidate_year and abs(int(year) - int(candidate_year)) > 1:
                continue
            ranked.append((score, provider, record, relation))
        ranked.sort(key=lambda item: item[0], reverse=True)
        if not ranked:
            return None
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < self.ambiguity_margin:
            first_doi = normalize_doi(ranked[0][2].get("doi"))
            second_doi = normalize_doi(ranked[1][2].get("doi"))
            if first_doi != second_doi:
                return None
        score, provider, record, relation = ranked[0]
        return provider, record, score, relation

    def _merge(
        self, base: dict[str, Any], resolved: dict[str, Any], source: str
    ) -> tuple[dict[str, Any], dict[str, str]]:
        values = dict(base)
        provenance = self._base_provenance(base)
        for key, value in resolved.items():
            if value not in (None, "", []):
                values[key] = value
                provenance[key] = source
        return values, provenance

    def enrich(
        self,
        base: dict[str, Any],
        first_pages_text: str,
        override: dict[str, Any],
    ) -> EnrichmentResult:
        working = dict(base)
        text_doi = normalize_doi(first_pages_text)
        if text_doi and not working.get("doi"):
            working["doi"] = text_doi
        if not working.get("year"):
            working["year"] = _first_year(first_pages_text)

        if override:
            for key, value in override.items():
                if key not in {"sha256", "source_file", "verified"} and value not in (None, ""):
                    working[key] = value
            if bool(override.get("verified")):
                if not working.get("authors") and working.get("publisher"):
                    working["authors"] = [str(working["publisher"])]
                required = working.get("title") and working.get("authors") and working.get("year")
                if not required:
                    raise ValueError(
                        "A verified metadata override requires title, year, and either authors "
                        "or an issuing organization in publisher"
                    )
                provenance = {
                    key: "manual_verified_override" for key, value in working.items() if value
                }
                return EnrichmentResult(
                    values=working,
                    verified=True,
                    confidence=1.0,
                    match_source="manual_verified_override",
                    provenance=provenance,
                )

        if not self.enabled:
            return EnrichmentResult(
                values=working,
                verified=False,
                confidence=0.35,
                match_source="enrichment_disabled",
                provenance=self._base_provenance(working),
                warnings=["metadata_enrichment_disabled"],
            )

        title = str(working.get("title") or "").strip()
        resolution_warnings: list[str] = []
        doi = normalize_doi(working.get("doi"))
        if doi:
            for provider, lookup in (
                ("crossref_doi", self._crossref_doi),
                ("openalex_doi", self._openalex_doi),
            ):
                record = lookup(doi)
                if record and normalize_doi(record.get("doi")) == doi:
                    score = title_similarity(title, record.get("title"))
                    relation = title_relation(
                        title,
                        record.get("title"),
                        self.threshold,
                        self.minimum_fragment_tokens,
                        self.minimum_fragment_characters,
                    )
                    if relation == "conflict":
                        resolution_warnings.append(
                            f"doi_title_mismatch:{provider}:{score:.4f}"
                        )
                        continue
                    if relation == "verified_truncation":
                        resolution_warnings.append(
                            f"verified_filename_title_truncation:{provider}"
                        )
                    values, provenance = self._merge(working, record, provider)
                    return EnrichmentResult(
                        values=values,
                        verified=True,
                        confidence=(0.97 if relation == "verified_truncation" else 0.99),
                        match_source=provider,
                        provenance=provenance,
                        warnings=resolution_warnings,
                    )

        if title:
            candidates = [
                ("crossref_title", record) for record in self._crossref_title(title)
            ] + [("openalex_title", record) for record in self._openalex_title(title)]
            matched = self._strict_title_match(title, working.get("year"), candidates)
            if matched:
                provider, record, score, relation = matched
                if relation == "verified_truncation":
                    resolution_warnings.append(
                        f"verified_filename_title_truncation:{provider}"
                    )
                values, provenance = self._merge(working, record, provider)
                return EnrichmentResult(
                    values=values,
                    verified=True,
                    confidence=(
                        0.95 if relation == "verified_truncation" else round(score, 4)
                    ),
                    match_source=provider,
                    provenance=provenance,
                    warnings=resolution_warnings,
                )

        return EnrichmentResult(
            values=working,
            verified=False,
            confidence=0.35,
            match_source="unresolved",
            provenance=self._base_provenance(working),
            warnings=[*resolution_warnings, "metadata_review_required"],
        )
