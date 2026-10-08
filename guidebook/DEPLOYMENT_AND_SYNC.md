# Canonical deployment and synchronization

Synchronized from the Mini-run SFT and Full_RAG/Pipeline code on 2026-10-08. Deployment paths are portable; runtime release manifests and credentials are not exported. Historical absolute Spark paths in policy examples describe that deployment, not mandatory checkout locations.

## Full-corpus ingestion

Use Linux with the declared Python dependencies, Poppler (`pdftotext`, `pdftoppm`) for metadata evidence/rendering, and a configured CUDA Docling/Granite runtime. Put PDFs under `Full_RAG/Full_Corpus` or customize `config/settings_full_rag.toml` and `GIIS_FULL_RAG_ROOT` consistently. Install the protected OpenAlex key with `python scripts/install_openalex_key.py` or supply its environment variable.

The configured metadata vision server is separate from SFT serving: port 8820, served identity `qwen122b-vision-metadata`, Qwen-122B vision-capable configuration. Its model identity is checked before local adjudication. Provision that server separately and retain deployment memory guards; do not co-load it with SFT models on a memory-constrained Spark. Missing or mismatched vision service cannot qualify metadata and produces a referral for authorized gold review.

```bash
bash scripts/run_full_rag_pipeline.sh continuous
# One diagnostic cycle instead:
bash scripts/run_full_rag_pipeline.sh once
```

Each cycle promotes completed source-bound adjudications, parses unique documents, materializes accepted passages and builds matching dense/lexical indexes. OpenAlex deferrals never stop the first scan; renewed allowance processes older backlog first. Gold referral is not automatic gold review. The separate historical staged-release publisher on Spark is not a reusable activation command and is not shipped here.

## V17-Turbo generation and review

Customize model identifiers, protected environment-file locations, corpus/index paths and tokenizer snapshots. Keep Qwen generation (8800) and OSS review (8600) mutually exclusive. The launchers retain model/profile checks and 100-GiB available-memory startup gates. Activate a Linux environment with vLLM, tmux, curl and ss available.

Refresh checkout bindings after customization, then freeze a fresh plan:

```bash
python scripts/refresh_v17_turbo_execution_binding.py
python -m giis_prod.v17_turbo_canary plan --out artifacts/v17_turbo/my_main_run
bash scripts/run_v17_turbo_pre_gold_pipeline.sh artifacts/v17_turbo/my_main_run artifacts/v17_turbo/my_heldout_run
```

Required corpus selections, partitions and index bindings must exist before planning. The launcher runs main generation with deterministic verification and bounded retries, OSS across every training/validation lane, then Qwen authoring across all held-out lanes. It stops at the joint hash-bound gold authorization checkpoint. Confident main semantic rejects enter blinded gold review; every generated held-out reference enters direct gold review. Exhausted Qwen/OSS failures remain terminal quarantine, and qualified survivors continue.

Configuration edits require fresh plans. Binding refresh never changes historical plans, authorizes gold review, activates an index, releases a dataset, or starts training. After actual operator authorization, use the pre-gold and semantic-finalization modules described in the canonical protocol.
