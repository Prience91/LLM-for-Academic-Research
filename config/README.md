# Configuration

`settings.toml`, `settings_v17_turbo.toml` and `settings.example.toml` contain the synchronized portable V17-Turbo settings. `settings_full_rag.toml` configures the independent full-corpus ingestion path. Customize deployment paths before creating a new frozen plan; never edit a running plan's inputs.

Paths resolve relative to the repository root. For a shared external corpus, set the ingestion path values consistently and export `GIIS_FULL_RAG_ROOT` for the full-corpus runner. Set `GIIS_ROOT` for orchestration when necessary. Add only your own source-bound completed ledgers to `metadata_adjudication.import_ledgers`; historical Spark migration ledgers are not dependencies.

Secrets are environment variables, including `GIIS__METADATA__OPENALEX_API_KEY`, `TEACHER_API_KEY` and `VLLM_API_KEY`. Protected environment files can be selected with `GIIS_OPENALEX_ENV` and `GIIS_TEACHER_ENV`. Never commit real credentials.

After configuration or bound code changes, run `python scripts/refresh_v17_turbo_execution_binding.py`, then create a fresh frozen plan. This refresh does not authorize a review or release. See the deployment guide for the full workflow.
