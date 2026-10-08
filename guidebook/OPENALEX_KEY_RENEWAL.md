# OpenAlex key renewal for Full_RAG

The active project credential is held only in `$HOME/.config/giis/openalex.env` (or `GIIS_OPENALEX_ENV`) (mode 600) as `GIIS__METADATA__OPENALEX_API_KEY`. Full_RAG OpenAlex workers source that file when they start; the credential is not stored in this project or its artifacts.

To renew it, run this single command from a terminal and type the new key at the hidden prompt:

```bash
python scripts/install_openalex_key.py
```

The installer validates the key against the OpenAlex rate-limit endpoint before replacing the active env file atomically. It prints remaining daily credits without printing the key. A key change does not affect already running workers until they are restarted from their saved ledgers. If an old key left a provider-429 state in the shared metadata cache, archive that state only after confirming the new key has available credits; then restart the OpenAlex workers. A rotation on the same account retains that account's daily budget.
