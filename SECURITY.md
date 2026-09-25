# Security

Do not commit API keys, access tokens, passwords, model-server keys, private corpora, or generated review packets.

Use environment variables or a protected environment file outside the repository. The default OpenAlex credential path is `~/.config/giis/openalex.env`; it can be changed with `GIIS_OPENALEX_ENV_FILE`. The GPT-OSS reviewer key file can be changed with `GIIS_OSS_REVIEWER_KEY_FILE`.

Before publishing changes, scan the complete Git history as well as the working tree. If a credential is ever committed, rotate it and remove it from history rather than relying on a later deletion commit.

