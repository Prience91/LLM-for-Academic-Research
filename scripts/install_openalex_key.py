#!/usr/bin/env python3
from __future__ import annotations

import getpass
import json
import os
from pathlib import Path
import tempfile
import urllib.error
import urllib.request

TARGET = Path(os.getenv("GIIS_OPENALEX_ENV", str(Path.home() / ".config/giis/openalex.env")))
VARIABLE = "GIIS__METADATA__OPENALEX_API_KEY"
ENDPOINT = "https://api.openalex.org/rate-limit"

key = getpass.getpass("Rotated OpenAlex API key (input hidden): ").strip()
if not key or any(character in key for character in "\r\n\x00"):
    raise SystemExit("A non-empty single-line key is required")

request = urllib.request.Request(
    ENDPOINT,
    headers={
        "Accept": "application/json",
        "Authorization": "Bearer " + key,
        "User-Agent": "GIIS-Metadata-Enrichment/2.0",
    },
)
try:
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
        headers = response.headers
except urllib.error.HTTPError as exc:
    raise SystemExit(f"OpenAlex rejected the key with HTTP {exc.code}; credential unchanged")
except (OSError, ValueError, urllib.error.URLError) as exc:
    raise SystemExit(f"OpenAlex validation failed; credential unchanged: {type(exc).__name__}")

TARGET.parent.mkdir(parents=True, exist_ok=True)
fd, temporary_name = tempfile.mkstemp(prefix=".openalex.", suffix=".env", dir=TARGET.parent)
temporary = Path(temporary_name)
try:
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(f"{VARIABLE}={key}\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(TARGET)
    TARGET.chmod(0o600)
finally:
    if temporary.exists():
        temporary.unlink()

safe = {
    "installed": True,
    "credential_file": str(TARGET),
    "mode": oct(TARGET.stat().st_mode & 0o777),
    "daily_limit": headers.get("X-RateLimit-Limit"),
    "daily_remaining": headers.get("X-RateLimit-Remaining"),
    "reset_after_seconds": headers.get("X-RateLimit-Reset"),
    "response_keys": sorted(payload) if isinstance(payload, dict) else [],
}
print(json.dumps(safe, indent=2))
