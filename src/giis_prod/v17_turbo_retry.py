"""Persisted one-retry authoring requests; terminal failures cannot reopen."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

class TerminalGenerationFailure(RuntimeError): pass

def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True)+"\n")
    temp.replace(path)

def author_request(directory, identity, call):
    directory=Path(directory)
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    path=directory/"attempts"/(key+".json")
    bundle_key=hashlib.sha256(json.dumps({k:identity[k] for k in ("partition","bundle_id")},sort_keys=True).encode()).hexdigest()
    terminal=directory/"terminal_failures"/(bundle_key+".json")
    if terminal.exists(): raise TerminalGenerationFailure("Author fail-closed request exhausted; no repair or resumed retry")
    state=json.loads(path.read_text()) if path.exists() else {"identity":identity,"attempts":0,"status":"new"}
    if state["identity"]!=identity: raise ValueError("Author request identity drift")
    if state["status"]=="complete": return state["result"]
    while state["attempts"]<2:
        state.update(attempts=state["attempts"]+1,status="in_progress")
        atomic(path,state)
        try:
            result=call()
        except Exception as exc:
            state.update(status="failed",error=f"{type(exc).__name__}: {exc}")
            atomic(path,state)
        else:
            state.update(status="complete",result=result)
            atomic(path,state)
            return result
    atomic(terminal,{"identity":identity,"attempts":2,"status":"terminal_quarantine",
                     "error":state.get("error","Interrupted request budget exhausted"),
                     "retry_allowed":False,"repair_allowed":False,"semantic_review_allowed":False})
    raise TerminalGenerationFailure("Author request failed after initial attempt plus one retry")
