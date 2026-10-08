#!/usr/bin/env python3
import concurrent.futures
import hashlib
import json
import pathlib
import threading
import time
import urllib.request
from collections import Counter, defaultdict

ROOT = pathlib.Path('/home/codeaccounting/Documents/Mini run')
OUT = ROOT / 'artifacts/v17_turbo/heldout_authoring/oss120b_single_source_review'
PACKETS = ROOT / 'artifacts/v17_turbo/heldout_authoring/review_packets_oss_single_source.jsonl'
RESULTS = OUT / 'axis_results.jsonl'
SUMMARY = OUT / 'summary.json'
MODEL = 'gpt-oss-120b-reviewer'
URL = 'http://127.0.0.1:8600/v1/chat/completions'
KEY_PATH = pathlib.Path('/tmp/giis_gptoss_reviewer_api_key')
CONCURRENCY = 1
TIMEOUT_SECONDS = 300
write_lock = threading.Lock()

COMMON = '''Treat the supplied scientific evidence as a closed world. The teacher ledger is fallible. Inspect question, evidence, and answer directly. Topic overlap is insufficient; absent reporting is not a negative finding; association is not causation; reference numbers are not empirical counts; respect source ownership and table boundaries. Reason privately and return only the requested compact JSON.'''

COVERAGE = COMMON + '''
You are the COVERAGE auditor. Judge what the CANDIDATE ANSWER actually states, not what its citations or the supplied evidence contain. Evidence availability never proves answer coverage.

First split every question and teacher-ledger requirement into atomic required components, especially clauses after words such as including, and, compare, outcomes, percentages, steps, dimensions, or qualifications. Then match each component to an exact phrase in the candidate answer. A citation marker alone is not a match. For tables, inspect every requested field and cell category. Add obligations the ledger missed.

Severity: minor_omission means only a secondary detail or supporting example is absent while every principal requested component and conclusion remains. materially_incomplete means a requested component, comparison dimension, primary outcome, table field, conclusion, or necessary qualification is absent. When a question asks for dominant or high-level findings, omitted additional examples are minor if principal categories and conclusions are present.

Return exactly: {"coverage":"complete|minor_omission|materially_incomplete","obligation_checks":[{"id":"ledger id or NEW1","required_components":1,"covered_components":1,"answer_anchor":"exact short candidate phrase or absent","missing_components":["short missing item"]}],"ledger_gap":"short text or none","confidence":0.0}. Include every teacher-ledger ID exactly once. complete is allowed only when covered_components equals required_components for every obligation and all missing_components arrays are empty. Keep phrases short.'''

SUPPORT = COMMON + '''\nYou are the ADVERSARIAL SUPPORT auditor. Attempt to falsify the answer by atomizing all material claims and locating exact evidence support. Scrutinize numbers, signs, counts, comparisons, categories, populations, places, methods, timeframes, causal language, uncertainty, and especially evaluative or comparative words such as robust, effective, structured, superior, comprehensive, or stronger. A plausible synthesis is unsupported when the evidence does not establish its exact characterization. Missing support for any material claim yields unsupported.
Return exactly: {"support":"supported|unsupported","unsupported_claims":[{"claim":"exact concise claim","evidence":"contradicting marker/span or absent"}],"risk_terms_checked":["at most 5 short terms"],"confidence":0.0}. Empty unsupported_claims is allowed only after checking every material claim.'''

def read_jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]

def extract_json(text):
    a,b=text.find('{'),text.rfind('}')
    if a<0 or b<a: raise ValueError('no JSON object')
    return json.loads(text[a:b+1])

def validate(axis, review, packet):
    if axis=='coverage':
        if review.get('coverage') not in {'complete','minor_omission','materially_incomplete'}: raise ValueError('invalid coverage')
        checks=review.get('obligation_checks')
        if not isinstance(checks,list) or not checks: raise ValueError('missing obligation checks')
        expected={str(x.get('obligation_id')) for x in packet.get('teacher_declared_obligations',[]) if x.get('obligation_id')}
        got=[str(x.get('id')) for x in checks if x.get('id')]
        if not expected.issubset(set(got)) or len(got) != len(set(got)): raise ValueError('ledger obligation missing or duplicated')
        for check in checks:
            req=check.get('required_components'); cov=check.get('covered_components'); missing=check.get('missing_components')
            if not isinstance(req,int) or not isinstance(cov,int) or req < 1 or cov < 0 or cov > req or not isinstance(missing,list): raise ValueError('invalid component accounting')
        if review.get('coverage')=='complete' and any(x['covered_components'] != x['required_components'] or x['missing_components'] for x in checks): raise ValueError('complete contradicts component accounting')
    else:
        if review.get('support') not in {'supported','unsupported'}: raise ValueError('invalid support')
        claims=review.get('unsupported_claims')
        if not isinstance(claims,list): raise ValueError('invalid unsupported claims')
        if review['support']=='unsupported' and not claims: raise ValueError('unsupported without claim')

def user_text(p):
    return ('QUESTION\n'+p['question']+'\n\nNUMBERED EVIDENCE\n'+p['evidence']+'\n\nCANDIDATE ANSWER\n'+p['answer']+'\n\nTEACHER LEDGER (fallible)\n'+json.dumps(p.get('teacher_declared_obligations',[]),ensure_ascii=False))

def review_axis(task):
    p,axis=task; started=time.time(); system=COVERAGE if axis=='coverage' else SUPPORT; max_tokens=2048 if axis=='coverage' else 1536
    rec={'review_id':p['review_id'],'candidate_sha256':p['candidate_sha256'],'lane':p['lane'],'axis':axis,'started_unix':started}
    body=json.dumps({'model':MODEL,'messages':[{'role':'system','content':system},{'role':'user','content':user_text(p)}],'temperature':0.0,'top_p':1.0,'max_tokens':max_tokens,'chat_template_kwargs':{'reasoning_effort':'low'},'response_format':{'type':'json_object'}}).encode()
    try:
        key=KEY_PATH.read_text().strip(); req=urllib.request.Request(URL,data=body,headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
        with urllib.request.urlopen(req,timeout=TIMEOUT_SECONDS) as response: raw=json.loads(response.read().decode())
        choice=raw['choices'][0]
        if choice.get('finish_reason')=='length': raise ValueError('Reviewer output truncated')
        content=choice['message'].get('content') or ''; review=extract_json(content); validate(axis,review,p)
        rec.update(status='ok',review=review,raw_content=content,finish_reason=choice.get('finish_reason'),usage=raw.get('usage'),reasoning_present=bool(choice['message'].get('reasoning_content') or choice['message'].get('reasoning')))
    except Exception as exc: rec.update(status='error',error=f'{type(exc).__name__}: {exc}')
    rec['elapsed_seconds']=round(time.time()-started,3)
    with write_lock:
        with RESULTS.open('a',encoding='utf-8') as f: f.write(json.dumps(rec,ensure_ascii=False,sort_keys=True)+'\n')
        metric=rec.get('review',{}).get('coverage') if axis=='coverage' else rec.get('review',{}).get('support')
        print(json.dumps({'review_id':p['review_id'],'axis':axis,'status':rec['status'],'decision':metric,'seconds':rec['elapsed_seconds'],'tokens':(rec.get('usage') or {}).get('completion_tokens')}),flush=True)
    return rec

def run():
    packets=read_jsonl(PACKETS); done={(r['review_id'],r['axis']) for r in read_jsonl(RESULTS)} if RESULTS.exists() else set()
    tasks=[(p,a) for p in packets for a in ('coverage','support') if (p['review_id'],a) not in done]
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool: list(pool.map(review_axis,tasks))

def score():
    packets=read_jsonl(PACKETS)
    axes={(r['review_id'],r['axis']):r for r in read_jsonl(RESULTS)} if RESULTS.exists() else {}
    cases=[]; counts=Counter()
    for p in packets:
        rid=p['review_id']; c=axes.get((rid,'coverage')); s=axes.get((rid,'support'))
        if not c or not s or c.get('status')!='ok' or s.get('status')!='ok':
            verdict='error_fail_closed'
        else:
            cov=c['review']['coverage']; sup=s['review']['support']
            verdict='unsupported' if sup=='unsupported' else cov
        disposition='eligible_for_operator_release' if verdict=='complete' else 'quarantine'
        counts[verdict]+=1
        cases.append({'review_id':rid,'candidate_sha256':p['candidate_sha256'],'lane':p['lane'],'verdict':verdict,'disposition':disposition,'coverage':c.get('review',{}).get('coverage') if c else None,'support':s.get('review',{}).get('support') if s else None})
    summary={'schema':'giis_v17_turbo_heldout_oss_review_v1','protocol':'gpt_oss_120b_component_mapped_dual_axis','packet_sha256':hashlib.sha256(PACKETS.read_bytes()).hexdigest(),'total':len(packets),'counts':dict(counts),'cases':cases,'single_source_gold_review_fraction':0.0,'training_admission_effect':'none','benchmark_freeze':False}
    SUMMARY.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='cases'},indent=2))

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    if not PACKETS.exists():
        raise FileNotFoundError(PACKETS)
    run()
    score()
