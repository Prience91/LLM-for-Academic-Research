"""Source-bound English Qwen adjudication and authorization-gated gold referrals."""
from __future__ import annotations
import argparse,base64,hashlib,json,os,pathlib,re,subprocess,time,urllib.request,fcntl
from datetime import datetime,UTC
from .metadata import EnrichmentResult
FIELDS=('title','authors','year','publisher','document_type','doi','journal','volume','issue','theme')
def now():return datetime.now(UTC).isoformat()
def digest(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def read(path):
 path=pathlib.Path(path)
 if not path.exists():return []
 content=path.read_text();lines=content.splitlines();out=[]
 for i,line in enumerate(lines):
  if not line.strip():continue
  try:out.append(json.loads(line))
  except json.JSONDecodeError:
   if i==len(lines)-1 and not content.endswith('\n'):break
   raise
 return out
def atomic(path,value):
 path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.new')
 with tmp.open('w') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def write_rows(path,rows):
 path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.new')
 with tmp.open('w') as f:
  for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
  f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def complete(m):
 return isinstance(m,dict) and all(isinstance(m.get(k),str) and m[k].strip() for k in ('title','publisher','document_type')) and isinstance(m.get('authors'),list) and bool(m['authors']) and all(isinstance(x,str) and x.strip() for x in m['authors']) and isinstance(m.get('year'),int) and not isinstance(m['year'],bool) and m['year']>0
class MetadataAdjudicator:
 def __init__(self,settings):
  self.settings=settings;self.config=dict(settings.data.get('metadata_adjudication',{}))
  def project_path(value):
   path=pathlib.Path(value)
   return str(path if path.is_absolute() else settings.root/path)
  for key in ('artifact_dir','prompt_file','selection_manifest'):
   if self.config.get(key):self.config[key]=project_path(self.config[key])
  self.config['import_ledgers']=[project_path(value) for value in self.config.get('import_ledgers',[])]
  self.enabled=bool(self.config.get('enabled',False));self.root=pathlib.Path(self.config.get('artifact_dir',str(settings.path('paths.metadata_overrides').parent/'adjudication')));self._external=None
 def external(self):
  if self._external is None:
   self._external={}
   for path in self.config.get('import_ledgers',[]):
    for row in read(path):
     key=row.get('sha256') or row.get('source_sha256')
     if pathlib.Path(path)==self.root/'gold_review_ledger.jsonl':
      row=dict(row,_gold_authorized=bool(row.get('authorization_reference') and row.get('reviewer') and row.get('reviewed_at')))
     if key:self._external[key]=row
  return self._external
 @staticmethod
 def non_english(text):
  letters=sum(c.isalpha() for c in text);dev=sum('\u0900'<=c<='\u097f' for c in text)
  return dev>50 and dev/max(letters,1)>0.15
 def source(self,path):
  root=self.settings.path('paths.pdf_dir').resolve();source=pathlib.Path(path).resolve();return source.relative_to(root).as_posix()
 def referral(self,path,key,base,reason,**extra):
  row=dict(sha256=key,source_file=self.source(path),reviewed_at=now(),status='needs_gold_review',routing='gold_review',gold_review_status='awaiting_user_authorization',reason=reason,**extra)
  atomic(self.root/'gold_review_queue'/f'{key}.json',row);atomic(self.root/'decisions'/f'{key}.json',row)
  return EnrichmentResult(values=dict(base),verified=False,confidence=0.0,match_source='awaiting_gold_metadata_adjudication',provenance={},warnings=['gold_review_requires_user_authorization',reason])
 def decision_result(self,row,base):
  prediction=row.get('prediction',{});m=prediction.get('metadata') or row.get('metadata') or {k:row[k] for k in FIELDS if k in row}
  qualified=row.get('status') in ('local_model_qualified','gold_qualified','qualified') or row.get('verified') is True
  if row.get('_gold_authorized') is False:return None
  if not qualified or not complete(m):return None
  if row.get('status')=='local_model_qualified' and str(prediction.get('language','')).lower() not in ('en','eng','english'):return None
  method='local_model_metadata_adjudication' if row.get('status')=='local_model_qualified' else str(row.get('verification_method') or 'manual_verified_override')
  values=dict(base);values.update({k:m.get(k) for k in FIELDS if k in m});values['doi']=m.get('doi')
  return EnrichmentResult(values=values,verified=True,confidence=0.0 if method=='local_model_metadata_adjudication' else 1.0,match_source=method,provenance={k:method for k,v in values.items() if v},warnings=['local_model_adjudication_not_gold_verified'] if method=='local_model_metadata_adjudication' else [])
 def adjudicate(self,path,key,base,first_pages_text):
  if not self.enabled:return None
  if digest(path)!=key:raise ValueError('Adjudication source hash mismatch')
  if self.non_english(first_pages_text):return self.referral(path,key,base,'Non-English/Devanagari document reserved for gold review',language='ne_or_other_devanagari')
  cached_path=self.root/'decisions'/f'{key}.json';row=json.loads(cached_path.read_text()) if cached_path.exists() else self.external().get(key)
  if row:
   accepted=self.decision_result(row,base)
   if accepted:return accepted
   return self.referral(path,key,base,'Prior local adjudication unresolved or failed; reserved for gold review',prior_decision=row)
  record=dict(sha256=key,source_file=self.source(path),reviewed_at=now(),reviewer=self.config['model'],status='needs_gold_review')
  try:
   text=subprocess.run(['pdftotext','-f','1','-l','12','-layout',str(path),'-'],capture_output=True,text=True,timeout=60)
   if text.returncode:raise ValueError('PDF evidence extraction failed')
   t='\n'.join(re.sub(r'[ \t]+',' ',line).strip() for line in text.stdout.splitlines());t=t[:int(self.config.get('max_source_text_characters',33000))]
   image=self.root/'images'/f'{key}.png';image.parent.mkdir(parents=True,exist_ok=True)
   if not image.exists():
    out=subprocess.run(['pdftoppm','-f','1','-l','1','-scale-to','1600','-singlefile','-png',str(path),str(image.with_suffix(''))],capture_output=True,timeout=60)
    if out.returncode:raise ValueError('PDF page rendering failed')
   prompt=pathlib.Path(self.config['prompt_file']).read_text();endpoint=self.config['endpoint'].rstrip('/')
   with urllib.request.urlopen(endpoint+'/v1/models',timeout=10) as r:models=json.load(r)
   if self.config['model'] not in {x['id'] for x in models['data']}:raise ValueError('Endpoint model identity mismatch')
   body=dict(model=self.config['model'],messages=[dict(role='system',content=prompt),dict(role='user',content=[dict(type='text',text='SOURCE EXTRACT (opening pages; may be incomplete):\n'+t),dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(image.read_bytes()).decode()))])],temperature=0,max_tokens=int(self.config.get('max_tokens',1200)),chat_template_kwargs=dict(enable_thinking=False))
   record.update(source_text_sha256=hashlib.sha256(t.encode()).hexdigest(),page_image_sha256=digest(image),request_sha256=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest())
   req=urllib.request.Request(endpoint+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
   with urllib.request.urlopen(req,timeout=int(self.config.get('timeout_seconds',180))) as response:raw=json.load(response)
   atomic(self.root/'responses'/f'{key}.json',raw);choice=raw['choices'][0]
   if choice['finish_reason']!='stop':raise ValueError('Response did not finish naturally')
   prediction=json.loads(re.sub(r'^```(?:json)?\s*|\s*```$','',choice['message']['content'].strip()));record['prediction']=prediction
   if prediction.get('decision') not in ('qualified','needs_gold_review'):raise ValueError('Invalid adjudication decision')
   if str(prediction.get('language','')).lower() not in ('en','eng','english'):return self.referral(path,key,base,'Non-English or uncertain document language',prediction=prediction)
   if prediction['decision']!='qualified' or not complete(prediction.get('metadata')) or prediction.get('unresolved_fields'):return self.referral(path,key,base,'Incomplete or unresolved Qwen adjudication',prediction=prediction)
   record['status']='local_model_qualified';atomic(self.root/'decisions'/f'{key}.json',record);atomic(self.root/'qualified'/f'{key}.json',record);return self.decision_result(record,base)
  except Exception as error:return self.referral(path,key,base,'Local adjudication failure; gold review required',error=str(error))
def promote(settings):
 a=MetadataAdjudicator(settings);dest=settings.path('paths.metadata_overrides');dest.parent.mkdir(parents=True,exist_ok=True)
 with dest.with_suffix('.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);existing={r['sha256']:r for r in read(dest)};rejection_path=dest.parent/'gold_rejected_manual_human_review_only.jsonl';rejection_rows=read(rejection_path);rejected={r['sha256'] for r in rejection_rows};root=settings.path('paths.pdf_dir').resolve();accepted={};referred={};skipped=[]
  source_rows=list(a.external().values())
  for folder in ['decisions']:
   for f in sorted((a.root/folder).glob('*.json')):source_rows.append(json.loads(f.read_text()))
  for row in source_rows:
   key=row.get('sha256') or row.get('source_sha256');rel=row.get('source_file')
   if not key or not rel:skipped.append('missing_source_identity');continue
   if key in rejected:skipped.append('gold_rejected:'+key);continue
   source=(root/rel).resolve()
   if not source.is_relative_to(root) or not source.is_file() or digest(source)!=key:skipped.append('source_hash_mismatch:'+str(key));continue
   if row.get('status')=='gold_rejected' and row.get('_gold_authorized') is True:
    if key not in rejected:
     rejection_rows.append(dict(row,recovery_policy='manual_human_review_only',automatic_retry_allowed=False,automatic_promotion_allowed=False));rejected.add(key)
    existing.pop(key,None)
   if key in rejected:skipped.append('gold_rejected:'+key);continue
   m=row.get('prediction',{}).get('metadata') or row.get('metadata') or {k:row[k] for k in FIELDS if k in row}
   qwen=row.get('status')=='local_model_qualified' or row.get('verification_method')=='local_model_metadata_adjudication'
   verified=(row.get('status') in ('gold_qualified','local_model_qualified','qualified') or row.get('verified') is True) and row.get('_gold_authorized') is not False
   if qwen and str(row.get('prediction',{}).get('language','')).lower() not in ('en','eng','english'):verified=False
   if not verified or not complete(m):referred[key]=dict(row,gold_review_status='awaiting_user_authorization');continue
   method='local_model_metadata_adjudication' if qwen else str(row.get('verification_method') or 'manual_verified_override')
   proof=hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False).encode()).hexdigest();candidate=dict(sha256=key,source_file=rel,**{k:m.get(k) for k in FIELDS if k in m and k!='doi'},doi=m.get('doi'),verified=True,verification_method=method,reviewer=row.get('reviewer',a.config.get('model') if qwen else 'source_corroboration'),evidence_manifest_sha256=proof,clear_unverified_doi=m.get('doi') is None,adjudicated_at=row.get('reviewed_at',now()))
   old=existing.get(key)
   if old and old.get('verified') and old.get('verification_method')=='manual_verified_override' and qwen:candidate=old
   existing[key]=candidate;accepted[key]=dict(sha256=key,source_file=rel,verification_method=candidate['verification_method'])
  for key in rejected:existing.pop(key,None);accepted.pop(key,None)
  write_rows(rejection_path,rejection_rows)
  write_rows(dest,[existing[k] for k in sorted(existing)])
  selection_path=a.config.get('selection_manifest')
  if selection_path:
   selected={r['sha256']:r for r in read(selection_path)}
   selected.update(accepted)
   for key in rejected:selected.pop(key,None)
   write_rows(selection_path,[selected[k] for k in sorted(selected)])
  write_rows(a.root/'ready_for_parsing.jsonl',[accepted[k] for k in sorted(accepted)]);write_rows(a.root/'gold_review_queue.jsonl',[referred[k] for k in sorted(referred) if k not in accepted]);summary=dict(updated_at=now(),qualified_promoted=len(accepted),gold_review_queued=len(set(referred)-set(accepted)),excluded_or_invalid=len(skipped),excluded_reasons=skipped,automatic_gold_review=False,parsing_started=False);atomic(a.root/'promotion_report.json',summary);return summary
def main():
 from .config import Settings
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['promote']);parser.add_argument('--settings',required=True);args=parser.parse_args();print(json.dumps(promote(Settings.load(args.settings)),indent=2))
if __name__=='__main__':main()
