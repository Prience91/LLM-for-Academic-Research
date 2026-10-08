import unittest,tempfile,pathlib,json,hashlib,copy,subprocess,tomllib
from unittest.mock import patch,Mock
from giis_prod.config import Settings
from giis_prod.metadata import MetadataResolver,MetadataQuotaDeferred,EnrichmentResult
from giis_prod.ingestion import CorpusBuilder
from giis_prod.metadata_adjudication import MetadataAdjudicator,promote,write_rows,read,digest
CONFIG=pathlib.Path(__file__).parents[1]/'config/settings_full_rag.toml'
class AdjudicationTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=pathlib.Path(self.tmp.name);self.pdf=self.root/'Paper.pdf';self.pdf.write_bytes(b'source fixture');self.key=digest(self.pdf);data=tomllib.loads(CONFIG.read_text());data['paths'].update(pdf_dir=str(self.root),metadata_overrides=str(self.root/'verified.jsonl'),metadata_cache=str(self.root/'cache'));data['metadata_adjudication'].update(artifact_dir=str(self.root/'adjudication'),import_ledgers=[str(self.root/'imports.jsonl'),str(self.root/'adjudication/gold_review_ledger.jsonl')],selection_manifest=str(self.root/'selection.jsonl'),prompt_file=str(self.root/'prompt.txt'));(self.root/'prompt.txt').write_text('Source evidence only');self.settings=Settings(self.root,data);self.metadata=dict(title='Paper',authors=['Author'],year=2020,publisher='Publisher',document_type='report',doi=None);self.row=dict(sha256=self.key,source_file='Paper.pdf',status='local_model_qualified',reviewer='qwen',prediction=dict(language='en',decision='qualified',metadata=self.metadata,unresolved_fields=[]))
 def test_promotion_preserves_local_provenance_and_doi_clearing(self):
  write_rows(self.root/'imports.jsonl',[self.row]);out=promote(self.settings);self.assertEqual(out['qualified_promoted'],1);r=read(self.root/'verified.jsonl')[0];self.assertEqual(r['verification_method'],'local_model_metadata_adjudication');self.assertTrue(r['clear_unverified_doi']);self.assertTrue(r['evidence_manifest_sha256']);self.assertEqual(len(read(self.root/'selection.jsonl')),1);resolver=MetadataResolver(self.settings);result=resolver.enrich(dict(self.metadata,doi='10.1234/cited'),'',r);self.assertTrue(result.verified);self.assertIsNone(result.values['doi']);self.assertNotEqual(result.confidence,1.0)
 def test_changed_pdf_cannot_be_promoted(self):
  write_rows(self.root/'imports.jsonl',[self.row]);self.pdf.write_bytes(b'changed source');out=promote(self.settings);self.assertEqual(out['qualified_promoted'],0);self.assertFalse(read(self.root/'verified.jsonl'))
 def test_existing_gold_rejection_blocks_local_qualification(self):
  write_rows(self.root/'imports.jsonl',[self.row]);write_rows(self.root/'gold_rejected_manual_human_review_only.jsonl',[dict(sha256=self.key)]);out=promote(self.settings);self.assertEqual(out['qualified_promoted'],0)
 def test_gold_authorization_is_required_for_new_gold_input(self):
  r=dict(sha256=self.key,source_file='Paper.pdf',status='gold_qualified',metadata=self.metadata,reviewer='Codex',reviewed_at='now');write_rows(self.root/'adjudication/gold_review_ledger.jsonl',[r]);self.assertEqual(promote(self.settings)['qualified_promoted'],0);r['authorization_reference']='user-approved-test';write_rows(self.root/'adjudication/gold_review_ledger.jsonl',[r]);self.assertEqual(promote(self.settings)['qualified_promoted'],1)
 def test_authorized_gold_rejection_is_persistent(self):
  write_rows(self.root/'imports.jsonl',[self.row]);promote(self.settings);r=dict(sha256=self.key,source_file='Paper.pdf',status='gold_rejected',reviewer='Codex',reviewed_at='now',authorization_reference='user-approved-test');write_rows(self.root/'adjudication/gold_review_ledger.jsonl',[r]);promote(self.settings);self.assertFalse(read(self.root/'verified.jsonl'));self.assertFalse(read(self.root/'selection.jsonl'));self.assertEqual(read(self.root/'gold_rejected_manual_human_review_only.jsonl')[0]['recovery_policy'],'manual_human_review_only')
 def test_gold_result_is_not_overwritten_by_local_result(self):
  original=dict(sha256=self.key,source_file='Paper.pdf',**self.metadata,verified=True,verification_method='manual_verified_override');write_rows(self.root/'verified.jsonl',[original]);write_rows(self.root/'imports.jsonl',[self.row]);promote(self.settings);self.assertEqual(read(self.root/'verified.jsonl')[0]['verification_method'],'manual_verified_override')
 def test_local_override_requires_lineage(self):
  resolver=MetadataResolver(self.settings);bad=dict(self.metadata,verified=True,verification_method='local_model_metadata_adjudication')
  with self.assertRaises(ValueError):resolver.enrich(self.metadata,'',bad)
 def test_cached_referral_does_not_repeat_model_review(self):
  r=dict(self.row,status='needs_gold_review');write_rows(self.root/'imports.jsonl',[r]);a=MetadataAdjudicator(self.settings)
  with patch('giis_prod.metadata_adjudication.urllib.request.urlopen') as http:
   result=a.adjudicate(self.pdf,self.key,self.metadata,'English source');self.assertFalse(result.verified);http.assert_not_called();self.assertTrue((a.root/'gold_review_queue'/f'{self.key}.json').exists())
 def test_devanagari_skips_model_and_queues_gold(self):
  a=MetadataAdjudicator(self.settings)
  with patch('giis_prod.metadata_adjudication.urllib.request.urlopen') as http:
   result=a.adjudicate(self.pdf,self.key,self.metadata,'à¤¨à¥‡à¤ªà¤¾à¤² à¤¸à¤°à¤•à¤¾à¤° '*30);self.assertFalse(result.verified);http.assert_not_called()
 def test_model_declared_non_english_is_not_promoted(self):
  r=copy.deepcopy(self.row);r['prediction']['language']='es';write_rows(self.root/'imports.jsonl',[r]);self.assertEqual(promote(self.settings)['qualified_promoted'],0)
 def test_truncated_model_response_queues_gold(self):
  a=MetadataAdjudicator(self.settings);(a.root/'images').mkdir(parents=True);(a.root/'images'/f'{self.key}.png').write_bytes(b'image');payloads=[dict(data=[dict(id=a.config['model'])]),dict(choices=[dict(finish_reason='length',message=dict(content='{}'))])]
  def response(*args,**kwargs):
   r=Mock();r.__enter__=Mock(return_value=r);r.__exit__=Mock(return_value=False);r.read.return_value=json.dumps(payloads.pop(0)).encode();return r
  with patch('giis_prod.metadata_adjudication.subprocess.run',return_value=subprocess.CompletedProcess([],0,stdout='English source',stderr='')),patch('giis_prod.metadata_adjudication.urllib.request.urlopen',side_effect=response):
   result=a.adjudicate(self.pdf,self.key,self.metadata,'English');self.assertFalse(result.verified);self.assertTrue((a.root/'gold_review_queue'/f'{self.key}.json').exists())
 def builder(self):
  b=CorpusBuilder(self.settings);b.metadata_adjudicator=Mock(enabled=True);b.metadata_adjudicator.non_english.return_value=False;return b
 def test_successful_provider_result_does_not_call_qwen(self):
  b=self.builder();b.metadata_resolver.enrich=Mock(return_value=EnrichmentResult(self.metadata,True,.99,'crossref_title',{},[]));r=b._resolve_metadata(self.pdf,self.key,self.metadata,1,[],'Paper');self.assertEqual(r.metadata_status.value,'verified');b.metadata_adjudicator.adjudicate.assert_not_called()
 def test_provider_quota_deferral_never_calls_qwen(self):
  b=self.builder();b.metadata_resolver.enrich=Mock(side_effect=MetadataQuotaDeferred('quota'))
  with self.assertRaises(MetadataQuotaDeferred):b._resolve_metadata(self.pdf,self.key,self.metadata,1,[],'Paper')
  b.metadata_adjudicator.adjudicate.assert_not_called()
 def test_unresolved_provider_result_calls_qwen(self):
  b=self.builder();b.metadata_resolver.enrich=Mock(return_value=EnrichmentResult(self.metadata,False,0,'unresolved',{},[]));b.metadata_adjudicator.adjudicate.return_value=EnrichmentResult(self.metadata,True,0,'local_model_metadata_adjudication',{},[]);r=b._resolve_metadata(self.pdf,self.key,self.metadata,1,[],'Paper');self.assertEqual(r.metadata_status.value,'verified');b.metadata_adjudicator.adjudicate.assert_called_once()
 def test_flat_provider_qualification_is_reused(self):
  row=dict(sha256=self.key,source_file='Paper.pdf',**self.metadata,verified=True,verification_method='crossref_title_pdf_corroborated');write_rows(self.root/'imports.jsonl',[row]);a=MetadataAdjudicator(self.settings);self.assertTrue(a.adjudicate(self.pdf,self.key,self.metadata,'Paper').verified)
 def test_unauthorized_gold_cannot_be_reused_from_cache_import(self):
  row=dict(sha256=self.key,source_file='Paper.pdf',status='gold_qualified',metadata=self.metadata,reviewer='Codex',reviewed_at='now');write_rows(self.root/'adjudication/gold_review_ledger.jsonl',[row]);a=MetadataAdjudicator(self.settings);self.assertFalse(a.adjudicate(self.pdf,self.key,self.metadata,'Paper').verified)
 def test_changed_source_blocks_new_gold_rejection(self):
  row=dict(sha256=self.key,source_file='Paper.pdf',status='gold_rejected',reviewer='Codex',reviewed_at='now',authorization_reference='user-approved-test');write_rows(self.root/'adjudication/gold_review_ledger.jsonl',[row]);self.pdf.write_bytes(b'changed');promote(self.settings);self.assertFalse(read(self.root/'gold_rejected_manual_human_review_only.jsonl'))
 def test_relative_adjudication_paths_resolve_against_project_not_cwd(self):
  data=copy.deepcopy(self.settings.data)
  data['metadata_adjudication'].update(artifact_dir='relative/adjudication',prompt_file='relative/prompt.txt',selection_manifest='relative/selection.jsonl',import_ledgers=['relative/imports.jsonl'])
  a=MetadataAdjudicator(Settings(self.root,data))
  self.assertEqual(a.root,self.root/'relative/adjudication')
  self.assertEqual(pathlib.Path(a.config['prompt_file']),self.root/'relative/prompt.txt')
  self.assertEqual(pathlib.Path(a.config['selection_manifest']),self.root/'relative/selection.jsonl')
  self.assertEqual(a.config['import_ledgers'],[str(self.root/'relative/imports.jsonl')])
if __name__=='__main__':unittest.main()
