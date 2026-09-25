from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from jsonschema import Draft202012Validator

from .candidate_validation import validate_answer_style
from .citations import citation_coverage, contains_non_corpus_citation
from .config import Settings
from .creative_audit import _deterministic_review, _request_options
from .io import read_jsonl, write_json_atomic, write_jsonl_atomic
from .teacher import EvidenceBundle, _is_substantive_chunk, _stratified_chunks
from .v17_turbo_builder import V17TurboDatasetBuilder
from .v17_turbo_layer import (
    build_support_units, candidate_schema, materialize_candidate,
    render_support_context, validate_structure,
)

_MARKER = re.compile(r"\[E(\d+)\]")


def _digest(value: Any) -> str:
    payload=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(payload).hexdigest()


def heldout_case_allocation(document_ids: list[str], settings: Settings) -> list[dict[str, Any]]:
    cfg=settings.section('v17_turbo_heldout_allocation')
    counts={
        'single_source':int(cfg['single_source_cases']),
        'multi_source':int(cfg['multi_source_cases']),
        'table':int(cfg['table_cases']),
        'comparison':int(cfg['comparison_cases']),
    }
    if sum(counts.values()) != int(cfg['case_count']):
        raise ValueError('Held-out case allocation does not sum to case_count')
    if len(document_ids) != int(settings.get('v17_turbo.frozen_heldout_documents')):
        raise ValueError('Held-out document count differs from V17-Turbo policy')
    shuffled=sorted(document_ids)
    random.Random(int(settings.get('teacher.seed'))).shuffle(shuffled)
    single_docs=shuffled[:counts['single_source']]
    uncovered=shuffled[counts['single_source']:]
    pair_count=sum(counts[k] for k in ('multi_source','table','comparison'))
    pairs=[]
    for offset in range(0,len(uncovered),2):
        first=uncovered[offset]
        second=uncovered[offset+1] if offset+1 < len(uncovered) else single_docs[0]
        pairs.append((first,second))
    cursor=0
    while len(pairs)<pair_count:
        first=shuffled[cursor % len(shuffled)]
        second=shuffled[(cursor*7+31) % len(shuffled)]
        cursor+=1
        if first != second and (first,second) not in pairs and (second,first) not in pairs:
            pairs.append((first,second))
    lane_order=(['multi_source']*counts['multi_source']+['table']*counts['table']+['comparison']*counts['comparison'])
    cases=[]
    for index,document_id in enumerate(single_docs,1):
        cases.append({'case_id':f'heldout_single_{index:03d}','candidate_type':'single_source','document_ids':[document_id]})
    for index,(lane,pair) in enumerate(zip(lane_order,pairs,strict=True),1):
        cases.append({'case_id':f'heldout_{lane}_{index:03d}','candidate_type':lane,'document_ids':list(pair)})
    covered={value for case in cases for value in case['document_ids']}
    if covered != set(document_ids):
        raise ValueError('Held-out allocation does not cover every frozen document')
    if len(cases)!=int(cfg['case_count']):
        raise ValueError('Held-out allocation produced the wrong case count')
    return cases


def _bundle(case: dict[str, Any], chunks: list[dict[str, Any]], total_count: int) -> EvidenceBundle:
    selected=[]
    per_source=total_count if len(case['document_ids'])==1 else max(2,total_count//2)
    for document_id in case['document_ids']:
        substantive=[row for row in chunks if str(row['document_id'])==document_id and _is_substantive_chunk(row)]
        if len(substantive)<3:
            raise ValueError(f'Held-out document {document_id} has only {len(substantive)} substantive chunks; at least 3 are required')
        selected.extend(_stratified_chunks(substantive,per_source))
    return EvidenceBundle(bundle_id='heldout_'+_digest({'case':case,'chunks':[x['chunk_id'] for x in selected]})[:24],document_ids=tuple(case['document_ids']),chunks=selected)


def _structured_options(settings: Settings,bundle: EvidenceBundle,answer_style: str) -> dict[str, Any]:
    units=build_support_units(bundle.chunks)
    schema=candidate_schema([f'[E{i}]' for i in range(1,len(bundle.chunks)+1)],[x.support_unit_id for x in units])
    if answer_style!='comparison':
        raise ValueError('Compact held-out structured authoring is reserved for comparisons')
    schema['properties']['answer_style']['enum']=[answer_style]
    rendering='an explicit point-by-point comparison whose factual points contain evidence markers'
    prompt=f'''Create one held-out scientific benchmark case from the supplied evidence.\n\nV17-Turbo contract:\n- Use only the numbered support units.\n- Form a coherent question requiring evidence from both source papers.\n- Declare every material obligation and at least one material obligation for each source.\n- Map every obligation to exact support units and to at least one answer unit.\n- Answer as {rendering}.\n- Cite every factual answer unit with exact [E#] markers.\n- Preserve differences in method, population, place, scale, timeframe, outcome, and causal status.\n- Do not expose chain-of-thought.\n\nEVIDENCE:\n{render_support_context(bundle.chunks,units)}'''
    extra={}
    if 'enable_thinking' in settings.section('teacher'):
        extra['extra_body']={'chat_template_kwargs':{'enable_thinking':bool(settings.get('teacher.enable_thinking'))}}
    return {'model':settings.get('models.teacher'),'messages':[{'role':'system','content':'Author one strictly grounded held-out benchmark case in the requested schema.'},{'role':'user','content':prompt}],'response_format':{'type':'json_schema','json_schema':{'name':'giis_v17_turbo_heldout_structured','strict':True,'schema':{'type':'object','additionalProperties':False,'required':['examples'],'properties':{'examples':{'type':'array','minItems':1,'maxItems':1,'items':schema}}}}},'temperature':float(settings.get('teacher.temperature')),'max_tokens':int(settings.get('teacher.max_output_tokens')),**extra}


def _table_request_options(
    settings: Settings,bundle: EvidenceBundle,source: dict[str,Any]
) -> tuple[dict[str,Any],str]:
    evidence=render_support_context(bundle.chunks,build_support_units(bundle.chunks))
    question=str(source['question']).strip()
    options=_request_options(
        settings,
        evidence=evidence,
        question=question,
        answer_style='table',
        instruction=(
            'Answer as one compact Markdown comparison table. Every substantive row must carry '
            'one or more evidence markers.'
        ),
    )
    required='\n'.join(
        f"- {str(row['obligation_id'])}: {str(row['requirement']).strip()}"
        for row in source['coverage_obligations']
    )
    insertion=(
        '\nRequired coverage inherited from the passing V17-Turbo prose source; address every '
        'item without adding unsupported requirements:\n'+required+'\n'
    )
    options['messages'][1]['content']=options['messages'][1]['content'].replace(
        '\nEVIDENCE:\n',insertion+'\nEVIDENCE:\n',1
    )
    return options,evidence


def _source_coverage(bundle: EvidenceBundle,answer: str) -> set[str]:
    indexes={int(x) for x in _MARKER.findall(answer)}
    return {str(bundle.chunks[i-1]['document_id']) for i in indexes if 1<=i<=len(bundle.chunks)}


def _validate_candidate(builder: V17TurboDatasetBuilder,bundle: EvidenceBundle,candidate: dict[str, Any],candidate_type: str) -> tuple[bool,str]:
    if candidate_type in {'single_source','multi_source'}:
        valid,reason=builder._validate_deterministic(bundle,candidate)
        if not valid:return valid,reason
    else:
        units=build_support_units(bundle.chunks)
        schema=candidate_schema([f'[E{i}]' for i in range(1,len(bundle.chunks)+1)],[x.support_unit_id for x in units])
        schema['properties']['answer_style']['enum']=[candidate_type]
        raw={k:candidate[k] for k in schema['properties'] if k in candidate}
        if list(Draft202012Validator(schema).iter_errors(raw)):return False,'v17_turbo_schema_invalid'
        candidate.update(materialize_candidate(raw))
        valid,reason=builder._normalize_capacity_mode(candidate)
        if not valid:return False,reason
        valid,reason=validate_answer_style(candidate)
        if not valid:return False,reason
        answer=str(candidate['answer'])
        used={int(x) for x in _MARKER.findall(answer)}
        if not used:return False,'missing_evidence_markers'
        if any(x<1 or x>len(bundle.chunks) for x in used):return False,'out_of_range_evidence_markers'
        if contains_non_corpus_citation(answer):return False,'non_corpus_secondary_citation'
        if citation_coverage(answer)<float(builder.teacher['minimum_training_citation_coverage']):return False,'insufficient_citation_coverage'
        if builder._chat_token_count(builder._training_messages(bundle,candidate['question'],answer))>int(builder.settings.get('training.max_length')):return False,'sample_exceeds_training_context'
        coverage=validate_structure(candidate,units,lane='qualification')
        if not coverage['passed']:return False,str(coverage['reasons'][0])
    if _source_coverage(bundle,str(candidate['answer'])) != set(bundle.document_ids):
        return False,'heldout_source_coverage_incomplete'
    expected_style='prose' if candidate_type in {'single_source','multi_source'} else candidate_type
    if str(candidate.get('answer_style'))!=expected_style:return False,'heldout_answer_style_mismatch'
    return True,'ok'


def _rows(case: dict[str, Any],bundle: EvidenceBundle,candidate: dict[str, Any],settings: Settings) -> tuple[dict[str,Any],dict[str,Any],dict[str,Any]]:
    used=set(); obligations=[]
    for obligation in candidate['coverage_obligations']:
        chunk_ids=[]
        for support in obligation['supports']:
            index=int(str(support['evidence_id'])[2:-1]); used.add(index); chunk_ids.append(str(bundle.chunks[index-1]['chunk_id']))
        obligations.append({'obligation_id':str(obligation['obligation_id']),'requirement':str(obligation['requirement']),'supporting_chunk_ids':sorted(set(chunk_ids))})
    case_row={'case_id':case['case_id'],'query':candidate['question'],'category':'heldout_scientific_evidence','candidate_type':case['candidate_type'],'expected_document_ids':case['document_ids'],'expect_abstention':False,'requested_mode':candidate['response_mode'],'question_authoring_model':settings.get('v17_turbo_heldout_benchmark.authoring_model')}
    reviewer=(settings.get('v17_turbo_heldout_benchmark.single_source_reviewer') if case['candidate_type']=='single_source' else settings.get(f"v17_turbo_heldout_benchmark.{case['candidate_type']}_reviewer"))
    reference={'case_id':case['case_id'],'candidate_type':case['candidate_type'],'expected_document_ids':case['document_ids'],'supporting_chunk_ids':[str(bundle.chunks[i-1]['chunk_id']) for i in sorted(used)],'required_points':[x['requirement'] for x in obligations],'semantic_guardrails':['Use only the frozen supporting evidence.','Preserve source ownership, scope, method, timeframe, uncertainty, and causal boundaries.','Satisfy every atomic obligation without unsupported claims.'],'coverage_obligations':obligations,'reference_answer':candidate['answer'],'reference_answer_supplied_to_model':False,'reviewer':reviewer,'review_verdict':'pending','authoring_model':settings.get('v17_turbo_heldout_benchmark.authoring_model')}
    packet={'review_id':case['case_id'],'candidate_sha256':_digest(candidate),'lane':case['candidate_type'],'question':candidate['question'],'evidence':render_support_context(bundle.chunks,build_support_units(bundle.chunks)),'answer':candidate['answer'],'teacher_declared_obligations':candidate['coverage_obligations'],'expected_document_ids':case['document_ids'],'assigned_reviewer':reviewer}
    return case_row,reference,packet


def author_heldout(settings: Settings,output_dir: Path) -> dict[str,Any]:
    output_dir.mkdir(parents=True,exist_ok=True); checkpoints=output_dir/'checkpoints'; checkpoints.mkdir(exist_ok=True)
    heldout=json.loads(settings.path('paths.heldout_manifest').read_text()); ids=[str(x) for x in heldout['heldout_document_ids']]
    cases=heldout_case_allocation(ids,settings); allocation_digest=_digest(cases)
    documents={str(x['document_id']):x for x in read_jsonl(settings.path('paths.document_registry'))}; chunks=list(read_jsonl(settings.path('paths.chunk_file')))
    if not set(ids)<=set(documents):raise ValueError('Held-out document missing from registry')
    status_path=output_dir/'status.json'; allocation_path=output_dir/'allocation.json'; lock=Lock()
    if allocation_path.exists():
        prior_allocation=json.loads(allocation_path.read_text())
        if prior_allocation.get('allocation_digest')!=allocation_digest or prior_allocation.get('cases')!=cases:
            raise ValueError('Existing held-out allocation does not match the canonical plan')
        prior_status=json.loads(status_path.read_text()) if status_path.exists() else {}
        if prior_status.get('selection_digest') not in {None,heldout['selection_digest']}:
            raise ValueError('Existing held-out selection digest does not match the frozen split')
        started=str(prior_status.get('started_at') or datetime.now(UTC).isoformat())
    else:
        if any(path.name not in {'checkpoints'} for path in output_dir.iterdir()):
            raise FileExistsError(f'Unrecognized held-out output prevents safe resume: {output_dir}')
        started=datetime.now(UTC).isoformat()
        write_json_atomic(allocation_path,{'schema_version':1,'seed':int(settings.get('teacher.seed')),'counts':settings.section('v17_turbo_heldout_allocation'),'allocation_digest':allocation_digest,'cases':cases})
    existing_results=[]
    case_by_id={str(case['case_id']):case for case in cases}
    for checkpoint_path in sorted(checkpoints.glob('*.json')):
        payload=json.loads(checkpoint_path.read_text())
        case_id=str(payload.get('case_spec',{}).get('case_id',''))
        if case_id not in case_by_id or payload.get('case_spec')!=case_by_id[case_id]:
            raise ValueError(f'Checkpoint case mismatch: {checkpoint_path}')
        if payload.get('allocation_digest')!=allocation_digest:
            raise ValueError(f'Checkpoint allocation mismatch: {checkpoint_path}')
        if payload.get('review_packet',{}).get('candidate_sha256')!=_digest(payload.get('candidate')):
            raise ValueError(f'Checkpoint candidate digest mismatch: {checkpoint_path}')
        existing_results.append(payload)
    progress={'status':'running','started_at':started,'total':len(cases),'completed':len(existing_results),'failed':0,'selection_digest':heldout['selection_digest'],'allocation_digest':allocation_digest,'resumed_at':datetime.now(UTC).isoformat()}
    write_json_atomic(status_path,progress)
    builder=V17TurboDatasetBuilder(settings)
    def one(case: dict[str,Any])->dict[str,Any]:
        bundle=_bundle(case,chunks,int(settings.get('teacher.evidence_chunks_per_document'))); failures=[]
        for attempt in range(1,int(settings.get('teacher.maximum_core_requests_per_document'))+1):
            try:
                proven_table_review=None
                if case['candidate_type'] in {'single_source','multi_source'}:
                    examples=builder._request_examples(bundle,requested_examples=1,partition='heldout',request_number=attempt)
                elif case['candidate_type']=='table':
                    source_examples=builder._request_examples(bundle,requested_examples=1,partition='heldout',request_number=attempt)
                    if len(source_examples)!=1:raise ValueError(f'Expected one prose source candidate, received {len(source_examples)}')
                    source=source_examples[0]
                    source_valid,source_reason=_validate_candidate(builder,bundle,source,'multi_source')
                    if not source_valid:
                        failures.append({'attempt':attempt,'reason':f'table_source_{source_reason}'})
                        continue
                    options,evidence=_table_request_options(settings,bundle,source)
                    response=builder.client.chat.completions.create(**options)
                    content=response.choices[0].message.content
                    if not content:raise ValueError('Teacher returned empty creative table output')
                    rendered=list(json.loads(content).get('examples') or [])
                    if len(rendered)!=1:raise ValueError(f'Expected one rendered table, received {len(rendered)}')
                    proven_table_review=_deterministic_review(
                        rendered[0],
                        question=str(source['question']).strip(),
                        answer_style='table',
                        evidence=evidence,
                        minimum_coverage=float(builder.teacher['minimum_training_citation_coverage']),
                    )
                    if not proven_table_review['passed']:
                        failures.append({'attempt':attempt,'reason':','.join(proven_table_review['issues'])})
                        continue
                    table=dict(rendered[0])
                    table['coverage_obligations']=source['coverage_obligations']
                    examples=[table]
                else:
                    response=builder.client.chat.completions.create(**_structured_options(settings,bundle,case['candidate_type']))
                    content=response.choices[0].message.content
                    if not content:raise ValueError('Teacher returned empty structured output')
                    examples=list(json.loads(content).get('examples') or [])
                if len(examples)!=1:raise ValueError(f'Expected one candidate, received {len(examples)}')
                if case['candidate_type']=='table':
                    if _source_coverage(bundle,str(examples[0]['answer'])) != set(bundle.document_ids):
                        failures.append({'attempt':attempt,'reason':'heldout_source_coverage_incomplete'})
                        continue
                else:
                    valid,reason=_validate_candidate(builder,bundle,examples[0],case['candidate_type'])
                    if not valid:failures.append({'attempt':attempt,'reason':reason});continue
                case_row,reference,packet=_rows(case,bundle,examples[0],settings)
                payload={'case_spec':case,'allocation_digest':allocation_digest,'bundle':{'bundle_id':bundle.bundle_id,'chunk_ids':[x['chunk_id'] for x in bundle.chunks]},'candidate':examples[0],'case':case_row,'reference':reference,'review_packet':packet,'attempt':attempt,'failures':failures}
                write_json_atomic(checkpoints/f"{case['case_id']}.json",payload);return payload
            except Exception as exc:failures.append({'attempt':attempt,'reason':f'{type(exc).__name__}: {exc}'})
        raise RuntimeError(f"Held-out authoring exhausted for {case['case_id']}: {failures[-1] if failures else 'unknown'}")
    results=list(existing_results);errors=[]
    completed_case_ids={str(payload['case_spec']['case_id']) for payload in existing_results}
    pending_cases=[case for case in cases if str(case['case_id']) not in completed_case_ids]
    with ThreadPoolExecutor(max_workers=int(settings.get('teacher.generation_concurrency'))) as pool:
        futures={pool.submit(one,case):case for case in pending_cases}
        for future in as_completed(futures):
            case=futures[future]
            try:results.append(future.result())
            except Exception as exc:errors.append({'case_id':case['case_id'],'error':f'{type(exc).__name__}: {exc}'})
            with lock:
                progress.update(completed=len(results),failed=len(errors),updated_at=datetime.now(UTC).isoformat());write_json_atomic(status_path,progress)
    if errors:
        progress.update(status='failed',errors=errors,failed_at=datetime.now(UTC).isoformat());write_json_atomic(status_path,progress);raise RuntimeError(f'Held-out authoring failed for {len(errors)} cases')
    results.sort(key=lambda x:x['case']['case_id']);write_jsonl_atomic(output_dir/'benchmark_pre_review.jsonl',(x['case'] for x in results));write_jsonl_atomic(output_dir/'benchmark_reference_pre_review.jsonl',(x['reference'] for x in results));write_jsonl_atomic(output_dir/'review_packets.jsonl',(x['review_packet'] for x in results))
    counts={lane:sum(x['case']['candidate_type']==lane for x in results) for lane in ('single_source','multi_source','table','comparison')}
    summary={'status':'awaiting_routed_independent_review','started_at':started,'completed_at':datetime.now(UTC).isoformat(),'case_count':len(results),'cases_by_type':counts,'heldout_document_coverage':len({d for x in results for d in x['case']['expected_document_ids']}),'heldout_selection_digest':heldout['selection_digest'],'allocation_digest':allocation_digest,'authoring_model':settings.get('v17_turbo_heldout_benchmark.authoring_model'),'reviewer_routing':{'single_source':settings.get('v17_turbo_heldout_benchmark.single_source_reviewer'),'multi_source':settings.get('v17_turbo_heldout_benchmark.multi_source_reviewer'),'table':settings.get('v17_turbo_heldout_benchmark.table_reviewer'),'comparison':settings.get('v17_turbo_heldout_benchmark.comparison_reviewer')}}
    write_json_atomic(output_dir/'authoring_summary.json',summary);progress.update(status=summary['status'],completed=len(results),failed=0,completed_at=summary['completed_at']);write_json_atomic(status_path,progress);return summary


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='config/settings_v17_turbo.toml');parser.add_argument('--out',required=True);args=parser.parse_args();settings=Settings.load(args.config);print(json.dumps(author_heldout(settings,(settings.root/args.out).resolve()),indent=2))
if __name__=='__main__':main()
