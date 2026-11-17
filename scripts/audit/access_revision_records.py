"""Verify saved decision records without rerunning the summarizer."""
from __future__ import annotations

import argparse
from collections import Counter
import numpy as np
import yaml

from scripts.audit.access_revision_common import *


def numeric_hash(values):
    return hashlib.sha256(np.asarray(values,dtype='<i8').tobytes()).hexdigest()


def test_directory(ds):
    return ROOT/f'runs_v2/{ds}_final_test_v1/predictions/proposed'


def fixed_directory(ds,condition):
    if condition.startswith('A08'):
        return ROOT/f'runs_v2/postfreeze_zero_lexical_weight_v1/{ds}/dev/{condition}'
    return run_dir(ds,condition)


def fixed_pool():
    verify_frozen_inputs()
    conditions=['A04_equal_rank_rrf','A05_index_only_lexical','A08_exact_pool_zero_lexical_weight']
    files=[fixed_directory(ds,c)/f for ds in DATASETS for c in conditions for f in ['predictions.jsonl','resolved_config.yaml']]
    frozen={str(p.relative_to(ROOT)):sha(p) for p in files}
    writej(OUT/'audit/fixed_pool_started.json',{'at_utc':utc(),'inputs':frozen,'code_sha256':sha(Path(__file__))})
    all_results=[]; summaries=[]
    for ds in DATASETS:
        # Retain just the fields needed for comparison, not repeated route metadata.
        anchor={x['id']:{'indices':[r['original_index'] for r in x['candidate_records']],
                'route_scores':{r['original_index']:{k:(v['raw'],v['rank']) for k,v in r['route_scores'].items()} for r in x['candidate_records']}}
                for x in rows(run_dir(ds,'C01')/'predictions.jsonl')}
        for condition in conditions:
            cfg=yaml.safe_load((fixed_directory(ds,condition)/'resolved_config.yaml').read_text())
            weights={r:cfg['candidates'].get('route_weights',{}).get(r,1.0) for r in cfg['compute_budget']['enabled_routes']}
            seen=set(); group=[]
            for pred in rows(fixed_directory(ds,condition)/'predictions.jsonl'):
                key=pred['id']
                if key in seen or key not in anchor: raise ValueError('fixed-pool ID mismatch')
                seen.add(key); records=pred['candidate_records']; ids=[r['original_index'] for r in records]
                allocation=pred['candidate_pool']['allocation']
                checks={'same_C01_ordered_membership':ids==anchor[key]['indices'],
                    'fixed_pool_flag':allocation.get('audit_fixed_candidate_pool') is True,
                    'fixed_pool_hash':allocation.get('audit_fixed_candidate_original_indices_sha256')==numeric_hash(anchor[key]['indices']),
                    'full_eligible_reconstruction_flag':allocation.get('route_scores_recomputed_over_full_eligible_source') is True,
                    'fixed_pool_reasons':all(r['inclusion_reasons']==['audit:fixed_C01_candidate_pool'] for r in records),
                    'configured_weights':allocation['route_weights']==weights,
                    'selector_salience_source':pred['candidate_pool']['selector_salience_source']==cfg['selector']['salience_source']}
                feature_ok=True; salience_ok=True; rrf_error=0.0
                for r in records:
                    base=anchor[key]['route_scores'].get(r['original_index'],{})
                    feature_ok &= set(base)==set(r['route_scores']) and all(k in base and abs(v['raw']-base[k][0])<=1e-8 and v['rank']==base[k][1] for k,v in r['route_scores'].items())
                    score=sum(weights[k]/(cfg['candidates']['rrf_constant']+v['rank']) for k,v in r['route_scores'].items())
                    rrf_error=max(rrf_error,abs(score-r['fusion_score']))
                    expected=r['route_scores']['lexical']['percentile'] if condition.startswith('A05') else r['fusion_normalized']
                    salience_ok &= abs(expected-r['selector_salience'])<=1e-8
                checks.update(unchanged_route_features_and_ranks=bool(feature_ok),configured_RRF_arithmetic=rrf_error<=1e-8,selector_salience_arithmetic=bool(salience_ok))
                result={'dataset':ds,'condition':condition,'id':key,'passed':all(checks.values()),'checks':checks,'max_RRF_error':rrf_error}
                group.append(result); all_results.append(result)
            if seen!=set(anchor): raise ValueError('incomplete fixed-pool comparison')
            summaries.append({'dataset':ds,'condition':condition,'N':len(group),'passed':sum(r['passed'] for r in group),'mismatches':sum(not r['passed'] for r in group)})
            print(summaries[-1],flush=True)
    write_rows(OUT/'audit/fixed_pool_per_document.jsonl',all_results)
    write_tsv(OUT/'audit/fixed_pool_summary.tsv',summaries)
    if any(sha(ROOT/p)!=h for p,h in frozen.items()): raise ValueError('fixed-pool input changed')
    writej(OUT/'audit/fixed_pool_complete.json',{'at_utc':utc(),'N':len(all_results),'all_passed':all(r['passed'] for r in all_results),'input_hashes_rechecked':True,'scope':'stored membership, flags, ranks/features and salience arithmetic; not replay of every ablation'})


def stored_reservations(pred):
    """Check exclusive-proposal-first reservation using saved proposal lists."""
    pool=pred['candidate_pool']; props=pool['route_proposals']; a=pool['allocation']
    sets={r:{x['original_index'] for x in values} for r,values in props.items()}
    if any(len(sets[r])!=len(values) for r,values in props.items()): return False
    records={r['original_index']:r for r in pred['candidate_records']}
    for route,values in props.items():
        order=sorted(values,key=lambda x:(sum(x['original_index'] in s for s in sets.values())!=1,x['rank'],x['original_index']))
        quota=a['effective_min_per_route'][route]
        chosen={x['original_index'] for x in order[:quota]}
        if chosen!={x['original_index'] for x in values if x['reserved']}: return False
        if chosen!={idx for idx,r in records.items() if 'reserve:'+route in r['inclusion_reasons']}: return False
        if not chosen<=set(records): return False
    return True


def stored_mmr(pred):
    diag=pred['optimizer_diagnostics']; steps=diag.get('steps')
    if steps is None: return None
    pool=pred['candidate_records']; chosen=[]; error=0.0
    for number,step in enumerate(steps,1):
        idx=step['candidate_relative_index']
        if step['step']!=number or idx in chosen or not 0<=idx<len(pool): return False
        if not chosen and step['max_similarity_to_selected']!=0: return False
        error=max(error,abs(step['relevance']-pool[idx]['selector_salience']),
                  abs(step['mmr_score']-(diag['lambda_relevance']*step['relevance']-(1-diag['lambda_relevance'])*step['max_similarity_to_selected'])))
        chosen.append(idx)
    return error<=1e-8 and chosen==diag['selection_order'] and sorted(pool[i]['original_index'] for i in chosen)==pred['selected_indices']


def all_test_records():
    files=[p for ds in DATASETS for p in [canonical_path(ds,'test'),test_directory(ds)/'predictions.jsonl',test_directory(ds)/'resolved_config.yaml',test_directory(ds)/'prediction_evidence.json']]
    inputs={str(p.relative_to(ROOT)):sha(p) for p in files}
    writej(OUT/'audit/L0_L1_started.json',{'at_utc':utc(),'inputs':inputs,'code_sha256':sha(Path(__file__))})
    all_results=[]; summaries=[]; descriptions=[]
    for ds in DATASETS:
        docs=unique_map(rows(canonical_path(ds,'test'))); seen=set(); counts=Counter()
        directory=test_directory(ds); cfg=yaml.safe_load((directory/'resolved_config.yaml').read_text())
        ev=readj(directory/'prediction_evidence.json')
        if ev['predictions_sha256']!=sha(directory/'predictions.jsonl') or ev['config_sha256']!=sha(directory/'resolved_config.yaml'): raise ValueError('original execution evidence hash mismatch')
        weights={r:cfg['candidates'].get('route_weights',{}).get(r,1.0) for r in cfg['compute_budget']['enabled_routes']}
        for pred in rows(directory/'predictions.jsonl'):
            key=pred['id']
            if key in seen or key not in docs: raise ValueError('test ID mismatch')
            seen.add(key); compact=validate_mapping(pred,docs[key]); pool=pred['candidate_records']; budget=pred['output_budget']; evaluation=pred['selection_evaluation']
            feasible=bool(pred['selected_indices']) and compact['selected_words']>=budget['effective_min_words'] and compact['selected_words']<=budget['max_words']
            checks={'L0_source_mapping':True,'L0_output_order':pred['selected_indices']==sorted(pred['selected_indices']),
                'L0_length_and_status':feasible==pred['feasible']==evaluation['feasible'] and evaluation['selected_words']==compact['selected_words'] and compact['selected_words']<=budget['max_words'] and (pred['infeasible_code'] is None if feasible else pred['infeasible_code'] is not None),
                'L0_selected_metadata':all(r['text']==s['text'] and r['original_index']==idx and r['selection_order']==i for i,(r,s,idx) in enumerate(zip(pred['selected_sentences'],[flatten(docs[key])[j] for j in pred['selected_indices']],pred['selected_indices']))) and len(pred['selected_sentences'])==len(pred['selected_indices']),
                'L1_pool_RRF_arithmetic':all(abs(r['fusion_score']-sum(weights[k]/(cfg['candidates']['rrf_constant']+v['rank']) for k,v in r['route_scores'].items()))<=1e-8 for r in pool),
                'L1_reservation_rule':stored_reservations(pred),
                'L1_Gov_guard_from_stored_rank':None if ds!='govreport' else {r['original_index'] for r in pool if 'guard:document' in r['inclusion_reasons']}=={r['original_index'] for r in pool if r['fused_rank']==1},
                'L1_selected_MMR_arithmetic':stored_mmr(pred),
                'L2_full_rrf_fill':None,'L2_global_minmax':None,'L2_global_route_ranks':None,'L2_all_feasible_argmax':None,
                'L2_Multi_document_guards':None}
            all_results.append({'dataset':ds,'id':key,'passed_checkable':all(v for v in checks.values() if v is not None),'checks':checks})
            for name,value in checks.items(): counts[(name,'not_checkable' if value is None else 'pass' if value else 'fail')]+=1
            a=pred['candidate_pool']['allocation']; props=pred['candidate_pool']['route_proposals']
            nominated=set(x['original_index'] for values in props.values() for x in values)
            selected_docs={flatten(docs[key])[i]['document_id'] for i in pred['selected_indices']}
            descriptions.append(dict(compact,dataset=ds,proposal_union_n=len(nominated),proposal_not_in_pool_n=len(nominated-set(compact['pool_indices'])),
                at_cap_80=compact['pool_size']==80,guard_n=sum('guard:document' in r['inclusion_reasons'] for r in pool),
                reserved_union_n=sum(any(t.startswith('reserve:') for t in r['inclusion_reasons']) for r in pool),
                summary_document_count=len(selected_docs),min_words_relaxed=budget['min_words_relaxed'],
                proposal_counts={r:len(v) for r,v in props.items()},proposal_membership_multiplicity=dict(Counter(sum(i in {x['original_index'] for x in v} for v in props.values()) for i in nominated))))
        if seen!=set(docs) or len(seen)!=ev['rows']: raise ValueError('incomplete test population')
        for name in sorted({k[0] for k in counts}): summaries.append({'dataset':ds,'check':name,'N':len(seen),**{s:counts[(name,s)] for s in ['pass','fail','not_checkable']}})
        print({'stage':'L0_L1','dataset':ds,'N':len(seen),'failed_checks':sum(n for (k,s),n in counts.items() if s=='fail')},flush=True)
    write_rows(OUT/'audit/L0_L1_per_document.jsonl',all_results)
    write_tsv(OUT/'audit/L0_L1_summary.tsv',summaries)
    write_rows(OUT/'audit/test_pool_descriptions.jsonl',descriptions)
    if any(sha(ROOT/p)!=h for p,h in inputs.items()): raise ValueError('test inputs changed')
    writej(OUT/'audit/L0_L1_complete.json',{'at_utc':utc(),'N':len(all_results),'all_checkable_passed':all(x['passed_checkable'] for x in all_results),'source_inputs_unchanged':True,'scope':'stored arithmetic and mapping only; L2 fields deliberately marked not_checkable here'})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['fixed_pool','all_test']);a=ap.parse_args()
    (fixed_pool if a.stage=='fixed_pool' else all_test_records)()
