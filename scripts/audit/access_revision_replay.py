"""Run a frozen-code development replay, then independently verify decisions.

Each dataset executes in a fresh process with its recorded source snapshot.
Read-only cache access is enforced; the original predictions are never replaced.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path

from scripts.audit.access_revision_common import *


def worker(ds,phase='dev40'):
    formal=phase=='formal200'
    prepared=readj(OUT/('manifests/formal_preparation_complete.json' if formal else 'manifests/preparation_complete.json'))
    if sha(OUT/f'manifests/{phase}.jsonl')!=prepared[phase+'_sha256']:
        raise ValueError('sample manifest changed')
    if formal:
        for p,h in prepared['inputs'].items():
            if sha(ROOT/p)!=h: raise ValueError('frozen formal input changed')
    snap=ROOT/prepared['snapshots'][ds]['root']
    if sha(snap/'source.zip')!=prepared['snapshots'][ds]['archive_sha256']:
        raise ValueError('frozen source archive changed')
    import zipfile
    with zipfile.ZipFile(snap/'source.zip') as archive:
        if any(archive.read(n)!=(snap/n).read_bytes() for n in archive.namelist() if not n.endswith('/')):
            raise ValueError('extracted frozen code drift')
    sys.path.insert(0,str(snap))
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    cache=ROOT/(f'runs_v2/{ds}_final_test_v1/_embedding_cache' if formal else f'runs_v2/gate2_baseline_matrix_v1/{ds}/dev/plm/_embedding_cache')
    os.environ['META_SUM_EMBEDDING_CACHE_DIR']=str(cache)
    import numpy as np
    import torch
    import yaml
    from src.pipeline import select_sentences as selection
    from src.pipeline import candidate_builder as builder
    from src.models.extractive import encoder_rank as encoder
    from scripts.audit.access_revision_math import independent_pool, independent_selection, independent_capacity
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    directory=ROOT/f'runs_v2/{ds}_final_test_v1/predictions/proposed' if formal else run_dir(ds,'C01')
    cfg=yaml.safe_load((directory/'resolved_config.yaml').read_text())
    if formal:
        env=readj(ROOT/f'configs/preregistrations/{ds}_final_execution_freeze_v1.json')['environment']
        expected=dict(env['dependencies'],python=env['python'])
    else:
        expected=readj(directory/'candidate_summary.json')['dependency_versions']
    for key,value in expected.items():
        actual=sys.version.split()[0] if key=='python' else importlib.metadata.version(key)
        if actual!=value: raise ValueError(f'package drift: {key}')
    samples=[x for x in rows(OUT/f'manifests/{phase}.jsonl') if x['dataset']==ds]
    wanted={x['id'] for x in samples}
    docs=unique_map(x for x in rows(canonical_path(ds,'test' if formal else 'validation')) if x['id'] in wanted)
    predictions=unique_map(x for x in rows(directory/'predictions.jsonl') if x['id'] in wanted)
    if set(docs)!=wanted or set(predictions)!=wanted: raise ValueError('incomplete replay sample')
    captured={}; used_caches={}
    original_embeddings=encoder.encoder_document_embeddings

    def read_only_embeddings(sentences,**kw):
        key,identity=encoder._embedding_cache_identity(sentences,**kw)
        path=encoder._cache_path(cache,key)
        if not path.exists(): raise FileNotFoundError(f'frozen numeric cache missing: {key}')
        used_caches[str(path.relative_to(ROOT))]=sha(path)
        return original_embeddings(sentences,**kw)

    encoder.encoder_document_embeddings=read_only_embeddings
    selection.encoder_document_embeddings=read_only_embeddings
    original_score=builder._score_route
    def score_route(*args,**kw):
        values,metadata=original_score(*args,**kw)
        captured.setdefault('route_values',{})[args[0]]=list(values)
        return values,metadata
    builder._score_route=score_route
    original_pool=selection.build_candidate_pool
    def build_pool(*args,**kw):
        captured['eligible_records']=args[0]
        result=original_pool(*args,**kw)
        captured['pool_result']=result
        return result
    selection.build_candidate_pool=build_pool
    original_dispatch=selection.dispatch_optimizer
    def dispatch(*args,**kw):
        captured['selector_args']=args
        return original_dispatch(*args,**kw)
    selection.dispatch_optimizer=dispatch
    results=[]
    for sample in samples:
        key=sample['id']; original=predictions[key]; captured.clear()
        replay=selection.summarize_one(docs[key],cfg)
        folder=OUT/f'audit/{phase}/{ds}/{key}'
        folder.mkdir(parents=True,exist_ok=False)
        writej(folder/'replayed_prediction.json',replay)
        p=independent_pool(captured['eligible_records'],captured['route_values'],cfg)
        writej(folder/'independent_pool.json',p)
        writej(folder/'full_eligible_route_values.json',{'original_indices':[x['original_index'] for x in captured['eligible_records']], 'route_values':captured['route_values']})
        recs=replay['candidate_records']; original_recs=original['candidate_records']
        row_by_id={r['original_index']:r for r in original_recs}
        pool_ids=[r['original_index'] for r in recs]
        checks={'pool_matches_original':pool_ids==[r['original_index'] for r in original_recs],
                'selected_ids_match_original':replay['selected_indices']==original['selected_indices'],
                'summary_exact':replay['summary']==original['summary'],
                'infeasibility_status':(replay['feasible'],replay['infeasible_code'])==(original['feasible'],original['infeasible_code']),
                'independent_pool_membership':p['indices']==pool_ids,
                'independent_inclusion_reasons':all(p['reasons'][r['original_index']]==sorted(r['inclusion_reasons']) for r in recs),
                'independent_fused_ranks':all(p['ranks'][r['original_index']]==r['fused_rank'] for r in recs),
                'independent_full_route_ranks':all(p['route_ranks'][route][r['original_index']]==values['rank'] for r in recs for route,values in r['route_scores'].items()),
                'selector_input_fingerprints_exact':replay['selector_inputs']==original['selector_inputs']}
        differences=[]
        for r in recs:
            idx=r['original_index']
            for name in ['fusion_score','fusion_normalized','selector_salience']:
                if idx in row_by_id: differences.append(abs(r[name]-row_by_id[idx][name]))
            differences.extend([abs(r['fusion_score']-p['scores'][idx]),abs(r['fusion_normalized']-p['normalized'][idx])])
        checks['record_numeric_error_within_1e_8']=max(differences,default=0)<=1e-8
        args=captured['selector_args']
        method,sents,salience,sim,cap,_,_,_,_,spec,min_words,_,_,coverage=args
        independently_selected,trace=independent_selection(sents,salience,sim,coverage,cfg,spec,min_words)
        writej(folder/'reconstructed_selection_steps.json',{'status':'reconstructed; not an original Greedy log','method':method,'steps':trace,'selection_order':independently_selected})
        np.savez_compressed(folder/'selector_numerical_inputs.npz',salience=np.asarray(salience),similarity=sim,coverage=coverage,original_indices=np.asarray(pool_ids))
        checks['independent_selector_output']=sorted(pool_ids[i] for i in independently_selected)==replay['selected_indices']
        if method=='mmr':
            checks['independent_MMR_decision_order']=independently_selected==original['optimizer_diagnostics']['selection_order']
        source_lengths=[len(s['text'].split()) for s in flatten(docs[key])]
        capacity=independent_capacity(source_lengths,cfg['length_control']['max_words'])
        checks['independent_effective_min_words']=min(cfg['length_control']['min_words'],capacity)==replay['output_budget']['effective_min_words']
        eligible_indices=[i for i,n in enumerate(source_lengths) if 0<n<=cap]
        checks['independent_eligibility']=eligible_indices==[r['original_index'] for r in captured['eligible_records']]
        result={'dataset':ds,'id':key,'checked_at_utc':utc(),'passed':all(checks.values()),'checks':checks,
                'max_record_numeric_error':max(differences,default=0),'near_tie_steps_le_1e_10':sum(x['margin'] is not None and x['margin']<=1e-10 for x in trace),
                'scope':'original frozen code reproduction plus independent downstream rank/fusion/pool/selector arithmetic; shared features and cached embeddings remain common inputs'}
        writej(folder/'verification.json',result); results.append(result)
        print({'dataset':ds,'id':key,'passed':result['passed'],'failed':[k for k,v in checks.items() if not v]},flush=True)
    if any(sha(ROOT/p)!=h for p,h in used_caches.items()): raise ValueError('numeric cache changed during replay')
    writej(OUT/f'audit/{phase}/{ds}_summary.json',{'finished_at_utc':utc(),'dataset':ds,'rows':len(results),'passed_rows':sum(r['passed'] for r in results),'all_passed':all(r['passed'] for r in results),'results':results,'cache_inputs':used_caches,'cache_hashes_rechecked':True,'snapshot':prepared['snapshots'][ds],'analysis_code_sha256':{p.name:sha(p) for p in (ROOT/'scripts/audit').glob('access_revision*.py')}})


def run(phase='dev40'):
    verify_frozen_inputs()
    for ds in DATASETS:
        subprocess.run([sys.executable,'-m','scripts.audit.access_revision_replay','--worker',ds,'--phase',phase],cwd=ROOT,check=True)
    summaries=[readj(OUT/f'audit/{phase}/{ds}_summary.json') for ds in DATASETS]
    result={'finished_at_utc':utc(),'n':sum(x['rows'] for x in summaries),'passed_n':sum(x['passed_rows'] for x in summaries),'all_passed':all(x['all_passed'] for x in summaries),'scope':f'{phase} sample only; no all-test replay or human-interpretability certification','source_input_hashes_rechecked_after':True}
    verify_frozen_inputs()
    if phase=='formal200':
        for p,h in readj(OUT/'manifests/formal_preparation_complete.json')['inputs'].items():
            if sha(ROOT/p)!=h: raise ValueError('formal inputs changed')
    writej(OUT/f'audit/{phase}_gate.json',result)
    print(result,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',choices=DATASETS);ap.add_argument('--phase',choices=['dev40','formal200'],default='dev40');args=ap.parse_args()
    worker(args.worker,args.phase) if args.worker else run(args.phase)
