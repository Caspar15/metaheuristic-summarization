"""Preserve and extract development evidence, including unsuccessful conditions."""
from __future__ import annotations

import zipfile
from scripts.audit.access_revision_common import *


def run():
    prior=readj(PLAN/'Historical_Evidence_Verified_20261110.json')
    for name,h in prior['source_sha256'].items():
        if sha(ROOT/name)!=h: raise ValueError('historical source changed: '+name)
    table=[]; files=set(ROOT/p for p in prior['source_sha256'])
    for ds in DATASETS:
        for stage,base,anchor in [
            ('D1',ROOT/f'runs_v2/d1_greedy_sensitivity/{ds}/dev/cheap_multiroute','G00_lexical_graph'),
            ('D3a',ROOT/f'runs_v2/d3a_router_fusion_full_dev_v1/{ds}/dev','R00_anchor')]:
            summary=readj(base/'study_summary.json'); anchor_score=summary['candidates'][anchor]['metrics']['macro_rouge']
            for key,c in summary['candidates'].items():
                metrics=c.get('metrics',{}); score=metrics.get('macro_rouge')
                table.append({'dataset':ds,'stage':stage,'candidate':key,'status':c['status'],
                    'N':metrics.get('rows'),'macro_ROUGE':score,'delta_vs_stage_anchor':None if score is None else score-anchor_score,
                    'anchor':anchor,'evaluation_protocol':metrics.get('evaluation_protocol'),'declared_delta':json.dumps(c.get('declared_delta'),sort_keys=True),
                    'failure_reason':c.get('error',c.get('reason')),'source':str((base/'study_summary.json').relative_to(ROOT))})
                for path_key,hash_key in [('config_path','config_sha256'),('evidence_path','evidence_sha256'),('per_example_path','per_example_sha256')]:
                    if path_key in c:
                        path=ROOT/c[path_key]
                        if hash_key in c and sha(path)!=c[hash_key]: raise ValueError('historical artifact mismatch: '+str(path))
                        # Large per-row files remain in place with their verified hashes.
                        files.add(path)
            files.update(p for p in base.rglob('*') if p.is_file() and (p.name in ['resolved_config.yaml','evidence.json','candidate_summary.json'] or 'analysis' in p.name and p.suffix=='.json'))
    files.update((ROOT/'configs/preregistrations').glob('d1_*.json'))
    files.update((ROOT/'configs/preregistrations').glob('d3a_*.json'))
    files.update((ROOT/'runs_v2').glob('d1*/*/*/paired_analysis.json'))
    catalog=[]
    (OUT/'historical').mkdir(parents=True,exist_ok=True)
    # A ZIP preserves metadata copies without advertising them as new run roots
    # to repository-wide provenance discovery (whose sibling paths are original).
    with zipfile.ZipFile(OUT/'historical/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(files):
            relative=p.relative_to(ROOT); copied=p.suffix in {'.json','.yaml','.md','.py'} and p.stat().st_size<5_000_000
            if copied: archive.write(p,relative.as_posix())
            catalog.append({'path':str(relative),'sha256':sha(p),'bytes':p.stat().st_size,'copied_to_snapshot':copied})
    write_tsv(OUT/'historical/development_conditions.tsv',table)
    writej(OUT/'historical/source_manifest.json',{'at_utc':utc(),'sources':catalog,'prior_ledger_hashes_all_verified':True,'condition_rows':len(table),'scope':'D1 cheap-multiroute complete matrix and D3a full-dev finalists; historical development, not final-config exhaustive sensitivity'})
    examples=[]
    for weights in [(1.,1.,1.),(.5,1.,1.),(1.,1.,2.)]:
        for ranks in [(1,258,258),(258,1,258),(20,20,20)]:
            examples.append({'n':258,'K':60,'weights':weights,'ranks':ranks,'RRF':sum(w/(60+r) for w,r in zip(weights,ranks))})
    writej(OUT/'historical/RRF_numeric_examples.json',{'at_utc':utc(),'status':'hypothetical rank patterns using real example source size; not measured rank tuples',
        'examples':examples,'top40_rank1_to_rank40_contribution_ratio_K60':100/61,
        'top40_rank1_to_rank40_contribution_ratio_K30':70/31,
        'generic_mandatory_upper_bound':3*20+20,'Gov_single_document_upper_bound':3*20+1})
    print({'stage':'historical','sources':len(catalog),'conditions':len(table)},flush=True)


if __name__=='__main__': run()
