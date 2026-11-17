"""Freeze local reviewer diagnostics, then make reference-blind manifests."""
from __future__ import annotations

import argparse
import importlib.metadata
import platform
import shutil
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

import yaml

from scripts.audit.access_revision_common import *


def freeze():
    if (OUT/'protocol/execution_freeze.json').exists():
        raise FileExistsError('execution freeze already exists; never overwrite')
    files = []
    for ds in DATASETS:
        files += [canonical_path(ds), partition_path(ds)]
        for condition in CONDITIONS:
            d = run_dir(ds, condition)
            files += [d/'predictions.jsonl', d/'resolved_config.yaml']
        files += [run_dir(ds, 'C01')/'candidate_summary.json']
        # Byte hashing only, no greedy targets parsed before pool membership freeze.
        files += [ROOT/f'runs_v2/gate2_greedy_reference_v1/{ds}/dev/{t}/rows.jsonl' for t in TARGETS]
    plan_files = sorted(f for f in PLAN.iterdir() if f.suffix in ['.md', '.json'])
    snapshot = OUT/'protocol/plan_v4_1'
    snapshot.mkdir(parents=True, exist_ok=False)
    for f in plan_files:
        shutil.copyfile(f, snapshot/f.name)
    sources = sorted((ROOT/'scripts/audit').glob('access_revision*.py'))
    payload = {'created_at_utc': utc(), 'protocol_version': '4.1',
               'authorization': 'Requesting author instructed implementation in this conversation; no new final configuration or paid API authorized.',
               'original_test_scores_known': True, 'new_recall_scores_observed': False,
               'git_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
               'python': platform.python_version(),
               'dependencies': {k:importlib.metadata.version(k) for k in ['numpy','scikit-learn','nltk','rouge-score','PyYAML','torch','transformers','tokenizers','sentence-transformers']},
               'inputs': [{'path':f.relative_to(ROOT).as_posix(),'sha256':sha(f)} for f in files],
               'plan_sha256': {f.name:sha(f) for f in plan_files},
               'initial_analysis_code_sha256': {f.relative_to(ROOT).as_posix():sha(f) for f in sources},
               'analysis_code_policy': 'Subsequent code changes retained and recorded per stage; scientific rules cannot be silently changed after results.',
               'paid_api_calls':0, 'frozen_predictions_overwrite':False}
    writej(OUT/'protocol/execution_freeze.json',payload)
    print({'stage':'freeze','inputs':len(files),'created_at_utc':payload['created_at_utc']},flush=True)


def prepare():
    frozen = verify_frozen_inputs()
    if (OUT/'manifests/preparation_complete.json').exists():
        raise FileExistsError('preparation already completed')
    sample, joins, parity, memberships, compact = [], [], {}, [], []
    for ds in DATASETS:
        part = readj(partition_path(ds))
        if sha(canonical_path(ds)) != part['input_sha256']:
            raise ValueError('canonical and original partition hash disagree')
        all_docs = unique_map(rows(canonical_path(ds)))
        ids = part['partitions']['dev']['selected_ids']
        if len(ids) != len(set(ids)) or not set(ids) <= set(all_docs):
            raise ValueError('bad development manifest')
        docs = {k:all_docs[k] for k in ids}
        sample += sample_manifest(list(docs.values()),ds,5,'pilot')
        anchor_cfg = yaml.safe_load((run_dir(ds,'C01')/'resolved_config.yaml').read_text())
        pools = {}
        for condition in CONDITIONS:
            cfg = yaml.safe_load((run_dir(ds,condition)/'resolved_config.yaml').read_text())
            diff = config_diff(anchor_cfg,cfg)
            parity[f'{ds}/{condition}'] = diff
            if condition == 'A06_no_route_reservation' and diff != {'candidate_budget.min_per_route':{'C01':20,'control':0}}:
                raise ValueError(f'A06 is not a pure reservation configuration: {diff}')
            if cfg['candidate_budget']['total'] != 80:
                raise ValueError('configured cap is not 80')
            indexed = {}
            for pred in rows(run_dir(ds,condition)/'predictions.jsonl'):
                key = pred['id']
                if key in indexed or key not in docs:
                    raise ValueError(f'duplicate/extra prediction ID: {ds}/{condition}/{key}')
                if pred['candidate_pool']['total_cap'] != 80:
                    raise ValueError('prediction configured cap drift')
                record = validate_mapping(pred,docs[key])
                indexed[key] = record
                compact.append(dict(record,dataset=ds,condition=condition))
            if set(indexed) != set(ids):
                raise ValueError(f'incomplete predictions: {ds}/{condition}')
            pools[condition] = indexed
            joins.append({'dataset':ds,'condition':condition,'expected_rows':len(ids),'verified_rows':len(indexed),'mapping_pass':True})
            print({'stage':'mapping','dataset':ds,'condition':condition,'rows':len(indexed)},flush=True)
        for condition in CONDITIONS[1:]:
            for key in ids:
                left,right = pools['C01'][key],pools[condition][key]
                group,child,relation=structural_group(left['pool_indices'],right['pool_indices'])
                memberships.append({'dataset':ds,'control':condition,'id':key,'size_group':group,
                                    'equal_size_child':child,'set_relation':relation,
                                    'C01_size':left['pool_size'],'control_size':right['pool_size'],
                                    'C01_words':left['pool_words'],'control_words':right['pool_words'],
                                    'source_words':left['source_words']})
    write_rows(OUT/'manifests/dev40.jsonl',sample)
    write_rows(OUT/'coverage/pool_records.jsonl',compact)
    write_rows(OUT/'coverage/pool_size_membership.jsonl',memberships)
    writej(OUT/'manifests/config_parity.json',parity)
    write_tsv(OUT/'manifests/input_joins.tsv',joins)
    sizes,relations=[],[]
    for ds in DATASETS:
        for condition in CONDITIONS[1:]:
            group=[x for x in memberships if x['dataset']==ds and x['control']==condition]
            for name in ['C01_smaller','equal_size','C01_larger','both_at_80','equal_positive_below_80','both_empty']:
                sizes.append({'dataset':ds,'control':condition,'stratum':name,'N':sum(x['size_group']==name or x['equal_size_child']==name for x in group),'total_N':len(group)})
            for name in ['identical','C01_strict_subset','control_strict_subset','neither_contains_other']:
                relations.append({'dataset':ds,'control':condition,'relation':name,'N':sum(x['set_relation']==name for x in group),'total_N':len(group)})
    write_tsv(OUT/'coverage/pool_size_strata.tsv',sizes)
    write_tsv(OUT/'coverage/pool_set_relations.tsv',relations)
    # Obtain each C01 implementation exactly as recorded, without altering this checkout.
    snapshots={}
    for ds in DATASETS:
        summary=readj(run_dir(ds,'C01')/'candidate_summary.json')
        if any(frozen['dependencies'].get(k)!=v for k,v in summary['dependency_versions'].items() if k!='python') or frozen['python']!=summary['dependency_versions']['python']:
            raise ValueError('dependency parity failed')
        commit=summary['implementation_commit']
        snap=OUT/f'protocol/frozen_src_{commit}'
        if not snap.exists():
            snap.mkdir()
            archive=snap/'source.zip'
            with archive.open('xb') as f:
                subprocess.run(['git','archive','--format=zip',commit,'src'],cwd=ROOT,stdout=f,check=True)
            with zipfile.ZipFile(archive) as z:
                if any(not (snap/n).resolve().is_relative_to(snap.resolve()) for n in z.namelist()):
                    raise ValueError('unsafe archive path')
                z.extractall(snap)
        snapshots[ds]={'implementation_commit':commit,'root':snap.relative_to(ROOT).as_posix(),'archive_sha256':sha(snap/'source.zip')}
    completed={'completed_at_utc':utc(),'new_recall_scores_observed':False,'greedy_reference_rows_parsed':False,
               'dev40_n':len(sample),'prediction_rows_verified':len(compact),'pool_pair_memberships':len(memberships),
               'A06_config_parity_pass':True,'snapshots':snapshots,
               'dev40_sha256':sha(OUT/'manifests/dev40.jsonl'),
               'membership_sha256':sha(OUT/'coverage/pool_size_membership.jsonl'),
               'pool_records_sha256':sha(OUT/'coverage/pool_records.jsonl'),
               'code_sha256':{p.name:sha(p) for p in (ROOT/'scripts/audit').glob('access_revision*.py')}}
    writej(OUT/'manifests/preparation_complete.json',completed)
    print(completed,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('stage',choices=['freeze','prepare']); args=ap.parse_args()
    {'freeze':freeze,'prepare':prepare}[args.stage]()
