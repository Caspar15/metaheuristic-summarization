"""Freeze reference-blind formal samples, population clusters and replay inputs."""
from __future__ import annotations

import zipfile
from collections import Counter,defaultdict
from scripts.audit.access_revision_common import *


def dependency_groups(documents):
    """Transitive components on full inputs or identical >=100-word source docs."""
    parent={d['id']:d['id'] for d in documents}; owner={}
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    for doc in sorted(documents,key=lambda d:d['id']):
        texts=[' '.join(' '.join(s['text'].split()) for sec in d['sections'] for s in sec['sentences']) for d in sorted(doc['documents'],key=lambda d:d['source_order'])]
        keys=[('full',hashlib.sha256(' '.join(texts).encode()).hexdigest())]
        keys += [('document',hashlib.sha256(t.encode()).hexdigest()) for t in texts if len(t.split())>=100]
        for key in keys:
            if key in owner:
                a,b=find(doc['id']),find(owner[key]); parent[max(a,b)]=min(a,b)
            else: owner[key]=doc['id']
    return {key:find(key) for key in parent}


def prepare_formal():
    gate=readj(OUT/'audit/dev40_gate.json')
    if not gate['all_passed'] or gate['n']!=40: raise ValueError('dev40 gate has not passed')
    samples=[]; populations=[]; snapshots={}; inputs={}; reports=[]; subsets=[]
    for ds in DATASETS:
        directory=ROOT/f'runs_v2/{ds}_final_test_v1/predictions/proposed'
        evidence=readj(directory/'prediction_evidence.json')
        policy=readj(ROOT/f'configs/data_policies/{ds}_test_v1.json')
        canon=canonical_path(ds,'test')
        if sha(canon)!=policy['analyses']['main']['expected_file_sha256']: raise ValueError('test canonical policy mismatch')
        for path in [canon,directory/'predictions.jsonl',directory/'resolved_config.yaml',directory/'prediction_evidence.json',ROOT/f'configs/preregistrations/{ds}_final_execution_freeze_v1.json']:
            inputs[str(path.relative_to(ROOT))]=sha(path)
        if evidence['predictions_sha256']!=sha(directory/'predictions.jsonl') or evidence['config_sha256']!=sha(directory/'resolved_config.yaml'): raise ValueError('test prediction or config hash mismatch')
        documents=list(rows(canon)); unique_map(documents)
        if len(documents)!=policy['analyses']['main']['expected_rows']: raise ValueError('test row count mismatch')
        clusters=dependency_groups(documents); sizes=Counter(clusters.values())
        sample=[dict(x,cluster_id=clusters[x['id']],cluster_population_n=sizes[clusters[x['id']]],analysis_weight=(x['stratum_population_n']/x['population_n'])/25) for x in sample_manifest(documents,ds,25,'formal')]
        samples+=sample
        populations += [{'dataset':ds,'id':k,'cluster_id':c,'cluster_population_n':sizes[c]} for k,c in sorted(clusters.items())]
        for name in ['stability','source-position','manual-qc']:
            for h,quota in enumerate([3,2,2,3]):
                choices=[dict(x,subset=name,subset_sha256=hashlib.sha256(f'access-2026-46834|{name}-v1|20261109|{ds}|{x["id"]}'.encode()).hexdigest()) for x in sample if x['stratum']==h]
                subsets+=sorted(choices,key=lambda x:(x['subset_sha256'],x['id']))[:quota]
        reports.append({'dataset':ds,'population_N':len(documents),'population_clusters':len(sizes),'population_non_singleton_clusters':sum(n>1 for n in sizes.values()),'sample_N':len(sample),'sample_unique_clusters':len({x['cluster_id'] for x in sample}),'sampled_members_per_cluster':dict(Counter(x['cluster_id'] for x in sample))})
        commit=evidence['implementation_commit']; snap=OUT/f'protocol/frozen_src_{commit}'
        snap.mkdir(exist_ok=False)
        with (snap/'source.zip').open('xb') as f:
            subprocess.run(['git','archive','--format=zip',commit,'src'],cwd=ROOT,stdout=f,check=True)
        with zipfile.ZipFile(snap/'source.zip') as z:
            if any(not (snap/n).resolve().is_relative_to(snap.resolve()) for n in z.namelist()): raise ValueError('unsafe source archive')
            z.extractall(snap)
        snapshots[ds]={'implementation_commit':commit,'root':str(snap.relative_to(ROOT)),'archive_sha256':sha(snap/'source.zip')}
    write_rows(OUT/'manifests/formal200.jsonl',samples)
    write_rows(OUT/'manifests/formal200_subsets.jsonl',subsets)
    write_rows(OUT/'manifests/test_population_clusters.jsonl',populations)
    writej(OUT/'manifests/formal_sampling_summary.json',reports)
    writej(OUT/'manifests/formal_preparation_complete.json',{'at_utc':utc(),'N':len(samples),'formal200_sha256':sha(OUT/'manifests/formal200.jsonl'),'inputs':inputs,'snapshots':snapshots,'cluster_sha256':sha(OUT/'manifests/test_population_clusters.jsonl'),'subset_sha256':sha(OUT/'manifests/formal200_subsets.jsonl'),'code_sha256':sha(Path(__file__)),'new_quality_judgments_observed':False,'paid_api_calls':0,'purpose':'fixed v4.1 formal L2 and future LLM sample; no summary regeneration for outcome evaluation or final tuning'})
    print({'stage':'formal_prepared','N':len(samples),'clusters':[(x['dataset'],x['sample_unique_clusters']) for x in reports]},flush=True)


if __name__=='__main__': prepare_formal()
