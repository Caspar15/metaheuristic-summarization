"""Compute all predeclared candidate-coverage descriptions after dev40 passes."""
from __future__ import annotations

import statistics
from scripts.audit.access_revision_common import *


def safe_div(a,b):
    return a/b if b else None


def describe(values):
    values=[v for v in values if v is not None]
    if not values: return {'N':0,'mean':None,'median':None,'q1':None,'q3':None,'min':None,'max':None}
    q=[values[0]]*3 if len(values)==1 else statistics.quantiles(values,n=4,method='inclusive')
    return {'N':len(values),'mean':statistics.fmean(values),'median':statistics.median(values),'q1':q[0],'q3':q[2],'min':min(values),'max':max(values)}


def counts(g,c,s):
    g,c,s=set(g),set(c),set(s)
    if not s<=c: raise ValueError('selected set must be a subset of candidate pool')
    return {'reference_n':len(g),'pool_hits':len(g&c),'selected_hits':len(g&s),
            'entry_missing_n':len(g-c),'pool_not_selected_n':len(g&(c-s)),
            'pool_recall':safe_div(len(g&c),len(g)),
            'selected_recall':safe_div(len(g&s),len(g)),
            'conditional_retention':safe_div(len(g&s),len(g&c))}


def aggregate(records):
    out={'N':len(records)}
    for metric,num,den in [('pool_recall','pool_hits','reference_n'),('selected_recall','selected_hits','reference_n'),('conditional_retention','selected_hits','pool_hits')]:
        valid=[x for x in records if x[den]>0]
        out.update({metric+'_N':len(valid),metric+'_NA':len(records)-len(valid),
                    metric+'_macro':statistics.fmean(x[metric] for x in valid) if valid else None,
                    metric+'_numerator_sum':sum(x[num] for x in valid),
                    metric+'_denominator_sum':sum(x[den] for x in valid),
                    metric+'_micro':safe_div(sum(x[num] for x in valid),sum(x[den] for x in valid))})
    for metric in ['pool_size','pool_words','selected_size','selected_words','eligible_size','eligible_words','source_size','source_words','entry_missing_n','pool_not_selected_n']:
        out.update({metric+'_'+k:v for k,v in describe(x[metric] for x in records).items()})
    out['at_configured_cap_80_n']=sum(x['pool_size']==80 for x in records)
    out['at_configured_cap_80_fraction']=safe_div(out['at_configured_cap_80_n'],len(records))
    out['source_document_representation_macro']=statistics.fmean(safe_div(x['pool_document_count'],x['source_document_count']) for x in records) if records else None
    return out


def contrast(left,right,ids):
    a,b=aggregate([left[k] for k in ids]),aggregate([right[k] for k in ids])
    out={'N':len(ids)}
    for metric in ['pool_recall','selected_recall','conditional_retention']:
        diffs=[left[k][metric]-right[k][metric] for k in ids if left[k][metric] is not None and right[k][metric] is not None]
        out.update({metric+'_paired_delta_'+k:v for k,v in describe(diffs).items()})
        out[metric+'_positive_n']=sum(x>0 for x in diffs)
        out[metric+'_zero_n']=sum(x==0 for x in diffs)
        out[metric+'_negative_n']=sum(x<0 for x in diffs)
        for name,value in [('C01',a),('control',b)]:
            for suffix in ['N','NA','macro','micro','numerator_sum','denominator_sum']:
                out[name+'_'+metric+'_'+suffix]=value[metric+'_'+suffix]
        out[metric+'_micro_delta']=None if a[metric+'_micro'] is None or b[metric+'_micro'] is None else a[metric+'_micro']-b[metric+'_micro']
    for label,value in [('C01',a),('control',b)]:
        for k,v in value.items():
            if k.startswith(('pool_size_','pool_words_','source_words_')): out[label+'_'+k]=v
    return out


def run():
    verify_frozen_inputs()
    gate=readj(OUT/'audit/dev40_gate.json')
    if not gate['all_passed'] or gate['n']!=40: raise ValueError('dev40 replay has not passed')
    prep=readj(OUT/'manifests/preparation_complete.json')
    for name,key in [('pool_size_membership.jsonl','membership_sha256'),('pool_records.jsonl','pool_records_sha256')]:
        if sha(OUT/'coverage'/name)!=prep[key]: raise ValueError('pre-score membership changed')
    code={p.name:sha(p) for p in (ROOT/'scripts/audit').glob('access_revision*.py')}
    writej(OUT/'coverage/scoring_started.json',{'started_at_utc':utc(),'protocol':'4.1','preparation_sha256':sha(OUT/'manifests/preparation_complete.json'),'membership_sha256':prep['membership_sha256'],'dev40_gate_sha256':sha(OUT/'audit/dev40_gate.json'),'code_sha256':code})
    pool_list=list(rows(OUT/'coverage/pool_records.jsonl'))
    membership=list(rows(OUT/'coverage/pool_size_membership.jsonl'))
    output_rows=[]; main=[]; full=[]; subsets=[]; checks=[]
    for ds in DATASETS:
        manifest=readj(partition_path(ds)); expected=manifest['partitions']['dev']['selected_ids']; ids=set(expected)
        source={x['id']:flatten(x) for x in rows(canonical_path(ds)) if x['id'] in ids}
        pools={c:unique_map(x for x in pool_list if x['dataset']==ds and x['condition']==c) for c in CONDITIONS}
        for target in TARGETS:
            directory=ROOT/f'runs_v2/gate2_greedy_reference_v1/{ds}/dev/{target}'
            ev=readj(directory/'evidence.json'); cfg=readj(directory/'resolved_config.json')
            if ev['input_sha256']!=sha(canonical_path(ds)) or ev['rows_sha256']!=sha(directory/'rows.jsonl'):
                raise ValueError('greedy reference source or prediction hash mismatch')
            if cfg['partition_selected_ids_sha256']!=manifest['partitions']['dev']['selected_ids_sha256']:
                raise ValueError('greedy reference belongs to another partition')
            if cfg['max_words']!=({'govreport':650,'multinews':250}[ds]) or cfg['optimization_target']!=target:
                raise ValueError('greedy reference contract differs')
            reference=unique_map(rows(directory/'rows.jsonl'))
            if set(reference)!=ids: raise ValueError('greedy reference ID mismatch')
            for key,x in reference.items():
                indices=x['selected_indices']; src=source[key]
                if len(set(indices))!=len(indices) or any(not isinstance(i,int) or not 0<=i<len(src) for i in indices):
                    raise ValueError('invalid greedy original indices')
                if sum(len(src[i]['text'].split()) for i in indices)!=x['selected_words']:
                    raise ValueError('greedy words do not match canonical source')
                if any(len(src[i]['text'].split())>cfg['max_words'] for i in indices):
                    raise ValueError('greedy reference includes ineligible sentence')
            zero=sum(not x['selected_indices'] for x in reference.values())
            checks.append({'dataset':ds,'target':target,'N':len(reference),'empty_reference_N':zero,'evidence_sha256':sha(directory/'evidence.json'),'config_sha256':sha(directory/'resolved_config.json')})
            scored={}
            for condition in CONDITIONS:
                scored[condition]={}
                for key in expected:
                    record=pools[condition][key]
                    row=dict(record,target=target,**counts(reference[key]['selected_indices'],record['pool_indices'],record['selected_indices']))
                    scored[condition][key]=row; output_rows.append(row)
                main.append({'dataset':ds,'target':target,'condition':condition,**aggregate(scored[condition].values())})
            for condition in CONDITIONS[1:]:
                full.append({'dataset':ds,'target':target,'control':condition,'subset':'all_dev',**contrast(scored['C01'],scored[condition],expected)})
                matched=[x for x in membership if x['dataset']==ds and x['control']==condition]
                for group in ['C01_smaller','equal_size','C01_larger','both_at_80','equal_positive_below_80','both_empty']:
                    chosen=[x['id'] for x in matched if x['size_group']==group or x['equal_size_child']==group]
                    subsets.append({'dataset':ds,'target':target,'control':condition,'subset':group,'fraction_of_full_dev':len(chosen)/len(expected),**contrast(scored['C01'],scored[condition],chosen)})
    if len(main)!=30 or len(full)!=24 or len(subsets)!=144: raise AssertionError('incomplete predeclared tables')
    if next(x for x in checks if x['dataset']=='multinews' and x['target']=='rouge2')['empty_reference_N']!=6:
        raise ValueError('unexpected number of empty greedy references')
    write_rows(OUT/'coverage/recall_per_document.jsonl',output_rows)
    write_tsv(OUT/'coverage/recall_conditions.tsv',main)
    write_tsv(OUT/'coverage/recall_contrasts_all.tsv',full)
    write_tsv(OUT/'coverage/recall_contrasts_by_size.tsv',subsets)
    writej(OUT/'coverage/reference_validation.json',checks)
    verify_frozen_inputs()
    writej(OUT/'coverage/scoring_complete.json',{'finished_at_utc':utc(),'new_recall_scores_observed':True,'condition_rows':len(main),'full_contrasts':len(full),'size_subset_rows':len(subsets),'per_document_rows':len(output_rows),'source_inputs_unchanged':True,'code_sha256':code})
    print({'stage':'coverage_complete','conditions':len(main),'contrasts':len(full),'size_subsets':len(subsets),'raw_rows':len(output_rows)},flush=True)


if __name__=='__main__': run()
