"""Independent downstream calculations; no production rank/pool/selector calls.

Inputs are frozen-pipeline features and matrices. This does not independently
validate encoder, TF-IDF, graph feature implementations or human usability.
"""
from __future__ import annotations

import numpy as np


def independent_pool(source_records, route_values, cfg):
    n=len(source_records)
    if not n:
        return {'indices':[], 'scores':{}, 'normalized':{}, 'ranks':{}, 'reasons':{}}
    routes=list(route_values)
    budget=cfg['candidate_budget']; k=min(n,budget['route_top_k'])
    weights={r:cfg['candidates'].get('route_weights',{}).get(r,1.0) for r in routes}
    ranks={r:{idx:j+1 for j,idx in enumerate(sorted(range(n),key=lambda i:(-route_values[r][i],i)))} for r in routes}
    proposals={r:{i for i in range(n) if ranks[r][i]<=k} for r in routes}
    scores={i:sum(weights[r]/(cfg['candidates']['rrf_constant']+ranks[r][i]) for r in routes) for i in range(n)}
    order=sorted(range(n),key=lambda i:(-scores[i],i))
    fused={i:j+1 for j,i in enumerate(order)}
    low,high=min(scores.values()),max(scores.values())
    normalized={i:1.0 if low==high else (scores[i]-low)/(high-low) for i in range(n)}
    reasons={i:{'route:'+r for r in routes if i in proposals[r]} for i in set().union(*proposals.values())}
    guard=cfg['coverage_guard']
    if guard.get('section') or guard.get('position') or not guard.get('enabled') or not guard.get('document'):
        raise ValueError('independent checker only supports the frozen document guard')
    docs={r['document_id'] for r in source_records}
    guarded={min((i for i in range(n) if source_records[i]['document_id']==d),key=lambda i:fused[i]) for d in docs}
    guarded=set(sorted(guarded,key=lambda i:fused[i])[:guard['max_items']])
    for i in guarded: reasons.setdefault(i,set()).add('guard:document')
    mandatory=set(guarded)
    for route in routes:
        members=sorted(proposals[route],key=lambda i:(sum(i in proposals[r] for r in routes)!=1,ranks[route][i]))
        count=budget['min_per_route']
        count=count[route] if isinstance(count,dict) else count
        for i in members[:min(count,k)]:
            mandatory.add(i); reasons.setdefault(i,set()).add('reserve:'+route)
    chosen=set(mandatory); cap=min(budget['total'],n)
    if len(chosen)>cap: raise ValueError('mandatory pool exceeds cap')
    universe=set(reasons)
    for i in order:
        if len(chosen)>=cap: break
        if i in universe and i not in chosen:
            chosen.add(i); reasons[i].add('rrf_fill')
    orig=lambda i:source_records[i]['original_index']
    return {'indices':sorted(orig(i) for i in chosen),'scores':{orig(i):scores[i] for i in range(n)},
            'normalized':{orig(i):normalized[i] for i in range(n)},'ranks':{orig(i):fused[i] for i in range(n)},
            'reasons':{orig(i):sorted(reasons[i]) for i in chosen},'guard_indices':sorted(orig(i) for i in guarded),
            'route_ranks':{r:{orig(i):rank for i,rank in values.items()} for r,values in ranks.items()},
            'global_fusion_min':low,'global_fusion_max':high}


def independent_capacity(lengths, budget):
    reachable={0}
    for length in lengths:
        reachable |= {old+length for old in tuple(reachable) if old+length<=budget}
    return max(reachable)


def independent_selection(sentences,salience,similarity,coverage,cfg,spec,min_words):
    if spec['importance_aggregation']!='length_normalized' or spec['coverage_method'] not in ['max','set']:
        raise ValueError('unsupported objective; do not silently approximate')
    weights=spec['weights']; lengths=[len(t.split()) for t in sentences]
    if cfg['length_control']['unit']!='words': raise ValueError('expected word budget')
    cap=cfg['length_control']['max_words']; method=cfg['optimizer']['method']
    if method not in ['mmr','greedy']: raise ValueError('unsupported selector')
    a=np.asarray(salience,dtype=float); sim=np.asarray(similarity,dtype=float); cov=np.asarray(coverage,dtype=float)

    def utility(indices):
        ids=sorted(indices)
        if not ids: return 0.0
        importance=float(np.sum(a[ids]))/sum(max(1,lengths[i]) for i in ids)
        overlap=float(np.mean([sim[i,j] for j in ids for i in ids if i<j])) if len(ids)>1 else 0.0
        represented=float(np.mean(np.max(cov[:,ids],axis=1)))
        return weights['salience']*importance+weights['facility_coverage']*represented-weights['redundancy']*overlap

    chosen=[]; trace=[]; remaining=set(range(len(sentences)))
    while remaining:
        used=sum(lengths[i] for i in chosen)
        feasible=[i for i in sorted(remaining) if used+lengths[i]<=cap]
        options=[]
        for i in feasible:
            if method=='mmr':
                red=float(max((sim[i,j] for j in chosen),default=0.0))
                w=cfg['optimizer'].get('lambda_relevance',.7)
                value=w*a[i]-(1-w)*red
            else:
                red=None; value=utility(chosen+[i])
            options.append({'relative_index':i,'value':float(value),'relevance':float(a[i]),'max_similarity':red})
        if not options: break
        ordered=sorted(options,key=lambda x:(-x['value'],x['relative_index']))
        best=ordered[0]
        stop=method=='greedy' and bool(chosen) and used>=min_words and best['value']<=utility(chosen)+1e-12
        trace.append({'step':len(chosen)+1,'used_words':used,'current_utility':utility(chosen),
                      'best_relative_index':best['relative_index'],'stopped_without_addition':stop,
                      'margin':None if len(ordered)<2 else ordered[0]['value']-ordered[1]['value'],
                      'all_feasible_candidates':options})
        if stop: break
        chosen.append(best['relative_index']); remaining.remove(best['relative_index'])
    return chosen,trace
