"""Guard against outcome-driven strata, wrong mappings and implicit ID joins."""
import pytest
from scripts.audit.access_revision_common import structural_group, unique_map, sample_manifest, config_diff


def test_size_does_not_imply_containment():
    assert structural_group([1,2],[3,4,5]) == ('C01_smaller',None,'neither_contains_other')
    assert structural_group([1,2],[1,2,3]) == ('C01_smaller',None,'C01_strict_subset')
    assert structural_group([],[]) == ('equal_size','both_empty','identical')
    assert structural_group(range(80),range(1,81)) == ('equal_size','both_at_80','neither_contains_other')


def test_duplicate_join_fails():
    with pytest.raises(ValueError,match='duplicate ID'):
        unique_map([{'id':'a'},{'id':'a'}])


def test_sampling_order_invariant_and_reference_blind():
    docs=[{'id':str(i),'documents':[{'source_order':0,'document_id':str(i),'sections':[{'section_id':'s','sentences':[{'text':'word '*(i+1)}]}]}],'references':['unused']} for i in range(40)]
    before=sample_manifest(docs,'govreport',5,'pilot')
    for d in docs: d['references']=['changed']
    assert sample_manifest(list(reversed(docs)),'govreport',5,'pilot')==before
    assert [sum(r['stratum']==h for r in before) for h in range(4)]==[5]*4


def test_config_diff_only_ignores_declared_study_metadata():
    left={'study':{'id':'x'},'candidate_budget':{'min_per_route':20},'seed':1}
    right={'study':{'id':'y'},'candidate_budget':{'min_per_route':0},'seed':2}
    assert config_diff(left,right)=={'candidate_budget.min_per_route':{'C01':20,'control':0},'seed':{'C01':1,'control':2}}


def test_recall_empty_denominators_and_decomposition():
    from scripts.audit.access_revision_coverage import counts
    assert counts([], [1,2], [1])['pool_recall'] is None
    assert counts([3], [1,2], [1])['conditional_retention'] is None
    x=counts([1,2,3,4], [1,2,3,8], [2,8])
    assert x['pool_recall']==.75 and x['selected_recall']==.25
    assert x['conditional_retention']==1/3
    assert x['reference_n']==x['selected_hits']+x['pool_not_selected_n']+x['entry_missing_n']


def test_mmr_tie_break_and_word_budget():
    import numpy as np
    from scripts.audit.access_revision_math import independent_selection,independent_capacity
    cfg={'length_control':{'unit':'words','max_words':3},'optimizer':{'method':'mmr','lambda_relevance':.7}}
    spec={'importance_aggregation':'length_normalized','coverage_method':'max','weights':{'salience':1.,'facility_coverage':.8,'redundancy':.7}}
    ids,trace=independent_selection(['a b','c d','e'],[1.,1.,.1],np.eye(3),np.eye(3),cfg,spec,0)
    assert ids==[0,2] and trace[0]['margin']==0
    assert independent_capacity([4,6,9],8)==6


def test_greedy_stops_after_floor_but_can_fill_to_floor():
    import numpy as np
    from scripts.audit.access_revision_math import independent_selection
    cfg={'length_control':{'unit':'words','max_words':3},'optimizer':{'method':'greedy'}}
    spec={'importance_aggregation':'length_normalized','coverage_method':'max','weights':{'salience':1.,'facility_coverage':0.,'redundancy':0.}}
    args=(['a','b'],[1.,0.],np.eye(2),np.eye(2),cfg,spec)
    assert independent_selection(*args,1)[0]==[0]
    assert independent_selection(*args,2)[0]==[0,1]


def test_reservation_prefers_exclusive_nominees_over_rank():
    from scripts.audit.access_revision_records import stored_reservations
    pred={'candidate_records':[{'original_index':2,'inclusion_reasons':['reserve:a']},{'original_index':3,'inclusion_reasons':['reserve:b']}],
          'candidate_pool':{'allocation':{'effective_min_per_route':{'a':1,'b':1}},'route_proposals':{
              'a':[{'original_index':1,'rank':1,'reserved':False},{'original_index':2,'rank':2,'reserved':True}],
              'b':[{'original_index':1,'rank':1,'reserved':False},{'original_index':3,'rank':2,'reserved':True}]}}}
    assert stored_reservations(pred)
    pred['candidate_pool']['route_proposals']['a'][0]['reserved']=True
    assert not stored_reservations(pred)


def test_stored_mmr_checks_decision_order_and_arithmetic():
    from scripts.audit.access_revision_records import stored_mmr
    pred={'candidate_records':[{'original_index':5,'selector_salience':1.0}], 'selected_indices':[5],
          'optimizer_diagnostics':{'lambda_relevance':.7,'selection_order':[0],
              'steps':[{'step':1,'candidate_relative_index':0,'relevance':1.,'max_similarity_to_selected':0.,'mmr_score':.7}]}}
    assert stored_mmr(pred)
    pred['optimizer_diagnostics']['steps'][0]['mmr_score']=.8
    assert not stored_mmr(pred)


def test_dependency_groups_include_transitive_unsampled_bridge():
    from scripts.audit.access_revision_formal import dependency_groups
    def doc(key,texts):
        return {'id':key,'documents':[{'source_order':i,'sections':[{'sentences':[{'text':t}]}]} for i,t in enumerate(texts)]}
    a='alpha '*100; b='beta '*100
    docs=[doc('a',[a]),doc('b',[a,b]),doc('c',[b]),doc('d',['short']),doc('e',['short']),doc('f',['short','different'])]
    groups=dependency_groups(docs)
    assert groups['a']==groups['b']==groups['c']
    assert groups['d']==groups['e'] and groups['f']!=groups['d']
    assert dependency_groups(list(reversed(docs)))==groups
