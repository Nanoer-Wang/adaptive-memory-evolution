"""Check manuscript evidence, citations, inputs, and central numerical claims."""
from pathlib import Path
from collections import Counter
import hashlib, json, re

P=Path(__file__).resolve().parents[1]
ROOT=P.parent
def read(s): return json.loads((P/'evidence'/s).read_text(encoding='utf-8'))
def one(rows, **q):
    out=[r for r in rows if all(r.get(k)==v for k,v in q.items())]
    assert len(out)==1, q
    return out[0]
def near(a,b): assert abs(a-b)<1e-9, (a,b)

manifest=read('source_manifest.json')
found={k:hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v
       for k,v in manifest.items() if (ROOT/k).exists()}
assert all(found.values()), 'Frozen source changed'
pairs=read('C1/tradeoff_pairs.json')
assert len(pairs)==84
counts=Counter(r['condition'] for r in pairs if r['supported_tradeoff'])
assert dict(counts)=={'mixed-c8':9,'rapid_high-c8':12,'rapid_high-c16':6}, counts
assert len(read('C1/comparisons.json'))==168
effect=[]
for metric in ('stable_retention','adaptation_score'):
    r=one(read('C1/comparisons.json'), condition='rapid_high-c8',a='full_1',b='full_2',metric=metric)
    effect.append({'metric':metric, 'delta_alpha2_minus_alpha1':-r['delta'],
                   'ci95':[-r['ci95'][1],-r['ci95'][0]],'holm_p':r['p_holm']})
near(effect[0]['delta_alpha2_minus_alpha1'],-.3337673611111111)
near(effect[1]['delta_alpha2_minus_alpha1'],.197265625)
b1=one(read('B1A/paired_comparisons.json'),dynamics='rapid_shift',pressure='high',baseline='replacement_matched',metric='accuracy')
near(b1['delta_adaptive_minus_baseline'],.2526041666666667)
fv=read('characterization/final_validation.json')
assert fv['total_formal_runs']==17568
assert sum(r['runs'] for r in fv['evidence'])==17568
assert all(r['mini_passed'] for r in fv['evidence'])
assert '251 passed' in (P/'evidence/characterization/tests-final.txt').read_text()
worlds={}
for n in ['B2','B3','B4','B5','B6','B7','C1','C2','C3','C4']:
    d=read(n+'/protocol.json');w=d['world_seeds']
    assert len(w)==24
    assert d['policy_seeds']==[11,23,42]
    worlds[n]={'worlds':len(w),'first':min(w),'last':max(w)}
texfiles=[P/'main.tex',*sorted((P/'sections').glob('*.tex')),*sorted((P/'tables').glob('*.tex'))]
text='\n'.join(f.read_text(encoding='utf-8') for f in texfiles)
inputs=re.findall(r'\\input\{([^}]+)\}', text)
assert all((P/(s+'.tex' if not s.endswith('.tex') else s)).is_file() for s in inputs)
figures=re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',text)
assert all((P/f).is_file() for f in figures)
cites=set(k.strip() for group in re.findall(r'\\cite\w*\{([^}]+)\}',text) for k in group.split(','))
bibkeys=set(re.findall(r'@\w+\{([^,]+),',(P/'references.bib').read_text(encoding='utf-8')))
assert cites<=bibkeys,cites-bibkeys
assert (P/'sections/introduction.tex').read_text().count('\\item ')==3
report={
  'scope':'Read-only audit of archived evidence and manuscript dependencies; no experiments rerun.',
  'original_sources_verified':len(found),'source_manifest_entries':len(manifest),
  'all_available_source_hashes_match':all(found.values()),
  'formal_characterization_runs':17568,'supported_tradeoff_pairs':sum(counts.values()),
  'supported_pairs_by_condition':dict(counts),'c1_alpha_1_to_2':effect,
  'b1a_replacement_control_accuracy_contrast':b1,
  'world_seed_ranges':worlds,'contribution_count':3,'references':len(bibkeys),
  'cited_references':len(cites),'tex_inputs_verified':len(inputs),'figure_inputs_verified':len(figures),
  'limitations_preserved':[
    'Replacement mean matching does not identify a complete causal effect independent of replacement.',
    'C1 compares a finite rate grid against fixed baseline configurations, not equally tuned policy families.',
    'Alpha zero retains event jumps and temporal aging.',
    'Larger capacities can favor fixed policies; capacity and interference identities co-vary.',
    'Dormant retention, misleading feedback, and semantic addressing expose failures.',
    'B8 is a small custom subset with one answer/judge backend; NFCats quality is not binary QA accuracy.',
    'Content slots do not bound all metadata; measured dynamics adds computation.'
  ]
}
(P/'evidence/claim_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:report[k] for k in ['original_sources_verified','supported_tradeoff_pairs','contribution_count','references','tex_inputs_verified']}))
