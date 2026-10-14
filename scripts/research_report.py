"""Export every research seed and a development-results figure, without reranking runs."""
import csv
import json
from pathlib import Path
import numpy as np
from src.config import ROOT,atomic_json


def main():
    roots=[ROOT/'runs/research-round-01',ROOT/'runs/research-round-02-timefix']
    destination=ROOT/'runs/research-analysis'
    destination.mkdir(parents=True,exist_ok=True)
    rows=[]
    audits=[]
    for root in roots:
        protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
        summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
        assert summary['completed_runs']==summary['planned_runs']
        assert summary['completed_runs']==len(protocol['seeds'])*len(protocol['variants'])
        for variant in protocol['variants']:
            for seed in protocol['seeds']:
                run=root/f'{variant}-seed{seed}'
                result=json.loads((run/'result.json').read_text(encoding='utf-8'))
                metadata=json.loads((run/'validation/run.json').read_text(encoding='utf-8'))
                predictions=[json.loads(line) for line in (run/'validation/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
                assert metadata['split']=='val' and metadata['code_sha256']==protocol['source_sha256']
                assert len(predictions)==len(set(r['question_id'] for r in predictions))==93
                assert sum(r['session_recall_at_k'] is not None for r in predictions)==91
                rows.append({'round':root.name,**result})
        audits.append({'round':root.name,'runs':summary['completed_runs'],'source_consistent':True,
                       'unique_validation_records_per_run':93,'answerable_per_run':91,'screen_passed':summary['screen_passed']})
    with (destination/'all_seeds.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    atomic_json(destination/'audit.json',{'total_runs':len(rows),'rounds':audits,'api_calls':0,
        'limitations':['Development selection and scoring share validation; no independent claim.',
                       'Seed standard deviation is not a confidence interval across datasets.',
                       'Wall time and RSS are process observations, not isolated speed benchmarks.']})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={'liquid':'Liquid','reset_state':'State reset','fixed_time':'Fixed gaps',
            'no_consolidation':'No consolidation','gru_matched':'Matched GRU','linear':'Linear'}
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharex=True,layout='constrained')
    for axis,root,title in zip(axes,roots,['Round 1: legacy dates','Round 2: clock-time fix']):
        variants=list(json.loads((root/'protocol.json').read_text(encoding='utf-8'))['variants'])
        for index,name in enumerate(variants):
            values=np.array([r['recall_at_5']*100 for r in rows if r['round']==root.name and r['variant']==name])
            axis.scatter(values,index+np.linspace(-.12,.12,len(values)),s=20,color='#8699ab',alpha=.65,zorder=2)
            axis.errorbar(values.mean(),index,xerr=values.std(ddof=1),fmt='o',capsize=4,color='#006d77',zorder=3)
        axis.set_yticks(range(len(variants)),[labels[v] for v in variants])
        axis.invert_yaxis();axis.set_title(title,fontsize=12);axis.set_xlabel('Validation Recall@5 (%)')
        axis.grid(axis='x',alpha=.2);axis.spines[['top','right']].set_visible(False)
    fig.suptitle('Mechanism screening: five seeds per setting\nDots: individual seeds; teal: mean +/- sample SD (not confidence intervals)',fontsize=11)
    for extension in ['png','svg','pdf']:
        fig.savefig(destination/f'mechanism_screen.{extension}',dpi=180)
    plt.close(fig)
    print(f'Validated {len(rows)} runs; exported all seeds, audit and figures to {destination}')


if __name__=='__main__':
    main()
