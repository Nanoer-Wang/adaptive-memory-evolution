"""Evaluate frozen read-budget policies against measured paired task outcomes."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json,hashlib,sqlite3,html
from collections import defaultdict,Counter
import numpy as np
import torch
from src.config import atomic_json,load_config,resolve
from src.provenance import snapshot_source
from src.memory.task_controller import TaskController
from scripts.context_utility import OUT,SEEDS,KINDS,LAMBDA,read

def summary(records):
    grouped=defaultdict(list)
    for r in records:grouped[r['split'],r['method']].append(r)
    result=[]
    for (split,method),rr in grouped.items():
        # One record/question/seed; equal seeds, not independent observations.
        result.append(dict(split=split,method=method,questions=len({r['question_id'] for r in rr}),seeds=sorted({r['seed'] for r in rr}),
            accuracy=float(np.mean([r['correct'] for r in rr])),prompt_tokens=float(np.mean([r['prompt_tokens'] for r in rr])),
            completion_tokens=float(np.mean([r['completion_tokens'] for r in rr])),total_answer_tokens=float(np.mean([r['prompt_tokens']+r['completion_tokens'] for r in rr])),
            utility=float(np.mean([r['utility'] for r in rr])),compact_fraction=float(np.mean([r['action']=='compact' for r in rr])),
            answer_calls=float(np.mean([r.get('answer_calls',1) for r in rr])),abstention_fraction=float(np.mean([r['abstain'] for r in rr])),
            action_oracle_regret=float(np.mean([r['action_oracle_regret'] for r in rr])),
            harmful_compact_rate=float(np.mean([r['harmful_compact'] for r in rr])),unnecessary_full_rate=float(np.mean([r['unnecessary_full'] for r in rr]))))
    return result

def paired(records,a,b):
    group=defaultdict(list)
    for r in records:
        if r['split']=='test' and r['method'] in (a,b):group[r['question_id'],r['method']].append(r)
    ids=sorted({q for q,m in group if (q,a) in group and (q,b) in group});rng=np.random.default_rng(20261013);out={}
    for metric in ['correct','utility','prompt_tokens']:
        d=np.array([np.mean([r[metric] for r in group[q,a]])-np.mean([r[metric] for r in group[q,b]]) for q in ids]);bs=d[rng.integers(0,len(d),(10000,len(d)))].mean(1)
        out[metric]=dict(mean=float(d.mean()),ci95=np.quantile(bs,[.025,.975]).tolist(),question_differences=dict(zip(ids,d.tolist())),
                        caution='Descriptive small-sample bootstrap; degenerate intervals do not prove equality. No multiple-comparison adjustment.')
    return dict(a=a,b=b,n=len(ids),metrics=out)

def main():
    torch.set_num_threads(1);assert (OUT/'qa_complete.json').exists() and (OUT/'training_complete.json').exists()
    protocol=read(OUT/'protocol.json');features=read(OUT/'features.json');normal=np.load(OUT/'normalization.npz');rows=[read(p) for p in (OUT/'qa').glob('*.json')];by={(r['question_id'],r['method']):r for r in rows}
    assert all(r['correct'] in (0.,1.) for r in rows)
    checkpoints={};models={};parameters={}
    for kind in KINDS:
        for seed in SEEDS:
            path=OUT/'checkpoints'/f'{kind}-{seed}'/'best.pt';checkpoint=torch.load(path,weights_only=True)
            assert not(set(checkpoint['train_ids'])|set(checkpoint['val_ids']))&set(protocol['question_ids']['test'])
            m=TaskController(kind);m.load_state_dict(checkpoint['state_dict']);m.eval();models[kind,seed]=m;parameters[kind]=sum(p.numel() for p in m.parameters());checkpoints[kind,seed]=dict(epoch=checkpoint['epoch'],sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    threshold=float(read(OUT/'rule.json')['threshold']);records=[]
    with torch.inference_mode():
        for q,d in features.items():
            x=torch.tensor((np.array(d['x'])-normal['mean'])/normal['std'],dtype=torch.float32);dt=torch.tensor(d['dt'],dtype=torch.float32)
            choices=[('always_full',0,'full',None),('always_compact',0,'compact',None),('concentration_rule',0,'compact' if d['concentration']>=threshold else 'full',None)]
            for (kind,seed),m in models.items():
                pred=float(m(x,dt));choices.append((kind,seed,'compact' if pred>=0 else 'full',pred))
                if kind=='liquid':
                    for name,kw in [('liquid_reset',dict(reset=True)),('liquid_zero_time',dict(zero_time=True))]:
                        p=float(m(x,dt,**kw));choices.append((name,seed,'compact' if p>=0 else 'full',p))
            for method,seed,action,pred in choices:
                r=by[q,action];tokens=r['generation']['usage'];records.append(dict(question_id=q,split=d['split'],category=d['category'],method=method,seed=seed,action=action,predicted_utility_difference=pred,
                    correct=r['correct'],abstain=r['parsed']['abstain'],prompt_tokens=tokens['prompt_tokens'],completion_tokens=tokens['completion_tokens'],utility=r['correct']-LAMBDA*tokens['prompt_tokens']/4000,
                    generation_hash=r['generation']['request_hash'],controller_parameters=parameters.get(method)))
            compact=by[q,'compact'];fallback=compact['parsed']['abstain'];final=by[q,'full'] if fallback else compact
            prompt=compact['generation']['usage']['prompt_tokens']+(by[q,'full']['generation']['usage']['prompt_tokens'] if fallback else 0)
            completion=compact['generation']['usage']['completion_tokens']+(by[q,'full']['generation']['usage']['completion_tokens'] if fallback else 0)
            records.append(dict(question_id=q,split=d['split'],category=d['category'],method='abstention_cascade',seed=0,action='full' if fallback else 'compact',predicted_utility_difference=None,
                correct=final['correct'],abstain=final['parsed']['abstain'],prompt_tokens=prompt,completion_tokens=completion,utility=final['correct']-LAMBDA*prompt/4000,
                generation_hash=final['generation']['request_hash'],answer_calls=1+int(fallback),controller_parameters=0))
    for r in records:
        q=r['question_id'];f,c=by[q,'full'],by[q,'compact'];oracle=max(t['correct']-LAMBDA*t['generation']['usage']['prompt_tokens']/4000 for t in [f,c])
        r['action_oracle_regret']=oracle-r['utility']
        r['harmful_compact']=r['action']=='compact' and f['correct']==1 and c['correct']==0
        r['unnecessary_full']=r['action']=='full' and f['correct']==c['correct']
    ss=summary(records);pairs=[paired(records,a,b) for a,b in [('always_compact','always_full'),('abstention_cascade','always_full'),('liquid','abstention_cascade'),('linear','always_compact'),('gru','always_compact'),('liquid','always_compact'),('liquid','linear'),('liquid','gru'),('liquid','liquid_reset'),('liquid','liquid_zero_time')]]
    differences=[]
    for q,d in features.items():
        a,b=by[q,'full'],by[q,'compact'];differences.append(dict(question_id=q,split=d['split'],category=d['category'],full_correct=a['correct'],compact_correct=b['correct'],compact_minus_full=b['correct']-a['correct'],
            prompt_reduction=1-b['generation']['usage']['prompt_tokens']/a['generation']['usage']['prompt_tokens'],full_abstain=a['parsed']['abstain'],compact_abstain=b['parsed']['abstain']))
    keys={r[k]['request_hash'] for r in rows for k in ['generation','judge_response'] if k in r};attempts=[]
    with sqlite3.connect(resolve(load_config('configs/minute_precision.yaml')['api']['cache_db'])) as db:
        for key in keys:attempts += [dict(request_hash=key,started=s,status=status,elapsed=t,usage=json.loads(u)) for s,status,t,u in db.execute('SELECT started,status,elapsed,usage_json FROM requests_log WHERE request_hash=?',(key,))]
    attempts.sort(key=lambda r:r['started']);left=0;peak=0
    for right,r in enumerate(attempts):
        while attempts[left]['started']<=r['started']-60:left+=1
        peak=max(peak,right-left+1)
    api=dict(generations=len({r['generation']['request_hash'] for r in rows}),judge_requests=len({r['judge_response']['request_hash'] for r in rows if 'judge_response' in r}),http_attempts=len(attempts),statuses=dict(Counter(r['status'] for r in attempts)),reported_tokens=sum(r['usage'].get('total_tokens',0) or 0 for r in attempts),rolling60_max=peak)
    atomic_json(OUT/'policy_records.json',records);atomic_json(OUT/'summary.json',dict(methods=ss,comparisons=pairs,api=api,parameters=parameters));atomic_json(OUT/'paired_actions.json',differences);atomic_json(OUT/'api_attempts.json',attempts)
    atomic_json(OUT/'checkpoint_selection.json',[dict(kind=k,seed=s,**v) for (k,s),v in checkpoints.items()])
    # Per-action raw predictions remain directly inspectable, including failures.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    shown=[r for r in ss if r['split']=='test' and r['method'] in ['always_full','always_compact','abstention_cascade','concentration_rule','linear','mlp','gru','liquid']]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8));y=np.arange(len(shown));labels=[r['method'] for r in shown]
    axes[0].barh(y,[r['accuracy']*12 for r in shown]);axes[0].set_yticks(y,labels);axes[0].invert_yaxis();axes[0].set_xlim(0,12);axes[0].set_xlabel('Correct / 12 (seed-averaged policies)')
    axes[1].barh(y,[r['prompt_tokens'] for r in shown],color='#53a18e');axes[1].set_yticks(y,labels);axes[1].invert_yaxis();axes[1].set_xlabel('Mean deployed answer prompt tokens')
    fig.suptitle('Measured task utility, frozen context-budget decisions');fig.tight_layout();fig.savefig(OUT/'results.png',dpi=160);plt.close(fig)
    table=''.join(f"<tr><td>{r['method']}</td><td>{r['accuracy']:.1%}</td><td>{r['prompt_tokens']:.0f}</td><td>{r['compact_fraction']:.1%}</td></tr>" for r in shown)
    cases=[]
    for q in protocol['question_ids']['test']:
        a,b=by[q,'full'],by[q,'compact'];cases.append(f"<details><summary>{html.escape(q)} · {html.escape(a['category'])} · {a['correct']:.0f}→{b['correct']:.0f}</summary><p>{html.escape(a['question'])}</p><p>参考：{html.escape(a['answer'])}</p><h4>完整</h4><p>{html.escape(a['hypothesis'])}</p><h4>精简</h4><p>{html.escape(b['hypothesis'])}</p><details><summary>完整上下文</summary><pre>{html.escape(a['context'])}</pre></details><details><summary>精简上下文</summary><pre>{html.escape(b['context'])}</pre></details></details>")
    doc=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>任务收益监督的上下文决策</title><style>body{{max-width:1050px;margin:36px auto;font:16px/1.7 system-ui;color:#203040}}img{{width:100%}}table{{width:100%;border-collapse:collapse}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}pre{{white-space:pre-wrap;background:#f4f6f8;padding:14px}}details{{margin:12px 0}}</style><h1>直接用任务收益学习上下文取舍</h1>
    <p>24训练题、12验证题、12此前未进入问答链路的测试题，均来自已暴露离线开发数据。每题实际生成完整和精简两种答案；相同64KB强线性记忆、相同top5来源。训练标签为正确率收益减输入token代价，而不是证据召回。</p>
    <p>12个控制器检查点先训练冻结，随后才生成测试答案。线性/MLP/GRU/液态均输入同样14种存储与查询特征。三个训练种子不算三个独立测试集。液态为现有指数门控递推，不是精确CfC；这是查询时阅读预算控制，每次查询重新初始化状态，不是新存储策略或已证实的长期动力学。重置消融只去除递推状态，输入中的历史统计仍在；去时间消融只去除单元额外时间参数。</p>
    <img src="results.png"><table><tr><th>测试方法</th><th>任务正确率</th><th>平均输入token</th><th>选择精简比例</th></tr>{table}</table>
    <p>始终精简是必要强基线；只有超过它才能说明条件决策有额外价值。测试生成前另登记“精简拒答后重试完整”的简单级联基线；其成本包含两次调用，按实际已生成答案回放。一次固定生成的潜在结果被所有策略共享，避免把相同答案算多次；缺少重复生成和大样本确证，统计区间仅探索性。原始共同压缩造成的事实丢失仍不能恢复。生成时间受请求顺序、限流和服务波动影响，不作因果速度结论。</p>
    <p>API：{html.escape(json.dumps(api,ensure_ascii=False))}。显式拒答对本组可回答任务直接计为未完成；其他答案由更严格的参考裁判评估，仍需复核。</p><h2>全部测试题</h2>{''.join(cases)}</html>'''
    (OUT/'report.html').write_text(doc,encoding='utf-8');sig,archive=snapshot_source();atomic_json(OUT/'analysis_provenance.json',dict(source=sig,archive=archive))
    print(json.dumps(dict(test=shown,api=api,parameters=parameters),indent=2))

if __name__=='__main__':main()
