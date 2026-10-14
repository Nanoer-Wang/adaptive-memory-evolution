"""Measured fallback utility, omission-feature ablation and source-history controls."""
import sys,json,hashlib,sqlite3,html,re
from pathlib import Path
from collections import defaultdict,Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.config import atomic_json,load_config,resolve
from src.provenance import snapshot_source
from src.memory.task_controller import TaskController
from src.memory.semantic_reading import FEATURES
from scripts.semantic_fallback import OUT,read,data,transform,KINDS,SEEDS,LAMBDA,BUDGETS
ACTION_DIR=OUT/'qa'
EXTRA_ACTION_DIRS=[]

def main():
    torch.set_num_threads(1)
    supplemental=ROOT/'runs/semantic-fallback-augmentation-v1'
    if (supplemental/'protocol.json').exists():assert (supplemental/'training_complete.json').exists(), 'Do not inspect test outcomes until supplemental models frozen'
    assert (OUT/'qa_complete.json').exists();ds=data(True);normal=np.load(OUT/'normalization.npz');models={};checkpoints=[]
    ids=read(OUT/'protocol.json')['question_ids']
    for kind in KINDS:
        for seed in SEEDS:
            p=OUT/'checkpoints'/f'{kind}-{seed}'/'best.pt';c=torch.load(p,weights_only=True)
            assert set(c['train_ids'])==set(ids['train']) and set(c['val_ids'])==set(ids['val']) and not(set(c['train_ids'])|set(c['val_ids']))&set(ids['test'])
            m=TaskController('linear' if kind=='linear_no_omission' else kind,features=len(FEATURES));m.load_state_dict(c['state_dict']);m.eval();models[kind,seed]=m
            checkpoints.append(dict(kind=kind,seed=seed,epoch=c['epoch'],parameters=sum(p.numel() for p in m.parameters()),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    records=[];signal=[]
    with torch.inference_mode():
        for key,d in ds.items():
            compact,full=d['compact'],d['full'];uncertain=compact['parsed']['abstain'] or bool(re.search(r"insufficient|cannot (?:answer|determine|find)|do not have|don't have|not (?:contain|provided)|can't",compact['hypothesis'],re.I))
            choices=[('never_fallback',0,False,None),('always_fallback',0,True,None),('full_only',0,True,None),('abstention_fallback',0,compact['parsed']['abstain'],None),('uncertainty_rule',0,uncertain,None)]
            for (kind,seed),m in models.items():
                x,dt=transform(d,normal,kind);pred=float(m(x,dt));choices.append((kind,seed,pred>0,pred))
                if kind=='liquid':
                    for name,p in [('liquid_reset',m(x,dt,reset=True)),('liquid_zero_time',m(x,dt,zero_time=True)),('liquid_reverse',m(x.flip(0),dt.flip(0)))]:
                        pred=float(p);choices.append((name,seed,pred>0,pred))
            for method,seed,fallback,pred in choices:
                answer=full if fallback else compact;repairs=compact.get('repair_attempts',[])
                prompt=compact['generation']['usage']['prompt_tokens']+sum(r['usage']['prompt_tokens'] for r in repairs)+(full['generation']['usage']['prompt_tokens'] if fallback else 0)
                completion=compact['generation']['usage']['completion_tokens']+sum(r['usage']['completion_tokens'] for r in repairs)+(full['generation']['usage']['completion_tokens'] if fallback else 0);calls=1+len(repairs)+int(fallback)
                if method=='full_only':prompt=full['generation']['usage']['prompt_tokens'];completion=full['generation']['usage']['completion_tokens'];calls=1
                records.append(dict(question_id=d['question_id'],split=d['split'],category=d['category'],budget=d['budget'],method=method,seed=seed,fallback=bool(fallback and method!='full_only'),
                    predicted_gain=pred,correct=answer['correct'],abstain=answer['parsed']['abstain'],prompt_tokens=prompt,completion_tokens=completion,answer_calls=calls,
                    utility=answer['correct']-LAMBDA*prompt/4000,helpful_fallback=method!='full_only' and fallback and full['correct']>compact['correct'],
                    wasteful_fallback=method!='full_only' and fallback and full['correct']<=compact['correct'],missed_helpful_fallback=(not fallback) and full['correct']>compact['correct']))
            signal.append(dict(question_id=d['question_id'],split=d['split'],category=d['category'],budget=d['budget'],intervention=d.get('intervention',False),compact_correct=compact['correct'],full_correct=full['correct'],
                quality_gain=full['correct']-compact['correct'],target_gain=d['target'],compact_abstain=compact['parsed']['abstain'],omitted_risk_max=float(d['x'][:,6].max()),missing_numbers=float(d['x'][:,8].mean())))
    groups=defaultdict(list)
    for r in records:
        groups[r['split'],r['budget'],r['method']].append(r);groups[r['split'],'pooled',r['method']].append(r)
    summaries=[]
    for (split,budget,method),rr in groups.items():
        summaries.append(dict(split=split,budget=budget,method=method,questions=len({r['question_id'] for r in rr}),
            **{k:float(np.mean([r[k] for r in rr])) for k in ['correct','prompt_tokens','completion_tokens','answer_calls','utility','fallback','helpful_fallback','wasteful_fallback','missed_helpful_fallback']}))
    comparisons=[];rng=np.random.default_rng(20261012);testids=ids['test']
    for a,b in [('linear','linear_no_omission'),('liquid','linear'),('liquid','gru'),('liquid','abstention_fallback'),('liquid','uncertainty_rule'),('liquid','full_only'),('liquid','liquid_reset'),('liquid','liquid_zero_time'),('liquid','liquid_reverse')]:
        metrics={}
        for metric in ['correct','prompt_tokens','utility','answer_calls']:
            values=np.array([np.mean([r[metric] for r in records if r['question_id']==q and r['method']==a])-np.mean([r[metric] for r in records if r['question_id']==q and r['method']==b]) for q in testids])
            boots=values[rng.integers(0,len(values),(10000,len(values)))].mean(1)
            metrics[metric]=dict(mean=float(values.mean()),ci95=np.quantile(boots,[.025,.975]).tolist())
        comparisons.append(dict(a=a,b=b,question_units=len(testids),metrics=metrics,caution='Question-clustered descriptive bootstrap; average budgets/seeds within question; no multiplicity correction.'))
    actions=[read(p) for directory in [ACTION_DIR]+EXTRA_ACTION_DIRS for p in directory.glob('*.json')];keys={r[k]['request_hash'] for r in actions if not(r['split']!='test' and r['method']=='full') for k in ['generation','judge_response'] if k in r}
    repair_keys={g['request_hash'] for r in actions for g in r.get('repair_attempts',[])};repair_judge_keys={g['request_hash'] for r in actions for g in r.get('repair_judges',[])};keys|=repair_keys|repair_judge_keys;attempts=[]
    with sqlite3.connect(resolve(load_config('configs/minute_precision.yaml')['api']['cache_db'])) as db:
        for key in keys:attempts.extend(dict(request_hash=key,started=s,status=st,elapsed=e,usage=json.loads(u)) for s,st,e,u in db.execute('SELECT started,status,elapsed,usage_json FROM requests_log WHERE request_hash=?',(key,)))
    attempts.sort(key=lambda r:r['started']);left=0;peak=0
    for right,r in enumerate(attempts):
        while attempts[left]['started']<=r['started']-60:left+=1
        peak=max(peak,right-left+1)
    api=dict(new_answer_requests=len({r['generation']['request_hash'] for r in actions if not(r['split']!='test' and r['method']=='full')}|repair_keys),contract_repairs=len(repair_keys),reused_development_full_answers=30,
        new_judge_requests=len({r['judge_response']['request_hash'] for r in actions if not(r['split']!='test' and r['method']=='full') and 'judge_response' in r}|repair_judge_keys),http_attempts=len(attempts),
        statuses=dict(Counter(r['status'] for r in attempts)),reported_tokens=sum(r['usage'].get('total_tokens',0) or 0 for r in attempts),rolling60_max=peak)
    atomic_json(OUT/'policy_records.json',records);atomic_json(OUT/'summary.json',dict(methods=summaries,comparisons=comparisons,api=api));atomic_json(OUT/'signal.json',signal);atomic_json(OUT/'api_attempts.json',attempts);atomic_json(OUT/'checkpoint_selection.json',checkpoints)
    shown=[r for r in summaries if r['split']=='test' and r['budget']=='pooled' and r['method'] in ['never_fallback','full_only','always_fallback','abstention_fallback','uncertainty_rule','linear_no_omission','linear','gru','liquid']]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(13,5.8));y=np.arange(len(shown));labels=[r['method'] for r in shown]
    ax[0].barh(y,[r['correct']*100 for r in shown],color='#338c8d');ax[0].set_yticks(y,labels);ax[0].invert_yaxis();ax[0].set_xlim(0,100);ax[0].set_xlabel('Factual reference accuracy (%)')
    ax[1].barh(y,[r['prompt_tokens'] for r in shown],color='#476c9b');ax[1].set_yticks(y,labels);ax[1].invert_yaxis();ax[1].set_xlabel('Prompt tokens including optional full call')
    fig.suptitle('Semantic omissions and actual fallback benefit; 12 held-out QA questions');fig.tight_layout();fig.savefig(OUT/'results.png',dpi=160);plt.close(fig)
    table=''.join(f"<tr><td>{r['method']}</td><td>{r['correct']:.1%}</td><td>{r['prompt_tokens']:.0f}</td><td>{r['answer_calls']:.2f}</td><td>{r['missed_helpful_fallback']:.1%}</td></tr>" for r in shown)
    counts=[]
    for split in ['train','val','test']:
        for budget in BUDGETS:
            for intervention in [False,True]:
                rr=[r for r in signal if r['split']==split and r['budget']==budget and r['intervention']==intervention]
                if rr:counts.append(dict(split=split,budget=budget,intervention=intervention,n=len(rr),full_better=sum(r['quality_gain']>0 for r in rr),compact_better=sum(r['quality_gain']<0 for r in rr),equal=sum(r['quality_gain']==0 for r in rr)))
    atomic_json(OUT/'signal_summary.json',counts)
    cases=[]
    for q in testids:
        rows=[r for r in actions if r['question_id']==q];content=''.join(f"<h4>{r['method']} · {r['correct']:.0f}</h4><p>{html.escape(r['hypothesis'])}</p><details><summary>实际上下文</summary><pre>{html.escape(r['context'])}</pre></details>" for r in rows)
        cases.append(f"<details><summary>{q} · {html.escape(rows[0]['category'])}</summary><p>{html.escape(rows[0]['question'])}</p><p>参考：{html.escape(rows[0]['answer'])}</p>{content}</details>")
    doc=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>语义缺失与补读价值</title><style>body{{max-width:1120px;margin:36px auto;padding:0 18px;font:16px/1.7 system-ui;color:#234}}img{{width:100%}}table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;text-align:left;border-bottom:1px solid #ddd}}pre{{white-space:pre-wrap;background:#f4f6f8;padding:12px}}details{{margin:14px 0}}</style><h1>语义缺失 + 实际任务损失 → 补读决策</h1>
    <p>先读600或1800字符，再预测完整读取是否值得。42道公开事实题：20训练、10验证、12此前未调用问答的测试题；两档预算按问题聚类，不能当24道独立测试。测试包含四种题型；助手回忆题的新QA池已耗尽，没有冒充新题重用。30份既有开发集完整回答明确复用，本报告涉及{api['new_answer_requests']}份新回答。{'本组训练另含20个扣除最相关来源的干预样本，测试没有这类干预；测试回答可能在补充模型冻结前已生成，但训练只读取开发集，全部补充模型冻结后才分析测试答案。' if EXTRA_ACTION_DIRS else '本组只用自然压缩的40个训练动作对。'}</p>
    <p>新增18维可观察输入：语义覆盖、被省略内容的查询相关度、数字与查询词缺失、来源时间，以及已获得的精简回答是否拒答。相似度不等于蕴含或事实正确；这些都是风险代理。所有模型隐藏维度仍为8，线性去缺失特征对照与完整线性分别训练。训练标签是实际补读正确率增益减新增输入token成本，始终计入精简和可选完整两次调用。</p>
    <img src="results.png"><table><tr><th>测试方法</th><th>正确率</th><th>平均输入token</th><th>平均回答调用</th><th>漏掉有益补读比例</th></tr>{table}</table>
    <h2>是否补上了真实任务损失监督</h2><pre>{html.escape(json.dumps(counts,ensure_ascii=False,indent=2))}</pre>
    <h2>历史机制的范围</h2><p>本轮状态沿同一问题的检索来源日期序列递推，在不同问题之间重置。它检验的是证据历史整合，不是跨任务持续Agent记忆。没有把无关用户的问题拼成一条历史。反转来源、重置状态和去时间是同权重诊断；反转也反转对应间隔，并不构成真实时间过程。只有相对强静态/GRU的收益和对时间干预的稳定响应，才可能支持更强机制声明。</p>
    <p>原始自然训练模型先冻结再生成测试答案；补充训练模型只读开发数据，在任何测试结果分析前冻结。单次缓存潜在结果供各策略共享，不是独立重复生成。两档预算、多种子先在问题内平均。样本小，区间仅探索性，数据仍来自已有开发集。语义提取使用本地MiniLM，原始准备开销另存父实验prepared.json，不把token下降当作端到端提速。</p>
    <h2>成对比较</h2><pre>{html.escape(json.dumps(comparisons,ensure_ascii=False,indent=2))}</pre><p>API：{html.escape(json.dumps(api,ensure_ascii=False))}</p><h2>全部测试回答与上下文</h2>{''.join(cases)}</html>'''
    (OUT/'report.html').write_text(doc,encoding='utf-8');source,archive=snapshot_source();atomic_json(OUT/'analysis_provenance.json',dict(source=source,archive=archive))
    print(json.dumps(dict(test=shown,signal=counts,api=api),indent=2))

if __name__=='__main__':main()
