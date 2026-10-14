"""All predefined hard budgets, order sensitivity, measured QA outcomes and costs."""
import sys,json,hashlib,html,sqlite3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.config import atomic_json
from src.memory.call_budget import CallBudget
from src.memory.task_controller import TaskController
from scripts import bidirectional_fallback as exp
from scripts import semantic_fallback as base
OUT=exp.OUT


def main():
    torch.set_num_threads(1);meta=exp.read(OUT/'training_complete.json');ds=exp.data(True);ds.update(exp.stage_data(True));val=[d for d in ds.values() if d['split']=='val'];test=[d for d in ds.values() if d['split']=='test']
    predictions={};thresholds=[];records=[];hashes={}
    for recipe in ['previous','staged','bidirectional']:
        normal=np.load(OUT/recipe/'normalization.npz')
        for kind in base.KINDS:
            for seed in base.SEEDS:
                path=OUT/recipe/f'{kind}-{seed}.pt';ck=torch.load(path,weights_only=False);hashes[str(path.relative_to(ROOT))]=base.sha(path)
                m=TaskController('linear' if kind=='linear_no_omission' else kind,features=18,hidden=8);m.load_state_dict(ck['state_dict']);m.eval()
                modes=['normal','reset','zero_time'] if kind=='liquid' else ['normal']
                for mode in modes:
                    with torch.no_grad():scores={id(d):float(m(*base.transform(d,normal,kind),reset=mode=='reset',zero_time=mode=='zero_time')) for d in val+test}
                    name=kind if mode=='normal' else kind+'_'+mode
                    for cap in [.1,.25,.5]:
                        vv=sorted([scores[id(d)] for d in val],reverse=True);threshold=max(0.,vv[int(cap*len(vv))])
                        thresholds.append(dict(recipe=recipe,method=name,seed=seed,cap=cap,threshold=threshold))
                        predictions[recipe,name,seed,cap]=(scores,threshold)
    for cap in [.1,.25,.5]:
        for name in ['never','always_request','abstention']:
            predictions['rule',name,0,cap]=({id(d):float(name=='always_request' or (name=='abstention' and d['compact']['parsed']['abstain'])) for d in test},.5)
    for (recipe,method,seed,cap),(scores,threshold) in predictions.items():
        for reading in base.BUDGETS+['600_to_1800']:
            tt=[d for d in test if d['budget']==reading]
            for order_seed in base.SEEDS:
                ordered=sorted(tt,key=lambda d:hashlib.sha256(f"budget-v1:{order_seed}:{d['question_id']}".encode()).hexdigest());limiter=CallBudget(cap)
                for position,d in enumerate(ordered,1):
                    requested=scores[id(d)]>threshold;admitted=limiter.step(requested);c,f=d['compact'],d['full'];chosen=f if admitted else c
                    tokens=c['generation']['usage']['prompt_tokens']+(f['generation']['usage']['prompt_tokens'] if admitted else 0)
                    records.append(dict(recipe=recipe,method=method,seed=seed,cap=cap,reading_budget=reading,order_seed=order_seed,position=position,question_id=d['question_id'],score=scores[id(d)],threshold=threshold,requested=requested,admitted=admitted,blocked=requested and not admitted,correct=chosen['correct'],prompt_tokens=tokens,calls=1+int(admitted),gain=f['correct']-c['correct'],helpful=admitted and f['correct']>c['correct'],harmful=admitted and f['correct']<c['correct'],used=limiter.used))
                    assert limiter.used<=int(cap*position+1e-8)
    summary=[]
    for endpoint in ['full','stage']:
        for recipe,method,cap in sorted({(r['recipe'],r['method'],r['cap']) for r in records}):
            rr=[r for r in records if (r['recipe'],r['method'],r['cap'])==(recipe,method,cap) and (r['reading_budget']=='600_to_1800')==(endpoint=='stage')]
            summary.append(dict(endpoint=endpoint,recipe=recipe,method=method,cap=cap,questions=12,**{k:float(np.mean([r[k] for r in rr])) for k in ['correct','prompt_tokens','calls','admitted','blocked','helpful','harmful']}))
    outcomes=[];keys=set()
    for p in sorted((OUT/'qa').glob('*.json')):
        r=exp.read(p);outcomes.append(dict(question_id=r['question_id'],method=r['method'],correct=r['correct'],question=r['question'],reference=r['answer'],hypothesis=r['hypothesis'],record_sha256=base.sha(p)))
        for response in [r.get('generation'),r.get('judge_response')]+r.get('repair_attempts',[])+r.get('repair_judges',[]):
            if response:keys.add(response['request_hash'])
    attempts=[]
    with sqlite3.connect(ROOT/'artifacts/api/responses.sqlite3') as db:
        for key in keys:
            attempts.extend(dict(request_hash=key,started=t,status=s,usage=json.loads(u)) for t,s,u in db.execute('select started,status,usage_json from requests_log where request_hash=?',(key,)))
    times=sorted(r['started'] for r in attempts);peak=max((sum(t-60<z<=t for z in times) for t in times),default=0)
    api=dict(unique_requests=len(keys),http_attempts=len(attempts),reported_tokens=sum(r['usage'].get('total_tokens',0) for r in attempts),rolling60_peak=peak)
    paired=[];rng=np.random.default_rng(20261013)
    for endpoint in ['full','stage']:
        for cap in [.1,.25,.5]:
            for recipe in ['previous','staged','bidirectional']:
                liquid=[r for r in records if r['recipe']==recipe and r['method']=='liquid' and r['cap']==cap and (r['reading_budget']=='600_to_1800')==(endpoint=='stage')]
                for comparator in ['linear_no_omission','linear','gru','abstention']:
                    other=[r for r in records if r['recipe']==('rule' if comparator=='abstention' else recipe) and r['method']==comparator and r['cap']==cap and (r['reading_budget']=='600_to_1800')==(endpoint=='stage')]
                    questions=sorted({r['question_id'] for r in liquid})
                    delta=np.array([np.mean([r['correct'] for r in liquid if r['question_id']==q])-np.mean([r['correct'] for r in other if r['question_id']==q]) for q in questions])
                    boot=delta[rng.integers(0,len(delta),size=(2000,len(delta)))].mean(1)
                    paired.append(dict(endpoint=endpoint,cap=cap,recipe=recipe,comparator=comparator,questions=len(delta),accuracy_delta=float(delta.mean()),exploratory_ci95=np.quantile(boot,[.025,.975]).tolist()))
    atomic_json(OUT/'paired_comparisons.json',paired)
    signals=[]
    for split in ['val','test']:
        for endpoint in ['full','stage']:
            rr=[d for d in ds.values() if d['split']==split and (d['budget']=='600_to_1800')==(endpoint=='stage')]
            signals.append(dict(split=split,endpoint=endpoint,pairs=len(rr),harmful=sum(d['target']<0 for d in rr),neutral=sum(d['target']==0 for d in rr),helpful=sum(d['target']>0 for d in rr)))
    atomic_json(OUT/'evaluation_signal.json',signals)
    atomic_json(OUT/'policy_records.json',records);atomic_json(OUT/'summary.json',summary);atomic_json(OUT/'thresholds.json',thresholds);atomic_json(OUT/'outcome_review.json',outcomes);atomic_json(OUT/'api_attempts.json',attempts)
    atomic_json(OUT/'audit.json',dict(checkpoint_sha256=hashes,every_stream_prefix_obeys_cap=True,api=api,training=meta,scope='Previously exposed12 questions; seeds, budgets and orderings are not independent samples. Hard cap applies to extra answer calls, not token or RPM budgets. Same-query recurrence only.'))
    table=''.join(f"<tr><td>{r['endpoint']}</td><td>{r['recipe']}</td><td>{r['method']}</td><td>{r['cap']:.0%}</td><td>{r['correct']:.2%}</td><td>{r['admitted']:.2%}</td><td>{r['prompt_tokens']:.0f}</td><td>{r['harmful']:.2%}</td></tr>" for r in summary)
    doc=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>双向收益与硬调用预算</title><style>body{{max-width:1150px;margin:40px auto;font:16px/1.8 system-ui}}table{{border-collapse:collapse;width:100%}}td,th{{padding:7px;border-bottom:1px solid #ddd}}code{{overflow-wrap:anywhere}}</style><h1>真实补读收益监督与硬预算</h1><p>固定18维观测、8维隐藏状态，线性/GRU/液态各3种子。比较之前60对训练数据、加入全部20对600→1800逐级补读后的80对、再加入20对标注辅助证据扩展后的100对。标注只用于训练干预；未把参考答案放入上下文。新增输入使用原始历史，属于额外信息的训练干预，不冒充原64KB存储策略。</p><p>实际标签计数（-1有害，0无益，1有益）：<code>{html.escape(json.dumps(meta['label_counts']))}</code>。这些是实际生成评分，不保证人为配平。模型目标为质量差，验证MSE选择训练轮次，验证分数设置正收益门槛。</p><p>硬限制：每个查询到达后，累计额外调用不超过 floor(预算比例×已到达查询数)。无初始额度、不能预支；有额度也只执行超过门槛的请求。分别运行600/1800字符阅读预算及3个固定哈希到达顺序。每个前缀已验证。10%预算在12题里最多1次，25%最多3次，50%最多6次；不是每分钟API限速，也不是token上限。</p><p>使用此前已暴露的12道测试题，属于开发性对照，不是新独立确认。两种预算、种子、顺序不增加独立样本数。历史状态只在单题来源间延续。reset和zero_time分别重新按验证分数设门槛，属于同权重预算匹配诊断。</p><table><tr><th>补读终点</th><th>训练</th><th>方法</th><th>硬预算</th><th>准确率</th><th>实际补读率</th><th>输入token</th><th>有害补读率</th></tr>{table}</table><p>本轮API：{html.escape(json.dumps(api))}。所有反事实答案用于研究均产生费用，表中是各部署策略回放的条件成本。</p><p><a href="outcome_review.json">新增逐题答案与参考</a> · <a href="policy_records.json">逐决策记录</a> · <a href="audit.json">审计</a> · <a href="protocol.json">方案登记</a></p></html>'''
    (OUT/'index.html').write_text(doc,encoding='utf-8');print(json.dumps(dict(training=meta,api=api,summary=[r for r in summary if r['method'] in ['liquid','linear','gru','abstention']]),indent=2))


if __name__=='__main__':main()
