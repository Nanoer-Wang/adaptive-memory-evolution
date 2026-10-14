"""Post-hoc operating curves: thresholds use validation scores only, never test labels."""
import sys,json,html,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.config import atomic_json
from scripts.semantic_fallback import OUT as P,read,KINDS,SEEDS,LAMBDA
E=ROOT/'runs/semantic-fallback-augmentation-v1'

def main():
    protocol=dict(created_at=time.time(),scope='Post-hoc exploratory diagnostic prompted by excessive fallback in primary results. Not a new held-out confirmation; no model refit and no new API requests.',
        budgets=[.1,.25,.5],rule='For each model/seed use validation predicted gains only. Threshold=max(0,(k+1)th largest gain),k=floor(rate*N);strict greater-than handles ties conservatively. Actual test rate need not meet validation rate.',
        reporting='Save every recipe/model/budget, do not select a winner on test labels. Primary results unchanged.')
    if not (P/'calibration_protocol.json').exists():atomic_json(P/'calibration_protocol.json',protocol)
    rows=[];thresholds=[]
    for recipe,out in [('natural',P),('augmented',E)]:
        records=read(out/'policy_records.json')
        for method in KINDS:
            for seed in SEEDS:
                val=[r for r in records if r['split']=='val' and r['method']==method and r['seed']==seed];test=[r for r in records if r['split']=='test' and r['method']==method and r['seed']==seed]
                gains=sorted([r['predicted_gain'] for r in val],reverse=True)
                for cap in protocol['budgets']:
                    k=int(cap*len(val));threshold=max(0.,gains[k]);thresholds.append(dict(recipe=recipe,method=method,seed=seed,validation_budget=cap,threshold=threshold,validation_rate=sum(g>threshold for g in gains)/len(gains)))
                    for r in test:
                        q=r['question_id'];c=read(P/'qa'/f"{q}-compact{r['budget']}.json");f=read(P/'qa'/f'{q}-full.json');fallback=r['predicted_gain']>threshold
                        prompt=c['generation']['usage']['prompt_tokens']+(f['generation']['usage']['prompt_tokens'] if fallback else 0)
                        correct=(f if fallback else c)['correct']
                        rows.append(dict(recipe=recipe,method=method,seed=seed,validation_budget=cap,question_id=q,reading_budget=r['budget'],fallback=fallback,correct=correct,prompt_tokens=prompt,answer_calls=1+int(fallback),utility=correct-LAMBDA*prompt/4000))
    summary=[]
    for recipe in ['natural','augmented']:
        for method in KINDS:
            for cap in protocol['budgets']:
                rr=[r for r in rows if r['recipe']==recipe and r['method']==method and r['validation_budget']==cap]
                summary.append(dict(recipe=recipe,method=method,validation_budget=cap,questions=12,**{k:float(np.mean([r[k] for r in rr])) for k in ['correct','prompt_tokens','answer_calls','fallback','utility']}))
    atomic_json(P/'calibration_thresholds.json',thresholds);atomic_json(P/'calibration_records.json',rows);atomic_json(P/'calibration_summary.json',summary)
    table=''.join(f"<tr><td>{r['recipe']}</td><td>{r['method']}</td><td>{r['validation_budget']:.0%}</td><td>{r['fallback']:.1%}</td><td>{r['correct']:.1%}</td><td>{r['prompt_tokens']:.0f}</td></tr>" for r in summary)
    (P/'calibration.html').write_text(f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>补读预算诊断</title><style>body{{max-width:1000px;margin:36px auto;font:16px/1.7 system-ui}}td,th{{padding:8px;border-bottom:1px solid #ddd}}table{{width:100%;border-collapse:collapse}}</style><h1>冻结权重，只用验证集分数限定补读率</h1><p>这是看到主结果过度补读后追加的探索性分析，不是新的独立确认。未重训模型，未新增API调用；阈值仅由验证集预测分数确定，不使用测试标签选阈值。完整报告三个预算及全部模型，不选择测试最优点。验证集限额不保证测试集实际频率。</p><p>现有目标是正确率收益减0.02×额外输入token/4000。约3244token的完整调用对应0.0162的惩罚，因此很小的预测正收益也可能触发调用。这解释了成本偏好为何需要明确，而不是不断增加网络结构。</p><table><tr><th>训练配方</th><th>方法</th><th>验证集限额</th><th>测试补读率</th><th>测试正确率</th><th>输入token</th></tr>{table}</table><p><a href="index.html">主报告</a></p></html>''',encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
