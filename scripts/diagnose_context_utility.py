"""Post-run diagnostics of supervision, policy behavior and preference-label sensitivity.

Never modifies action labels or checkpoints. Preference bounds treat both action
labels as unknown, sharing each unknown across policies on the same question.
These are worst-case sensitivity ranges, not confidence intervals.
"""
import sys,json,hashlib,html
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.config import atomic_json
from scripts.context_utility import OUT,read

PREFERENCE='single-session-preference'

def main():
    records=read(OUT/'policy_records.json');pairs=read(OUT/'paired_actions.json')
    signal=[]
    for split in ['train','val','test']:
        for subset in ['all','factual','preference']:
            rr=[r for r in pairs if r['split']==split and (subset=='all' or (r['category']==PREFERENCE)==(subset=='preference'))]
            signal.append(dict(split=split,subset=subset,n=len(rr),both_correct=sum(r['full_correct']==r['compact_correct']==1 for r in rr),
                both_wrong=sum(r['full_correct']==r['compact_correct']==0 for r in rr),compact_wins=sum(r['compact_minus_full']>0 for r in rr),
                compact_losses=sum(r['compact_minus_full']<0 for r in rr),quality_contrast_pairs=sum(r['compact_minus_full']!=0 for r in rr)))
    test=[r for r in records if r['split']=='test'];methods=sorted({r['method'] for r in test});by=defaultdict(list)
    for r in test:by[r['question_id'],r['method']].append(r)
    ids=sorted({q for q,m in by});sensitivity=[]
    for a,b in [('always_compact','always_full'),('liquid','always_compact'),('liquid','linear'),('liquid','gru'),('liquid','abstention_cascade')]:
        fixed=0.;low=0.;high=0.
        for q in ids:
            aa,bb=by[q,a],by[q,b]
            if aa[0]['category']!=PREFERENCE:
                fixed+=np.mean([r['correct'] for r in aa])-np.mean([r['correct'] for r in bb])
            else:
                # An outcome is shared by every policy selecting the same action.
                for action in ['full','compact']:
                    coefficient=np.mean([r['action']==action for r in aa])-np.mean([r['action']==action for r in bb])
                    low+=min(coefficient,0);high+=max(coefficient,0)
        sensitivity.append(dict(a=a,b=b,n=len(ids),accuracy_difference_bounds=[(fixed+low)/len(ids),(fixed+high)/len(ids)],
            interpretation='Unknown preference action labels allowed to be 0 or 1; shared outcomes respected. Not a statistical confidence interval.'))
    behaviors=[]
    for baseline in ['always_compact','linear','gru','liquid_reset','liquid_zero_time']:
        comparisons=[]
        for q in ids:
            for r in by[q,'liquid']:
                other=next(t for t in by[q,baseline] if t['seed']==r['seed'] or t['seed']==0)
                comparisons.append(dict(question_id=q,seed=r['seed'],different_action=r['action']!=other['action'],
                    accuracy_difference=r['correct']-other['correct'],prompt_token_difference=r['prompt_tokens']-other['prompt_tokens']))
        behaviors.append(dict(a='liquid',b=baseline,decisions=len(comparisons),independent_questions=len(ids),
            different_actions=sum(r['different_action'] for r in comparisons),rows=comparisons))
    categories=[]
    for method in methods:
        for category in sorted({r['category'] for r in test}):
            rr=[r for r in test if r['method']==method and r['category']==category]
            categories.append(dict(method=method,category=category,questions=len({r['question_id'] for r in rr}),accuracy=float(np.mean([r['correct'] for r in rr])),
                prompt_tokens=float(np.mean([r['prompt_tokens'] for r in rr]))))
    # This explicit flag arose from reading both development answers before any
    # test answer was inspected. It is not a general-purpose automated regrader.
    flagged=[]
    reasons={
        ('1a1907b4','full'):'Flagged judge inconsistency: candidate explicitly offers a creative Pimm\'s Cup variation aligned with the reference, but receives 0. Its compact counterpart offers more generic suggestions yet receives 1. Requires independent adjudication.',
        ('c8090214','compact'):'Final numeric answer matches the reference, but its explanation says Black Friday is one day after the last Sunday of November. Final-answer correctness does not imply all explanation claims are reliable.',
        ('195a1a1b','full'):'Test preference score is rubric-sensitive: answer explicitly respects9:30PM wind-down and offers breathing/relaxation, but also recommends a meditation app while reference discourages phone use. Retain0; do not silently adjudicate it to favor a policy.'}
    for (q,action),reason in reasons.items():
        p=OUT/'qa'/f'{q}-{action}.json';r=read(p)
        flagged.append(dict(question_id=q,action=action,split=r['split'],original_correct=r['correct'],reason=reason,
            record_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),action_taken='No label replacement and no retraining. Assistant review, not an independent human assessment.'))
    cases=[]
    for q,needles,interpretation in [
        ('6a1eabeb',['27:12','25:50'],'Both old and newer personal-best values survive in both contexts. Compact succeeds while full refuses. This supports investigating interference/focus, not a claim that removing the old value resolved a contradiction. One paired generation cannot establish the causal mechanism.'),
        ('gpt4_1916e0ea',['FarmFresh','Instacart'],'Both dated source passages survive in both contexts. Compact matches the54-day reference and full refuses. Cancellation is described retrospectively, so matching the benchmark date convention is not proof that its exact event date is explicit.'),
        ('06878be2',['Sony','Godox'],'Full context contains camera-specific preferences; compact removes that session entirely and refuses. Simple lexical extraction can lose semantically related details, illustrating harmful compression; full fallback recovers this case.')]:
        item=dict(question_id=q,interpretation=interpretation,actions={})
        for action in ['full','compact']:
            p=OUT/'qa'/f'{q}-{action}.json';r=read(p);lines=r['context'].splitlines()
            item['actions'][action]=dict(correct=r['correct'],abstain=r['parsed']['abstain'],hypothesis=r['hypothesis'],
                record_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),source_headers=[s for s in lines if s.startswith('[session=')],
                needle_presence={s:s.lower() in r['context'].lower() for s in needles},evidence_lines=[s for s in lines if any(n.lower() in s.lower() for n in needles)])
        cases.append(item)
    atomic_json(OUT/'mechanism_cases.json',cases)
    result=dict(supervision=signal,preference_sensitivity=sensitivity,behavior=behaviors,category_metrics=categories,review_flags=flagged,
        scope='Post-hoc mechanism diagnostics. All original primary scores retained. Factual subset excludes advice tasks but is not an independent benchmark. No test-based model selection.')
    atomic_json(OUT/'diagnostics.json',result)
    table=''.join(f"<tr><td>{r['split']}</td><td>{r['subset']}</td><td>{r['n']}</td><td>{r['both_correct']}</td><td>{r['both_wrong']}</td><td>{r['compact_wins']}</td><td>{r['compact_losses']}</td></tr>" for r in signal)
    body=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>监督与机制诊断</title><style>body{{max-width:1050px;margin:36px auto;font:16px/1.7 system-ui;color:#203040}}table{{width:100%;border-collapse:collapse}}td,th{{padding:8px;border-bottom:1px solid #ddd}}pre{{white-space:pre-wrap;background:#f4f6f8;padding:14px}}</style><h1>学到的究竟是任务收益还是评分偏差？</h1>
    <p>本页为固定实验结束后的诊断，不改写原始标签、不重新选模型。比较同一问题完整与精简输入的真实回答。</p><table><tr><th>划分</th><th>题目范围</th><th>题数</th><th>都对</th><th>都错</th><th>精简更好</th><th>精简更差</th></tr>{table}</table>
    <p>训练集中事实题如果没有正确率差异，模型无法从这些样本学到“什么时候压缩会丢掉关键事实”，只能看到token节省。少量偏好题的差异也可能包含裁判偏差。这是监督信号的可辨识性限制，不能用更复杂的递推结构自动解决。</p>
    <h2>偏好评分敏感性</h2><p>将全部偏好题的两种答案评分视为未知，保持同一答案在不同策略间共享，计算最坏情况差值范围；不是置信区间，也没有从主结果删掉这些题。</p><pre>{html.escape(json.dumps(sensitivity,ensure_ascii=False,indent=2))}</pre>
    <h2>明确标记的复核问题</h2><p>全部96条训练/验证/测试短回答均由本助手对照参考答案阅读。事实题最终答案未发现需要反转的标签；这不等于完整证据核验或独立人工盲评。偏好题整体有口径敏感性，以下为明确标记的例子。</p><pre>{html.escape(json.dumps(flagged,ensure_ascii=False,indent=2))}</pre>
    <h2>改善与退步的内容核查</h2><pre>{html.escape(json.dumps(cases,ensure_ascii=False,indent=2))}</pre>
    <p>各题型结果、逐种子的策略动作差异及原始答案均已保存。<a href="report.html">返回主实验</a></p></html>'''
    (OUT/'diagnostics.html').write_text(body,encoding='utf-8')
    print(json.dumps(dict(supervision=signal,sensitivity=sensitivity,behavior=[{k:v for k,v in b.items() if k!='rows'} for b in behaviors]),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
