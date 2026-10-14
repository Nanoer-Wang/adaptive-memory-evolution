"""Task-level analysis: paired effects, recall disagreements and diagnostic interventions."""
import sys,json,html,math,sqlite3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import defaultdict,Counter
import numpy as np
from src.config import atomic_json,load_config,resolve
from src.provenance import snapshot_source

OUT=ROOT/'runs/task-utility-v1'
LABELS={'linear':'Strong linear','liquid':'Liquid','gru':'GRU','risk_liquid':'Sparse liquid','full_reference':'Unlimited store, top5','linear_addback':'Linear + evicted','linear_retained_control':'Linear + retained','liquid_addback':'Liquid + evicted','liquid_retained_control':'Liquid + retained'}
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def mean(x):return float(np.mean(x)) if x else None
def abstention_phrase(text):
    # Exploratory flag, not a semantic classifier or hallucination verdict.
    return any(s in text.lower() for s in ['cannot determine','unable to determine','insufficient information','not enough information','cannot be determined'])
def pair(rows,a,b):
    by={(r['question_id'],r['method']):r for r in rows};ids=sorted({r['question_id'] for r in rows if (r['question_id'],a) in by and (r['question_id'],b) in by})
    valid=[q for q in ids if by[q,a].get('custom_judge_correct') is not None and by[q,b].get('custom_judge_correct') is not None]
    d=np.array([by[q,a]['custom_judge_correct']-by[q,b]['custom_judge_correct'] for q in valid]);n=len(d)
    if not n:return dict(a=a,b=b,n=0)
    rng=np.random.default_rng(20261012);boot=d[rng.integers(0,n,(10000,n))].mean(1)
    wins=int((d>0).sum());losses=int((d<0).sum());discord=wins+losses
    p=min(1.,2*sum(math.comb(discord,k) for k in range(min(wins,losses)+1))/2**discord) if discord else 1.
    return dict(a=a,b=b,n=n,delta_accuracy=float(d.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),wins=wins,losses=losses,ties=n-discord,exact_discordant_p=p,
                zero_discordance_absolute_difference_bound95=float(1-.05**(1/n)) if discord==0 else None,
                interval_warning='Zero-discordance bootstrap degenerates; it does not establish population equality or noninferiority. Absolute-difference bound uses zero observed discordances and independent-question assumption.' if discord==0 else 'Exploratory unadjusted small-sample bootstrap.',
                question_differences=[dict(question_id=q,difference=float(x)) for q,x in zip(valid,d)])

def main():
    escape=html.escape
    tasks=read(OUT/'qa_tasks.json');rows=[read(p) for p in sorted((OUT/'qa').glob('*.json'))];complete=[r for r in rows if 'judge_response' in r]
    by={(r['question_id'],r['method']):r for r in complete};groups=defaultdict(list)
    for r in complete:r['explicit_abstention_phrase']=abstention_phrase(r['hypothesis'])
    for r in complete:groups[r['method']].append(r)
    methods=[]
    for method in LABELS:
        rr=groups[method];valid=[r for r in rr if r.get('custom_judge_correct') is not None]
        methods.append(dict(method=method,n=len(rr),graded=len(valid),correct=sum(r['custom_judge_correct']==1 for r in valid),accuracy=mean([r['custom_judge_correct'] for r in valid]),
            recall=mean([r['recall'] for r in rr]),context_chars=mean([r['context_chars'] for r in rr]),truncated_answers=sum(r['generation'].get('finish_reason')=='length' for r in rr),
            incorrect_with_abstention_phrase=sum(r['custom_judge_correct']==0 and r['explicit_abstention_phrase'] for r in valid),incorrect_without_abstention_phrase=sum(r['custom_judge_correct']==0 and not r['explicit_abstention_phrase'] for r in valid),
            prompt_tokens=mean([r['generation']['usage'].get('prompt_tokens',0) for r in rr]),completion_tokens=mean([r['generation']['usage'].get('completion_tokens',0) for r in rr]),
            provider_latency_seconds=mean([r['generation']['latency_seconds'] for r in rr]),
            caveat='Latency copied from originating cached response if reused; not independent end-to-end measurements.'))
    comparisons=[pair(complete,a,b) for a,b in [('liquid','linear'),('liquid','gru'),('risk_liquid','linear'),('liquid','full_reference'),('linear','full_reference'),('linear_addback','linear'),('liquid_addback','liquid'),('linear_retained_control','linear'),('liquid_retained_control','liquid'),('linear_addback','linear_retained_control'),('liquid_addback','liquid_retained_control')]]
    interventions=[];disagreements=[]
    for q in sorted({r['question_id'] for r in complete}):
        for base in ['linear','liquid']:
            a=by.get((q,base));b=by.get((q,base+'_addback'));c=by.get((q,base+'_retained_control'))
            if not(a and b and c):continue
            if any(r.get('custom_judge_correct') is None for r in [a,b,c]):continue
            da=b['custom_judge_correct']-a['custom_judge_correct'];dc=c['custom_judge_correct']-a['custom_judge_correct']
            interventions.append(dict(question_id=q,base=base,category=a['category'],base_correct=a['custom_judge_correct'],addback_correct=b['custom_judge_correct'],control_correct=c['custom_judge_correct'],
                effect='addback_helped' if da>0 else 'addback_harmed' if da<0 else 'no_binary_change',addback_delta=da,control_delta=dc,difference_in_differences=da-dc,
                abstention_to_incorrect_assertion_candidate=a['custom_judge_correct']==0 and a['explicit_abstention_phrase'] and b['custom_judge_correct']==0 and not b['explicit_abstention_phrase'],
                recall_change=b['recall']-a['recall'],added_ids=b['added_ids'],control_ids=c['added_ids'],addback_prompt_extra=b['generation']['usage'].get('prompt_tokens',0)-a['generation']['usage'].get('prompt_tokens',0),control_prompt_extra=c['generation']['usage'].get('prompt_tokens',0)-a['generation']['usage'].get('prompt_tokens',0),
                attribution='Appending evicted text changes context length/order and content; retained control is character-matched not token-identical. Binary tie is not proof of harmless long-term forgetting.'))
        a=by.get((q,'liquid'));b=by.get((q,'linear'))
        if a and b and a.get('custom_judge_correct') is not None and b.get('custom_judge_correct') is not None:
            disagreements.append(dict(question_id=q,category=a['category'],liquid_minus_linear_recall=a['recall']-b['recall'],liquid_minus_linear_correct=a['custom_judge_correct']-b['custom_judge_correct'],same_context=a['context_sha256']==b['context_sha256']))
    unique={}
    for r in complete:
        for k in ['generation','judge_response']:
            response=r[k];unique[response['request_hash']]=response
    api_config=load_config('configs/minute_precision.yaml')['api'];attempts=[]
    with sqlite3.connect(resolve(api_config['cache_db'])) as db:
        for key in unique:
            attempts += [dict(request_hash=key,started=s,status=status,elapsed=elapsed,usage=json.loads(usage)) for s,status,elapsed,usage in db.execute('SELECT started,status,elapsed,usage_json FROM requests_log WHERE request_hash=?',(key,))]
    attempts.sort(key=lambda x:x['started']);max60=0;left=0
    for right,a in enumerate(attempts):
        while attempts[left]['started']<=a['started']-60:left+=1
        max60=max(max60,right-left+1)
    api=dict(unique_generation_requests=len({r['generation']['request_hash'] for r in complete}),unique_judge_requests=len({r['judge_response']['request_hash'] for r in complete}),http_attempts=len(attempts),statuses=dict(Counter(a['status'] for a in attempts)),rolling60_max=max60,
        reported_tokens=sum(a['usage'].get('total_tokens',0) or 0 for a in attempts),evaluation_assignments=len(complete),failed_usage_may_be_unreported=True)
    category=[]
    for method,rr in groups.items():
        for cat in sorted({r['category'] for r in rr}):
            cc=[r for r in rr if r['category']==cat and r.get('custom_judge_correct') is not None]
            category.append(dict(method=method,category=cat,n=len(cc),accuracy=mean([r['custom_judge_correct'] for r in cc])))
    summary=dict(status='complete' if len(complete)==len(tasks) else 'partial',expected=len(tasks),completed=len(complete),methods=methods,comparisons=comparisons,api=api,
        interpretation='Task accuracy is primary. Recall is only diagnostic. Effects are exploratory on12 exposed-development histories with same-provider judge; bootstrap intervals are descriptive and comparisons are not multiplicity-adjusted. No multi-step agent success claim.')
    review_path=OUT/'reference_audit.json';review_html=''
    if review_path.exists():
        review=read(review_path);lookup={(r['question_id'],r['method']):r for r in review['records']};reviewed=[]
        for r in complete:
            a=lookup.get((r['question_id'],r['method']))
            if a is not None:
                import hashlib
                assert a['answer_sha256']==hashlib.sha256(r['hypothesis'].encode()).hexdigest()
                reviewed.append(dict(r,custom_judge_correct=a['review_correct']))
        reviewed_methods=[]
        for method in LABELS:
            rr=[r for r in reviewed if r['method']==method]
            reviewed_methods.append(dict(method=method,n=len(rr),correct=sum(r['custom_judge_correct']==1 for r in rr),accuracy=mean([r['custom_judge_correct'] for r in rr])))
        summary['reference_audit_methods']=reviewed_methods
        summary['reference_audit_comparisons']=[pair(reviewed,p['a'],p['b']) for p in comparisons]
        summary['reference_audit_disagreements']=[a for a in review['records'] if a['review_correct']!=a['original_judge_correct']]
        summary['reference_audit_limit']='Assistant review of every frozen answer against references, not blinded or independent human adjudication. Raw judge unchanged; preference grading remains rubric-sensitive.'
        review_html='<h2>逐题参考答案复核</h2><p>下表为本助手逐题复核结果；非盲评，也非独立人工评审。原始自动评分保持不变。偏好题的部分满足情况单独标注。</p><table><tr><th>方法</th><th>复核正确</th></tr>'+''.join(f"<tr><td>{escape(LABELS[m['method']])}</td><td>{m['correct']}/{m['n']}</td></tr>" for m in reviewed_methods)+'</table>'
    sensitivity_html=''
    if (OUT/'length_sensitivity/summary.json').exists():
        sensitivity=read(OUT/'length_sensitivity/summary.json');summary['length_sensitivity']=sensitivity
        sensitivity_html='<h2>输出预算敏感性：512→1536</h2><p>所有出现任一输出截断的题目，全条件统一复测，共2题18条记录。下列为参考答案复核结果。原始主实验不替换；此次重新生成同时包含输出预算变化与服务生成波动。不能把选出的2题当作独立优势验证。</p><table><tr><th>题目</th><th>方法</th><th>原始→复测</th></tr>'+''.join(f"<tr><td>{escape(r['question_id'])}</td><td>{escape(r['method'])}</td><td>{r['primary_review_correct']:.0f} → {r['expanded_review_correct']:.0f}</td></tr>" for r in sensitivity['rows'])+'</table>'
    if (OUT/'combined_api.json').exists():summary['combined_api']=read(OUT/'combined_api.json')
    atomic_json(OUT/'summary.json',summary);atomic_json(OUT/'interventions.json',interventions);atomic_json(OUT/'recall_task_disagreements.json',disagreements);atomic_json(OUT/'category.json',category);atomic_json(OUT/'api_attempts.json',attempts)
    atomic_json(OUT/'analysis_rows.json',complete)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot=[m for m in methods if m['graded']];fig,ax=plt.subplots(1,2,figsize=(12,5))
    display_correct={m['method']:m['correct'] for m in summary.get('reference_audit_methods',methods)}
    colors=['#2c6b9d' if 'addback' not in m['method'] and 'control' not in m['method'] else '#9ab7c9' for m in plot]
    labels=[LABELS[m['method']] for m in plot];ys=np.arange(len(plot))
    ax[0].barh(ys,[display_correct[m['method']] for m in plot],color=colors);ax[0].set_yticks(ys,labels);ax[0].invert_yaxis();ax[0].set_xlim(0,12);ax[0].set_xlabel('Correct / 12 ('+('reference audit' if review_path.exists() else 'custom judge')+')')
    ax[1].barh(ys,[m['prompt_tokens'] for m in plot],color=colors);ax[1].set_yticks(ys,labels);ax[1].invert_yaxis();ax[1].set_xlabel('Mean answer prompt tokens')
    fig.suptitle('Task utility before evidence recall — exploratory public QA');fig.tight_layout();fig.savefig(OUT/'task_utility.png',dpi=160);plt.close(fig)
    escape=html.escape
    table=''.join(f"<tr><td>{escape(LABELS[m['method']])}</td><td>{m['correct']}/{m['graded']}</td><td>{m['recall']:.1%}</td><td>{m['prompt_tokens']:.0f}</td></tr>" for m in plot)
    effects=''.join(f"<tr><td>{escape(p['a'])} − {escape(p['b'])}</td><td>{p['delta_accuracy']:+.1%}</td><td>[{p['ci95'][0]:+.1%}, {p['ci95'][1]:+.1%}]</td><td>{p['wins']}/{p['losses']}/{p['ties']}</td></tr>" for p in comparisons if p['n'])
    cases=[]
    for q in sorted({r['question_id'] for r in complete}):
        rr=[r for r in complete if r['question_id']==q];first=rr[0]
        body=''.join(f"<h4>{escape(r['method'])} · correct={r.get('custom_judge_correct')} · recall={r['recall']:.2f}</h4><p>{escape(r['hypothesis'])}</p><details><summary>Context and lineage</summary><pre>{escape(r['context'])}</pre></details>" for r in rr)
        cases.append(f"<details><summary>{escape(q)} · {escape(first['category'])}</summary><p>{escape(first['question'])}</p><p>Reference: {escape(first['answer'])}</p>{body}</details>")
    doc=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>任务效用与遗忘干预</title><style>body{{max-width:1100px;margin:40px auto;font:16px/1.7 system-ui;color:#203040}}table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #ddd;padding:8px;text-align:left}}img{{width:100%}}pre{{white-space:pre-wrap;background:#f4f6f8;padding:14px}}details{{margin:12px 0}}.note{{background:#edf3fa;padding:16px}}</style><h1>任务效用与遗忘干预</h1>
    <p>状态：{summary['status']}；12 道公开 LongMemEval 问题，六类各两题；已有离线开发集中的新问答子集。固定模型与种子，未按召回率筛选方案。</p>
    <p class="note">主要终点是最终答案正确性。补回分支使用额外归档访问与上下文，仅用于机制诊断；同数量、近似字符长度的已保留内容是扩展上下文对照。不能把这些分支视为同预算部署方案。当前是单次检索—回答组件实验，不是多步工具 agent 完整成功率。</p>
    <img src="task_utility.png"><table><tr><th>方法</th><th>自动评分正确</th><th>证据 session 召回</th><th>平均输入 token</th></tr>{table}</table>{review_html}
    <h2>成对差异</h2><table><tr><th>比较</th><th>准确率差</th><th>95% 描述性 bootstrap 区间</th><th>胜/负/平</th></tr>{effects}</table>
    <p>问题级成对 bootstrap10000次；12题区间不稳定，比较未做多重校正，不作确证性结论。若所有问题结果相同，bootstrap区间会退化为[0,0]，不证明总体相等或非劣；JSON同时给出零分歧样本下的保守差异界。相同上下文与答案的缓存复用不构成独立重复实验。类别每类仅2题，不用于选择机制。</p>
    <h2>机制发现与边界</h2><p>《夜莺》题中，线性基础上下文答对190页，补回后拒答，而同数量已保留内容对照答对，三者召回均为1，且答案未截断；这是信息选择影响任务的个例，尚无重复生成验证。韩国餐厅题补回后召回均为1，线性使用最新累计4家，液态分支错误计算3+4=7，显示证据齐全也可能使用错误。两道助手事实题的答案在原始会话存在，却被共同压缩器删掉；稀疏液态的一道时间题保存了正确事实，但下游模型误算时间后拒答。</p>
    <p>这些发现支持把任务效用、版本关系和推理失败分开分析，不支持已经建立液态独有优势。输入记忆与任务质量不是单调关系。108条主回答逐题复核发现5条明显误接收拒答，另2条偏好题存在评分口径分歧。复测又发现1条误接收拒答；历史定向检查发现5条同类记录。原始判分不覆盖。</p>{sensitivity_html}
    <h2>可复核记录</h2><p>原始任务、写入轨迹、逐题答案、评审、干预变化、召回与任务分歧、调用记录均保存。主实验API：{escape(json.dumps(api,ensure_ascii=False))}</p><p>含复测API：{escape(json.dumps(summary.get('combined_api',{}),ensure_ascii=False))}</p>
    <p>自动评审使用同一服务，仅评判答案与标准答案是否一致。未将所有更新类错误自动标成过期事实误用；需要逐条核对版本证据。单次遗忘无害不能推出长期无害。</p>
    <h2>全部问题与上下文</h2>{''.join(cases)}</html>'''
    (OUT/'report.html').write_text(doc,encoding='utf-8')
    sig,archive=snapshot_source();atomic_json(OUT/'analysis_provenance.json',dict(source=sig,archive=archive,protocol_frozen_before_generation=True))
    print(json.dumps(dict(status=summary['status'],methods=methods,api=api),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
