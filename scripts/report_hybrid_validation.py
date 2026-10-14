"""Frozen-control attribution, paired QA and metadata-only request audit."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json,html,hashlib,sqlite3
from collections import defaultdict,Counter
import numpy as np
from src.config import atomic_json,load_config,resolve
from scripts.hybrid_validation import OUT,read,CONDITIONS

def interval(groups):
    sums=np.array([sum(v) for v in groups.values()]);ns=np.array([len(v) for v in groups.values()]);rng=np.random.default_rng(20261012)
    ix=rng.integers(0,len(ns),(5000,len(ns)));bs=sums[ix].sum(1)/ns[ix].sum(1)
    return dict(difference=float(sums.sum()/ns.sum()),ci95=np.quantile(bs,[.025,.975]).tolist(),ci9875=np.quantile(bs,[.00625,.99375]).tolist(),clusters=len(ns),questions=int(ns.sum()))

def qa_summary(rows):
    result=[]
    for name in ['liquid_full','liquid_hybrid','liquid_linear','gru_hybrid']:
        rs=[r for r in rows if r['method']==name and 'judge_response' in r]
        result.append(dict(method=name,completed=len(rs),correct=sum(r['custom_judge_correct']==1.0 for r in rs),invalid=sum(r['custom_judge_correct'] is None for r in rs)))
    lookup={(r['question_id'],r['method']):r for r in rows if 'judge_response' in r};paired=[]
    for other in ['liquid_full','liquid_linear','gru_hybrid']:
        diffs=[];details=[]
        for q,method in lookup:
            if method!='liquid_hybrid' or (q,other) not in lookup:continue
            a=lookup[q,method]['custom_judge_correct'];b=lookup[q,other]['custom_judge_correct']
            if a is None or b is None:continue
            delta=a-b;diffs.append(delta);details.append(dict(question_id=q,difference=delta))
        if diffs:
            paired.append(dict(other=other,**interval({str(i):[x] for i,x in enumerate(diffs)}),wins=sum(x>0 for x in diffs),losses=sum(x<0 for x in diffs),ties=sum(x==0 for x in diffs),cases=details))
    return result,paired

def main():
    rr=[json.loads(s) for s in (OUT/'records.jsonl').read_text(encoding='utf-8').splitlines()];groups=defaultdict(list);byq=defaultdict(lambda:defaultdict(list));clusters={}
    for r in rr:
        if r['recall'] is None:continue
        groups[r['dataset'],r['budget'],r['method']].append(r)
        byq[r['dataset'],r['budget'],r['question_id']][r['method']].append(r['recall']);clusters[r['dataset'],r['question_id']]=r['cluster']
    summary=[dict(dataset=k[0],budget=k[1],method=k[2],recall=float(np.mean([r['recall'] for r in v])),retention=float(np.mean([r['retention'] for r in v]))) for k,v in groups.items()];diffs=[]
    for dataset in ['longmemeval','locomo']:
        for budget in [64000,128000]:
            for other in [m for m in CONDITIONS if m!='liquid_hybrid']:
                cs=defaultdict(list)
                for (ds,b,q),v in byq.items():
                    if (ds,b)==(dataset,budget):cs[clusters[ds,q]].append(float(np.mean(v['liquid_hybrid'])-np.mean(v[other])))
                diffs.append(dict(dataset=dataset,budget=budget,other=other,**interval(cs)))
    atomic_json(OUT/'summary.json',summary);atomic_json(OUT/'comparisons.json',diffs)
    phase=[]
    for dataset in ['longmemeval','locomo']:
        for budget in [64000,128000]:
            cs=defaultdict(list)
            for (ds,b,q),v in byq.items():
                if (ds,b)!=(dataset,budget):continue
                liquid=(np.mean(v['liquid_hybrid'])+np.mean(v['liquid_shift']))/2
                gru=(np.mean(v['gru_hybrid'])+np.mean(v['gru_shift']))/2
                cs[clusters[ds,q]].append(float(liquid-gru))
            phase.append(dict(dataset=dataset,budget=budget,comparison='two_phase_average_liquid_minus_gru',status='posthoc descriptive robustness diagnostic; not a new primary endpoint',**interval(cs)))
    atomic_json(OUT/'phase_robustness.json',phase)
    calls=read(OUT/'calls.json');costs=[]
    for dataset in ['longmemeval','locomo']:
        for name in CONDITIONS:
            rows=[r for r in calls if (r['dataset'],r['method'])==(dataset,name)];events=sum(r['events'] for r in rows);count=sum(r['calls'] for r in rows)
            costs.append(dict(dataset=dataset,method=name,events=events,calls=count,saved=1-count/events))
    atomic_json(OUT/'cost_summary.json',costs)
    prior={}
    with (ROOT/'runs/content-fallback-v1/records.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);name=r['kind']+('_linear' if r['policy']=='linear_only' else '_hybrid');prior[r['dataset'],r['question_id'],name,r['seed'],r['budget']]=r
    matched=0
    for r in rr:
        if r['method'] in ['liquid_hybrid','liquid_linear','gru_hybrid','static_hybrid']:
            old=prior[r['dataset'],r['question_id'],r['method'],r['seed'],r['budget']];assert old['recall']==r['recall'] and old['payload_bytes']==r['payload_bytes'];matched+=1
    protocol=read(OUT/'protocol.json')
    assert hashlib.sha256((ROOT/'runs/content-fallback-v1/students.npz').read_bytes()).hexdigest()==protocol['student_sha256']
    for name,digest in protocol['checkpoints'].items():assert hashlib.sha256((ROOT/'runs/preservation-study-v2/checkpoints'/name/'best.pt').read_bytes()).hexdigest()==digest
    qa=[read(p) for p in (OUT/'qa').glob('*.json')] if (OUT/'qa').exists() else [];qs,qp=qa_summary(qa);atomic_json(OUT/'qa_summary.json',qs);atomic_json(OUT/'qa_pairs.json',qp)
    historical=[]
    for directory,mapping in [('writer-efficiency-v1',{'liquid_full':'liquid_full'}),('content-fallback-v1',{'liquid_hybrid2':'liquid_hybrid','liquid_linear_only':'liquid_linear','gru_hybrid2':'gru_hybrid'})]:
        for p in (ROOT/'runs'/directory/'qa').glob('*.json'):
            r=read(p)
            if r['method'] in mapping:historical.append(dict(r,method=mapping[r['method']]))
    assert not {r['question_id'] for r in historical}&set(protocol['qa_ids'])
    pooled,poolpairs=qa_summary(qa+historical);atomic_json(OUT/'qa_pooled_exploratory.json',dict(summary=pooled,paired=poolpairs,note='Secondary pooling prior6 and new12; independent model calls across stages; custom same-provider grading, not fresh confirmation.'))
    atomic_json(OUT/'qa_failures.json',[r for r in qa if 'judge_response' in r and r['custom_judge_correct']!=1.0])
    failure_stages=Counter()
    for r in qa:
        if 'judge_response' not in r or r['custom_judge_correct']==1.0:continue
        stage='no_gold_session_retained' if r['retention']==0 else ('no_gold_session_retrieved' if r['recall']==0 else 'gold_session_retrieved_but_answer_failed')
        failure_stages[r['method']+':'+stage]+=1
    atomic_json(OUT/'failure_stages.json',dict(failure_stages))
    hashes={r[k]['request_hash'] for r in qa for k in ['generation','judge_response'] if k in r}
    config=load_config('configs/minute_precision.yaml')['api']
    with sqlite3.connect(resolve(config['cache_db'])) as db:attempts=[dict(zip(['hash','started','status','usage'],r)) for r in db.execute('SELECT request_hash,started,status,usage_json FROM requests_log') if r[0] in hashes]
    for r in attempts:r['usage']=json.loads(r['usage'])
    times=sorted(r['started'] for r in attempts);left=0;maximum=0
    for i,t in enumerate(times):
        while times[left]<=t-60:left+=1
        maximum=max(maximum,i-left+1)
    api=dict(attempts=len(attempts),statuses=dict(Counter(r['status'] for r in attempts)),reported_tokens=sum(r['usage'].get('total_tokens',0) for r in attempts),rolling60_max=maximum)
    atomic_json(OUT/'api_attempts.json',attempts);atomic_json(OUT/'api_summary.json',api)
    primary=[d for d in diffs if d['dataset']=='longmemeval' and d['budget']==64000 and d['other'] in ['liquid_linear','gru_hybrid','liquid_reset','liquid_zero_time']]
    audit=dict(records=len(rr),prior_matches=matched,weights_unchanged=True,capacity_violations=sum(r['payload_bytes']>r['budget'] for r in rr),qa_questions=len(protocol['qa_ids']),qa_complete=sum(r['completed'] for r in qs),mechanism_criterion_passed=all(d['ci9875'][0]>0 for d in primary))
    atomic_json(OUT/'audit.json',audit)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for axis,dataset in zip(axes,['longmemeval','locomo']):
        rows=[next(r for r in summary if (r['dataset'],r['budget'],r['method'])==(dataset,64000,m)) for m in CONDITIONS]
        axis.barh(np.arange(len(rows)),[r['recall']*100 for r in rows],color=['#176b87' if r['method']=='liquid_hybrid' else '#91a4ad' for r in rows]);axis.set(yticks=np.arange(len(rows)),yticklabels=[r['method'] for r in rows],xlabel='Evidence retrieval recall (%)',title=dataset+' | 64 KB');axis.set_xlim(0,75)
        for i,r in enumerate(rows):axis.text(r['recall']*100+.5,i,f"{r['recall']*100:.2f}",va='center',fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=170);plt.close(fig)
    def table(headers,rows):return '<table><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'
    mainrows=[[r['other'],f"{r['difference']*100:+.3f}",f"[{r['ci95'][0]*100:+.3f},{r['ci95'][1]*100:+.3f}]",f"[{r['ci9875'][0]*100:+.3f},{r['ci9875'][1]*100:+.3f}]"] for r in primary]
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>混合评分机制验证与扩大问答</title><style>body{{max-width:1150px;margin:35px auto;padding:0 24px;font:16px/1.75 system-ui;color:#24323b}}table{{border-collapse:collapse;width:100%;margin:20px 0}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}th{{background:#eef3f5}}img{{width:100%}}</style><h1>混合评分：液态历史、时间与调用起点归因</h1>
    <p>冻结教师和学生权重；103组历史、2,067道公开问题、3种子、64/128KB，9条件，共{len(rr):,}条记录。所有数据此前用于开发，本轮属于归因验证和新增问答测量，不能称新独立数据集。预设主判据是否通过：{audit['mechanism_criterion_passed']}。</p><img src="comparison.png">
    <h2>主比较：LongMemEval64KB</h2><p>以下都是liquid_hybrid减去对照的百分点差值。同权重reset保留当前文本评分、只重置递推历史；zero_time只移除真实时间间隔。先按问题平均三个种子，再按历史簇做5000次配对bootstrap。四主比较同时报告98.75%保守校正区间。</p>{table(['对照','差值/百分点','95%区间','98.75%区间'],mainrows)}
    <p>shift把调用从偶数记录移到奇数记录，不事后选择更好起点。奇数长度历史少一次调用，实际数量见cost_summary.json。液态与GRU都做此检查。LongMemEval64KB液态61.82%→57.47%，GRU58.21%→59.78%；收益依赖周期起点。额外保存两个起点均值的事后描述性诊断，不改变主判据，也不挑最好起点。容量128KB的LoCoMo几乎全历史装得下，不能用零差异证明机制非劣。完整保留所有预算和不利比较。</p>
    {table(['数据/容量','两个起点均值：液态减GRU/百分点','95%区间'],[[r['dataset']+'/'+str(r['budget']),f"{r['difference']*100:+.3f}",str([round(x*100,3) for x in r['ci95']])] for r in phase])}
    <h2>新增固定问答</h2><p>每个可用题型按固定带盐哈希选择两题，排除之前六题；共{len(protocol['qa_ids'])}道新问题，四种方案，seed23、64KB。选择规则在评分前保存，没有根据答案或检索结果筛题。自定义同提供商判分，非官方benchmark评分；小样本不证明非劣。</p>{table(['方案','完成','正确','无效判分'],[[r['method'],r['completed'],r['correct'],r['invalid']] for r in qs])}
    {table(['液态混合对照','胜','负','平','配对差95%区间'],[[r['other'],r['wins'],r['losses'],r['ties'],str([round(x*100,2) for x in r['ci95']])] for r in qp])}
    <h2>与此前六题合并（次要探索）</h2>{table(['方案','完成','正确'],[[r['method'],r['completed'],r['correct']] for r in pooled])}<p>只合并同种子、同预算、同方案，未重复计入同一题。新增题与历史题分别留档，合并不改变本轮主评价规则。</p>
    <h2>本轮决定与下一项假设</h2><p>固定隔条调用未显示稳定的液态独有收益，不接入默认Agent。新增12题中完整液态6/12、混合液态5/12、纯线性学生8/12、GRU混合4/12；相对纯线性，液态混合0胜3负9平。样本很小且为自定义判分，不能宣称纯线性普遍优越，但足以拒绝在此基础上声称混合方案已保住问答质量。</p><p>候选下一步是让便宜内容评分承担常规处理，依据不确定性或修订风险触发递推计算，并对液态、GRU和简单历史统计使用相同预算。另一个待检验的建模问题是：原教师在完整历史上训练，稀疏执行时却只看到一部分观察；学生拟合的是完整教师的分数，两者混用可能产生偏差。应在训练中显式模拟相同稀疏轨迹或校准残差，而不是继续挑选有利周期起点。这些是下一轮假设，本轮未声称已实现或验证。</p>
    <h2>工程与费用审计</h2><p>{matched:,}条相同条件与上一轮逐条一致；教师和学生哈希未变，容量越界为零。API当前{api['attempts']}次尝试，状态{html.escape(str(api['statuses']))}，报告tokens {api['reported_tokens']:,}，滚动60秒最多{api['rolling60_max']}次；未包含未知失败费用。完整答案、判分和错误均保存。没有部署新的稀疏策略。</p>
    <ul>{''.join(f'<li><a href="{name}">{name}</a></li>' for name in ['protocol.json','summary.json','comparisons.json','phase_robustness.json','cost_summary.json','qa_summary.json','qa_pairs.json','qa_pooled_exploratory.json','qa_failures.json','failure_stages.json','api_summary.json','audit.json','resume_audit.json','tests.txt'])}</ul></html>'''
    (OUT/'report.html').write_text(page,encoding='utf-8');print(json.dumps(primary,indent=2));print(json.dumps(qs,indent=2))

if __name__=='__main__':main()
