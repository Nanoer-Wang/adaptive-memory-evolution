"""Repeat every frozen test action once, without fitting or selecting policies."""
import sys,json,hashlib,sqlite3,html,time,argparse
from pathlib import Path
from collections import defaultdict,Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.config import atomic_json,load_config,resolve
from src.provenance import snapshot_source
from scripts import context_utility as original
PARENT=original.OUT
OUT=ROOT/'runs/context-utility-repeat-v1'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():return
    assert (PARENT/'qa_complete.json').exists()
    source,archive=snapshot_source()
    atomic_json(OUT/'protocol.json',dict(registered_at=time.time(),parent=str(PARENT.relative_to(ROOT)),source=source,archive=archive,
        parent_protocol_sha256=digest(PARENT/'protocol.json'),parent_policy_records_sha256=digest(PARENT/'policy_records.json'),
        generator_source_sha256=digest(ROOT/'scripts/context_utility.py'),
        selection='All12 parent test questions, both actions; one additional independent API generation request per action. No selection by outcome.',
        inference='Reuse original.qa verbatim with a separate output folder and namespace derived from this protocol; original prompts, model, temperature and output budget unchanged. rpm9.',
        policies='Freeze original policy choices for all learned and static methods. Recompute abstention cascade from this repeat\'s own compact answer; charge both calls when fallback is used. No fitting, retuning or checkpoint selection.',
        analysis='Report both repeats separately and average within each question before intervals. 12 independent question units, not24; descriptive paired bootstrap. This is repeatability on the same development subset, not new independent data.',
        budget='Exactly24 new answer requests plus judges only for non-abstentions, subject to retries; resumable. Stop after this repeat regardless of results.'))
    atomic_json(OUT/'tasks.json',[t for t in read(PARENT/'tasks.json') if t['split']=='test'])
    atomic_json(OUT/'training_complete.json',dict(reuse_only=True,parent=str(PARENT.relative_to(ROOT)),
        explanation='No training in this folder; reuse all12 already frozen parent checkpoints.',
        checkpoints={str(p.relative_to(PARENT)):digest(p) for p in (PARENT/'checkpoints').glob('*/best.pt')}))

def qa():
    protocol=read(OUT/'protocol.json')
    assert digest(ROOT/'scripts/context_utility.py')==protocol['generator_source_sha256']
    assert digest(PARENT/'policy_records.json')==protocol['parent_policy_records_sha256']
    previous=original.OUT
    try:
        original.OUT=OUT
        original.qa('test')
    finally:original.OUT=previous

def analyze():
    assert (OUT/'qa_complete.json').exists()
    parent=[r for r in read(PARENT/'policy_records.json') if r['split']=='test'];rr=[dict(r,repeat=0) for r in parent]
    answers=[read(p) for p in (OUT/'qa').glob('*.json')];by={(r['question_id'],r['method']):r for r in answers}
    assert len(by)==24 and all(r['correct'] in (0,1) and r['evaluation_status'] in ['graded','explicit_abstention'] for r in answers)
    for p in parent:
        q=p['question_id'];cascade=p['method']=='abstention_cascade';fallback=cascade and by[q,'compact']['parsed']['abstain']
        action=('full' if fallback else 'compact') if cascade else p['action'];r=by[q,action]
        tokens=r['generation']['usage'];extra=by[q,'compact']['generation']['usage'] if fallback else dict(prompt_tokens=0,completion_tokens=0)
        prompt=tokens['prompt_tokens']+extra['prompt_tokens'];completion=tokens['completion_tokens']+extra['completion_tokens']
        rr.append(dict(question_id=q,split='test',category=p['category'],method=p['method'],seed=p['seed'],repeat=1,action=action,
            correct=r['correct'],abstain=r['parsed']['abstain'],prompt_tokens=prompt,completion_tokens=completion,
            utility=r['correct']-original.LAMBDA*prompt/4000,answer_calls=1+int(fallback),generation_hash=r['generation']['request_hash']))
    summaries=[]
    for repeat in [0,1,'pooled']:
        for method in sorted({r['method'] for r in rr}):
            xs=[r for r in rr if r['method']==method and (repeat=='pooled' or r['repeat']==repeat)]
            summaries.append(dict(repeat=repeat,method=method,questions=12,accuracy=float(np.mean([r['correct'] for r in xs])),
                prompt_tokens=float(np.mean([r['prompt_tokens'] for r in xs])),completion_tokens=float(np.mean([r['completion_tokens'] for r in xs])),
                answer_calls=float(np.mean([r.get('answer_calls',1) for r in xs])),utility=float(np.mean([r['utility'] for r in xs]))))
    comparisons=[];ids=sorted({r['question_id'] for r in rr});rng=np.random.default_rng(20261012)
    for a,b in [('always_compact','always_full'),('abstention_cascade','always_full'),('liquid','always_compact'),('liquid','linear'),('liquid','gru')]:
        metrics={}
        for metric in ['correct','prompt_tokens','utility']:
            d=np.array([np.mean([r[metric] for r in rr if r['question_id']==q and r['method']==a])-np.mean([r[metric] for r in rr if r['question_id']==q and r['method']==b]) for q in ids])
            boots=d[rng.integers(0,len(d),(10000,len(d)))].mean(1)
            metrics[metric]=dict(mean=float(d.mean()),ci95=np.quantile(boots,[.025,.975]).tolist())
        comparisons.append(dict(a=a,b=b,questions=12,metrics=metrics,caution='Two repeats and three training seeds averaged within question; descriptive unadjusted bootstrap, not confirmatory significance.'))
    drift=[]
    for r in answers:
        p=read(PARENT/'qa'/f"{r['question_id']}-{r['method']}.json")
        assert r['context']==p['context'] and r['question']==p['question']
        assert r['generation']['request_hash']!=p['generation']['request_hash']
        drift.append(dict(question_id=r['question_id'],action=r['method'],category=r['category'],first_correct=p['correct'],repeat_correct=r['correct'],
            identical_answer=r['generation']['content']==p['generation']['content'],first_abstain=p['parsed']['abstain'],repeat_abstain=r['parsed']['abstain']))
    keys={r[k]['request_hash'] for r in answers for k in ['generation','judge_response'] if k in r};attempts=[]
    with sqlite3.connect(resolve(load_config('configs/minute_precision.yaml')['api']['cache_db'])) as db:
        for key in keys:attempts.extend(dict(request_hash=key,started=s,status=st,elapsed=e,usage=json.loads(u)) for s,st,e,u in db.execute('SELECT started,status,elapsed,usage_json FROM requests_log WHERE request_hash=?',(key,)))
    attempts.sort(key=lambda r:r['started']);left=0;peak=0
    for right,r in enumerate(attempts):
        while attempts[left]['started']<=r['started']-60:left+=1
        peak=max(peak,right-left+1)
    assert min(r['started'] for r in attempts)>=read(OUT/'protocol.json')['registered_at']
    api=dict(generations=24,judges=len({r['judge_response']['request_hash'] for r in answers if 'judge_response' in r}),judge_assignments=sum('judge_response' in r for r in answers),http_attempts=len(attempts),statuses=dict(Counter(r['status'] for r in attempts)),
        reported_tokens=sum(r['usage'].get('total_tokens',0) or 0 for r in attempts),rolling60_max=peak)
    # Explicit assistant review of this completed run, not a new automatic judge.
    # Preserve every primary label and apply the same factual constraint to every
    # policy that uses this exact answer. No fitting uses this post-hoc sensitivity.
    camera=by['06878be2','full'];assert 'CFexpress' in camera['hypothesis']
    review=dict(question_id='06878be2',action='full',repeat=1,original_correct=camera['correct'],strict_correct=0.,
        record_sha256=digest(OUT/'qa/06878be2-full.json'),reviewer='Assistant; not independent human or blind review',
        reason='Candidate recommends CFexpress cards for Sony A7R IV. Sony model-specific documentation states both slots support only SD cards. Personalized-topic agreement alone misses this material compatibility error.',
        source_title='Sony ILCE-7RM4 Help Guide: Inserting/removing a memory card',source_url='https://helpguide.sony.net/ilc/1930/v1/en/contents/TP0002711790.html',verified_on='2026-10-12',
        rule='Secondary sensitivity: a recommendation containing a verified incompatible component does not count as fully correct. Original reference-judge scores remain primary and unchanged.',
        scope='All24 repeat answers inspected; this specific factual error externally checked. Not exhaustive factual verification of every claim.')
    reviewed=[]
    for r in rr:
        audited=0. if r['repeat']==1 and r['question_id']=='06878be2' and r['action']=='full' else r['correct']
        reviewed.append(dict(question_id=r['question_id'],method=r['method'],seed=r['seed'],repeat=r['repeat'],original_correct=r['correct'],strict_review_correct=audited))
    reviewed_summary=[]
    for repid in [0,1,'pooled']:
        for m in sorted({r['method'] for r in rr}):
            xs=[r for r in reviewed if r['method']==m and (repid=='pooled' or r['repeat']==repid)]
            reviewed_summary.append(dict(repeat=repid,method=m,accuracy=float(np.mean([r['strict_review_correct'] for r in xs]))))
    atomic_json(OUT/'review.json',dict(flags=[review],methods=reviewed_summary,records=reviewed))
    atomic_json(OUT/'policy_records.json',rr);atomic_json(OUT/'summary.json',dict(methods=summaries,comparisons=comparisons,api=api));atomic_json(OUT/'drift.json',drift);atomic_json(OUT/'api_attempts.json',attempts)
    shown=['always_full','always_compact','abstention_cascade','linear','mlp','gru','liquid']
    table=''
    for m in shown:
        a,b,c=[next(r for r in summaries if r['method']==m and r['repeat']==rep) for rep in [0,1,'pooled']]
        table+=f"<tr><td>{m}</td><td>{a['accuracy']*12:.2f}</td><td>{b['accuracy']*12:.2f}</td><td>{c['accuracy']:.1%}</td><td>{c['prompt_tokens']:.0f}</td><td>{c['answer_calls']:.2f}</td></tr>"
    cases=''.join(f"<details><summary>{html.escape(r['question_id'])} · {r['method']} · {r['correct']:.0f}</summary><p>{html.escape(r['question'])}</p><p>参考：{html.escape(r['answer'])}</p><p>{html.escape(r['hypothesis'])}</p></details>" for r in sorted(answers,key=lambda r:(r['question_id'],r['method'])))
    doc=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>全部测试题固定复测</title><style>body{{max-width:1050px;margin:36px auto;font:16px/1.7 system-ui;color:#203040}}table{{width:100%;border-collapse:collapse}}td,th{{padding:8px;border-bottom:1px solid #ddd}}pre{{white-space:pre-wrap}}details{{margin:12px 0}}</style><h1>全部12道测试题：固定一次复测</h1>
    <p>完整与精简两种输入全部复测，无论第一轮成绩如何。沿用原始生成器、上下文、模型设置、策略及检查点；只更换缓存命名空间，产生24次新的回答请求。简单级联根据各自重复的拒答状态决定补读，成本含两次回答。未重新训练或根据测试结果选模型。</p>
    <table><tr><th>方法</th><th>首次正确/12</th><th>复测正确/12</th><th>两次平均正确率</th><th>平均输入token</th><th>平均调用数</th></tr>{table}</table>
    <p>学习型策略按三个种子平均，因此正确数可能为小数。仍只有12道不同问题；两次回答不能当作24道独立题。偏好题的判分争议保留，所有原始答案可核查。API：{html.escape(json.dumps(api,ensure_ascii=False))}</p>
    <h2>实际可用性复核</h2><p>上表是原始自动裁判分数。复测的摄影答案推荐了CFexpress卡，但<a href="{review['source_url']}">Sony该型号官方说明</a>明确两个卡槽仅支持SD卡。按“含确定不兼容配件不算完全正确”的次要复核，复测完整输入及级联均由6/12变为5/12，始终精简仍5/12。对使用同一答案的所有学习型策略同样处理；原始评分不覆盖，模型不重训。这是本助手完成的已知错误敏感性分析，不是独立人工盲评。</p>
    <h2>逐问题聚类的探索性区间</h2><pre>{html.escape(json.dumps(comparisons,ensure_ascii=False,indent=2))}</pre><h2>全部复测回答</h2>{cases}<p><a href="../context-utility-v1/report.html">首轮报告</a></p></html>'''
    (OUT/'report.html').write_text(doc,encoding='utf-8')
    print(json.dumps(dict(methods=[r for r in summaries if r['method'] in shown],api=api,changed_scores=sum(r['first_correct']!=r['repeat_correct'] for r in drift)),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['all','prepare','qa','analyze'],default='all');a=p.parse_args()
    if a.phase in ['all','prepare']:prepare()
    if a.phase in ['all','qa']:qa()
    if a.phase in ['all','analyze']:analyze()
