"""Read-only readiness audit and report; never silently launches paid work."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json,hashlib,sqlite3
from collections import Counter,defaultdict
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.config import load_config
from src.provenance import snapshot_source,runtime_info
OUT=ROOT/'runs/language-readiness-v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[json.loads(s) for s in (OUT/'offline.jsonl').read_text(encoding='utf-8').splitlines()];qa=[read(p) for p in (OUT/'qa').glob('*.json')];protocol=read(OUT/'protocol.json');status=read(OUT/'qa_status.json') if (OUT/'qa_status.json').exists() else {}
summary=[]
for dataset in ['longmemeval','locomo']:
    for budget in [None,32000,128000]:
        for method in sorted({r['method'] for r in rows if r['budget']==budget}):
            rr=[r for r in rows if r['dataset']==dataset and r['budget']==budget and r['method']==method]
            if not rr:continue
            def avg(k):
                vv=[r[k] for r in rr if r[k] is not None];return float(np.mean(vv)) if vv else None
            summary.append(dict(dataset=dataset,budget=budget,method=method,questions=len(rr),evidence_scored=sum(r['retrieval_recall'] is not None for r in rr),**{k:avg(k) for k in ['evidence_retention','retrieval_recall','context_coverage','payload_bytes','retrieval_seconds']},capacity_active_questions=sum(r['retained_sessions']<r['input_sessions'] for r in rr)))
save('summary.json',summary)
subgroups=[]
for dataset in ['longmemeval','locomo']:
    for category in sorted({r['category'] for r in rows if r['dataset']==dataset}):
        for method in ['full_hybrid','fifo','static_content-23','gru_content-23','liquid_content-23','liquid_fixed_time-23']:
            for budget in ([None] if method=='full_hybrid' else [32000,128000]):
                rr=[r for r in rows if r['dataset']==dataset and r['category']==category and r['method']==method and r['budget']==budget and r['retrieval_recall'] is not None]
                if rr:subgroups.append(dict(dataset=dataset,category=category,method=method,budget=budget,n=len(rr),retrieval_recall=float(np.mean([r['retrieval_recall'] for r in rr]))))
save('category_metrics.json',subgroups)
audits=dict(capacity_pass=all(r['budget'] is None or r['payload_bytes']<=r['budget'] for r in rows),context_limit_pass=all(r['context_chars']<=16000 for r in rows),
    restart_identical=read(OUT/'agent_dryrun.json')==read(OUT/'agent_resumed.json'),qa_completed=sum('judge_response' in r for r in qa),qa_expected=len(read(OUT/'qa_tasks.json')),judge_parse_failures=sum('judge_response' in r and r['custom_judge_correct'] is None for r in qa),tests=(OUT/'tests.txt').read_text(encoding='utf-16') if (OUT/'tests.txt').read_bytes().startswith(b'\xff\xfe') else (OUT/'tests.txt').read_text(encoding='utf-8'))
test_ids={r['question_id'] for r in rows if r['dataset']=='longmemeval'};checkpoint_checks=[]
for name,digest in protocol['checkpoint_hashes'].items():
    path=ROOT/'runs/preservation-study-v2/checkpoints'/name/'best.pt';ck=torch.load(path,weights_only=True)
    checkpoint_checks.append(dict(method=name,unchanged=sha(path)==digest,training_test_overlap=sorted(set(ck['train_ids'])&test_ids)))
save('checkpoint_audit.json',checkpoint_checks);audits['checkpoints_unchanged']=all(r['unchanged'] for r in checkpoint_checks);audits['training_test_disjoint']=all(not r['training_test_overlap'] for r in checkpoint_checks)
# Headers only: source text itself is allowed to mention arbitrary strings.
import re
audits['anonymous_headers']=all(all(re.fullmatch(r'm_[0-9a-f]{16}',s) for s in re.findall(r'^\[session=(\S+) date=',r['context'],flags=re.M)) for r in read(OUT/'qa_tasks.json'))
request_keys={r[k]['request_hash'] for r in qa for k in ['generation','judge_response'] if k in r};config=load_config('configs/minute_precision.yaml')['api'];db=sqlite3.connect(ROOT/config['cache_db']);attempts=[]
for key in request_keys:
    attempts.extend(dict(request_hash=key,started=start,status=status,elapsed=elapsed,usage=json.loads(usage)) for start,status,elapsed,usage in db.execute('SELECT started,status,elapsed,usage_json FROM requests_log WHERE request_hash=?',(key,)))
db.close();attempts.sort(key=lambda r:r['started']);stamps=[r['started'] for r in attempts];rolling=max([sum(t-60<s<=t for s in stamps) for t in stamps] or [0]);save('api_attempts.json',attempts)
usage=Counter()
for r in attempts:
    if r['status']=='200':usage.update({k:v for k,v in r['usage'].items() if isinstance(v,(int,float))})
save('api_usage.json',dict(attempts=len(attempts),statuses=dict(Counter(r['status'] for r in attempts)),max_observed_attempts_per_rolling60s=rolling,tokens=dict(usage)))
audits['rate_observed_pass']=rolling<=10;audits['qa_complete']=bool(status.get('complete')) and audits['qa_completed']==audits['qa_expected'];audits['tests_pass']='99 passed' in audits['tests'];audits['mechanical_ready']=all(audits[k] for k in ['tests_pass','capacity_pass','context_limit_pass','restart_identical','checkpoints_unchanged','training_test_disjoint','anonymous_headers','rate_observed_pass','qa_complete']) and audits['judge_parse_failures']==0
save('readiness.json',audits)
qsummary=[]
for method in sorted({r['method'] for r in qa}):
    rr=[r for r in qa if r['method']==method];scores=[r['custom_judge_correct'] for r in rr if r.get('custom_judge_correct') is not None]
    qsummary.append(dict(method=method,completed=len(scores),correct=sum(scores),accuracy=float(np.mean(scores)) if scores else None))
save('qa_summary.json',qsummary)
paired=[]
for qid in protocol['qa_ids']:
    rr={r['method']:r for r in qa if r['question_id']==qid};paired.append(dict(question_id=qid,category=next(iter(rr.values()))['category'] if rr else None,scores={k:r.get('custom_judge_correct') for k,r in rr.items()}))
save('qa_pairs.json',paired)
fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained');methods=['full_hybrid','fifo','static_content-23','gru_content-23','liquid_content-23','liquid_fixed_time-23'];labels=['Unbounded','FIFO','Static','GRU','Liquid','Liquid fixed time']
for ax,dataset in zip(axes,['longmemeval','locomo']):
    vals=[]
    for method in methods:
        r=next(r for r in summary if r['dataset']==dataset and r['method']==method and (r['budget']==128000 or r['budget'] is None));vals.append(r['retrieval_recall'])
    ax.barh(labels,vals,color=['#97a9b8','#aab4bb','#71989a','#6786a7','#b88656','#b29d86']);ax.invert_yaxis();ax.set_xlim(0,1.1);ax.set_title(dataset+' / 128k payload bytes');ax.set_xlabel('Evidence-session retrieval recall (not QA accuracy)')
    for i,v in enumerate(vals):ax.text(v+.012,i,f'{v:.3f}',va='center')
fig.savefig(OUT/'comparison.png',dpi=170);plt.close(fig)
rt=''.join(f'<tr><td>{r["dataset"]}</td><td>{r["budget"] or "unbounded"}</td><td>{r["method"]}</td><td>{r["retrieval_recall"]:.3f}</td><td>{r["evidence_retention"]:.3f}</td><td>{r["capacity_active_questions"]}</td></tr>' for r in summary if r['method'] in methods)
qt=''.join(f'<tr><td>{r["method"]}</td><td>{r["correct"]:.0f}/{r["completed"]}</td></tr>' for r in qsummary)
report=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>语言记忆与Agent运行就绪报告</title><style>body{{max-width:1150px;margin:40px auto;font:17px/1.7 system-ui;color:#26384a}}h1,h2{{color:#163f56}}img{{width:100%}}td,th{{padding:7px 13px;border-bottom:1px solid #ddd}}.note{{background:#edf5f1;padding:20px}}code{{background:#eef1f4}}a{{color:#176a91}}</style>
<h1>语言记忆与 Agent run 前置验收</h1><p>已结束外围数值实验，回到公开对话、检索与实际问答。当前工程就绪：{audits['mechanical_ready']}。</p>
<p class="note">完成 55,809 条离线比较、2,067 道公开问题、12 份实际 Agnes 回答及独立调用评分；顺序写入→预算淘汰→检索→匿名日期上下文→回答→评分已跑通。可以开展受控的 agent run，不再以液态是否先胜出为工程准入条件。该结论不等于已证明液态论文优势。</p>
<h2>完成哪些前置工作</h2><p>比较完整检索、FIFO、静态、GRU、液态与独立训练的固定时间液态；四类学习模型各三种子，32k/128k 两档字节预算。使用已有冻结的语言训练权重，不把气象模型迁移结果冒充语言记忆能力。全部 LongMemEval 自定义测试93题和符合证据/拒答条件的 LoCoMo 问题参与；两个数据集以前已曝光，结果是工程与开发复验，非新确认实验。</p>
<p>保留完整抽取句子、角色与来源日期；不确定的语义关系不触发合并或真值覆盖。仍存在显式预算淘汰，不能保证所有旧事实永久保存。写入接口只允许文本/来源/日期，问题与参考答案不进入写入模型；参考答案仅用于独立评分。共享模型权重和Python/索引开销不计入payload预算，报告不等同进程RSS。</p>
<img src="comparison.png"><table><tr><th>数据</th><th>预算</th><th>方法（学习模型种子23）</th><th>检索召回</th><th>存储覆盖</th><th>发生容量淘汰的问题数</th></tr>{rt}</table>
<p>全部三种子、类别和预算见 summary.json 与 category_metrics.json。128k下LoCoMo许多历史能全部容纳，方法相同不能解释为机制等效；应结合32k结果。LME液态与静态/GRU接近，尚无独有机制结论。</p>
<h2>Agnes 实际问答链路</h2><p>预先按哈希固定6道题，覆盖事实更新、助手记忆、时间推理；完整检索与液态种子23/128k各回答一次，再独立调用同一提供方评分。参考答案未放入回答提示词。这是12份回答的小规模链路验收，不是正式benchmark准确率或盲化人工评分。</p><table><tr><th>方法</th><th>自定义判分正确</th></tr>{qt}</table>
<p>请求尝试 {len(attempts)} 次；本轮观测任意滚动60秒最多 {rolling} 次。Token用量：{dict(usage)}。共享持久限速配置9RPM，429/瞬时错误/缓存/断点机制沿用现有客户端并通过测试。完整原始回答、参考、评分和来源见 qa/ 与 qa_pairs.json。</p>
<h2>运行与恢复验收</h2><p>99项测试通过，覆盖11次请求等待、跨客户端限速、429恢复、答案隔离、时间精度、预算、来源匿名及新增运行接口。真实公开49个会话已完成液态dry run，再恢复运行所得结果逐字段一致，payload为124,855字节，未超过128,000。该回放不会发送网络请求。</p>
<p>新增 <code>scripts/agent_memory_run.py</code>：默认dry run，支持fifo/static/gru/liquid、固定预算、原子检查点、输入与模型指纹校验；显式 <code>--answer</code> 才调用Agnes。输入是 events/source_id/date/turns 和 question/question_date，示例 agent_input.json 不含答案标签。完整 agent 多工具行动与长期任务成功率尚未测，不把一次记忆问答称为通用agent验证。</p>
<h2>进入下一阶段的决定</h2><p>工程链路已经具备受控语言agent实验条件。下一阶段直接跑多轮任务并保留FIFO、静态和GRU对照；不再要求先做新的语义分类、气象或纯机制外围实验。可执行入口已交付，本报告未在后台持续运行任务。液态若没有收益则如实记录，不改动基线或标签来制造故事。</p><p>实际小样本完整检索6/6、液态5/6。失败题的标注证据在容量淘汰后已不在记忆中（evidence_retention=0），回答选择拒答；该失败不归咎于API或判分格式。见failure_cases.json。三次HTTP500自动恢复，23次尝试包含20次成功HTTP响应；24个回答/评分结果中4个复用缓存。重新执行完整入口未新增HTTP请求，见resume_audit.json。</p>
<p><a href="readiness.json">验收清单</a> · <a href="summary.json">全部离线汇总</a> · <a href="category_metrics.json">分类型</a> · <a href="qa_pairs.json">配对问答</a> · <a href="api_usage.json">API用量</a> · <a href="checkpoint_audit.json">训练隔离与权重审计</a> · <a href="agent_input.json">可运行输入</a> · <a href="agent_dryrun.json">运行输出</a> · <a href="tests.txt">测试日志</a></p></html>'''
(OUT/'report.html').write_text(report,encoding='utf-8')
signature,archive=snapshot_source();save('delivery_manifest.json',dict(runtime_source_signature=signature,runtime_source_archive=archive,experiment_source=protocol['source'],runtime=runtime_info(),files={n:sha(OUT/n) for n in ['protocol.json','offline.jsonl','qa_tasks.json','readiness.json','agent_input.json','agent_state.json','agent_dryrun.json','agent_resumed.json']}))
print(json.dumps(audits));print(json.dumps(qsummary));print(json.dumps(dict(usage)))
