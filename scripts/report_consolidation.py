"""Architecture, comparisons and provenance for the consolidation refactor."""
import sys,json,html,hashlib,sqlite3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.config import atomic_json
from src.provenance import snapshot_source
from scripts import consolidation_study as study
OUT=study.OUT
read=study.read

def table(rows,fields):
    return '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in fields)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(f'{r[k]:.4f}' if isinstance(r[k],float) else str(r[k]))+'</td>' for k in fields)+'</tr>' for r in rows)+'</table>'

def main():
    streams=read(OUT/'stream_records.json');summary=[]
    for kind in list(study.KINDS)+['fifo','decay']:
        for budget in study.BUDGETS:
            for condition in study.CONDITIONS:
                rr=[r for r in streams if (r['kind'],r['budget'],r['condition'])==(kind,budget,condition)]
                summary.append(dict(kind=kind,budget=budget,condition=condition,stories=len({r['story'] for r in rr}),accuracy=float(np.mean([r['accuracy'] for r in rr])),mean_ms=float(np.mean([r['elapsed_seconds'] for r in rr])*1000),mean_evictions=float(np.mean([r['evicted'] for r in rr])),mean_merges=float(np.mean([r['merged'] for r in rr])),max_payload=max(r['peak_bytes'] for r in rr)))
    atomic_json(OUT/'summary.json',summary)
    continual=read(ROOT/'runs/consolidation-continual-v1/summary.json')
    controls=read(ROOT/'runs/consolidation-mechanism-v2/summary.json')
    languages=[];keys=set();review=[]
    for run in ['consolidation-language-v1','consolidation-language-controls-v1','consolidation-language-hybrid-v1']:
        rr=[]
        for p in sorted((ROOT/'runs'/run/'qa').glob('*.json')):
            r=read(p);assert r['evaluation_status'] in ['graded','explicit_abstention'];rr.append(r)
            review.append(dict(run=run,record=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),question=r['question'],reference=r['answer'],answer=r['hypothesis'],correct=r['correct'],label_changed=False))
            for response in [r.get('generation'),r.get('judge_response')]+r.get('repair_attempts',[])+r.get('repair_judges',[]):
                if response:keys.add(response['request_hash'])
        for method in sorted({r['method'] for r in rr}):
            rows=[r for r in rr if r['method']==method]
            languages.append(dict(run=run,method=method,questions=len(rows),correct=sum(r['correct'] for r in rows),mean_prompt_tokens=float(np.mean([r['generation']['usage']['prompt_tokens'] for r in rows]))))
    attempts=[]
    with sqlite3.connect(ROOT/'artifacts/api/responses.sqlite3') as db:
        for key in keys:attempts.extend(dict(request_hash=key,started=t,status=s,usage=json.loads(u)) for t,s,u in db.execute('select started,status,usage_json from requests_log where request_hash=?',(key,)))
    times=sorted(r['started'] for r in attempts)
    api=dict(unique_requests=len(keys),attempts=len(attempts),tokens=sum(r['usage'].get('total_tokens',0) for r in attempts),rolling60_max=max(sum(t-60<z<=t for z in times) for t in times))
    atomic_json(OUT/'api_attempts.json',attempts);atomic_json(OUT/'language_summary.json',languages)
    atomic_json(OUT/'reference_review.json',dict(reviewer='Assistant reference review of36 valid answers; not independent human blind review. No label changes. Invalid-format retry only.',records=review))
    diagnoses=[
        dict(component='MemoryRetriever',file='src/models/memory_model.py',finding='全部候选仍可用，液态状态调整检索查询，不等于有限容量记忆巩固。'),
        dict(component='BudgetStore',file='src/memory/budget_store.py',finding='文本已经外置，但写入优先级固定，缺少后续巩固与遗忘。'),
        dict(component='TaskController',file='src/memory/task_controller.py',finding='单题补读控制与跨任务长期巩固不是同一研究目标。'),
        dict(component='LiquidMemory',file='src/memory/liquid_memory.py',finding='exp(-rate*(1+delta))混合事件更新和时间流；并非精确连续积分。'),
        dict(component='EvidenceDynamics / VersionLedger',file='src/memory/evidence_dynamics.py',finding='已有精确分段时间流和版本语义，这是复用基础，不能宣称首次提出。')]
    atomic_json(OUT/'architecture_diagnosis.json',diagnoses)
    assert all(r['peak_bytes']<=r['budget'] for r in streams)
    for run in ['consolidation-continual-v1','consolidation-mechanism-v2']:
        assert all(r['peak_bytes']<=r['budget'] for r in read(ROOT/'runs'/run/'stream_records.json'))
    assert all(r['payload_bytes']<=32000 for r in read(ROOT/'runs/consolidation-language-v1/tasks.json'))
    raw=read(OUT/'stories.json')
    digest=lambda d:hashlib.sha256(json.dumps(d['events'],sort_keys=True).encode()).hexdigest()
    train={digest(d) for d in raw.values() if d['split']=='train'};test={digest(d) for d in raw.values() if d['split']=='test'}
    checkpoints={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/'checkpoints').glob('*.pt')}
    sizes={kind:sum(p.numel() for p in study.load_model(kind,23).parameters()) for kind in study.KINDS}
    atomic_json(OUT/'audit.json',dict(all_budget_checks_passed=True,exact_train_test_story_duplicates=len(train&test),checkpoint_sha256=checkpoints,allocated_model_parameters=sizes,parameter_note='Hidden8 for all; NOT parameter-matched. Simple controls retain unused allocated dynamic heads. No parameter-efficiency claim.',api=api,tests=152))
    main=[]
    for kind in list(study.KINDS)+['fifo','decay']:
        row=dict(method=kind)
        for phase in ['immediate','delayed']:row[phase]=next(r['accuracy'] for r in continual if r['kind']==kind and r['budget']==8192 and r['phase']==phase)
        main.append(row)
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained');names=[r['method'] for r in main];x=np.arange(len(names))
    ax.bar(x-.19,[r['immediate']*100 for r in main],.38,label='Immediate');ax.bar(x+.19,[r['delayed']*100 for r in main],.38,label='Delayed revisit')
    ax.set_xticks(x,names,rotation=25,ha='right');ax.set_ylim(0,105);ax.set_ylabel('Exact task accuracy (%)');ax.set_title('Persistent scoped-task replay:100 public worlds,8192-byte payload');ax.legend()
    fig.savefig(OUT/'continual_comparison.png',dpi=170);plt.close(fig)
    doc='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Liquid Memory Consolidation System</title><style>body{max-width:1160px;margin:36px auto;font:16px/1.8 system-ui;color:#182438}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:6px;border-bottom:1px solid #d9e0e8;text-align:left}pre{background:#eef3f8;padding:18px;white-space:pre-wrap}img{width:100%}h2{margin-top:32px}</style><h1>Liquid Memory Consolidation System</h1><p>研究定位：Adaptive Liquid Memory Consolidation for Lifelong Agents。外部存储与动态控制解耦；这是可运行的实验框架，不是液态独有优势或论文新颖性已获证明。</p><h2>架构诊断</h2>'''
    doc+=table(diagnoses,['component','file','finding'])
    doc+='''<p>不能据此断言LNN无法表示记忆。更具体的问题是学习目标与任务不一致、强静态内容信号掩盖动态机制，以及信息抽取/存储损失先于控制决策。这些是可检验解释，不是所有负结果的单一原因。</p><h2>重构方案与数学定义</h2><pre>experience → past-only features → continuous control state → α,w,β → external evidence
flow: h(t+Δ)=e+(h(t)-e)⊙exp(-rΔ)
event: h⁺=h⁻+sigmoid(Gx)⊙(tanh(Ux)-h⁻)
policy: α=sigmoid(a), w=2sigmoid(i), β=(softplus(b)+0.0001)/30
strength: s(t+Δ)=s(t)exp(-βΔ)
repeat / successful task feedback: s⁺=min(20,s⁻+αwR)
capacity: provisional admission → consolidate → evict weakest if needed</pre><p>时间系数在事件间固定，因此分段时间流精确；这不是标准LTC/CfC，也不是任意非线性ODE的精确解。事件跳变与时间流分开。隐藏维度8，32维投影内容与8维历史统计；β=0和w=1分别独立重训。运行时所有观察都更新控制状态，训练也如此，不存在完整历史训练、稀疏历史执行的差异。</p><p>学习采用训练集未来支持证据与使用次数代理目标，尚未端到端优化自然语言Agent收益。成功反馈直接强化证据强度，不作为液态状态的单独训练输入。训练/验证是短故事，持续任务测试是零样本长度迁移，尚未解决长期训练分布对齐。</p><p>ConsolidationMemory提供observe/retrieve/reinforce/checkpoint/restore；ConsolidatingAgentMemory提供完整归属切片和语言接口。默认混合检索仅重编码存活文本，不能读取被淘汰档案。结构化更新依赖可信key/value，不能把相似度当真伪判断。完全重复文本合并且保留计费来源；跨经验语义摘要尚未实现。</p><h2>实验和工程修复</h2><p>bAbI qa1使用120训练、40验证、官方200测试故事，3种子、3容量、3时间条件。时钟是受控注入，不是原始真实日期。21个学习模型与FIFO、固定衰减对照。测试集此前已在项目中使用，不能宣称全新独立确认。所有方法统一512字节控制元数据预留，预算为定义的紧凑载荷，不是Python RSS/JSON文件尺寸；同隐藏维度不代表参数完全匹配。</p><p>v1固定0.5拒收门槛在有空位时丢失信息，已完整保留。v2先候选写入再竞争空间，不把这项修复算作液态收益。额外修正了一次重复归一化误差；能力测试覆盖零时间、分段积分、版本防复活、字节预算、恢复一致性和防止读取淘汰档案。</p><h2>持续任务：8192字节</h2><p>100个公开bAbI世界共用持续控制状态和存储，显式世界命名空间防止事实混淆。每完成任务，回访5个任务前的最后问题；原问题及原答案不变。属于模拟多任务Agent，不是自然同用户长期日志。一个固定任务顺序与3训练种子不构成300个独立长期实验。</p>'''
    doc+=table(main,['method','immediate','delayed'])+'<img src="continual_comparison.png" alt="Immediate and delayed task accuracy">'
    doc+='''<p>液态回访44.67%超过FIFO3%，但低于线性64%与固定衰减63%。β=0造成旧强记忆占据空间、新证据饥饿；这只说明本机制需要释放存储压力，不说明学习β比固定β更好。w=1和无时间流结果也没有支持重要性权重或液态时间机制的必要性。</p><h2>同权重控制：事后机制诊断</h2>'''
    doc+=table([r for r in controls if r['budget']==8192],['kind','phase','accuracy'])
    doc+='''<p>固定同一液态权重，仅将β改为1/30，回访提高至61.33%。训练正例使用窗口中位数2天、最大14.05天；持续测试跨5个完整任务。后20任务预测半衰期中位数17.91天，固定规则为20.79天。目标窗口/前缀统计分布偏移值得优先修复；它是有证据的诊断线索，不是排除其他原因的证明。</p><h2>自然语言迁移与检索诊断</h2><p>6道此前未进入QA链路的LongMemEval事实题，4方法共24回答；策略仅在bAbI训练，故仅是跨域诊断。32000字节+32维投影检索均0/6。19/24条件完全丢失标注证据来源，其余来源存在也不保证关键片段存在。无容量限制下原检索1/6，完整384维语义加词法混合检索4/6。两项额外诊断各6题，不是同预算竞品；检索更改也伴随新一次生成波动。</p>'''
    doc+=table(languages,['run','method','questions','correct','mean_prompt_tokens'])
    doc+='''<p>默认语言接口已切换为混合检索；旧投影检索由显式参数保留。未重新训练或替换主实验的有效错误答案，不能把无容量结果当成有限记忆结果。额外查询重编码计算未用于宣称速度优势。</p><h2>交付范围与结论</h2><p>已完成存储/控制解耦、时间流、候选写入、重复合并、反馈增强、遗忘消融、可信版本处理、持久恢复、公开实验与实际QA链路。尚未建立液态独有收益；自然语言重要性学习、语义抽象、真实多步工具Agent奖励、独立长期数据确认仍缺失。下一步应将训练改为包含延迟回访的持续任务轨迹，先对齐目标，而非扩大网络。</p><h2>复现</h2><pre>python scripts/consolidation_study.py --phase all
python scripts/consolidation_continual.py
python scripts/consolidation_continual.py --frozen-controls
python scripts/consolidation_language_probe.py --phase all
python scripts/consolidation_language_controls.py
python scripts/consolidation_language_controls.py --hybrid
python scripts/report_consolidation.py</pre><p>已有完成标记/缓存用于恢复；语言生成缺少缓存时才调用API。保留错误答案，不按成绩重采样。全部方案、原始答案、控制参数、模型、容量及调用记录分目录保存。</p>'''
    doc+=f'<p>152项测试通过。API统计：{html.escape(json.dumps(api))}。36份有效答案均对照参考复核，未改标签；1次无效格式重试保留原文。不是独立人工盲评。</p>'
    doc+='<details><summary>全部结构化条件</summary>'+table(summary,['kind','budget','condition','accuracy','mean_ms','mean_merges','mean_evictions'])+'</details><p><a href="audit.json">审计</a> · <a href="reference_review.json">问答复核</a> · <a href="horizon_diagnostic.json">时间窗口诊断</a> · <a href="tests.txt">测试记录</a></p></html>'
    resume_path=OUT/'resume_audit.json'
    if resume_path.exists():
        resume=read(resume_path)
        doc=doc.replace('</html>','<h2>恢复与运行入口</h2><p>恢复审计：新增HTTP请求为0，21个模型校验值不变，持续Agent恢复输出完全一致。<a href="resume_audit.json">完整恢复审计</a></p><pre>python -m src.cli consolidation --phase report\npython scripts/consolidation_agent_run.py --input runs/liquid-consolidation-v2/agent_example.json --state runs/liquid-consolidation-v2/agent_state.json --output runs/liquid-consolidation-v2/agent_output.json</pre><p>持续运行脚本默认仅检索；显式添加 --answer 才调用Agnes生成答案。</p></html>')
    (OUT/'index.html').write_text(doc,encoding='utf-8')
    source,archive=snapshot_source()
    atomic_json(OUT/'delivery.json',dict(source=source,archive=archive,main_report='index.html',api=api,tests=152,trained_models=21,language_answers=36,claim='Experimental framework implemented; liquid-specific benefit and natural-language lifelong effectiveness NOT established.'))
    print(json.dumps(dict(api=api,language=languages,main=main),indent=2))

if __name__=='__main__':main()

