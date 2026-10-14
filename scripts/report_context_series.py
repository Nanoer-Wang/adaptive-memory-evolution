"""Present the complete frozen pilot and its all-question repeat together."""
import sys,json,html,hashlib,sqlite3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.config import atomic_json,load_config,resolve
from src.provenance import snapshot_source
from scripts import context_utility as original
from scripts import repeat_context_utility as repeat
P=original.OUT;R=repeat.OUT
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    first=read(P/'summary.json');rep=read(R/'summary.json');diag=read(P/'diagnostics.json');review=read(R/'review.json')
    reviewed={(r['repeat'],r['method']):r for r in review['methods']}
    # Repeat the completed execution path: cache/resume must issue no new requests.
    dbpath=resolve(load_config('configs/minute_precision.yaml')['api']['cache_db'])
    def count():
        with sqlite3.connect(dbpath) as db:return db.execute('SELECT COUNT(*) FROM requests_log').fetchone()[0]
    before=count();original.qa('development');original.qa('test');repeat.qa();after=count();assert before==after
    hashes=read(R/'training_complete.json')['checkpoints']
    assert all(hashlib.sha256((P/path).read_bytes()).hexdigest()==sha for path,sha in hashes.items())
    assert hashlib.sha256((P/'policy_records.json').read_bytes()).hexdigest()==read(R/'protocol.json')['parent_policy_records_sha256']
    atomic_json(P/'resume_audit.json',dict(requests_before=before,requests_after=after,additional_http_requests=after-before,
        original_checkpoints_unchanged=True,original_policy_records_unchanged=True,repeated_all_original_and_repeat_qa_paths=True))
    methods=['always_full','always_compact','abstention_cascade','concentration_rule','linear','mlp','gru','liquid']
    labels=['完整检索内容','始终精简','精简拒答后补读','简单集中度规则','线性决策器','MLP决策器','GRU决策器','液态决策器']
    by={(r['repeat'],r['method']):r for r in rep['methods']};base=by['pooled','always_full'];compact=by['pooled','always_compact'];cascade=by['pooled','abstention_cascade']
    attempts=sorted(read(P/'api_attempts.json')+read(R/'api_attempts.json'),key=lambda r:r['started']);left=0;peak=0
    for right,r in enumerate(attempts):
        while attempts[left]['started']<=r['started']-60:left+=1
        peak=max(peak,right-left+1)
    tokens=sum(r['usage'].get('total_tokens',0) or 0 for r in attempts)
    api=dict(answer_requests=120,judge_requests=first['api']['judge_requests']+rep['api']['judges'],http_attempts=len(attempts),reported_tokens=tokens,rolling60_max=peak)
    assert peak<=9
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    english=['Full context','Always compact','Abstention cascade','Concentration rule','Linear','MLP','GRU','Liquid']
    y=np.arange(len(methods));fig,ax=plt.subplots(1,2,figsize=(12.5,5.6),gridspec_kw={'width_ratios':[1.2,1]})
    for i,m in enumerate(methods):
        a,b=by[0,m],by[1,m];ax[0].plot([a['accuracy']*12,b['accuracy']*12],[i,i],color='#b6c5d2',lw=2)
        ax[0].scatter(a['accuracy']*12,i,color='#2563a6',marker='o',s=65,label='First generation' if i==0 else None)
        ax[0].scatter(b['accuracy']*12,i,color='#e29b31',marker='x',s=75,label='Fixed repeat' if i==0 else None)
        ax[0].scatter(reviewed[1,m]['accuracy']*12,i,facecolors='none',edgecolors='#398a68',marker='s',s=120,label='Repeat + known-error review' if i==0 else None)
    ax[0].set_yticks(y,english);ax[0].invert_yaxis();ax[0].set_xlim(-.25,12.25);ax[0].set_xticks(range(0,13,2));ax[0].set_xlabel('Correct / 12 (training seeds averaged)');ax[0].legend(loc='lower right')
    vals=[by['pooled',m]['prompt_tokens'] for m in methods];ax[1].barh(y,vals,color=['#718096','#38a39a','#397ca8','#96a6b4','#96a6b4','#96a6b4','#96a6b4','#bd7762']);ax[1].set_yticks(y,english);ax[1].invert_yaxis();ax[1].set_xlim(0,max(vals)*1.25)
    for i,v in enumerate(vals):ax[1].text(v+35,i,f'{v:,.0f}',va='center',fontsize=9)
    ax[1].set_xlabel('Mean prompt tokens, both generations')
    fig.suptitle('Task quality and reading cost: frozen policies, all-question repeat');fig.tight_layout();fig.savefig(P/'series.png',dpi=160);plt.close(fig)
    table=''
    for m,label in zip(methods,labels):
        a,b,c=by[0,m],by[1,m],by['pooled',m]
        table+=f"<tr><td>{label}</td><td>{a['accuracy']*12:.2f}</td><td>{b['accuracy']*12:.2f}</td><td>{reviewed[1,m]['accuracy']*12:.2f}</td><td>{c['prompt_tokens']:.0f}</td><td>{c['completion_tokens']:.0f}</td><td>{c['answer_calls']:.2f}</td></tr>"
    devtable=''
    for split in ['train','val','test']:
        row=[next(r for r in first['methods'] if r['split']==split and r['method']==m) for m in ['always_full','always_compact','abstention_cascade']]
        devtable+=f"<tr><td>{split}</td><td>{row[0]['questions']}</td>"+''.join(f"<td>{r['accuracy']:.1%}</td>" for r in row)+'</tr>'
    cases=[]
    for c in read(P/'mechanism_cases.json'):
        q=c['question_id'];second={a:read(R/'qa'/f'{q}-{a}.json') for a in ['full','compact']}
        cases.append(f"<details><summary>{q} · 首次完整/精简 {c['actions']['full']['correct']:.0f}/{c['actions']['compact']['correct']:.0f} · 复测 {second['full']['correct']:.0f}/{second['compact']['correct']:.0f}</summary><p>{html.escape(c['interpretation'])}</p><p>复测完整：{html.escape(second['full']['hypothesis'])}</p><p>复测精简：{html.escape(second['compact']['hypothesis'])}</p></details>")
    savings=1-compact['prompt_tokens']/base['prompt_tokens'];cs=1-cascade['prompt_tokens']/base['prompt_tokens']
    doc=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>任务质量与阅读成本：完整实验记录</title><style>body{{max-width:1100px;margin:36px auto;padding:0 20px;font:16px/1.75 system-ui;color:#223548}}h1,h2{{line-height:1.3}}.lead{{background:#edf5f8;padding:20px;border-radius:12px}}img{{width:100%;height:auto}}table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;text-align:left;border-bottom:1px solid #dce3e8}}details{{margin:14px 0}}a{{color:#176294}}.note{{color:#536577}}code{{overflow-wrap:anywhere}}</style>
    <h1>让记忆服务任务：质量、阅读成本与重复性</h1><p class="lead">完整输入首次/复测答对 {by[0,'always_full']['accuracy']*12:.0f}/{by[1,'always_full']['accuracy']*12:.0f} 题；始终精简 {by[0,'always_compact']['accuracy']*12:.0f}/{by[1,'always_compact']['accuracy']*12:.0f} 题，输入token减少 {savings:.1%}；拒答后补读 {by[0,'abstention_cascade']['accuracy']*12:.0f}/{by[1,'abstention_cascade']['accuracy']*12:.0f} 题，输入token减少 {cs:.1%}。这是12道相同题目的两次生成，不能视为24道独立测试。</p>
    <p>已完成48道公开数据问题的96份完整/精简回答，24训练、12验证、12测试；训练12个决策器后才生成测试答案。再对全部12道测试题统一复测，增加24份新回答，不筛选改善案例。数据来自已有离线开发划分，尚非新的独立数据集。</p>
    <p>所有方法使用同一个64KB强线性记忆写入器、同一top5检索来源。这里的“完整”是检索出的已压缩内容，最多16000字符，不是全部原始历史；“精简”是4200字符的查询相关摘录。线性/MLP/GRU/液态只控制阅读预算，液态并未替换本轮底层记忆写入。</p>
    <p>复测摄影答案含不兼容配件，自动评分未发现。按已知错误复核，完整输入和补读的复测成绩均为5/12，精简仍5/12。上面的原始评分保留；后面的表格单列复核，不能把自动通过直接当作实际任务成功。</p>
    <img src="series.png" alt="首次与统一复测的答对数和阅读token成本"><table><tr><th>方法</th><th>首次自动/12</th><th>复测自动/12</th><th>复测复核/12</th><th>平均输入token</th><th>平均输出token</th><th>平均回答调用</th></tr>{table}</table>
    <p class="note">学习型策略按3个训练种子平均；同一动作答案被所有策略共享，不增加独立样本。级联基线按已生成答案回放，每次拒答重试计入完整的两次调用和token；它可能降低token用量但增加网络往返，不能据此声称更快。原始自动裁判评分保留，偏好题存在争议。严格拒答契约防止把显式拒答判为完成任务。</p>
    <h2>液态机制为何尚未产生额外收益</h2><p>训练集只有4/24对回答存在正确率差异，而且全部是偏好题，其中发现明确评分疑点。20道事实训练题的完整/精简正确性全部相同，无法提供何时因压缩而丢失关键事实的直接监督。首轮去掉液态单元额外时间参数没有改变任何测试动作；重置递推状态只改变36个种子×问题决策中的2个。当前输入是查询相关度、长度和时间等统计，也没有直接描述精简前后究竟丢了哪些关键语义。这些证据不支持把当前收益归于液态时间动力学。</p>
    <h2>不要只看表现较好的划分</h2><p>首轮训练/验证/测试全部保留。验证集上精简和级联均低于完整输入；本轮还不能得出普遍非劣或显著优越结论。</p><table><tr><th>划分</th><th>题数</th><th>完整</th><th>精简</th><th>拒答补读</th></tr>{devtable}</table>
    <h2>内容级核查</h2>{''.join(cases)}<p>个人最好成绩的旧值和新值在两种上下文中都存在，不能把首轮改善解释成“删除旧值自动消除矛盾”。日期题沿用参考答案的会话日期约定，不代表原文明确给出了精确取消日期。摄影题则是明显的词面匹配漏掉相机语义，补读可以恢复。</p>
    <p>两道首轮精简改善的事实题，在复测中完整输入也答对，因此尚不能把改善归为稳定的抗干扰优势。复测完整摄影答案推荐CFexpress卡，<a href="https://helpguide.sony.net/ilc/1930/v1/en/contents/TP0002711790.html">Sony A7R IV官方说明</a>指出两个卡槽仅支持SD卡。含该不兼容建议的回答在严格复核中计为不完全正确，对所有使用它的策略一致处理。复核由本助手完成，并非独立人工盲评；其他偏好判分争议保留。</p>
    <h2>可以保留的研究方向</h2><p>继续研究“少量读取是否足够，以及何时值得补读”。下一轮需要让训练数据包含真实的关键信息缺失、更新与干扰对照，并把精简内容的语义覆盖纳入决策观察；否则复杂递推只是对粗粒度统计再拟合。若研究连续历史，状态应记录以往读取的收益与风险，在跨任务连续执行中更新，再与具有相同观察、时间特征和预算的线性统计/GRU公平比较。以上为后续设计约束，尚不是已验证的新模型或论文创新结论。</p>
    <h2>复核与保存</h2><p>本轮共 {api['answer_requests']} 次实际回答请求、{api['judge_requests']} 次判分请求，{api['http_attempts']} 次HTTP尝试，提供商报告 {tokens:,} token；滚动60秒最多 {peak} 次。完整恢复检查没有新增HTTP请求；原检查点和策略记录未变；130项测试通过。</p>
    <p><a href="report.html">首轮逐题与统计</a> · <a href="../context-utility-repeat-v1/report.html">全部复测与区间</a> · <a href="diagnostics.html">监督、判分敏感性和内容核查</a> · <a href="audit.json">隔离与冻结审计</a> · <a href="resume_audit.json">恢复审计</a></p></html>'''
    (P/'index.html').write_text(doc,encoding='utf-8');source,archive=snapshot_source();atomic_json(P/'delivery.json',dict(api=api,source=source,archive=archive,
        prompt_savings_compact=savings,prompt_savings_cascade=cs,primary='index.html',repeat_scope='All12 test questions once more; fixed original policies; no refitting.'))
    print(json.dumps(dict(api=api,savings_compact=savings,savings_cascade=cs,resume_new_requests=after-before),indent=2))

if __name__=='__main__':main()
