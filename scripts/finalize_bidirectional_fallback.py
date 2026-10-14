"""Delivery audit, source archive, uncertainty caveats and static research figure."""
import sys,json,hashlib,sqlite3,subprocess,html,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.config import atomic_json
from src.provenance import snapshot_source
from scripts import bidirectional_fallback as exp
OUT=exp.OUT


def main():
    audit=exp.read(OUT/'audit.json');review=exp.read(OUT/'reference_review.json')
    for name,r in review['records'].items():assert exp.base.sha(OUT/'qa'/f'{name}.json')==r['sha256']
    def requests():
        with sqlite3.connect(ROOT/'artifacts/api/responses.sqlite3') as db:return db.execute('select count(*) from requests_log').fetchone()[0]
    before=requests();run=subprocess.run([sys.executable,'scripts/bidirectional_fallback.py','--phase','all'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8');after=requests()
    (OUT/'resume_output.txt').write_text(run.stdout+'\n'+run.stderr,encoding='utf-8')
    unchanged=all(exp.base.sha(ROOT/k)==v for k,v in audit['checkpoint_sha256'].items())
    assert run.returncode==0 and before==after and unchanged
    atomic_json(OUT/'resume_audit.json',dict(exit_code=run.returncode,new_http_requests=after-before,all36_checkpoints_unchanged=unchanged))
    rows=exp.read(OUT/'summary.json');selected=[r for r in rows if r['endpoint']=='full' and r['cap']==.25 and ((r['recipe']=='bidirectional' and r['method'] in ['linear_no_omission','linear','gru','liquid']) or (r['recipe']=='rule' and r['method']=='abstention'))]
    table=''.join(f"<tr><td>{html.escape(r['method'])}</td><td>{r['correct']:.2%}</td><td>{r['admitted']:.2%}</td><td>{r['prompt_tokens']:.0f}</td></tr>" for r in selected)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for recipe,method in [('rule','abstention'),('bidirectional','linear_no_omission'),('bidirectional','linear'),('bidirectional','gru'),('bidirectional','liquid')]:
        rr=sorted([r for r in rows if r['endpoint']=='full' and r['recipe']==recipe and r['method']==method],key=lambda r:r['cap'])
        for ax,metric in zip(axes,['correct','prompt_tokens']):ax.plot([r['cap']*100 for r in rr],[r[metric]*(100 if metric=='correct' else 1) for r in rr],marker='o',label=method)
    for ax in axes:ax.set_xlabel('Maximum extra answer calls (% of queries)');ax.set_xticks([10,25,50]);ax.grid(alpha=.2)
    axes[0].set_ylabel('Reference accuracy (%)');axes[1].set_ylabel('Mean prompt tokens');axes[1].legend(fontsize=8)
    fig.suptitle('12 exposed development questions; identical hard caps, differing actual usage');fig.savefig(OUT/'budget_curves.png',dpi=170);plt.close(fig)
    insertion=f'''<h2>结果解读</h2><p>训练监督为20对有益、77对无益、3对有害，覆盖20道独立训练题。3条有害来自3道题：鱼缸数量3变2、事件先后颠倒、正确6个月变为拒答。后者是新生成的完整扩展，前两条是既有600→1800的真实结果。每种固定干预都保留全部题目，没有只挑负面样本。</p><p>25%是预定三个预算之一，不是验证选定的最终部署点；以下便于阅读，完整10%/25%/50%结果均保留。同上限不代表同实际调用量。</p><table><tr><th>方法</th><th>准确率</th><th>实际补读</th><th>输入token</th></tr>{table}</table><p>同样预算/训练流程下，旧60对训练的液态为39.35%与1098token；加20对逐级补读后40.28%与1097token；全部100对后39.81%与1035token。没有单调的质量改善。全部训练的液态在50%预算下下降至35.19%，有害补读率6.02%。预算约束有效，三类监督并未消除误触发。</p><p>完整补读测试24对中仅2对有益（来自同一道题）、3对有害；逐级补读测试12对仅1对有益、0对有害。验证集逐级补读全部无益。各模型在逐级补读上的表现不能证明会普遍识别有害扩展。负面监督仅3道题，仍不足以确认泛化。</p><p>液态25%预算时重置状态准确率35.65%，但调用量也改变，不能把差异全部归因于状态价值；取消时间输入后40.28%，仍不支持液态时间机制独有优势。与线性/GRU/规则的逐题配对探索区间见附件：预算及顺序先在题内平均，区间条件于这些固定回放，未重模拟预算队列，且没有多重比较校正。</p><p>新40份答案已对照参考复核。原裁判把明确回答600粉丝判为错误，训练采用独立保存的纠正标签，原始文件未改；另保留扩展答案推测性措辞、矛盾事件排序的评分敏感性标记。不是独立人工盲评。141项测试通过，36检查点断点续跑保持不变，新增HTTP为0。一次中途数据拼装错误修复后复用全部生成记录重训，未因答错重新采样。</p><img src="budget_curves.png" style="width:100%" alt="Budget quality and token curves"><p><a href="reference_review.json">参考复核与纠正</a> · <a href="paired_comparisons.json">配对探索区间</a> · <a href="evaluation_signal.json">各终点标签覆盖</a> · <a href="resume_audit.json">恢复检查</a> · <a href="tests.txt">测试记录</a></p><details><summary>展开全部预定预算、模型与状态诊断</summary>'''
    path=OUT/'index.html';doc=path.read_text(encoding='utf-8');marker='<table><tr><th>补读终点</th>'
    doc=re.sub('<!-- delivery-start -->.*?<!-- delivery-end -->','',doc,flags=re.S).replace('</details><!-- delivery-close -->','')
    assert marker in doc
    doc=doc.replace(marker,'<!-- delivery-start -->'+insertion+'<!-- delivery-end -->'+marker,1).replace('</body>','')
    doc=doc.replace('</html>','</details><!-- delivery-close --></html>');path.write_text(doc,encoding='utf-8')
    source,archive=snapshot_source();atomic_json(OUT/'delivery.json',dict(source=source,archive=archive,tests_passed=141,reviewed_answers=40,corrected_raw_grades=1,checkpoints=36,api=audit['api'],main_report='index.html',limitations='20train questions,12 previously exposed test questions; not publication-level independent evidence; no liquid-specific advantage established.'))
    print('Report, figure, source archive and zero-request resume audit complete.')


if __name__=='__main__':main()
