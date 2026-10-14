"""Analysis only: never modifies frozen checkpoints or primary scores."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json, hashlib
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.memory.bounded_codec import BoundedCodec
from scripts.bounded_codec_study import tensorize

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/bounded-codec-v2'


def main():
    results=json.loads((OUT/'summary.json').read_text())
    protocol=json.loads((OUT/'protocol.json').read_text())
    rows=json.loads((OUT/'examples.json').read_text())
    predictions=[json.loads(s) for s in (OUT/'predictions.jsonl').read_text().splitlines()]
    aggregate=[]
    for kind in ('input_flow','gru','ltc'):
        for width in (8,32):
            group=[r for r in results if r.get('kind')==kind and r.get('width')==width]
            aggregate.append(dict(kind=kind,width=width, state_bytes=4*width,weight_bytes=group[0]['weight_bytes'],
                **{k:float(np.mean([r[k] for r in group])) for k in ('accuracy','write_us_p50','read_us_p50','train_seconds')},
                accuracy_min=min(r['accuracy'] for r in group),accuracy_max=max(r['accuracy'] for r in group)))
    story_ids=sorted({p['story'] for p in predictions})
    story_scores={}
    for kind in ('input_flow','gru','ltc'):
        for width in (8,32):
            group=[p for p in predictions if p['method'].startswith(f'{kind}-{width}-')]
            story_scores[f'{kind}-{width}']=np.array([np.mean([p['prediction']==p['label'] for p in group if p['story']==s]) for s in story_ids])
    rng=np.random.default_rng(123); indices=rng.integers(0,len(story_ids),(5000,len(story_ids))); intervals=[]
    for width in (8,32):
        for control in ('gru','input_flow'):
            diff=story_scores[f'ltc-{width}']-story_scores[f'{control}-{width}']
            intervals.append(dict(width=width,contrast=f'ltc - {control}',accuracy_difference=float(diff.mean()),
                                  ci95=np.quantile(diff[indices].mean(1),[.025,.975]).tolist()))
    # Train-only learnability diagnostic. Does NOT select/tune primary checkpoints.
    torch.set_num_threads(1)
    small=[r for r in rows if r['split']=='train'][:32]
    x,v,q,y=tensorize(small,4,6); overfit=[]
    for kind in ('input_flow','gru','ltc'):
        torch.manual_seed(101); model=BoundedCodec(kind,32); opt=torch.optim.Adam(model.parameters(),lr=.01)
        for step in range(500):
            opt.zero_grad(); logits=model(x,v,q); loss=torch.nn.functional.cross_entropy(logits,y)
            loss.backward(); opt.step()
        with torch.no_grad():
            acc=float((model(x,v,q).argmax(-1)==y).float().mean())
        overfit.append(dict(kind=kind,examples=32,steps=500,train_accuracy=acc,note='Train-only debugging, not benchmark score'))
    analysis=dict(aggregate=aggregate,paired_story_bootstrap=intervals,train_only_overfit=overfit,
        inference='No efficiency or quality advantage demonstrated. Tiny sufficient statistic makes qa1 a diagnostic, not adequate flagship capacity benchmark.',
        uncertainty='200 test stories, 1000 questions; three seeds averaged within story; unadjusted exploratory intervals, previously inspected test.',
        memory='Dynamic payload excludes shared weights/dictionary and runtime. Exact table 4 payload bytes plus shared dictionary. RSS is process context, not isolated peak memory.',
        timing='Table uses amortized Python loops; neural timings eager batch1. Different execution paths, not algorithmic complexity bounds. No end-to-end generation speed claim.')
    (OUT/'analysis.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    names={'input_flow':'Input-only flow','gru':'GRU','ltc':'LTC (4 substeps)'}
    for kind,color in [('input_flow','#64748b'),('gru','#059669'),('ltc','#dc2626')]:
        group=[g for g in aggregate if g['kind']==kind]
        axes[0].plot([g['state_bytes'] for g in group],[g['accuracy']*100 for g in group],'-o',label=names[kind],color=color)
        axes[1].plot([g['state_bytes'] for g in group],[g['write_us_p50'] for g in group],'-o',label=names[kind],color=color)
    axes[0].scatter([4],[100],color='#111827',marker='*',s=140,label='Exact table')
    axes[0].set(ylabel='Answer accuracy (%)',xlabel='Dynamic state payload (bytes)',ylim=(0,105),title='Public bAbI qa1: state-only read')
    axes[1].set(ylabel='Median write latency (microseconds)',xlabel='Dynamic state payload (bytes)',title='CPU, batch 1; excludes parsing / LLM')
    for ax in axes: ax.set_xscale('log',base=2); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)
    table=''.join(f'<tr><td>{names[g["kind"]]}</td><td>{g["state_bytes"]}</td><td>{g["weight_bytes"]}</td><td>{g["accuracy"]:.1%} ({g["accuracy_min"]:.1%}–{g["accuracy_max"]:.1%})</td><td>{g["write_us_p50"]:.1f}</td><td>{g["read_us_p50"]:.1f}</td></tr>' for g in aggregate)
    diag='、'.join(f'{r["kind"]}: {r["train_accuracy"]:.1%}' for r in overfit)
    html=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>有限记忆第一轮实验</title><style>body{{font:16px/1.7 system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#172033}}table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #ddd;padding:9px;text-align:left}}img{{width:100%}}.lead{{background:#edf3f7;padding:20px}}code{{background:#eee}}</style>
    <h1>有限容量在线记忆：第一轮诊断</h1><p>2026-10-12 · 公开 bAbI qa1 · CPU · 0 API 请求</p>
    <p class="lead">本轮没有发现液态优势。精确的实体—位置表只需 4 字节动态载荷即可达到 100%；当前神经压缩方案更大、更慢且准确率较低。此结果否定当前方案在这个任务上的效率主张，不证明所有液态设计无效，也不代表已完成自然语言有限记忆研究。</p>
    <h2>研究对象与读取边界</h2><p>研究给定信息的在线保存，不预测世界如何变化。实体与位置由公开训练文本确定，所有方法共享同一解析结果。每次写入不接收问题；神经方法只保留一个 float32 状态，最终序列化后从恢复的状态回答。原文、训练前缀与答案只属于实验驱动及评估，不传入读取器。所有 18 个模型在 1000 个测试问题上均通过独立逐事件写入—序列化—读取与批处理预测一致性检查。</p>
    <img src="comparison.png"><h2>三种子平均结果</h2><table><tr><th>方法</th><th>状态字节</th><th>权重字节</th><th>准确率（种子范围）</th><th>写入 μs</th><th>读取 μs</th></tr>
    <tr><td>精确表</td><td>4</td><td>0</td><td>100%</td><td>{results[0]['write_us_p50']:.3f}</td><td>{results[0]['read_us_p50']:.3f}</td></tr>{table}</table>
    <p>状态是动态载荷，不是进程内存。另有所有方法共享的词表序列化大小 {protocol['shared_dictionary_utf8_bytes']} 字节；Python/框架对象与临时工作区未计入状态数。权重另计；检查点文件大小和进程 RSS 保留在 summary.json，RSS 不是隔离测量的峰值内存。</p>
    <h2>协议与成本边界</h2><p>815 训练、185 验证、1000 测试问题，按完整故事划分；测试含 200 故事。三机制 × 两容量 × 三种子，最多 200 轮，验证 NLL 连续 40 轮不改善则停止，最佳验证检查点用于测试。初试 v1 的 20 轮结果完整保留，因训练/验证损失仍下降统一提高训练上限；测试此前已查看，因此所有结果属于探索性研究。19,000 条预测不是独立样本数。</p>
    <p>计时是单线程 CPU、batch=1、预编码输入的算子微基准。神经方法使用 eager PyTorch；精确表太快，改用 10,000 次操作的摊销时间避免计时分辨率产生零值。两者实现路径不同，不据此宣称硬件无关的速度倍数。未测完整解析、语言模型生成或部署端到端延迟，未证明峰值内存优势。</p>
    <h2>可靠性诊断与解释</h2><p>训练集前 32 个样本的独立 500 步记忆检查：{diag}。这是排查学习通路的诊断，不能替代泛化结果。完整训练轨迹、全部检查点、原始计时样本与配对故事 bootstrap 区间都保留。LTC 是状态依赖 conductance 的四子步半隐式更新；输入条件化流可精确更新，二者的计算路径不同。</p>
    <p>本任务查询仅需四个最新位置，因此真正必要的信息本就极少。当前结果不足以研究大容量档案中的信息取舍。下一阶段应先换用需要保存大量可独立查询事实的公开任务并核算信息量，再评估压缩；不能继续把这个简单任务的调参当作论文主实验。历史追溯、开放词汇、证据出处保存与长期序列容量尚未验证。</p>
    <h2>可复现文件</h2><p><a href="protocol.json">冻结协议</a> · <a href="analysis.json">分析与区间</a> · <a href="summary.json">逐模型指标</a> · <a href="timing.json">原始耗时</a> · <a href="predictions.jsonl">全部预测</a> · <a href="failures.json">失败案例</a></p></html>'''
    (OUT/'report.html').write_text(html,encoding='utf-8')
    failures=[p for p in predictions if p['method']=='ltc-32-11' and p['prediction']!=p['label']][:20]
    (OUT/'failures.json').write_text(json.dumps(failures,indent=2),encoding='utf-8')
    (OUT/'analysis_manifest.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),tests='73 passed in 7.65s',primary_protocol='protocol.json'),indent=2))
    print(json.dumps(aggregate,indent=2)); print('Train-only diagnostic:',diag)


if __name__=='__main__': main()
