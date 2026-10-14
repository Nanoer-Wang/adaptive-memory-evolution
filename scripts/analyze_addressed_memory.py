"""Saved analysis for addressing repair and duplicate guard resource checks."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,hashlib
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.memory.bounded_codec import BoundedCodec
from src.memory.addressed_codec import AddressedCodec
from scripts.addressed_transfer_study import parse

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/addressed-memory-v1'
TRANSFER=ROOT/'runs/addressed-transfer-v2'


def load(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    torch.set_num_threads(1)
    summary=load(OUT/'summary.json');old=load(ROOT/'runs/bounded-codec-v2/summary.json')
    tr=load(TRANSFER/'summary.json');protocol=load(OUT/'protocol.json');timing=load(TRANSFER/'timing.json')
    aggregate=[]
    for kind in ('input_flow','gru','ltc'):
        for width in (2,8):
            group=[r for r in summary if r['kind']==kind and r['slot_width']==width]
            baseline=[r for r in old if r.get('kind')==kind and r['state_bytes']==width*16]
            aggregate.append(dict(kind=kind,state_bytes=width*16,weight_bytes=group[0]['weight_bytes'],
                accuracy=float(np.mean([r['test_accuracy'] for r in group])),min_accuracy=min(r['test_accuracy'] for r in group),
                max_accuracy=max(r['test_accuracy'] for r in group),old_accuracy=float(np.mean([r['accuracy'] for r in baseline]))))
    grouped=[]
    for dataset in ('qa1','qa6'):
        for kind in ('input_flow','gru','ltc'):
            for repeats in (1,4):
                always=[r for r in tr if r['dataset']==dataset and r['method'].startswith(kind+'-') and r['repeats']==repeats and r['policy']=='always']
                dedup=[r for r in tr if r['dataset']==dataset and r['method'].startswith(kind+'-') and r['repeats']==repeats and r['policy']=='dedup']
                a=np.mean([r['write_us_per_observation'] for r in always]);d=np.mean([r['write_us_per_observation'] for r in dedup])
                grouped.append(dict(dataset=dataset,kind=kind,repeats=repeats,always_us=float(a),dedup_us=float(d),saving=float(1-d/a),
                    always_accuracy=float(np.mean([r['accuracy'] for r in always])),dedup_accuracy=float(np.mean([r['accuracy'] for r in dedup])),
                    skipped=float(np.mean([r['skip_fraction'] for r in dedup]))))
    # Measure unintended read changes under writes to OTHER entity slots.
    stories=parse(ROOT/'data/babi/qa1_single-supporting-fact_test.txt',protocol['entities'],protocol['values'],False)
    interference=[]
    with torch.no_grad():
        for kind in ('input_flow','gru','ltc'):
            global_model=BoundedCodec(kind,32);global_model.load_state_dict(torch.load(ROOT/f'runs/bounded-codec-v2/{kind}-32-11.pt',weights_only=True));global_model.eval()
            addressed=AddressedCodec(kind,8);addressed.load_state_dict(torch.load(OUT/f'{kind}-8-11.pt',weights_only=True));addressed.eval()
            trials=0;global_changes=0;addressed_changes=0;global_correct_before=0;global_correct_to_wrong=0
            for story in stories:
                h=torch.zeros(1,32);slots=addressed.initial();known={}
                for event in story:
                    if event['type']!='event':continue
                    owner=event['owner'];value=event['value'];x=torch.zeros(1,10);x[0,owner]=1;x[0,4+value]=1
                    hnew=global_model.write(h,x);newslots=addressed.write(slots,torch.tensor([owner]),torch.eye(6)[value:value+1])
                    for other,truth in known.items():
                        if other==owner:continue
                        q=torch.eye(4)[other:other+1]
                        before=int(global_model.read(h,q).argmax());after=int(global_model.read(hnew,q).argmax())
                        abefore=int(addressed.read(slots,torch.tensor([other])).argmax());aafter=int(addressed.read(newslots,torch.tensor([other])).argmax())
                        trials+=1;global_changes+=before!=after;addressed_changes+=abefore!=aafter
                        global_correct_before+=before==truth;global_correct_to_wrong+=(before==truth and after!=truth)
                    h=hnew;slots=newslots;known[owner]=value
            interference.append(dict(kind=kind,seed=11,unrelated_entity_probes=trials,global_prediction_changes=global_changes,
                addressed_prediction_changes=addressed_changes,global_correct_before=global_correct_before,global_correct_to_wrong=global_correct_to_wrong,
                caveat='Frozen test diagnostic, not training or a universal causality claim; new design changes addressing and reader, and parameter count.'))
    # Bootstrap cost ratio with paired stories, averaging three seeds first.
    rng=np.random.default_rng(20261012);bootstrap=[]
    for dataset in ('qa1','qa6'):
        for kind in ('input_flow','gru','ltc'):
            for repeats in (1,4):
                pair={}
                for policy in ('always','dedup'):
                    runs=[r for r in timing if r['dataset']==dataset and r['method'].startswith(kind+'-') and r['repeats']==repeats and r['policy']==policy]
                    pair[policy]=np.mean([[c['write_ns'] for c in r['stories']] for r in runs],axis=0)
                idx=rng.integers(0,len(pair['always']),(3000,len(pair['always'])))
                samples=1-pair['dedup'][idx].sum(1)/pair['always'][idx].sum(1)
                bootstrap.append(dict(dataset=dataset,kind=kind,repeats=repeats,saving_ci95=np.quantile(samples,[.025,.975]).tolist(),
                    caveat='Exploratory story resampling of single process runs; not repeated-machine runtime uncertainty; no multiple-testing correction.'))
    analysis=dict(capacity=aggregate,cost=grouped,interference=interference,cost_intervals=bootstrap,
        conclusion='Addressed storage solves these public location tasks reliably at 128 bytes. Duplicate guard saves cost only on inserted repeats; neither result establishes a unique liquid contribution.',
        next_gate='Require natural redundant semantic updates before studying learned/liquid update decisions. Keep exact guard and simple learned gate as strong controls.')
    (OUT/'analysis.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')
    names={'input_flow':'Input-only flow','gru':'GRU','ltc':'LTC'}
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.3))
    width128=[g for g in aggregate if g['state_bytes']==128];xs=np.arange(3)
    axes[0].bar(xs-.17,[g['old_accuracy']*100 for g in width128],width=.34,label='Previous shared state',color='#94a3b8')
    axes[0].bar(xs+.17,[g['accuracy']*100 for g in width128],width=.34,label='Addressed slots',color='#059669')
    axes[0].set(xticks=xs,xticklabels=[names[g['kind']] for g in width128],ylabel='qa1 accuracy (%)',ylim=(0,110),title='Same 128-byte dynamic payload')
    for offset,repeats,color,label in [(-.17,1,'#94a3b8','Native qa6'),(.17,4,'#2563eb','Inserted 4x repeats')]:
        group=[g for g in grouped if g['dataset']=='qa6' and g['repeats']==repeats]
        axes[1].bar(xs+offset,[g['saving']*100 for g in group],width=.34,color=color,label=label)
    axes[1].set(xticks=xs,xticklabels=[names[g['kind']] for g in width128],ylabel='Measured write-path saving (%)',ylim=(-10,85),title='Exact guard: +4 bytes; accuracy 100%')
    for ax in axes:ax.legend(fontsize=9);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)
    table=''.join(f'<tr><td>{names[g["kind"]]}</td><td>{g["state_bytes"]}</td><td>{g["weight_bytes"]}</td><td>{g["old_accuracy"]:.1%}</td><td>{g["accuracy"]:.1%} ({g["min_accuracy"]:.1%}–{g["max_accuracy"]:.1%})</td></tr>' for g in aggregate)
    costs=''.join(f'<tr><td>{g["dataset"]}</td><td>{names[g["kind"]]}</td><td>{"原始" if g["repeats"]==1 else "人为4倍重复"}</td><td>{g["always_us"]:.2f}</td><td>{g["dedup_us"]:.2f}</td><td>{g["saving"]:.1%}</td></tr>' for g in grouped)
    diagnostics=''.join(f'<li>{r["kind"]}：{r["unrelated_entity_probes"]} 次无关实体检查，旧结构改变 {r["global_prediction_changes"]} 次预测，其中原先正确后变错 {r["global_correct_to_wrong"]} 次；显式槽位改变 {r["addressed_prediction_changes"]} 次。</li>' for r in interference)
    html=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>内容寻址修复与写入成本验证</title><style>body{{font:16px/1.7 system-ui;max-width:1120px;margin:40px auto;padding:0 24px;color:#172033}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}img{{width:100%}}.lead{{padding:20px;background:#eaf5ef}}</style>
    <h1>内容寻址修复与写入成本验证</h1><p>2026-10-12 · 18 个新模型 · 公开 bAbI qa1 / qa6 · CPU · 0 API 请求</p>
    <p class="lead">可靠读写底座已建立：128 字节动态状态下，三种机制的全部三个种子在 qa1 与冻结迁移 qa6 均为 100%。这支持显式实体寻址的设计修复。人为重复条件下的去重加速不能当作原始数据收益，更不是液态独有贡献。</p>
    <img src="comparison.png"><h2>同动态容量的结构对照</h2><table><tr><th>机制</th><th>状态字节</th><th>权重字节</th><th>旧结构准确率</th><th>新结构准确率（种子范围）</th></tr>{table}</table>
    <p>每个实体使用独立固定槽位，写入不接收查询。读取只接收固定大小的序列化状态与问题键，不回查原文。四实体 × 每槽 2/8 个 float32 对应 32/128 字节。32 字节设置存在明显种子差异，全部保留。共享参数与词表另计；这是相同动态载荷的结构比较，不是相同参数量比较，也没有声称测得隔离进程峰值内存。</p>
    <h2>干扰诊断</h2><ul>{diagnostics}</ul><p>诊断使用 seed11 冻结模型和 qa1 测试流。只能说明此前结构存在跨实体干扰，不能把全部准确率提升归因于单个因素；新设计同时改变了写入路由与读取方式。</p>
    <h2>冻结迁移与计时</h2><p>所有 128 字节模型在下载 qa6 前冻结。qa6 原始测试 1000 问题、200 故事，包含 1375 条位置事件和 675 条物品持有事件。v1 因不支持物品句型在评分前停止；v2 根据 qa6 训练文本扩展显式句法，将物品操作作为位置任务的无操作，未知句型仍报错。没有重训或按 qa6 成绩选模，但这不是完全不接触目标句法的盲迁移。qa6 与 qa1 共用生成器、词表；未发现 qa6 完整事件序列与 qa1 训练故事相同，不等于开放域泛化。</p>
    <p>所有 9 个模型 × 两任务 × 两策略 × 两条件，共 72,000 条预测均为正确。两任务的精确 4 字节位置表仍为 100%，它是必须保留的强基线。去重缓存本身也保存全部四个值，因此在这种有限本体任务中神经编码不是必要条件。</p>
    <table><tr><th>任务</th><th>机制</th><th>条件</th><th>每事件完整写入 μs</th><th>含去重判断 μs</th><th>节省</th></tr>{costs}</table>
    <p>去重读取最近已观察到的实体值，未接收答案标签；缓存额外 4 字节，带守卫总状态 132 字节。四倍重复是明确标注的派生压力测试，不是官方数据。原始两任务没有可跳过的相同位置报告，零写入节省，小幅耗时波动不能当作收益。成本包括判断与神经写入，排除共同文本解析、张量编码、序列化和语言模型生成。审计计数不影响模型决策。故事级原始耗时和探索性配对区间已保存，不等于多机器重复基准。</p>
    <h2>现在能说与不能说的结论</h2><p>能说：共同读写故障得到实质修复，冻结模型可完成同生成器的新问答形式；重复事件的跳过能落实为可测计算节省。不能说：液态优于强门控或规则、自然数据已节省成本、已经得到可投稿的创新证据。下一研究门槛是公开自然信息流中的真实重复/修订和更大的可查询事实容量，随后比较强规则与简单学习门控，不能仅靠人工重复制造优势。</p>
    <h2>分析与复现</h2><p><a href="protocol.json">训练协议</a> · <a href="analysis.json">统计与干扰诊断</a> · <a href="summary.json">18 模型指标</a> · <a href="../addressed-transfer-v2/protocol.json">迁移协议</a> · <a href="../addressed-transfer-v2/predictions.jsonl">迁移全部预测</a> · <a href="../addressed-transfer-v2/timing.json">原始耗时</a></p></html>'''
    (OUT/'report.html').write_text(html,encoding='utf-8')
    (OUT/'analysis_manifest.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),primary='protocol.json',transfer='../addressed-transfer-v2/protocol.json'),indent=2))
    print(json.dumps(interference,indent=2));print('Report saved.')


if __name__=='__main__':main()
