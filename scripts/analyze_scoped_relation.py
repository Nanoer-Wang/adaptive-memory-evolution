"""Report frozen factorial experiment without retuning on external labels."""
import json,html
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/scoped-relation-v1'
def read(n):return json.loads((OUT/n).read_text(encoding='utf-8'))
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
m=read('metrics.json');pred=read('predictions.json');rows=read('external_examples.json');lookup={r['id']:r for r in rows}
by={(r['dataset'],r['method']):r for r in m}
methods=['roles','roles_balanced','scoped','scoped_balanced'];labels=['Prior roles','+ Class balance','+ Scope','Scope + balance']
fig,axes=plt.subplots(1,3,figsize=(14,4.9),layout='constrained')
for ax,d,title in zip(axes,['sick_reused_development','snli','hans'],['SICK (reused development)','SNLI (2,000 fresh project samples)','HANS (1,500 template challenges)']):
    mm=methods if d=='sick_reused_development' else ['frozen_nli']+methods
    ll=labels if len(mm)==4 else ['Frozen NLI']+labels
    v=[by[d,k]['macro_f1'] for k in mm];ax.bar(range(len(mm)),v,color=['#9fb3c4']*(len(mm)-1)+['#c46b47']);ax.set_ylim(0,1);ax.set_xticks(range(len(mm)),ll,rotation=40,ha='right');ax.set_title(title,fontsize=10);ax.set_ylabel('Macro F1')
    for i,x in enumerate(v):ax.text(i,x+.014,f'{x:.3f}',ha='center',fontsize=9)
fig.suptitle('Scope features and class balancing do not resolve generalization',fontsize=14);fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
# Paired errors and dataset/heuristic-specific diagnostics, no new model fitting.
errors=[];groups=[]
for d in ('snli','hans'):
    primary=[p for p in pred if p['dataset']==d and p['method']=='scoped_balanced'];base={p['id']:p for p in pred if p['dataset']==d and p['method']=='frozen_nli'}
    groups.append(dict(dataset=d,n=len(primary),baseline_correct_primary_wrong=sum(b['prediction']==b['label'] and p['prediction']!=p['label'] for p in primary for b in [base[p['id']]]),baseline_wrong_primary_correct=sum(b['prediction']!=b['label'] and p['prediction']==p['label'] for p in primary for b in [base[p['id']]])))
    for p in primary:
        if p['prediction']!=p['label']:errors.append(dict(**p,text1=lookup[p['id']]['text1'],text2=lookup[p['id']]['text2'],baseline=base[p['id']]))
cases=[]
for g in sorted({p['group'] for p in pred if p['dataset']=='hans'}):
    for name in ['frozen_nli']+methods:
        r=[p for p in pred if p['dataset']=='hans' and p['group']==g and p['method']==name]
        cases.append(dict(subcase=g,method=name,n=len(r),accuracy=sum(p['label']==p['prediction'] for p in r)/len(r)))
save('errors.json',errors);save('hans_subcases.json',cases)
analysis=dict(paired_outcomes=groups,contradictions_overridden=len(read('overridden_contradictions.json')),
    conclusion='Scope+balance primary fails to improve overall semantic reliability. No threshold deployment or history experiment. SNLI label order verified against Parquet schema; five-to-three mapping recorded in protocol.',
    diagnosis='SICK-trained relation adapter suppresses many correct frozen-encoder contradictions on SNLI. Domain/annotation mismatch or unconstrained calibration are hypotheses, not proven unique causes.',
    next_hypothesis='Keep calibrated base directional evidence separate from role-based duplicate approval; conflicts/uncertainty retain information. Requires new held-out evaluation, not post-hoc repair claims on current tests.')
save('analysis.json',analysis)
table=''.join(f'<tr><td>{r["dataset"]}</td><td>{r["method"]}</td><td>{r["accuracy"]:.3f}</td><td>{r["macro_f1"]:.3f}</td></tr>' for r in m)
examples=''.join(f'<li><p>旧：{html.escape(r["text1"])}</p><p>新：{html.escape(r["text2"])}</p><p>原模型矛盾概率 {r["frozen"][2]:.3f}，主方案 {r["adapted"][2]:.3f}。</p></li>' for r in read('overridden_contradictions.json')[:8])
report=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>作用范围与跨域可靠性实验</title>
<style>body{{max-width:1150px;margin:40px auto;font:17px/1.7 system-ui;color:#253648}}h1,h2{{color:#193f56}}img{{width:100%}}td,th{{padding:7px 14px;border-bottom:1px solid #ddd}}.note{{background:#fff0dd;padding:20px}}li{{margin:22px 0}}a{{color:#186b92}}</style>
<h1>作用范围特征与类别加权：跨域审计</h1><p>2026-10-12 · 4 个同条件训练方案 + 冻结底层基线 · CPU · 0 API 请求</p>
<p class="note">本轮主方案没有改善整体可靠性。SICK 的局部召回提升伴随明显的总体 F1 下降；SNLI 暴露出适配分类器压低原本正确的矛盾判断。未进入液态/GRU/历史统计阶段。</p>
<h2>实验设计</h2><p>保持上轮训练与选模样本不变，做 2×2 消融：原角色特征 / 新增作用范围特征，普通损失 / 类别均衡损失。新增 35 维特征，绑定名词修饰、动作副词、否定词依存中心、量词、介词、角色修饰及从句关系；所有特征都是自动句法分析，不能当作逻辑证明。四方案采用同一 C 搜索，预先指定“作用范围+类别均衡”为主方案。</p>
<p>旧 SICK 测试集已在开发中查看，本轮仅作重复开发对照。新外部数据为 SNLI 官方测试中 2,000 个不同前提代表，以及 HANS 官方评价集 30 子类各 50 对，共 1,500 对。按文本/ID 哈希选择，未使用测试结果选模型；去除与 SICK 完全相同的规范化句子。底层 NLI 曾在 SNLI 训练，因此 SNLI 是项目新评估，绝非预训练未见领域；HANS 是模板生成挑战，并非真实会话。</p>
<p>外部任务仅评估单向蕴含：五类输出映射为 E=等义+变弱，N=新增+未确定，C=矛盾；HANS 再合并 C/N。单向标签不能认证五类关系、等义删除或长期记忆。本轮不设一个数据量不足却必然失败的可靠性认证。</p>
<img src="comparison.png"><table><tr><th>数据</th><th>方案</th><th>准确率</th><th>宏 F1</th></tr>{table}</table>
<h2>怎样理解结果</h2><p>旧 SICK 上，主方案新增召回从 68.0% 到 79.2%、矛盾召回从 69.8% 到 81.2%，但宏 F1 从 0.785 降至 0.711；不是全面改善。单独加入作用范围特征的宏 F1 为 0.783，也未超过原角色方案。</p>
<p>SNLI 底层模型宏 F1 0.900，主方案仅 0.513；矛盾召回从 90.0% 降至 2.9%。595 个原模型正确的矛盾样本被主方案改错。核对了数据文件标签顺序及五类到三类映射，未发现标签序号交换。结果提示小数据适配的域偏移或重校准可能压制底层证据，但尚不能断定唯一原因。</p>
<p>HANS 主方案宏 F1 0.647，低于原角色模型 0.674，也未超过底层 0.656。各子类完整保存，未筛选有利子类作为主成绩。</p>
<h2>被覆盖的正确矛盾判断</h2><ol>{examples}</ol>
<h2>下一步应改什么</h2><p>应将底层方向证据与角色层的等义许可分开：角色层提出补充判断，矛盾或不确定证据触发保留/复核，不能被小数据分类器自由覆盖。这个约束方案尚未实验，不能提前宣称有效；需要另留新数据验证，同时保留未适配底层模型的强基线。当前先暂停单纯堆叠特征和类别加权，避免把召回率与精度的交换写成全面进步。</p>
<p>92 项测试通过。数据、模型系数、协议、原始概率、错误、子类指标及性能记录均保存。没有启动 Agnes 推理，也没有后台继续训练。</p>
<p>公开来源：<a href="https://nlp.stanford.edu/projects/snli/">SNLI</a> · <a href="https://github.com/tommccoy1/hans">HANS</a>。版本与哈希见 protocol.json。</p>
<p><a href="protocol.json">冻结协议</a> · <a href="metrics.json">完整指标</a> · <a href="analysis.json">成对错误诊断</a> · <a href="predictions.json">逐条概率</a> · <a href="errors.json">全部外部错误</a> · <a href="hans_subcases.json">HANS 子类</a> · <a href="overridden_contradictions.json">595 条被覆盖矛盾</a> · <a href="timing.json">性能记录</a></p></html>'''
(OUT/'report.html').write_text(report,encoding='utf-8');print(json.dumps(analysis))
