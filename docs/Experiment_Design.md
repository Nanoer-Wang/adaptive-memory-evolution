
# Experimental Design

## 当前B1-A与B1：替换控制归因矩阵

用户已确认交错流mini协议通过，进入新登记的正式评测：`scripts/b1_attribution.py`。此轮使用容量8，不扩展到容量16；每个压力—动态条件有24个配对世界和3个冻结策略种子（11/23/42），三档噪声键数为2/8/16。共8方法、5184次测试评估。旧版B0–B7更宽泛的模型训练与容量扫描仍属于后续计划，不冒称已全部执行。

**执行状态：已完成。** 5184次逐事件审计、216次预指定精确重放和212项项目测试通过。[完整结果解读](../runs/ame-validation-v2/B1-attribution-v1/interpretation.html)保留全部九条件结果。替换匹配通过8/9条件，stable/medium失败，不重调测试参数。全部6个非平稳条件通过匹配，四项主要端点相对匹配对照均改善且Holm校正显著；不能推广为对原始固定衰减所有端点均优。原先宽泛计划的`prepare`仍描述旧训练/比例/容量矩阵，本轮执行证据以独立protocol及evaluation_complete为准。

动态定义：stable没有主题切换与事实修订；slow_drift和rapid_shift共享A→B→A提示、查询顺序和时钟。前者每轮只修订一个活跃主题事实，后者在第0/2轮成批修订四个事实，每轨迹共8/16次修订。“慢漂移”指事实变化速率，不是连续变化的语义嵌入。

除原五方法外，增加三个归因对照。`static_state_adaptive`保留相同学习插入状态与评分公式，冻结每个驻留episode的I/C/U/T，关闭连续变化及观察/反馈跳变。`static_uniform_state`进一步将初始与新增条目的I/C/U/T统一固定为1/1/0/0，排查冻结对照的初始化锁死问题。两者不单独识别连续时间积分贡献。

固定衰减的正λ只缩放评分，不改变年龄排序，不能通过调λ改变替换。`replacement_matched`明确增加种子控制的独立写入节流，对通过门控的候选仍按固定衰减排序。每条件在独立4个校准世界、3种子上选择概率，仅最小化淘汰计数和平均观察寿命相对误差，不使用任务成绩。测试前锁定，测试阶段不再调整。只有测试条件的两项均值误差都≤10%才记为近似匹配；未匹配结果仍完整保存。

满容量条件下，全部驻留episode观察寿命之和等于容量×时长，平均观察寿命与淘汰数因此耦合，不能当作两份独立证据。额外保存年龄分布、右截尾寿命、淘汰时机及对象。匹配两项均值不等于匹配完整替换过程。

统计以世界为独立单位：先平均同世界3种子，再做24世界的配对bootstrap 5000次。主要端点为accuracy/stale_error/stable_retention/adaptation_lag，比较adaptive对fixed/static/matched/uniform_static；随机符号检验10000次，主比较统一Holm校正。稳定环境的适应lag不适用，不能记为0。首次恢复和末尾持续性分开。原始记录、完整状态轨迹、校准选择及源码快照均保存。

## 历史mini：受控压力与交错流（已通过，原记录保留）

为修正下文历史mini的问题，新建`src/memory/b1_interleaved.py`及`src/evaluation/b1_interleaved.py`；五种策略适配器不变，没有改参数、增加网络或训练。活跃事实固定8条（4稳定+4当前主题），休眠主题另4条；noise ratio明确为不同噪声键数/8，三档噪声池2/8/16条。每轮核心观察和噪声观察均8次，三档共享核心事件、查询序列和时间轴，阶段内实际覆盖全部配置噪声键。此设置考察工作集压力，不等同于改变噪声流量占比。

日程为交错知识介绍→A→B→A；每次观察后紧跟查询和单独的反馈事件，查询不问刚观察的同一个键。稳定与当前主题事实刷新次数相同；版本仅在实际观察时修订；评估标签和未来查询不提供给策略。记录活跃/噪声池数、实际驻留数、替换/拒收/原位刷新、寿命和驻留年龄分布。

三个新的世界种子60401–60403分别配对三档压力，共9条；60404的medium无切换阴性对照为第10条。策略种子11/23只读取旧冻结头和决定平分，不训练。正式评测容量仅8；容量16在五种策略接口单元检查中覆盖。100次评测及100次完整重放通过，全项目203项测试通过，旧两轮sanity与原始v1结果不变。

适应指标报告首次连续两个4查询窗口至少75%正确的查询数/时间、成功率与截尾（每阶段32次当前主题查询）；无切换对照为null。另记每次新版本首次被观察后，第一次答对用了多少查询机会/时间及是否经历失败后恢复。样例新版本4:2事件234拒收、259/283失败、297重观察后保存、307答对，说明不再只测查询开始时的存活内容。

**指标边界：**首次过阈值不等于一直恢复；事后另存末尾8查询与末尾两窗口诊断，不覆盖登记端点。阶段中途还有事实修订，末尾成绩同时包含再次适应，不能全部归因为恢复后的遗忘。固定衰减在mini中替换频繁且过阈值较快；自适应更保留稳定事实，但不是各项都更好。三档只有3个配对世界，不能用于正式显著性或优势结论。

**归因边界：**状态在无写入、同一驻留episode内确实随事件变化；但当前状态最终只影响保留评分，检索仍取驻留最新版本。这不能替代同替换预算与关闭演化的对照，也不能宣称已完全脱离动态cache replacement。剩余正式基线及训练器未宣称完成。

[最新审查报告](../runs/ame-validation-v2/B1-interleaved-review/index.html) · [完整可读event trace](../runs/ame-validation-v2/B1-interleaved-sanity-v1/event_trace.html) · [metrics JSON](../runs/ame-validation-v2/B1-interleaved-sanity-v1/metrics.json)。轨迹片段检查由助手完成，不是独立人工评审。`python scripts/b1_interleaved_sanity.py`对已完成目录只核验；正式B1仍未启动。

## 历史B1首次sanity：实现完成但协议有缺陷（保留原记录）

已实现`src/memory/b1_environment.py`（独立80键生成器）、`src/memory/b1_policy.py`（五种统一策略接口）、`src/evaluation/b1_sanity.py`（指标与事件审计）、`scripts/b1_sanity_check.py`（只运行mini、无训练入口）。以下完整批次计划保留；当前只完成用户本轮要求的fixed decay、frequency、current adaptive evolution、adaptive rule、immediate context五种策略，不将正式B1剩余的LRU/random/linear/GRU适配和训练器记为已完成。

运行8条条件轨迹：2世界种子×（3种稳定比例的快变环境+1个50%稳定比例的平稳环境），2策略种子，共80回放，容量8。正式训练/验证/测试种子未读取。当前自适应只加载旧123参数头的冻结11/23权重，不重新训练，不调用API。80次事件重建及80次确定性重放通过；全项目190项测试通过。

**必须先解决两个协议问题：**每阶段查询前固定最后6条观察都是干扰，固定衰减在所有96个阶段检查中至少75%槽位被干扰占据；每阶段随后40次查询期间没有新观察，因此逐阶段正确率完全由查询块开始时的内容存活决定，“适应窗口过阈值”不能解释为在线再学习速度。这一mini是协议缺陷的证据，不能用于证明演化优越。

全部原始结果保留在`runs/ame-validation-v2/B1-sanity-v1/`；只读诊断另存`B1-sanity-review/`，没有覆盖初次协议或数据。[审查报告](../runs/ame-validation-v2/B1-sanity-review/index.html)包含框架、策略差异、真实替换率、拒收、寿命与状态图。下一步应先登记交错到达/查询日程、把无提醒保留与有新证据的适应分开，再用新sanity种子复查；当前不继续正式B1。

## 当前执行计划：分批验证 Adaptive Memory Evolution（v2，尚未启动新训练）

本轮先完成拆批、冻结和配置，不一次运行全部实验。旧 `evolving-memory-v1` 已冻结，既有结果不再改写。当前目标是验证记忆演化的能力边界及可控的稳定性—可塑性权衡，不预设获胜。新的实验实现与旧实现分离；本节是执行计划，不是新增成绩。

机器可读配置：[ame_validation_batches.json](../configs/ame_validation_batches.json)。冻结清单：[frozen_reference.json](../runs/ame-validation-v2/frozen_reference.json)。当前存档约34.86 MiB，包含原源码快照、125个归档文件中的实验记录与历史文档，不包含`.env`。原运行目录仍保留，归档保存完整旧副本。

### 批次顺序与产物

| 批次 | 回答的问题与范围 | 运行前提 | 产物与推进条件 |
|---|---|---|---|
| B0：冻结与契约 | 固定当前方程、代码、数据、模型、全部原始结果；登记新协议 | 已完成 | 哈希清单、压缩归档、批次配置；验证旧结果未变 |
| B1：混合记忆与替换归因 | 稳定比例20%/50%/80%；容量8/16/32；平稳/慢变/快变；八种方法及无演化的即时上下文规则 | 独立生成器、八种策略适配、因果/容量小样例检查；验证选择锁定 | 全部条件表、稳定保留—适应图、真实替换量；有益、无益或负结果都可推进 |
| B2：容量响应 | 扩展到4/8/16/32/64槽 | B1完成并锁定模型 | 复用B1中8/16/32槽结果，只运行4/64新单元；不重新调参；五点曲线不包装为普遍scaling law |
| B3：循环回访 | A→B→A；休眠长度1/3/6阶段；8/16/32槽 | B1模型锁定 | 无提醒回访、提醒后恢复、内容生存率分开；不能用重新学习冒充持久保留 |
| B4：反馈/重要性噪声 | 10%/25%/50%反馈翻转；交替误导反馈；单列错误主题提示 | B1–B3完成 | 同世界、同查询配对下降；反馈与重要性信号扰动分开；先测干净训练后的迁移 |
| B5：重要性反转 | A/B需求反转与真实事实修订两个条件 | B1–B3完成 | 适应滞后、旧版本错误、新版本获取速度；保留全部基线，不只挑三种 |
| B6：需求条件化与消融 | 去I/C/U/T；常数状态、离散事件更新、仅时间连续流、完整条件化 | B4/B5完成，机制协议固定 | 同权重干预与等预算重训分开；记录计算成本，不能把上下文收益直接归因连续时间 |
| B7：语言记忆与Agent接口 | LongMemEval、LoCoMo；MemoryBench待确切来源锁定 | 先完成来源/暴露审计，再离线检索，小规模QA，最后正式评测 | 真实回答、错误类型、检索耗时、输入输出token、调用次数；不得用合成成绩替代真实Agent质量 |

B1–B3决定核心论点；B4–B5增强边界证据；B6–B7支撑机制归因和语言迁移。每批完成后都先检查完整性和解释，再启动下一批；是否正向不作为准入条件。B2/B3共享已锁定模型，不因看到B1测试结果而重新训练。B7的来源审计可以提前进行，QA正式评测仍按阶段执行。

### 当前已经能运行什么

```powershell
# 查看批次、核验冻结记录（不训练、不调用API）
.venv\Scripts\python.exe scripts/ame_validation_batches.py status
.venv\Scripts\python.exe scripts/ame_validation_batches.py verify

# 为指定批次生成带配置哈希的独立计划、比较单元和准备状态
.venv\Scripts\python.exe scripts/ame_validation_batches.py prepare --batch B1
```

`prepare`只生成执行清单，不生成测试轨迹、不训练、不评测。当前B0完成，B1–B7的新版运行器与适配器仍待实现，准备状态会明确返回`implementation_required`，不存在伪装为可用训练入口的空脚本。不能把旧的`evolving-memory --phase all`当作本计划的实现。计划变更产生新哈希目录，不覆盖旧版本；每一批最终还需登记实际运行代码哈希和验证锁定文件。

### B1的可实施步骤

1. 新建独立80键环境：40个核心事实、40个干扰事实。核心事实中稳定数为8/20/32，精确对应20%/50%/80%。稳定属性由评估器保存，不作为模型可见标签；稳定与易变事实的主题分布需平衡，避免“非主题=稳定”的旧构造偏差。不能沿用仅24键环境直接评价64槽。
2. 将事实构成比例与查询比例分开：主实验稳定/易变查询各50%，另测查询比例随事实比例变化。固定每阶段20个目标查询；平稳、慢变、快变的易变事实修订概率分别0、0.1、0.5，所有方法观察完全相同的经验与因果提示。过期版本和离题内容分开计数。
3. 小样例仅用60001/60002种子。训练61000起24个世界种子，验证62000起12个，测试63000起24个；不同条件和容量复用种子形成配对，不声称它们独立。训练/验证12阶段、测试24阶段。新日程是新的程序化样本，不是新的独立公开语料来源。
4. 保留七个必选基线并加简单自适应规则。固定40维输入，GRU隐藏维8，自适应学习头123参数。线性/GRU控制插入参数，之后不更新逐项优先级；自适应规则与学习型共享状态方程。所有方法使用相同内容和检索器，统一元数据访问，分别报告内容/索引/状态字节。
5. 采用统一训练日程覆盖混合与循环任务、8/16/32槽；各可学习方法同样训练数据、16epochs、批量8、优化器及更新次数。保持硬容量真实执行；策略梯度训练用随机动作探索，主部署验证采用确定性淘汰并据此选模，单列随机/确定性执行差异，不能宣称两者动作分布相同。先通过验证集上的小规模流程检查，再锁定模型并读取主测试结果。
6. B1主表648个方法—环境—容量—策略种子单元，每单元24测试世界，共15552条配置—轨迹记录；替换匹配与权重扫描另计。任务按单元落盘和断点恢复，不一次塞进单个不可恢复运行。先测吞吐再给总耗时估算，不凭空承诺耗时。
7. 输出所有方法、容量和环境，不只导出赢家。每个单元保存配置、世界种子、策略种子、轨迹哈希、模型哈希、原始分子分母、失败/拒答及资源记录。运行器需先验证这些记录可重放，再标记完成。

### 必须先校正的对照定义

- 用户给定的纯固定衰减为`exp(-lambda*age)`。确定性淘汰时任何正lambda产生同样的年龄排序；要保留零衰减的平分对照，但不能把正lambda网格当成不同排序能力。旧v1含正反馈强化的“固定衰减”是另一策略，历史成绩不与新定义直接拼表。
- 频率策略只统计返回条目的访问次数，不读正确性反馈；LRU只看最近观察/访问；随机策略均匀淘汰。三者不能意外获得主题或真实正确性标签。主表按各策略原义采用确定性淘汰（随机基线除外）；旧共同softmax淘汰作为明确标注的敏感性检查。
- 所有方法可访问同样的可观察上下文；额外的即时上下文规则使用当前相关性、已观察版本一致性和有限使用统计，却没有演化隐状态。若该规则也有同样收益，只能支持上下文响应，不能归因状态历史。
- 以共同、因果的候选准入掩码控制25%/50%/100%写入机会，不从未来测试记录倒推匹配。即使写入机会一样，各方法也可能拒收候选而不是替换驻留内容，所以必须报告真实替换率、候选拒收、内容寿命和淘汰稳定事实比例。只有在这些条件可比后，才能讨论“并非只是激进淘汰”。

### 指标、权衡与统计预先固定

主指标始终是整体与分组准确率、SR、过期错误、拒答率、适应滞后、恢复成功率及无提醒回访。`AS=1-min(lag,T)/T`，T=20目标查询机会；两次连续4查询窗口≥75%正确，最短可测lag=5，因此本定义AS最高0.75，不能误标为归一化完美适应。未恢复者按T截尾且单列失败率；没有实际变化的平稳阶段，AS记为不适用，不能机械算成零。

以世界/会话为统计簇，先平均同世界3个策略种子，再做配对bootstrap；容量副本、噪声副本和多条查询不当作独立样本。预定主要对比采用批次内Holm多重比较校正；探索性区间明确标注未校正。既报告差异也报告区间，不通过“一个显著另一个不显著”推断方法间显著差异。

要证明“可控权衡”，不能只展示一个适应较高、保留较低的点。预先扫描稳定保留损失权重`0/0.25/0.5/1/2`，可学习对照均等预算重训，各点只用验证集选检查点。事后改加权汇总分数不是改变记忆策略。探索性约束为验证SR至少达到固定衰减SR减2个百分点；报告无可行点及测试约束违例，不预设“可接受保留”已经达成。

机制图需在主测试之前用固定种子选择代表条目，画I/C/U/T、主题变化、反馈、存活状态及淘汰/重插入。淘汰后画空白，不画一条延续的状态线暗示已删除内容还在记忆中。图展示机制，不单独构成因果证据。

### 真实数据与论断边界

本地LongMemEval与LoCoMo都已有历史评估；尤其`data/locomo_reserved/source.json`明确标注已暴露，目录名reserved不代表仍是未见数据。后续可做冻结策略迁移和重复验证，但不能再称全新外部盲测。MemoryBench尚未锁定具体仓库/版本/许可，先把这一项标为待接入，不把现有MemOps文件冒充该基准。

语言测试按“来源与划分审计→离线检索/容量→少量成对QA→登记后的规模化评测”推进，共用抽取器、读者、提示及token预算。API阶段沿用全局滚动60秒限速和重试缓存，统计每次尝试及提供商token，不把免费、价格或服务可用性当作已确认事实。本轮规划没有API调用。

“所有已有Agent记忆都是静态记录”不作为未经文献核查的论文前提。即使本实验出现稳定性—可塑性曲线，也仍需与已有动态记忆、遗忘和巩固机制比较才能判断新颖性。正结果支持什么、负结果排除什么，应由每批证据决定。

## Current scope: Adaptive Memory Evolution

The current hypothesis concerns evolving per-item states, not LNN superiority. See [the protocol](../runs/evolving-memory-v1/protocol.json) and [current report](../runs/evolving-memory-v1/index.html) for stable/volatile mixture, paired noisy/misleading-feedback conditions, and cyclic shifts. Primary comparisons are fixed decay, retrieval frequency, adaptive rule, and a 123-parameter adaptive learned head. Stale errors, stable retention, censored adaptation lag, pre-reminder recall, and post-reminder recovery are reported separately. Current results do not by themselves establish research novelty or natural-language Agent effectiveness. The records below are historical and remain unchanged as evidence.

## Preservation continuation (completed 2026-10-11)

`runs/preservation-study-v2` freezes a matched downstream context experiment,
18 stratified old/new QA pairs, twelve retrained controls (static/reset,
liquid, independently trained fixed-time liquid, GRU; three seeds), and true
normalized payload budgets of 32,000 and 128,000 bytes. Same selected session
positions are held fixed for QA; duplicate upstream IDs require reproducing
the parent's ranking rather than taking the first matching ID. Whole-session
and extractive-unit eviction share identical learned insertion priorities.
Unit metadata is charged to the budget; parent embeddings are shared only
while at least one unit survives. Source timestamps are not inferred fact
validity times. No semantic fact key or contradiction resolver is implemented.

The predeclared QA continuation gate passed (10/18 to12/18, four wins/two losses),
but its question-bootstrap interval crosses zero. Evidence coverage improved
substantially; this is a representation result, not a liquid-mechanism result.
Neither budget passed the liquid-versus-static-and-GRU development gate.
LoCoMo checkpoints were frozen on LME and evaluated without external tuning:
1,528 eligible QA in10 previously exposed conversations, 41,256 records.
External comparisons also fail to establish a stable liquid advantage.
Intervals are descriptive and not multiple-comparison corrected.

The finer unit eviction probe reduced average context evidence coverage for
every method and both budgets. Its ordinal metadata overhead and inherited
coarse priorities were not separately ablated; do not attribute all loss to
one cause. The current architecture assigns priorities once at insertion:
new facts cannot directly revise older records' scores. A next research
hypothesis must test explicit version linkage and update-triggered reprioritization
against simple deterministic baselines before crediting liquid dynamics.
Keep temporal reader failures separate: both values can survive while the
generator still chooses the older one. All negative results and paired
regressions remain in the report and raw records.

## Content-memory experiment and resulting hypotheses (2026-10-11)

Online-evaluation repair: raw LongMemEval session IDs contain annotation-bearing
`answer_` prefixes. The first 46 online results exposed these headers and are
superseded diagnostic records only. `content-memory-qa-v2` regenerates every
preselected QA item with opaque hashed headers and unchanged excerpt bodies.
Shared context packing now anonymizes IDs by default, with regression tests.
The repair also increases the judge output allowance uniformly to avoid a
truncated explanation being mistaken for a verdict. It does not change offline
rankings, trained weights, external selection or the fixed question sample.

`runs/content-memory-v1/protocol.json` was frozen before inspecting LoCoMo QA.
Its execution manifest records the exact implementation, the correction from
"overlapping" to first/last token windows, source archive, encoder hashes and
the preselected final-QA IDs. The experiment trains query-independent retention
writers on LongMemEval train and selects checkpoints on its existing development
split. All ten LoCoMo conversations are the external evaluation; they are now
exposed and must not be called an untouched confirmation set in future rounds.

This experiment evaluates **liquid-state-assisted retention**. The persistent
state assigns insertion priorities; the reader can only retrieve retained text
and vectors. The state is not a decoder capable of reconstructing evicted facts.
The cell remains a time-aware gated approximation, not exact CfC/LTC.

The first external offline results do not establish a robust benefit. At eight
slots, liquid evidence retention is 29.37%; uniformly selecting eight of the
available sessions has analytic expected evidence retention 29.65%. For unique
session records, each gold session is retained with probability min(B,N)/N;
linearity gives that same expected fraction for multiple gold sessions. This is
a retention expectation, not a measured random-retrieval or QA score, and it is
not an upper bound for an informative learned policy. It diagnoses the current
policy's weak transfer, rather than proving all memory policies impossible.

A separate post-hoc, noncausal oracle diagnostic selects the best fixed eight
sessions per external conversation using all future evidence labels. Its maximum
weighted annotated-evidence retention is 45.44%. The optimum follows by summing
1/|gold(q)| for each session appearing in each valid question's gold set, then
selecting the eight largest session weights. This is a ceiling for the current
fixed-store annotated-coverage objective, not a deployable baseline, not a QA
accuracy upper bound, and not applicable to a system that changes its store
between questions or reconstructs facts from a separate latent memory. It
motivates examining content consolidation rather than claiming that a different
retention cell alone can recover full-storage performance.

Two additional post-hoc checks are saved independently of the frozen scoring
code. Fixed text compression preserves every annotated utterance in full for
62.17% of valid external questions (partial utterances can still contain an
answer). On development data, 98.55% of the liquid writer's last-logit local
gradient mass lies in the last five sessions. Neither diagnostic alone proves
a causal accuracy mechanism, but both point to limitations that cell scaling
alone will not address.

The next development hypothesis should distinguish durable facts from revisable
facts: preserve stable information across irrelevant intervening events while
allowing explicit updates to replace superseded values. Test finer-grained
content units, redundancy consolidation, and versioned fact retention under
equal byte budgets. Compare ordinary recency/diversity policies, static and
recurrent learned policies, and separately trained time-disabled controls.
Any design change must be selected without using external outcomes; new external
confirmation requires additional untouched data. GPU work still waits for the
user to announce availability.

Minute-resolution auditing also exposes five LongMemEval validation questions
with annotated evidence later than the question clock on the same calendar day.
Keep original benchmark-provided-history scores and report the saved exclusion
sensitivity. Do not present these timestamp annotations as a strictly causal
online deployment test. Natural history-span subgroups were predeclared;
evidence-age and gradient decompositions are explicitly post-hoc diagnostics.

## Publication-oriented research protocol (2026-10-11)

Planning constraint: target a relevant CCF A venue in 2027. The user has GPU resources,
but they are currently occupied; wait for the user to announce availability before
assigning GPU work. Current development experiments run on CPU. Specific venue
fit and official deadlines remain to be checked rather than guessed.

The research objective is to identify reproducible conditions in which continuous-time
memory helps, not to require a favorable result. Acceptance at a particular venue cannot
be guaranteed by this protocol or by a benchmark improvement.

### Current claim boundary

The implemented cell is a time-aware gated approximation, not a reproduction of LTC or
CfC. The v2 model reads five query-dependent lexical scores, resets per question, and
rescans the history. It is a recurrent reranker: it does not yet establish persistent
fact memory, bounded storage, lifelong adaptation, or semantic knowledge updates.
`memory_state_bytes` records only the recurrent tensor. The system still retains
raw histories, feature caches, indices and model weights; that field must never
be presented as total system storage or evidence of constant-memory inference.
The original linear control has five parameters; liquid v2 has 1,206. The existing
improvement over TF-IDF cannot isolate recurrence from extra model capacity.

### Round 01: isolate the mechanism before increasing model size

Run `python scripts/research_round.py` in the project environment. All runs are offline.
The executable freezes its protocol and source archive before the first experiment,
supports interruption recovery, and refuses reuse after source/protocol changes.
Outputs are under `runs/research-round-01/`; every seed retains best/last weights,
losses, performance, config, environment, validation predictions and failures.

- Fixed seeds: 11, 23, 42, 67, 89. Six epochs each, identical optimizer and features.
- Conditions: liquid, exactly equal-parameter state-reset control, fixed time gaps,
  no consolidation, approximately parameter-matched GRU, and the linear control.
- Liquid and state-reset: 1,206 parameters. GRU uses width 13, within 2% of liquid.
- Primary metric: macro unique-session Recall@5. Secondary: all-evidence@5,
  predefined question types, latency and parameter count.
- Best checkpoint within six epochs is selected on development validation. Report
  mean and sample standard deviation over all five seeds, not the best seed.
  These are development results, not independent generalization estimates.
- Before results, the engineering screening gate is fixed: at least +1 percentage
  point mean recall and wins in at least 4/5 seeds against BOTH the reset-state and
  GRU controls. This gate is a decision rule, not a statistical proof or p-value.
- If the gate fails, retain all results and diagnose representation/task mismatch.
  Do not add seeds until significant, silently remove negative variants, switch the
  primary metric post hoc, or treat a promising subgroup as a confirmed hypothesis.

### Round 02: repair timestamps, then repeat the key controls

Round 01 completed all 30 runs and failed its prespecified screening gate. Mean
validation Recall@5 was 93.4249% for liquid, 93.1502% for state-reset,
93.5897% for fixed-time, 93.6264% without consolidation, 93.8095% for matched GRU,
and 93.4066% for linear. These results do not establish a continuous-time benefit.

Diagnosis found a concrete timestamp defect: `date_days` discarded the clock time,
and same-day sessions were sorted only by date. Validation had 70.8829% zero gaps;
minute-aware sorting changes 45/93 histories and reduces zero gaps to 0.5489%.
Absolute day ordinals must remain float64 until subtraction, otherwise float32 also
loses minute-scale differences. This fix affects both elapsed time and ordering;
its effect must not be attributed solely to decay. Feature version 2 and separate
prepared data preserve the old checkpoints' day-only behavior for reproduction.

Round 02 is fixed BEFORE its results: same five seeds, six epochs, optimizer,
training/validation membership and screening rule. Repeat liquid, reset-state,
fixed-time and matched GRU (20 runs), with minute precision for all conditions.
No extra tuning or test access is allowed. Run:

```powershell
.\.venv\Scripts\python.exe -m src.cli prepare --config configs/minute_precision.yaml
.\.venv\Scripts\python.exe scripts/research_round.py --config configs/minute_precision.yaml --output runs/research-round-02-timefix --variants liquid reset_state fixed_time gru_matched
```

The preparation audit is `artifacts/validation/timestamp_precision_audit.json`.
Round 02 completed all 20 runs. Mean validation Recall@5 was 93.36996% for liquid,
93.20513% for state-reset, 93.42491% for fixed-time and 93.58974% for matched GRU.
The screening gate again failed. All seeds are in `runs/research-analysis/all_seeds.csv`;
these findings do not justify a positive claim about liquid dynamics. Stop expanding
this score-only architecture's hyperparameter search: the next experiment requires
a new representation/write-read hypothesis, not more attempts on the same validation set.
The earlier gradient diagnostic (`runs/research-round-01/memory_sensitivity.json`)
finds that the median terminal memory correction has about 99.18% of its input
gradient norm in the last five sessions. This is a local sensitivity diagnostic,
not proof of effective memory capacity or a causal measure of factual retention.

### Independent confirmation and a credible paper claim

LongMemEval's current test split has already been inspected in v1/v2 development;
it is not a fresh blind test. LoCoMo is downloaded from the official repository and
reserved at `data/locomo_reserved/`, with commit, hash, license and provenance.
Its QA JSON has not been parsed or evaluated as part of Round 01. Before confirmation,
freeze the architecture, checkpoint-selection rule, text/image policy, evidence mapping,
retrieval granularity, context budget, reader/judge, exclusions and scoring scripts.
Split or bootstrap by conversation, not random QA, because many questions share history.
Audit source overlap before calling it independent; use a frozen train-only vocabulary
or label-free per-history index. Any adapter/schema inspection must be logged.

Do not substitute LongMemEval-M for an independent question set: its longer histories
are useful for length stress testing, but independence from S must be checked.
Report official-compatible QA scoring separately from retrieval and custom Agnes grading.
Keep the same reader and budget across methods, use repeated generation where necessary,
and check a blinded human sample of disputed judge outcomes. A full paper also needs
strong lexical+dense retrieval, a capable reranker, memory baselines, and accuracy/cost
trade-offs under the same resource budget. Their absence is an outstanding gap.

### Follow-up architecture hypothesis (not an established finding)

If score-only recurrence fails, the next design should retain content/entity/value
representations and separate chronological writes from later queries. Evaluate insert
cost, persistent state size, query cost, conflict/update correctness and forgetting.
Use the same representations and training budget for liquid, recurrent, nonrecurrent,
fixed-decay and state-reset alternatives. Keep controlled perturbations separate from
public benchmark main results. Do not add an easy synthetic task and relabel it as
evidence of superiority on realistic memory workloads.

### Primary research references checked

Novelty risk identified in the literature search: adaptive decay and forgetting
already appear in agent-memory work. A future contribution must specify what its
learned continuous-time dynamics add beyond those policies and beyond ordinary GRU
state. "Adding a forgetting curve" or replacing a cell is not, by itself, an established
research gap. The following are related approaches, not independently reproduced
results in this project:

- [FadeMem (2026 preprint)](https://arxiv.org/abs/2601.18642): adaptive exponential
  decay with semantic, frequency and temporal factors, plus memory consolidation.
- [ScrubJay-MEM (2026 preprint)](https://arxiv.org/abs/2608.04746): type-conditioned
  decay for perishable facts and temporal generalization; the authors explicitly
  report narrower or reversed gains under other backbones/tasks. This motivates
  reporting the conditions and failure cases of a memory mechanism.

- [LongMemEval official repository](https://github.com/xiaowu0162/LongMemEval):
  public data, evidence labels, retrieval baselines and the upstream QA evaluator.
- [LoCoMo official repository](https://github.com/snap-research/locomo):
  ten annotated multi-session conversations; use conversation-level dependence units.
- [Closed-form continuous-time neural networks](https://www.nature.com/articles/s42256-022-00556-7)
  and [author code](https://github.com/raminmh/CfC): established continuous-time
  architectures to distinguish from the project's approximation; adoption alone is
  not a novel contribution.
- [Titans](https://arxiv.org/abs/2501.00663): neural test-time memorization is related
  work, not equivalent to this project's fixed-weight score reranker.
- [Mem0](https://arxiv.org/abs/2504.19413): a memory extraction/consolidation/retrieval
  baseline family; published numbers from different readers/protocols are not directly
  comparable to our custom split.
- [LongMemEval-V2 official repository](https://github.com/xiaowu0162/LongMemEval-V2):
  an additional agent-trajectory evaluation candidate. It has a different harness,
  data scale and scoring setup; it is not integrated or evaluated here.

## Implemented phase-one protocol (2026-10-11)

- Public source: xiaowu0162/longmemeval-cleaned, LongMemEval-S, 500 instances.
- Custom split: shared evidence IDs/content (including evidence used as another
  question's distractor) and matching abstention pairs stay in one group.
  Seed 42 yields 314 training, 93 validation, 93 test instances. This is NOT the
  official full-500 benchmark. Shared non-evidence distractors may remain.
- Train on answerable training instances with evidence labels; select checkpoints
  by validation session Recall@5. Test annotations never enter model inputs.
- Implemented comparisons: no_memory, recent, tfidf, hash_rag, gru, transformer,
  lcm. The first four are untrained. GRU/Transformer/LCM use the same hashing
  inputs and optimization settings; differences in parameter counts are saved.
- No training across users at evaluation: state resets for each question history.
- Generation uses the same Agnes model and bounded context-selection policy.
- Metrics: unique-session Recall@5, all-evidence hit, lexical EM/F1 diagnostics,
  optional custom Agnes judge, per-category results, tokens, queue/latency/RSS.
- LCM ablations currently supported: no consolidation, fixed time gaps.
- The proposed reward/transfer/forgetting tasks below are a roadmap. LongMemEval
  alone does not establish lifelong policy learning or improved task reward.
- Full benchmark comparability later requires independent training data, frozen
  test protocol, matching reader/judge settings and the upstream scoring script.

## Research Questions

RQ1:
Does liquid memory improve lifelong adaptation?

RQ2:
Does dynamic memory reduce forgetting?

RQ3:
Does consolidation improve memory efficiency?


## Baselines

1. No Memory
2. Vector RAG
3. Summary Memory
4. Reflexion Memory
5. GRU Memory
6. Transformer Memory
7. LCM


## Metrics

### Performance
Final reward / accuracy

### Forward Transfer

Improvement on future tasks from previous experience

### Forgetting Rate

Performance degradation on previous tasks

### Adaptation Speed

Steps required for new knowledge acquisition

### Memory Efficiency

Performance per memory size


## Ablations

- Replace liquid module with GRU
- Remove consolidation
- Remove episodic memory
- Fixed time constant
- Remove feedback signal


## Main Table

|Method|Reward|Forward Transfer|Forgetting|Memory Cost|
|-|-|-|-|-|
|No Memory|
|RAG|
|Summary|
|Reflexion|
|GRU|
|Transformer|
|LCM|


## 2026-10-12 混合信念实验解释边界

完整协议与结果见 `runs/liquid-belief-v1/report.html`。公开 bAbI en1k qa1 按完整故事划分训练/验证（163/37 故事），官方测试含 200 故事、1000 问题。三种条件为完整观察、40% 缺失、40% 缺失加 20% 报告噪声。后两者是预先固定的通道压力测试，不是官方 benchmark 成绩。时间是原始行号的逻辑步骤。测试集此前用于账本检查，不能称为全新盲测。

五种模型各训练三种子，共 15 个；只用验证 NLL 选择检查点。共保存 54,000 条测试预测，它们不是 54,000 个独立样本。置信区间先在故事内平均问题和种子，再对 200 故事作 5000 次配对重采样，属于未作多重比较校正的探索性区间。主要看 NLL、Brier、准确率与校准；零概率的 NLL 截断为 1e-8。

LTC 与固定风险率滤波的 Brier 差在两种压力条件下约为 -0.0009，95% 区间均跨零。输入条件化流也与 LTC 接近。相同权重关闭隐状态连续流会改变预测，但效用随种子不同而改变，不能把非零功能影响等同于可靠收益。

独立 qa9/qa10 原生句型检查见 `runs/signed-belief-v2/`。该检查仅验证显式带符号证据更新，未训练 LTC，也未启用时间演化，不属于液态收益。qa10 保留 18 个重复析取选项导致的全量测试错误，并单独报告该来源定义子集；不得用剔除后成绩替代全量 98.2%。


## 有限容量方向的第一轮诊断（2026-10-12）

研究目标改为保存已观察信息供未来读取，不再把世界状态预测与记忆压缩混合。`scripts/bounded_codec_study.py` 使用公开 qa1，对所有机制固定实体/位置表示，写入无查询，读取只接收恢复后的固定 float32 状态和查询；原文只用于外部实验驱动，不可由读取器访问。

v1 初试后统一提高 v2 训练上限，保留初试数据并声明测试已查看。v2 训练 18 模型（输入条件化流、GRU、LTC；8/32 维；3 种子）。全部按验证 NLL 选模。动态状态、共享词表、权重分别计账。端到端生成成本、运行时峰值内存、实体扩展与自然语言压缩不在本轮结论范围。

精确实体表只需 4 字节动态载荷即可无损回答该任务，神经模型没有资源或精度优势。该结果用于否定当前配置的效率主张，而不是否定所有液态模型。下一主实验必须先证明信息需求确实随历史增长，且不存在极小充分统计量直接解决全部查询；需要过去事实、多实体事实或证据追溯，不能只保留四个最新位置。不得为了获得正结果删去精确基线。


## 显式槽位修复与冻结迁移（2026-10-12）

`addressed-memory-v1` 对输入条件化流、GRU、LTC 各运行两容量三种子，沿用 qa1 原划分和验证选模规则。128 字节模型全部达到 100%；与前一轮约 52%–56% 相比，修复有效。由于同时改变写入路由、读出及参数量，不能把增益归于单一变化或液态动态。

`addressed-transfer-v2` 使用训练结束冻结的全部 9 个 128 字节检查点，未根据目标任务成绩选模型。新公开 qa6 有物品持有干扰句，v1 解析失败后根据 qa6 train 扩充明确无操作句法，保留失败记录；目标测试没有用于训练，但这不是完全句法盲迁移。qa6 完整位置事件序列与 qa1 训练故事无完全相同项；词表、生成器和短前缀可能共享。

原始条件与每条位置事件复制四次的压力条件分开报告。所有 72,000 条迁移/策略预测正确。原始位置报告没有重复，去重无稳定成本收益；人工重复下的显著节省不能冒充自然数据上的贡献。精确守卫也使 GRU 与输入流获益，未证实液态独有优势。下一步应先审计公开自然信息流中的冗余与语义更新，再研究学习式写入决策。


## 语义写入判别协议（2026-10-12）

`semantic-gate-v2` 使用 PAWS 等义标签训练 pair-local gate，输入仅含两段冻结 MiniLM384 向量与共享词法特征，模型看不到操作名、目标标签或 old/new value。MLP、GRU、输入条件化流与状态耦合 LTC 各三种子，选择 NLL 检查点后在独立分组校准集确定阈值。所有官方测试分数只在冻结后生成；同时保留完整关系判别 AUC 与误删/跳过指标。

主风险指标为“非重复样本中错误跳过的比例”，必须同时报告“已跳过样本中的错误比例”。LTC 前者约 0.63%，后者约 17.3%，不得混淆。校准阶段的 Wilson 上界筛选阈值只是启发式，不是适应阈值选择或域偏移下的保证。简单逻辑回归在测试超过 1% 目标，不能因跳过更多就判其可部署。

MemOps 修订由 upstream gold trace 提供目标绑定和引文位置，以原文包含关系、前驱值匹配和 confirmed 状态筛出 147 条。未标注的转述/闲聊不能直接作为 duplicate。该检查仅有必须保留的修订负类，0/147 不能独立比较模型，也不等于全系统事实修订成功。PAWS 为人工标注的构造句对，MemOps 为生成式对话，不称作真实人类长期会话。

三类边界：每对重置隐藏状态，未验证真正时间序列液态优势；内核耗时排除共同编码/检索/词法和后台写入，跳过比例不等于端到端节省；置信区间按句子连通分量重采样，文章/文件层面残余相关仍需考虑。
## relation-verifier-v1：先审计语义关系

运行脚本 `scripts/relation_verifier_study.py` 在推理前冻结数据、方法、模型文件哈希和主假设，按选择、独立校准、新 PAWS 审计、MRPC、复用修订集顺序运行。主方案预先定为双向 NLI 加数字/否定守卫，不在审计后换主方案。其余消融是探索性比较。

风险目标改为跳过集合中的错误比例。选择集要求至少 100 次跳过且经验错误不超过 0.5%；独立校准要求单侧 Clopper–Pearson 95% 上界不超过 1%。连通分量隔离不消除文章相关性或域偏移；该区间是 IID 诊断，不是无条件保证。所有方法在选择阶段即无合格阈值，因此全部回退到保留，不能报告低风险有效覆盖。

新 PAWS 样本来自未参与以前实验的官方训练集剩余连通分量，绝不称为官方测试。MRPC 为新外部测试；MemOps 147 条修订沿用旧挑战。主模型固定量化版本，未做全精度一致性验证；预训练数据重叠无法完全排除。独立审计已经查看，后续模型开发不能继续将它称为未见测试。

`scripts/analyze_relation_verifier.py` 仅生成事后诊断，不修改阈值。高置信错误样例保留官方标签，疑似歧义不删除。结果未证明 NLI 普遍优于词法分类，更未证明液态优势。下一步在训练侧补足实体—谓词—对象绑定及新增信息监督，再用新留出数据检验非零覆盖的风险，之后才做连续历史机制比较。

## role-relation-v1：关系监督与角色消融

使用公开 SICK 原始归档的两个方向人工蕴含标签，保留 611 对方向不一致记录，确定关系评价限定于余下样本。正反方向同组，新增/变弱标签互换。将 surface sentence 和 original caption 都加入分组连通关系，按哈希分配 60/15/10/15% 来源组；原始 caption、ID、标签均不进入模型特征。实际测试 2,402 方向例 / 101 来源组，为自定义来源隔离测试而非官方 split。

固定 NLI、NLI 加词法、NLI 加词法与角色三种轻量监督模型共享 C={0.1,1,10} 搜索，选择集 macro F1 定配置。预先指定角色模型为主方案，完整保存其他消融。角色来自固定版本句法模型，不能称为人工角色金标；词序变化也不自动产生矛盾标签。角色净增 macro F1 1.78pp，按来源组配对 3,000 次重采样的 95% 区间 [0.44,3.11]pp，仅表示本次已拟合模型的测试抽样不确定性。

SICK 的安全准入存在独立来源样本不足：选择 84 组无法满足至少 100 个接受样本；校准 60 组即使零错，单侧 95% 上界仍 4.87%。拟合前另存 feasibility_audit.json，保留原协议，不将其描述为充分能力的安全性试验。1% 上界至少要求 299 个独立零错误接受样本，不能用高度相关的正反句对凑数。

为扩展验证，固定关系模型后追加从未使用过的 PAWS 连通分量 2,000 个，一组一对，1,000 校准 / 1,000 测试。旧选择集仅拟合阈值；新审计未调参。该扩展仅检验等义/非等义，不能产生可靠的新增或矛盾标签。主模型 AUC 0.766 高于词法对照 0.740，但无候选阈值通过风险选择。文章相关及预训练污染仍未排除。

主方案测试 macro F1 0.785、新增召回约 0.680、矛盾召回 0.698，未达到预设 0.80 / 每类0.75 / 矛盾0.90 标准。历史机制比较未获本协议准入且未运行。以后使用此次错误进行开发，必须再建立未见测试，不能把本轮继续当作独立验证。

## scoped-relation-v1：作用范围与类别加权的因子实验

四种模型共享旧 SICK 训练/选择、固定 NLI、正则化 C 网格。两个因素为 42/77 维特征及普通/类别均衡损失；主方案预先固定为 scoped_balanced。原测试已在开发中查看，仅作开发对照。

外部在权重固定后评价：SNLI 官方测试中按文本哈希取 2,000 个唯一前提代表，HANS 官方评价按子类各取 50 对，共 1,500 对；数据版本、文件哈希、样本记录均保存。过滤与 SICK 规范化句子完全相同的例子。SNLI 曾用于底层预训练，HANS 是模板语料且子类内相关；不能夸大为预训练未见真实对话验证。

五类到单向三类映射：E=等义+变弱，N=新增+未确定，C=矛盾；HANS 合并 C/N。数据文件标签序号经 Parquet schema 核对。单向数据不能作为五类方向关系或自动删除安全性的认证，本轮不启动历史实验。

SICK 主方案宏 F1 从原角色 0.785 降至 0.711，尽管新增/矛盾召回上升。SNLI 底层宏 F1 0.900、主方案 0.513，595 条底层正确矛盾被覆盖；HANS 主方案 0.647 也未优于原角色 0.674。保留全部子类与失败，未在外部结果上重选主方案。后续若据这些错误改架构，当前外部数据将成为开发材料，必须另留新测试。

## evidence-history-v1/v2：不等待全局语义达标的限定场景实验

根据用户最新推进要求，取消“必须先全面通过旧语义准入才能开展任何历史实验”的研究阻塞，保留实际删除安全边界。所有后续结果仅为预先注明适用范围的离线机制实验，不能宣称通过旧认证。

任务为公开 SICK anchored replay：同一锚句下候选描述有放回抽样 20 次，间隔从 .25/1/4 取值；真值为人工等义或此前精确见过，其余新内容要保留。原来源隔离划分不变，训练800/选择200锚；v1审计277锚，v2审计159锚（原语义certificate池）。语义数据曾评估，因此不冒充新独立语义验证。所有模型共享证据、共同精确账本、同一保护条件；无需假设所有关系均已判准即可作探索。

静态/GRU/液态各三种子、相同宽度及训练预算，参数不同，非严格容量匹配。v1学习全部非重复事件；v2学习保护条件允许的非重复事件，以匹配真正可决策范围。v2用了不同审计池，不能将跨轮变化归因为损失修正。选择集阈值要求≥20次额外语义跳过、经验错误≤5%，测试固定阈值；这是探索操作点，非1%安全性认证。基线包括全写、精确去重、单次证据和三个预设 EMA 时间常数，完整报告，不能隐藏低收益基线。

v2精确去重777写、EMA(.25)754写，额外省23次、2.96%，667条需要保留的不同内容无缺失。单次判据少写32次但漏存2条。神经三种子均无合格额外跳过；液态无独有收益，且控制器更慢。保留重置历史及液态单位dt消融；零额外覆盖时不能解读为历史贡献已被证实或证伪。

指标同时包括语义误判、额外写入节省、不同内容缺失和时刻缺失。重复账本最大20个32字节hash载荷与公共锚另计，不能拿重复缓存免费信息作为神经记忆收益。控制器计时排除共同编码、账本和保护条件，写入次数不等于端到端成本。下一轮应换任务让多事件积累/真实时间有可验证作用，避免在本质pair-local真值上反复调机制。

## temporal-forecast-v1：真实时间与多观测累积

选择UCI公开单站小时数据，解决先前回放标签本质pair-local的问题。24小时窗口、6小时预测、6小时步幅；训练2010–12、验证2013、测试2014，各窗口及目标不跨split。训练统计标准化，缺失整行不观测，不使用未来填补/气象。原始小时和数值真实，额外50%观测稀疏化为确定性随机压力条件。

所有12个神经模型相同预算、宽度16、3种子；原始最后值、选tau的EMA、完整24小时masked向量岭回归、末次MLP、GRU、输入流与耦合液态全部保留。全部参数/配置冻结后才评价测试指标。参数数目不等，后续需要预算匹配；本轮只能报告数值取舍。

误差按3种子分别计算后平均；53周块配对bootstrap用种子平均绝对误差，而非预测集成误差。区间仅为该站/年、这些权重下的探索性不确定性，多比较及残余时序相关存在。单次窗口CPU计时含完整前向，不含训练与数据读取。

额外缺失条件液态MAE38.67，GRU38.47、输入流38.35；液态重置40.52支持模型利用历史，但重置不重新训练，也须结合末次MLP39.21看。液态单位dt38.64没有稳定下降，不能声称真实时间作用。液态参数1053/GRU1869，但约3倍慢；不把少参数等同少计算。输入流是必须保留的更小强基线。

该轮只证实可以推进紧凑历史状态方向，不能代替语言记忆实验证据或构成CCF-A创新证明。下一轮应做多站点和容量匹配，再决定语义证据接口如何迁移，避免因指标局部有利就跳过强基线。

## language-readiness-v1：结束外围诊断，进入语言与运行验收

用户要求集中做完进入语言/agent之前的准备，不再逐轮外围停滞。本轮完成55,809条离线评价：所有LME自定义测试93题、LoCoMo有效证据与拒答共2,067题，完整检索+两个预算下FIFO及4类语言训练writer各3种子。所有冻结权重来自preservation-study-v2，而非数值预测模型。训练ID与当前LME测试交集为空，但数据以前已评估，不称新独立科学确认。

2400 UTF8字节完整句子抽取、同一混合检索器、top5/16000上下文字符；标注仅在评分使用。各不同日期原文保留，未经验证的语义冲突不触发真值替换；显式容量淘汰仍有损失。计账含文本/ID/日期/向量/状态/元数据，排除共享模型和Python/TFIDF索引，不能当成RSS。

预先按哈希选更新/助手/时间推理各2题，共6题×完整/液态23种子128k。Agnes真实回答与独立评分调用分离，完整6/6、液态5/6。失败题evidence_retention=0，因存储淘汰导致无法回答。23次HTTP尝试中3次500自动恢复、20成功，4结果复用缓存，总36,091Token；重跑无新HTTP。小样本及同提供方自定义判分不作正式基准结论。

99项测试和实际顺序写入/恢复检查通过，运行链路机械就绪。之后直接开展受控多轮语言agent任务，保留FIFO/静态/GRU对照；无需再先做气象或语义分类外围实验。液态独有收益与完整agent长期成功率仍是待验证研究问题，工程通过不等于论文成功。

