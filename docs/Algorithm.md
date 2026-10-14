
# Algorithms

## Current model: Adaptive Memory Evolution

The research hypothesis is that each memory item is an evolving state. A neural controller is optional; LNN superiority is not a project claim. The sections after this definition preserve historical implementations and proposals, not current architecture requirements.

Let a versioned memory be

\[
m_i(t)=(\mathrm{content}_i,z_i(t)),\qquad z_i=(I_i,C_i,U_i,T_i).
\]

Content is immutable within a version. An observed revision creates a new version; the model never rewrites factual content from its hidden state. Importance, observed consistency, usage, and elapsed time form the dynamical state. On each interval between observations:

\[
\frac{dz_i}{dt}=f(z_i,c_{\mathrm{held}},\theta_{\mathrm{held}}),
\]

\[
\dot I_i=k(0.5+wq_i-I_i),\quad
\dot C_i=4k(v_i-C_i),\quad
\dot U_i=-kU_i/20,\quad \dot T_i=1.
\]

Here q is relevance to the already observed work context, v is consistency with the latest observed version, and k,w are rule-based or learned coefficients. The held inputs are fixed during an interval, giving an exact exponential transition. Current queries or new feedback must not be applied retroactively to the elapsed interval.

Feedback is a separate causal jump on the returned item: C becomes 0.8C+0.2y, U increases by observed feedback y, and T resets. In the robustness benchmark y can be misleading. Gold correctness is only an offline optimization/evaluation target. C is an operational consistency state, not a calibrated probability of factual truth.

The retention score is log(I)+log(max(C,0.05))+log(1+U)-0.05log(1+T), with a shared observed-version penalty. Admission and eviction operate under a common hard slot budget. Deleted content and its state are unavailable until a new observation supplies that content.

The learned coefficient head is a single 40-to-3 linear map (123 parameters), with no recurrent hidden network. It learns from realized future task utility through discrete eviction decisions. The state equations themselves are specified by design rather than learned wholesale.

Common execution order:

1. Advance resident states using previously observed held inputs.
2. Process the current observation or retrieve the newest version actually retained.
3. Apply observed feedback, including any benchmark corruption.
4. Update version metadata and admit an observed candidate.
5. Enforce capacity using the same eviction sampler across methods.
6. Hold the new context and coefficients for the next interval.

The current implementation is [evolving_memory.py](../src/memory/evolving_memory.py); experiments and metric definitions are in [the registered protocol](../runs/evolving-memory-v1/protocol.json). This is a controlled research engine; the existing natural-language Agent adapter has not been migrated or revalidated.

## Historical algorithms and experiments

## Executable phase-one pipeline

1. Download and checksum the official cleaned LongMemEval-S data.
2. Group related evidence histories before making train/validation/test splits.
3. Strip all answer flags from the inference view; keep only text and timestamps.
4. Encode complete history sessions with local hashing features.
5. Evolve a liquid/GRU/causal Transformer state over chronological sessions.
6. Condition a query on the final state and rank stored session vectors.
7. Select query-relevant text windows from retrieved sessions under a shared
   character budget; pass text (not latent vectors) to the frozen Agnes model.
8. Queue every HTTP attempt through the shared 10-per-60-second limiter.
9. Commit each completed prediction to SQLite and export analysis artifacts.

Offline optimization uses multi-positive evidence supervision, next-session
feature prediction, and state stability. Answer-session annotations are available
only to training loss/metric computation, never to inference retrieval or prompts.
The original conceptual algorithms below also describe unimplemented reward-based
extensions; see README and Experiment_Design for the current boundary.

## Algorithm 1: Liquid Cognitive Memory Update

Input:
Experience sequence Z

Initialize:
m0

for each interaction:

1. Encode:

e_t = Encoder(x_t,a_t,r_t)

2. Estimate importance:

w_t = Importance(e_t,r_t)

3. Update liquid state:

m_(t+1)=LiquidUpdate(m_t,e_t,w_t)

4. Store episodic memory

Return final state


## Algorithm 2: Lifelong Agent Inference

Input:
new task q

1. Read current cognitive state m

2. Generate retrieval query

3. Retrieve episodic experiences

4. Combine:

context=[m,E*]

5. Execute frozen LLM


## 2026-10-11 核心重设计：显式版本账本与事件驱动保留状态

实现：`src/memory/version_ledger.py`、`src/memory/evidence_dynamics.py`。
这是独立研究原型，旧实验与生产问答链路尚未切换，避免混淆历史结果。

每条证据保存实体、关系、范围、值、观察时间、原文位置与断言类型。当前版本由显式状态机维护；分数仅决定存储价值，不能决定事实真伪。这里的“当前”只表示输入中最近有效的明确断言，不等于现实世界真相，也尚未处理多来源可信度、讽刺、复杂否定或有效时间倒序更新。

相关新证据会触发同一键下所有仍驻留版本的状态更新。过去、计划、推测分别保留；撤销只作用于匹配值。新版本被淘汰后，保留计入预算的版本水位，防止旧版本复活。若该键全部证据都被淘汰，水位一并释放；不维护无限隐藏历史。

每个版本维护八维状态 z、平衡点 b、非负速率 λ。事件间使用精确流：

`z(t+Δ) = b + (z(t)-b) exp(-λΔ)`。

事件到达时以门控凸组合更新状态，并重估 b 与 λ。系数固定的事件间区间满足零时间恒等、时间分段一致、状态有界。这是分段线性连续时间模型，不是 CfC/LTC 实现，不据此宣称架构创新。当前只使用16维结构特征，不含通用语义内容表示；跨事实依赖图也未实现。

对照接口包含静态特征评分、GRU、连续时间状态、固定速率与无时间演化。各方法使用相同事实和版本信息；主比较预留相同96字节状态存储。模型有效参数量并未完全匹配，未来应单独报告权重、运行内存与质量/成本曲线。

已完成验证：54项测试；公开 bAbI en 1k qa1，训练与测试各1000题，三个规则策略×四个预算，总计24000条因果前缀问答记录。所有条件无错误事实返回；测试1024字节时固定写入规则86.6%、动态规则96.7%；2048字节或无限预算均100%。不足时留空。这只验证版本机制与重评分基线，不是液态优势实验，也不是自然语言通用能力证据。

LongMemEval窄规则抽取审计：289个可回答训练历史只抽出56条证据，命中标注答案证据的历史为0。验证93个历史抽出26条，只有1个命中；训练已阻止。必须先补足并审计语义证据抽取，再比较液态与强规则/GRU，并测试真实时间跨度、历史问答、冲突更新及存储压力；不能靠在已看过的验证集上调抽取规则证明优势。

设计先验参考：[A-MEM](https://arxiv.org/abs/2502.12110)、[Mem0](https://arxiv.org/abs/2504.19413)。更新已有记忆不是本项目独占的新颖性。科学目标是检验事件条件化时间演化是否在公平预算下有额外价值，允许零收益或负收益结论。


## 2026-10-11 语义内容表示与保留对照

实现 `src/memory/semantic_evidence.py`：双角色原文1000字符穷尽分段，每个片段保留精确引用、来源、时间。MiniLM384向量用于共同检索；固定32维投影输入控制器。相似度≥0.55的最多4条驻留记录收到事件更新，相关事件计数不代表事实支持次数。每个片段为独立观察，不把语义相似当成同一事实版本。修正基础账本：值相同但键不同不能增加事实支持计数。

向量、投影、原文、版本元数据和状态计入有效编码存储；此预算不是Python进程的实际堆内存。离线缓存只能准备输入，写入策略不能访问未来片段、问题或标注。读取才使用问题向量。结构化提案接口 `src/data/grounded_claims.py` 另行进行训练侧小样本检查：严格引用验证、保留原始响应、最多一次JSON格式恢复、对源文本中的不确定语气加强检查。所有提案保持unverified，不自动写成当前事实。

冻结机制实验：12个训练历史，6个开发历史，5机制×3种子，两档64/128KiB有效预算；210条离线结果。训练仅两轮、24个64事件窗口，只有17个标注正例，不能声称模型收敛。各方法信息/预算/检索器一致，但有效参数量不同。128KiB标注字符覆盖：静态6.124%、液态6.124%、GRU13.889%、固定速率5.556%、无时间3.346%。无压缩共同检索覆盖50%，因此检索与压缩仍是严重瓶颈。事后固定词法/语义混合检索诊断也未改善六题的无压缩覆盖；它单独保存，不替换冻结主结果。

Agnes固定seed11、64KiB每方法6题。原始裁判误将5个弃答判对；Codex逐条核对24个答案后：规则1/6，其余static/GRU/liquid均0/6。这不是独立人工标注；原始判分和复核依据并存。裁判格式合法不保证判断正确，后续不能直接信任此裁判的单次布尔输出。

全部结果见 `runs/semantic-mechanism-v2/report.html`。新增63项累计测试通过，原文覆盖不等于事实抽取精度。尚未完成结构化事实大规模验证、独立新数据集验证和最终液态优势证明。


## 2026-10-11 数学建模审查：撤回过强解释

原始LTC的一种形式是 `dh/dt = -(1/tau + f(h,u))*h + f(h,u)*A`，有效时间常数为 `1/(1/tau+f(h,u))`。状态与输入共同影响连续向量场；见 https://cdn.aaai.org/ojs/16936/16936-13-20430-1-2-20210518.pdf 。CfC是相关闭式神经架构，不等于任意指数衰减器，见 https://www.nature.com/articles/s42256-022-00556-7 。

项目当前 `dz/dt=lambda(phi)*(b(phi)-z)` 在事件之间固定系数，是有效的分段线性系统，精确积分也正确。但lambda与b不读取隐状态z，状态坐标在事件间不耦合。它不是原始状态耦合LTC的忠实实现，不能直接继承LTC的表达力结论。事件门的目标与系数也不读取z；因此此前实验主要检验输入条件化滤波和保留评分，而不是充分检验液态神经记忆。

更根本的任务抽象问题：
1. 旧语义片段的事件输入只含新旧余弦相似度等少量信息，没有新内容的完整表示，也没有支持/矛盾/时间修订的有符号关系。相似不等于支持。
2. 语义片段全部使用独立scope和past类型，未运行实体-关系级版本修订。此前语义实验没有检验真实的事实更新假设。
3. latent只影响淘汰分数，事实正文仍在外部片段中，读取也不是从latent解码内容。应称为记忆控制器，不能等同于语义内容记忆。
4. 对某一道问题的答案来源做BCE，是一种弱相关性学习；它不等于下一时刻信念更新、未来问答损失或预算下的集合价值。未标注片段也不等于对所有未来问题无用。
5. 永久历史事实、对当前状态的不确定性、未来访问需求、物理存储淘汰属于不同变量。过去发生的事不会因为时间流逝而变成未发生。
6. 无新证据时让h收敛到b，必须说明b有什么统计意义。否则仅因时间流逝就改变“重要性”没有可靠任务依据。隐状态稳定有界不保证语义正确。
7. 标准准确版本账本已能解决完整可观测、确定更新任务。LTC是否有价值应在部分可观测、噪声、上下文依赖变化率、非均匀时间条件下检验，不能因为使用LTC就预设收益。

可复现诊断 `runs/liquid-modeling-audit-v1/audit.py`：构造两个相距1.2且与旧向量余弦同为0.8的输入，旧记录状态/分数差均0。该反例证明旧记录更新通路的信息坍缩，不声称两个自然语言矛盾句在MiniLM中恰好如此，也不声称整个系统无法区分新记录。改变latent而保持特征不变，rate/equilibrium变化也均0。

三个已训练液态模型的同权重flow关闭诊断（训练窗口，不是新独立测试）：平均logit变化0.000365/0.000479/0.001060，top16重合率99.740%/99.740%/98.958%。rate梯度非零，不能说梯度断开；只能说当前任务/训练下连续流对这些排名影响很弱。梯度范数受参数化影响，不据此直接比较模型质量。

下一代应首先定义可测状态和可证伪假设，而不是立即替换一个更大网络。候选：不可变证据账本保存可追溯历史；实体-关系槽位维护对当前候选值的概率/不确定性；液态隐状态建模上下文依赖的变化过程；存储分配另用未来问答效用。新事件输入必须包含新内容、旧候选内容、关系类型、来源可信度与时间，不能只传相似度。完整LTC的非线性状态依赖通常需要数值积分；不得把冻结系数解宣称为完整LTC精确解。

可考虑混合过滤假设：事件间 `dp/dt=Q(h,c)^T p`，Q非对角非负且每行和为0，保证在适当积分下概率质量守恒；事件时根据明确观察似然更新候选概率，新增候选必须分配先验质量。Q=0对应稳定关系，不凭空遗忘历史。h可用状态耦合LTC；这仅是候选设计，概率模型的校准/正确性和语义似然均未证实。不能把普通Bayesian filtering重新命名为创新；必须与正确设定的解析滤波上界、估计参数的滤波基线、时间感知GRU和显式版本账本对照。

先做失败条件实验：同相似度的确认/否定是否可分；无新证据是否只改变合理的不确定性而不改写过去；同均值不同时间间隔是否影响预测；时间打乱与关闭连续流是否改变任务损失；只在训练部分学习，留独立验证。若完整信息下固定规则/滤波同样好，就撤回“此任务需要液态”假设，保留负结果。不要把强压缩下所有方法差误解为LTC的必然缺陷，也不要假设任意高熵事实都能无损压入小状态。


## 2026-10-12 已实现的混合信念原型

实现位于 `src/memory/liquid_belief.py`。不可变证据与当前状态分离；每个实体槽位维护候选位置概率 p、隐状态 h 和完整事件特征 c。事件包含候选似然、肯定/否定/析取类型、可信度、观察捕获率与逻辑间隔。明确新状态报告替换当前候选分布；同一状态上的约束使用似然乘法，二者不可混用。

事件间 LTC 为 `dh/dt=-(leak+f(h,c))*h+f(h,c)*tanh(A)`，正 conductance 使用 softplus，采用 4 子步半隐式积分。信念采用 `dp/dt=r(h,c)*(1-capture)*(pi(h,c)-p)`，pi 为 softmax。每个子步冻结系数作概率混合，再重新计算系数；这不是完整非线性耦合系统的精确闭式解。capture=1 时，无新报告不产生未观察的状态变化。

固定风险率、输入条件化流、时间 GRU、静态和关闭隐状态连续流使用同一证据接口与训练先验。相同隐宽不等于相同参数或计算量。每槽有效数值状态为 144 字节，解析基线为 32 字节；均不包括原文档案、共享权重与运行时开销，不能据此宣称总内存优势。

当前原型仅覆盖已知六候选位置的状态估计。报告可信度与捕获率已知；语义抽取使用严格公开合成句法。没有验证开放词汇抽取、自然对话问答、真实时间跨度、未知观察机制或有限证据档案。原始 bAbI 完整观察各方法均为 100%；缺失观察下概率建模改善评分，但尚未证明 LTC 状态耦合的独有价值。


## 2026-10-12 内容寻址修复

`src/memory/addressed_codec.py` 将记忆表示为固定实体槽位 M，每次事件 (e,v) 只更新 M[e]，其余槽位逐位保持不变；读取由问题实体选择 M[e] 再经共享解码器输出。更新和读取接口不接收历史原文。该设计是显式内容寻址的工程底座，不能本身宣称液态创新。

四个已知实体，每槽 2 或 8 个 float32，对应总动态载荷 32 或 128 字节；词表及神经权重另计。未知或开放实体分配、自然文本抽取、历史版本追溯和有限槽位淘汰尚未实现，不能据此声称完整通用记忆系统已解决。

GuardedWriter 的可选精确重复守卫保存每实体最近观察值的 uint8，共额外 4 字节；守卫只比较源事件值，不能访问答案。重复则跳过神经更新，读取仍只用神经状态。守卫也是存储信息，必须计账；在当前有限本体任务中该缓存本身足以回答，所以精确表必须作为基线。诊断计数不参与决策。这里不使用相似度作为事实相同的判据。


## 语义 gate 组件（2026-10-12）

`src/models/semantic_gate.py` 提供普通 MLP、两事件 GRU、输入条件化流及四子步半隐式 LTC 的共享内容判别接口。完整句向量经同一宽度投影，读出接收两个投影、差、乘积、隐状态、余弦与七个词法特征。各机制参数量不同。每一文本对独立重置，单位步没有实际时间意义，所以它只是“是否允许跳过当前报告”的判别器实验，不是长期记忆控制器。

`src/data/semantic_gate.py` 的 MemOps adapter 仅依据公开操作标注构造离线检查，严格核对原文与更新链；该 privileged target/trigger selection 不属于可部署抽取。此轮尚未训练补充/修订/撤回等多类操作，更未把判别器接入实际删除路径。负结果与不确定性用于下一轮表示和序列目标设计，不得将词法基线收益归因于液态动态。
## 双向关系验证器（2026-10-12）

`src/models/relation_verifier.py` 引入固定版本 CPU ONNX NLI 模型，联合编码文本对，分别输出旧到新与新到旧的矛盾、蕴含、中性概率。双向最小蕴含分数是等义候选分数，不是已校准的等义概率。数字集合或显式否定变化仅作为保守否决规则；它不识别全部实体关系改变，且实验未证明总体收益。

输入超过 512 token 时拒绝截断并保留信息。缓存绑定模型文件哈希和有序文本对。高 NLI 概率不授权事实覆盖；矛盾也不自动等于修订。离线可靠性门槛未通过时，策略回退到保留全部。本轮所有方法均触发回退。

液态连续历史实验仍需先固定可靠语义输入。后续状态应表示证据随时间的积累与不确定性，而非把两句顺序编码直接称为历史记忆；至少与无历史、指数平滑和 GRU 比较，并控制真实时间间隔、来源重复与实体绑定。

## 角色绑定关系监督（2026-10-12）

`role_relation.py` 将旧文本 A、新文本 B 的双向人工蕴含标签映射到五类：E/E 等义，N/E 严格新增，E/N 变弱，C/C 矛盾，N/N 未确定。其他组合为标注不一致，保存并排除确定标签训练/主评价，不将其判成模型错误或伪造新标签。矛盾不是覆盖授权，角色互换也未必矛盾。

冻结 spaCy 3.8.11 / en_core_web_sm 3.8.0 提取谓词、施事和受事，以主动/被动依存关系作近似归一化。谓词—角色对应、三元组匹配和交叉角色特征与双向 NLI、共同词法特征共同输入轻量多类分类器。该层属于自动角色特征上的人工关系监督；不是经过人工验证的 SRL 标注器，尚未覆盖全部量词、修饰、共指和多事件语义。

本轮增益属于关系表征层，不属于液态状态。语义未通过准入，未修改实际记忆删除/覆盖路径。后续历史比较必须冻结共同语义层，比较相同证据、时间与来源输入下的无历史、指数统计、GRU、液态；不能借不同抽取器或更大输入制造液态优势。

## 作用范围消融与证据覆盖问题（2026-10-12）

`scoped_relation.py` 增加 35 维自动依存特征，把名词修饰、副词、否定中心、量词、介词、带修饰的角色以及从句绑定到对应词元。它们不提供逻辑有效性保证。以相同 C 网格比较普通/均衡损失，不改变底层 NLI 参数。

外部检查发现，SICK 五类线性适配器可能大幅压低底层正确的矛盾输出。未来应检验保留底层方向证据、独立控制等义许可的约束架构；不得将当前分类器直接用于事实覆盖或语义删除。域偏移、标签体系差异和重校准失真是待区分的解释，尚未证明某一项为唯一原因。新架构尚未实现/验证，不得写成已经修复。

## 底层证据保护与历史接口（2026-10-12）

`evidence_history.py` 提供保留原始方向证据的实验接口：非精确重复仅在双向蕴含≥0.9、矛盾&lt;0.1、角色等义≥0.5 时允许历史分数进一步决定跳过。历史不可放宽保护条件。共同精确观察账本处理已见文字，非有限世界状态覆盖器。第一次误丢弃会在后续重复中持续造成内容缺失，故分析另计不同内容覆盖，而不是仅按重复标签统计。

神经历史模型共享 8 个输入、16 维表示与读出。GRU 使用普通更新；液态以输入及状态共同产生正速率，四个半隐式子步按输入 dt 更新；静态模型不携带历史。全部使用同一显式 dt 输入。合成时间仅作机制压力测试，本轮标签不依赖物理时间，不能据此声称连续时间建模有效。

本次只实现离线决策与内容覆盖审计，未接入生产删除或事实替换。EMA 在构造历史中有少量额外节省，是下一轮强基线；液态在当前操作点无额外收益。参数与状态、公共账本必须分别计账。

## 真实时间数值状态诊断（2026-10-12）

`TemporalForecast` 共用过去观测、mask、真实小时差及到目标的时间差，以最近PM2.5值作残差锚。输入流速率仅依赖当前输入，液态速率同时依赖当前状态；两者均采用四个半隐式子步。GRU显式接收同一dt，防止通过额外时间信息制造优势。未观测步骤保持状态且不能读取缺失位置数值；标准化与目标均由时间切分隔离。

当前数值实验支持历史状态有作用：液态重置消融变差。它尚不支持状态耦合或真实dt独有收益，输入流更小且误差略低。参数较GRU少不等于更快，四子步开销更高；全部16维递归状态64字节。此模型未接入语义记忆删除或事实更新。

## 可运行 Agent 记忆接口（2026-10-12）

`src/memory/agent_memory.py` 提供 query不可见的observe(source_id,date,turns)、仅读幸存内容的query及原子checkpoint/restore。文本经allowlist抽取完整句子，日期/来源保留；相同来源/日期不同内容报错，不同日期版本不作语义覆盖。超预算由已有冻结writer或FIFO淘汰，evicted诊断日志及时清空而非藏在在线记忆中。全局时钟/计数16字节另计。

快照验证编码器/写入器指纹、预算与状态形状；事件顺序检查使用分钟日期。同一完整事件输入可多次问不同问题。脚本`agent_memory_run.py`默认dry run，只有--answer调用Agnes；相同状态文件加锁，断点cursor与状态同文件原子保存。实际49会话恢复逐字段相同。该入口是可供多轮运行器调用的记忆组件，不声称完成通用工具调用agent。

