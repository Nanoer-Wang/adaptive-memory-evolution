# 分析文件说明

## 最新 preservation-study-v2

先阅读 `runs/preservation-study-v2/report.html`。这一轮已经完成，使用 CPU；39项工程测试与额外的恢复/预算检查通过。当前源码对应本轮执行清单，旧实验复现请恢复各自的 ZIP 快照到独立环境。

- `execution_manifest.json`：代码快照、数据摘要、18道预选问答、训练/验证划分、模型/预算/验收门槛。`preservation-study-v1/status.json` 解释启动前的重复ID修复。
- `context_metrics.jsonl`：91道验证题 × 新旧两种保存方式，相同检索位置；`component_audit.json` 与 `component_records.jsonl` 拆分压缩和拼接的影响。
- `qa.jsonl`、`qa_records/`：36份原始上下文、问题、参考、回答、评分响应及API用量/排队/延时。生成先落盘，评分失败不会导致再次生成。
- `qa_pairs.csv/jsonl`：18道逐题新旧对照，包含所有胜/负/平。`qualitative_failures.json` 为部分错误的人工阅读诊断候选，不是新的人工评分。
- `blind_human_review.jsonl`：36份打乱方法标识、尚未评分的复核表；`blind_review_mapping_analyst_only.json` 单独保管。不能称为已完成人工评测。
- `checkpoints/`：12组模型最佳/最后权重、三轮日志、训练题号；`encoding.json` 单独记录编码阶段开销，不能用缓存后的检索计时代表端到端延时。
- `mechanism.jsonl`、`mechanism_metrics.csv`、`time_span_metrics.csv`：会话存储逐题、按题型和跨度的指标。真实发生的byte-budget淘汰由在线插入顺序决定，不允许被淘汰记录后来复活。
- `fragment_protocol.json`、`fragment_probe.py`、`fragment_metrics.jsonl`、`fragment_analysis.json`：有单独源码哈希的抽取单元粒度实验。日期是来源时间；单元不等于解析后的语义事实，也没有自动解决新旧冲突。
- `external_protocol.json`、`external_probe.py`、`external_metrics.jsonl`、`external_analysis.json`、`external_conversation_metrics.csv`：LoCoMo41,256条冻结模型外部评估。按10个对话聚类统计；此前已查看该数据，不能标注为新盲测。128k预算可以保存全部会话，对照相同是预期结果。
- `analysis.json`：主统计和预设开发门槛。训练检查点选择与打分共用开发验证集；问答只有18对，自定义同提供商裁判。不得把覆盖率当成问答准确率。
- `api_attempts.csv`、`completion_audit.json`、`artifact_hashes.json`：72成功+4次429，全部恢复；滚动60秒峰值9次，返回124,698Token。不包含失败请求未返回的用量。记录源码无漂移、训练/验证分离、匿名提示头、预算合规与文件校验值。
- `resume_audit.json`、`budget_equivalence_audit.json`：4项无网络恢复检查、100组有变长UTF8/并列评分/超大记录的线上与离线选择等价检查。片段存储另有3项自检。

只读分析可重跑本目录的 `analyze.py`、`analyze_fragments.py`、`analyze_external.py`，随后 `render_report.py` 与 `completion_audit.py`。它们不调用API。`scripts/preservation_study.py --phase qa` 只补问答缺项；`--phase mechanism` 会重新计算并导出离线结果与计时，勿用于只读查看。源码一旦改变，运行器会拒绝修改已冻结实验。

## 内容记忆实验 content-memory-v1

**在线评分修正：** 初版上下文暴露了 LongMemEval 原始 ID 中的 `answer_` 前缀，旧 46 组在线问答已标记为排错数据，见 `qa_status.json`、`qa_superseded.jsonl`。正式在线结果改用 `runs/content-memory-qa-v2/`：全部 150 个上下文已检查匿名化，正文与所选会话完全不变；原始 ID 仅在本地审计记录中保留。修正版还将裁判输出限额统一提高到 512，并只接受一个完整的布尔 JSON 判决。三路网络调度共用每分钟 9 次限速与服务端冷却，仍低于用户要求的上限 10。

修正版可用 `.venv\Scripts\python.exe scripts/content_qa.py` 断点续跑。旧冻结实验的源代码摘要因泄漏修复而不同，直接重跑旧 `content_experiment.py --phase all` 会拒绝覆盖；它的原始代码仍保存在执行清单指向的 ZIP 中。已有离线结果不依赖 ID 的字符串内容，保持有效。本目录分析程序将原始离线结果与修正版在线结果联合报告，不把被替代的在线分数混进新表。

本轮将内容写入与问题读取分离，使用固定的 CPU MiniLM 内容向量；训练仅访问 LongMemEval train，验证选择仅访问既有 val。LoCoMo 外部数据现已打开并用于首次验证，以后继续围绕它调整就不再属于未查看的确认集。

`runs/content-memory-v1/` 中保留：

- `protocol.json`：查看外部问答结构前固定的设计。`execution_manifest.json` 补充运行细节、窗口措辞修正、源代码快照、数据/编码器摘要、样本清单和依赖环境。
- `qa_sample.json`：在评分前按题号 SHA256 固定的 30 道最终问答题；每方法相同题目，20 道 LoCoMo、10 道 LongMemEval 事实更新。
- `data_audit.json`：外部样本数、证据标注异常、不可回答题和长文本逐字重合审计。逐字无重合不等于语义独立的证明。
- `checkpoints/`：三种内容写入器 × 三个种子的训练过程、最佳/最后权重和训练题号。
- `results.sqlite3`：逐题持久化；生成后先保存，再请求裁判。重跑会跳过已完成题，调用失败不会丢弃已生成答案。
- 本目录 `retrieval.jsonl` 与修正版目录 `content-memory-qa-v2/qa.jsonl`：检索轨迹、留存内容 ID、存储大小、实际发送的上下文、回答、参考答案、API 用量与耗时。离线全量结果与抽样在线结果必须分开分析，旧在线数据只看 `qa_superseded.jsonl` 并保持排错标签。
- `aggregate_metrics.csv`、`conversation_metrics.csv`、`paired_comparisons.csv/json`：按预算、方法、种子、题型、时间跨度和可回答性分组；外部配对区间以对话为抽样单位，不把同一对话的问题或训练种子伪装成独立样本。
- `analysis_status.json`：完成数、真实成功请求用量、分析代码摘要和解释限制。图表有 PNG/SVG/PDF 三种格式。

`payload_bytes` 统计保留文字的 UTF-8、向量、ID、日期、优先级、序号及隐藏状态；不包含共享编码器/模型权重、Python 容器或检索索引的进程开销。权重大小与进程 RSS 另记。写入/检索计时不含提前缓存的内容编码，不能据此宣称端到端生产延迟。

LoCoMo 每段对话仅 19–32 个会话，32 条预算在该数据上等于完整存储；8 条预算才构成压缩。会话 Recall@5 不保证会话中具体证据片段仍在裁剪后的文本里。Agnes 自定义裁判、固定种子的小样本 QA 均不是官方 benchmark 得分。

只保留复现、错误分析、性能分析需要的文件；不记录密钥、Authorization、原始错误响应或无用途的调试转储。

## 每次评测：runs/<run-id>/

| 文件 | 用途 |
|---|---|
| summary.json | 总体与分类指标、有效评分数量、延迟分位数、内存采样、缓存与 token 汇总 |
| per_example.csv | 逐题标量，适合筛选失败题、画图、比较速度和内存 |
| predictions.jsonl | 问题、参考答案、实际回答、证据会话、检索结果、实际发送的上下文、截断情况、耗时及用量 |
| report.md | 可直接阅读的实验摘要与限制 |
| run.json | 方法、数据校验值、问题 ID 清单、seed、checkpoint SHA-256、代码摘要、检索和生成设置 |
| environment.json | Python/库版本、操作系统、CPU 运行标记、完整非敏感配置、源代码快照路径 |
| results.sqlite3 | 逐题事务存储，是断点恢复的依据，不应在运行中删除 |
| official_predictions.jsonl | 官方评测输入格式：question_id/hypothesis；不代表已经运行官方裁判 |

离线 retrieval-only 不包含答案、生成耗时或 API 用量，不把缺失答案计作零分。回答失败时停止并保留已完成记录，不能把未完成题悄悄当作成功样本。报告同时给出 completed/requested。

## 训练：artifacts/checkpoints/<method>/

- `history.json`、`history.csv`：逐 epoch 总损失、检索损失、未来会话代理损失、稳定性损失、验证 Recall@k、学习率、耗时和进程内存采样。
- `best.pt`：验证集 session Recall@k 最高的模型。
- `last.pt`：最近完整 epoch 的模型和优化器状态，用于继续训练。
- `environment.json`：结构配置、参数量、运行环境与源代码快照。

中断训练最多重做未完成的当前 epoch。测试集不用于选 epoch 或超参数。

## 数据与复现

- `data/longmemeval/source.json`：上游仓库、精确 revision、下载地址、大小、SHA-256、许可来源。
- `artifacts/prepared/manifest.json`：完整分组划分、特征配置、随机种子、数据签名。
- `artifacts/prepared/features/`：只含公开历史/问题得到的向量缓存，供训练与评测复用。
- `artifacts/source/<hash>.zip`：运行时源代码、脚本、配置和依赖清单的白名单快照；不包含 `.env`、数据集、模型权重或聊天记录。

## API 账目与总体比较

- `artifacts/api/responses.sqlite3`：成功响应缓存及 `requests_log` 请求账目。
- `runs/api_usage.csv`：运行 `scripts/export_api_usage.py` 导出实际 HTTP 尝试时间、服务区域、模型、状态、耗时、供应商返回的 token 用量。
- `runs/comparison.csv`：运行 `scripts/summarize_runs.py` 汇总不同实验。先核对相同测试 ID、模型、上下文预算、代码版本再比较；不同 limit 的结果不能直接排序下结论。

缓存响应保留最初调用的 token 记录。`api_tokens_recorded` 是这些回答对应的逻辑 token 总量；`uncached_answer_tokens` / `uncached_judge_tokens` 是本次结果中未命中本地缓存的用量。跨中断、失败重试的实际请求以请求账目为准，供应商未返回 usage 的失败请求不能据此断言免费。总请求日志从启用账目记录功能后开始，之前的连通性探测不补造记录。

## 指标解读

- `session_recall_at_k`：命中证据会话数 / 全部证据会话数，每题计算再宏平均；拒答题与无证据题排除，分母单独保存。
- `all_evidence_at_k`：该题的证据会话是否全部被检索到。
- 同编号的重复历史记录保留；评分按唯一会话 ID 去重。检索取 k 条记录，可能不足 k 个唯一 ID，记录可供检查。
- `exact_match` / `token_f1`：英文文本规范化后的诊断指标，不能当作官方语义准确率。
- `custom_judge_correct`：Agnes 自定义裁判。解析失败为 null，并记录失败数量，不当作正确或错误暗中计入。
- `retrieval_seconds`：历史整理和检索耗时；`context_seconds`：片段选择与上下文打包耗时；`generation_seconds` / `judge_seconds` 包含排队和重试。
- `queue_wait_seconds` 与 `judge_queue_wait_seconds` 分别记录回答和裁判的客户端队列等待；缓存命中时为零。
- `process_rss_mb`：逐题或 epoch 边界的进程驻留内存采样，不是 GPU 显存，也不是精确进程峰值。summary 中 max 仅指这些采样的最大值。
- `dense_feature_bytes` 是当前历史的稠密特征大小；`memory_state_bytes` 是状态向量理论字节数，不能等同整个记忆系统存储量。
- 输入预算以字符数控制，非精确 token 数；真实 token 以 API usage 为准。

建议优先分析：LCM 与同特征的 hash_rag/GRU 差异、TF-IDF 强基线差距、时间推理/知识更新类别、检索成功但回答失败的样本、上下文截断、拒答表现，以及建库与推理开销。5 题在线试运行只证明流程，不能用于有效性结论。

开发记录：最早的 5 题 test 冒烟测试暴露了提示词过度拒答；这些题已被人工查看，不再声称其问答结果是盲测。旧代码版本记录保留用于排查，后续在线开发试运行默认使用 val。正式盲测需排除已调试样本或使用独立公开测试数据。检索全量结果是阶段性观测，不应据此在相同 test 上反复调参。


## 语义机制实验（2026-10-11）

入口 `runs/semantic-mechanism-v2/report.html`。`protocol.json`保存最初冻结源码；`evaluations/*.jsonl`保存每个条件的驻留ID、检索ID、上下文、时间和证据覆盖；`prepared/*.json + *.npz`保存精确片段/来源与向量，可复原失误；`checkpoints`包含15个权重和训练记录。`analysis.json`按历史对种子平均后计算配对bootstrap，不能把三个种子当三个独立样本。

最终问答应读取 `qa_adjudication.json` 的复核值，原始 `qa_judges.jsonl` 有5条误判；所有原始答案保留在 `qa_predictions.jsonl`。`combined_api_usage.json`含本轮全部37次请求、Token与滚动窗口统计。`reader_diagnostic`是事后统一读取方式诊断，不属于主结果。`semantic-claim-audit-v1/v2/v3`分别保存初始提案、格式修复、源文本不确定性过滤结果；不能把引用对齐率称为语义事实精度。

本轮仅12训练/6开发历史；监督少、时间变化稀疏、编码向量占用较大、检索未覆盖全部证据，不足以得出液态机制一般有效或无效。重复调用已完成实验的all/qa/evaluate只提示读取报告；新研究须新目录与协议。若必须逐行复现历史实验，应使用protocol指向的源码快照，不能用后续源码覆盖旧结果。
