# Comparison with published memory systems

Checked on 2026-10-14. This is a primary-source protocol review, not a reproduction.

| Work | Relevant mechanism | Inspected evaluation | Direct performance comparison with AME? |
|---|---|---|---|
| [A-MEM, v1](https://arxiv.org/html/2502.12110v1) | Historical note attributes and links evolve when new notes arrive. | LoCoMo; multiple readers; F1/BLEU-1; category-dependent retrieval settings (Sections 3.3, 4.1-4.2 and Appendix B.4). | No: question selection, reader, metric, and memory budget differ. |
| [Mem0, v1](https://arxiv.org/html/2504.19413v1) | LLM extraction and ADD/UPDATE/DELETE/NOOP operations; graph variant. | LoCoMo excluding adversarial questions; GPT-4o-mini operations; lexical and judge metrics (Sections 2-3). | No: question set, scoring pipeline, feedback and budget do not match. |
| [LongMemEval official protocol](https://github.com/xiaowu0162/LongMemEval) | Benchmark for multi-session memory, temporal reasoning, updates and abstention. | 500 instances per variant; official evaluation entry point uses GPT-4o. | No: our custom test has 24 questions and an Agnes judge. |
| [LoCoMo official release](https://github.com/snap-research/locomo) | Long conversational histories and annotated QA. | Public ten-conversation release with its own scripts. | No: our fixed 24-question transfer subset and 64-slot policy are different. |

The defensible contribution is an explicit, low-dimensional state formulation
and controlled rate interventions on shared streams. Memory updating/evolution
already exists in earlier systems and is not claimed as unprecedented here.

No external scores are inserted into AME result tables, no cross-paper gain is
computed, and no favorable subset of another paper is presented as a matched
baseline. A fair performance study would first lock identical questions,
reader/judge, prompts, retrieval and storage budgets, and feedback availability.
All frozen AME outcomes, including negative language-transfer findings, remain.
