
# Historical proposal: Liquid Cognitive Memory

This proposal is superseded by **Adaptive Memory Evolution for Lifelong Agents**. It is retained for provenance, not as a current claim of LNN superiority. The active mathematical definition is in [Algorithm.md](Algorithm.md), and the latest protocol is [here](../runs/evolving-memory-v1/protocol.json).

> Implementation note (2026-10-11): the sections below preserve the original
> research proposal. The executable phase-one system uses a custom grouped
> LongMemEval-S split, lexical hashing features, and local LCM/GRU/Transformer
> training. Its future loss predicts the next session's feature direction, not
> task success. Importance is learned from history/state without a reward input.
> The liquid cell is a time-aware gated approximation, not an exact CfC or ODE
> solver. Latent state conditions retrieval; only retrieved text goes to Agnes.
> Real reward feedback, ALFWorld, MSC, and lifelong transfer/forgetting evaluation
> remain future research work; no results for them are claimed.

## 1. Problem Definition

Existing agent memory systems treat memory as external storage:

Experience -> Embedding -> Retrieval -> Context

We argue that lifelong agents require:

Experience -> Dynamic Cognitive State Evolution


## 2. Dual Memory Architecture

LCM contains:

### Episodic Memory

Stores concrete experiences.

Example:

(task, action, feedback)


### Liquid Cognitive Memory

Stores abstract evolving cognitive state.


## 3. Experience Encoding

For interaction t:

z_t=(x_t,a_t,r_t)

where:

x_t: task input
a_t: agent action
r_t: feedback


Experience embedding:

e_t = Encoder(z_t)


## 4. Liquid Memory Dynamics

Memory state:

m_t


Continuous formulation:

dm(t)/dt = F_theta(m(t),e(t))


Discrete update:

m_(t+1)=m_t+dt*F_theta(m_t,e_t)


CfC-style update:

m_(t+1)=alpha_t*m_t+(1-alpha_t)*candidate_t


## 5. Importance-aware Consolidation

Importance:

w_t = g(e_t,r_t)


Update:

m_(t+1)=m_t+w_t*F_theta(m_t,e_t)


## 6. Retrieval

Current query:

q_t


Conditioned retrieval:

q'_t = W[q_t,m_t]


Retrieve:

E*=Retrieve(q'_t,EpisodicMemory)


Context:

C=[m_t,E*]


## 7. Training Objective

Total loss:

L=L_future+lambda1*L_contrastive+lambda2*L_stability


Future objective:

predict future task success


Contrastive objective:

pull useful experiences closer


Stability:

prevent catastrophic drift


## 8. Inference

1. Receive task
2. Encode interaction
3. Update liquid state
4. Retrieve episodic memories
5. Provide context to LLM
