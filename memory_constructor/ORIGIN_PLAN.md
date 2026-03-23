# **Trainable Memory Constructor Project \- Implementation Plan**

## **Context**

This project aims to build a trainable memory constructor for a fixed downstream agent operating in the WebShop environment. The memory constructor learns to decide:

1. Whether to write a memory item at each timestep  
2. What keys to use for retrieval (multiple keys allowed)  
3. What value to write (compressed, faithful representation)

The system uses an append-only memory store with a fixed retriever and fixed downstream agent. Training proceeds through three stages:

1. **SFT** on static WebShop trajectories  
2. **Candidate sampling \+ best-of-n** on static trajectories  
3. **RL** on dynamic trajectory generation

This addresses the challenge of learning effective memory construction for long-horizon tasks where the agent must compress and store information for future retrieval.

---

## **Step 1: Formulation Critique**

### **What is Well-Defined ✓**

1. **Clear action space**: The constructor outputs structured JSON with `{write: bool, keys: [...], value: "..."}` \- this is concrete and inspectable.  
2. **Append-only constraint**: No update/merge/delete operations simplifies the problem and prevents credit assignment confusion from memory overwrites.  
3. **Fixed agent \+ fixed retriever**: Isolates the memory constructor as the only trainable component, making experiments interpretable.  
4. **Three-stage curriculum**: SFT → best-of-n → RL is a sensible progression from supervised to online learning.  
5. **Token budget constraints**: Explicit limits on `max_value_tokens`, `max_key_tokens`, `max_num_keys`, `episode_memory_budget` make the problem tractable.  
6. **Structured evaluation**: Clear metrics (write accuracy, retrieval hit rate, task success, redundancy, compression behavior).

### **What is Ambiguous ⚠️ → RESOLVED**

1. **"Future usefulness" scoring for best-of-n**: ✓ RESOLVED  
   * **Scope**: Until episode end  
   * **Method**: Use both hindsight (what was actually retrieved) and counterfactual (what would have been retrieved) with weighted merge  
   * **Dense rewards**: Sample multiple memory slots from different perspectives and use counterfactual evaluation to make rewards more dense  
2. **Retriever behavior**: ✓ RESOLVED  
   * **Algorithm**: Hybrid (BM25 \+ dense embedding)  
   * **Top-k**: 3-4 memories per query (default)  
   * **Retrieval features**: Key overlap \+ value semantic embedding (since we have multiple keys)  
   * **Deterministic**: Start with deterministic retrieval  
   * **Temporal ordering**: Yes, add time label for each memory slot  
3. **Fixed agent interface**: ✓ RESOLVED  
   * **Agent receives**: Just retrieved memories (not full memory store)  
   * **Query formatting**: Prompt the LLM (GPT-5) to format queries into a compact sentence, ensuring no key information is missed  
   * **API**: Use GPT-5 via API (key in `/home/yuhao/code/.env`)  
4. **WebShop trajectory format**: ✓ RESOLVED  
   * **Example file**: `/home/yuhao/code/WebShop_VanillaReactAgent_TrajectoryGeneration/logs/raw/webshop/react_v1/20260309_221152_baseline.jsonl`  
   * **Structure**:

{  
  "episode\_id": "...",  
  "task\_id": "...",  
  "instruction": "...",  
  "steps": 3,  
  "final\_reward": 1.0,  
  "is\_success": true,  
  "trajectory": \[  
    {"node\_type": "OBS\_TOOL", "payload": {"raw\_text": "...", "tool\_name": "env\_reset"}, "t": ...},  
    {"node\_type": "THOUGHT", "payload": {"raw\_text": "..."}, "context\_tokens": ..., "t": ...},  
    {"node\_type": "ACT\_TOOL", "payload": {"action\_str": "..."}, "t": ...},  
    {"node\_type": "OBS\_TOOL", "payload": {"raw\_text": "...", "tool\_name": "..."}, "t": ...},  
    ...  
  \]  
}

*   
  * **Rewards**: Check example file to determine if intermediate rewards exist or only terminal success/failure  
5. **Candidate scoring priorities**: ✓ RESOLVED  
   * **Default weights** (intuitive, adjust empirically later):  
     * Retrieval usefulness: 0.4  
     * Task success score: 0.3  
     * Compactness: 0.1  
     * Redundancy: 0.1  
     * Faithfulness: 0.1  
   * **Expensive rollouts**: Optimize later (start with simpler scoring)  
6. **no\_write as a candidate**: ✓ RESOLVED  
   * **Always include**: Start with always including no\_write  
   * **Scoring**: Use overall reward directly \- if it saves budget (increases budget-saving reward) and doesn't decrease utility much, it will add more no\_write actions  
7. **Future-relevant information for hindsight labeling**: ✓ RESOLVED  
   * **Reward model approach**: Design a reward model to define how much contribution one memory may contribute to the final result  
   * **Options**: Use LLM as a judge or other methods

### **What is Likely to Fail ❌ → UPDATED WITH MITIGATIONS**

1. **Credit assignment with very large max\_value\_tokens (2048)**:  
   * If values are 2048 tokens, the constructor might just copy raw observations  
   * The model won't learn compression \- it will learn "dump everything"  
   * **Risk**: Degenerate solution where every memory is a full transcript  
   * **Mitigation**: ✓ Start with small max\_value\_tokens (128-512), gradually increase  
2. **Sparse delayed reward in RL stage**:  
   * If task success is the only reward, and episodes are long (50+ steps), credit assignment will be extremely difficult  
   * The constructor won't know which memories helped vs. hurt  
   * **Risk**: RL training will be unstable and sample-inefficient  
   * **Mitigation**: ✓ Use dense auxiliary rewards combining counterfactual and hindsight:  
     * **Counterfactual**: Simulate retrieval for each candidate memory  
     * **Hindsight**: Track actual retrieval hits  
     * **Distillation signal**: Give LLM full context and ask which part is most useful for current task/turn  
     * **Weighted combination**: Merge all signals for dense reward  
3. **Mismatch between constructor output and retriever behavior**:  
   * If the constructor generates keys that don't match the retriever's indexing scheme, memories will never be retrieved  
   * Example: Constructor writes key="user wants red shoes", but retriever uses BM25 and agent queries "red footwear" → no match  
   * **Risk**: Constructor learns to write memories that are never used  
   * **Mitigation**:  
     * ✓ Use the same tokenizer for BM25 computing in retrieval  
     * ✓ Constructor doesn't need same tokenizer/model \- keys are natural language  
     * ✓ Hybrid retrieval (BM25 \+ semantic embedding) handles language variation  
     * ✓ Log retrieval hit rates during training  
4. **Mismatch between constructor output and agent usage**:  
   * If the agent doesn't actually use retrieved memories effectively, the constructor has no signal  
   * Example: Agent ignores memory and just uses current observation  
   * **Risk**: Constructor training is futile if agent doesn't benefit from memory  
   * **Mitigation**:  
     * ✓ Verify agent actually uses memory (ablation: agent with vs. without memory)  
     * ✓ Use distillation training: give LLM full context, ask which part is most useful  
     * ✓ Use this as extra signal for memory reward (sample \+ counterfactual approach)  
     * **Note**: Memory ablation is expensive, so list all training methods and try them one by one  
5. **Candidate sampling diversity collapse**:  
   * If temperature is too low, all N candidates will be nearly identical  
   * If temperature is too high, candidates will be incoherent  
   * **Risk**: Best-of-n degenerates to single-sample  
   * **Mitigation**:  
     * ✓ Use nucleus sampling (top-p) instead of temperature  
     * **Note**: For retrieval, use deterministic top-k (no diversity issue there)  
6. **Episode memory budget exhaustion**:  
   * If the constructor writes too many memories early, it runs out of budget  
   * If it writes too few, it misses important information  
   * **Risk**: Constructor learns degenerate "always write" or "never write" policy  
   * **Mitigation**:  
     * ✓ **No overall budget penalty** \- instead, writing a memory has a computational cost penalty  
     * ✓ If memory is useful, it gets a much larger reward than the cost penalty  
     * ✓ This naturally balances write vs. no\_write decisions  
     * ✓ Include budget-aware features (current budget, episode progress) in input  
7. **Hindsight labeling bias**:  
   * Hindsight labels assume the trajectory was optimal, but it might not be  
   * Example: Agent failed because it didn't have memory X, but hindsight labels say "don't write X" because it wasn't retrieved  
   * **Risk**: SFT learns from suboptimal demonstrations  
   * **Mitigation**: ✓ Use only successful trajectories for hindsight labeling

## **Training Methods Summary (Prioritized by Cost)**

Based on the discussion, here are all the training methods and signals we can use, ordered from cheapest to most expensive:

### **1\. Hindsight Labeling (Cheapest)**

* **Cost**: Offline, one-time processing  
* **Method**: Look ahead in successful trajectories to identify future-relevant information  
* **Signal**: Binary (write/no\_write) \+ keys \+ value  
* **Use for**: SFT stage

### **2\. Counterfactual Retrieval Simulation (Cheap)**

* **Cost**: Offline, deterministic computation  
* **Method**: For each candidate memory, simulate future retrieval queries using hybrid retriever  
* **Signal**: Retrieval score (0-1) for each candidate  
* **Use for**: Best-of-n candidate scoring  
* **Benefit**: Makes rewards more dense without expensive rollouts

### **3\. LLM-as-Judge for Contribution (Moderate)**

* **Cost**: One LLM call per step  
* **Method**: Prompt GPT-5 to evaluate how useful it would be to remember this information  
* **Signal**: Contribution score (0-1)  
* **Use for**: Hindsight labeling, candidate scoring  
* **Benefit**: More accurate than heuristics, cheaper than rollouts

### **4\. Distillation Training (Moderate)**

* **Cost**: One LLM call per step  
* **Method**: Give LLM full context, ask which part is most useful for current task/turn  
* **Signal**: Attention weights or importance scores  
* **Use for**: Auxiliary reward in RL, candidate scoring  
* **Benefit**: Provides dense signal about memory usefulness

### **5\. Hindsight Retrieval Tracking (Moderate)**

* **Cost**: Offline, requires parsing trajectory  
* **Method**: Track which memories were actually retrieved during the trajectory  
* **Signal**: Retrieval hit count per memory  
* **Use for**: Best-of-n candidate scoring  
* **Benefit**: Ground truth about memory usage

### **6\. Mini-Rollouts for Candidate Scoring (Expensive)**

* **Cost**: N rollouts per candidate (N candidates per step)  
* **Method**: Replay trajectory with each candidate memory, measure task success  
* **Signal**: Task success delta  
* **Use for**: Best-of-n candidate scoring (optional, optimize later)  
* **Benefit**: Most accurate signal, but very expensive

### **7\. Memory Ablation (Most Expensive)**

* **Cost**: 2 full rollouts per memory (with/without)  
* **Method**: Run agent with and without specific memory, compare performance  
* **Signal**: Memory utility \= success\_with \- success\_without  
* **Use for**: Evaluation, debugging (not for training)  
* **Benefit**: Ground truth about memory impact, but too expensive for training

### **Recommended Training Pipeline (v1):**

**Stage 1: SFT**

* Use hindsight labeling (method \#1)  
* Optionally enhance with LLM-as-judge (method \#3)  
* Train on successful trajectories only

**Stage 2: Best-of-N**

* Generate candidates with nucleus sampling  
* Score using:  
  * Counterfactual retrieval simulation (method \#2) \- primary signal  
  * Hindsight retrieval tracking (method \#5) \- secondary signal  
  * Weighted merge: 0.5 \* counterfactual \+ 0.5 \* hindsight  
  * Optionally add LLM-as-judge (method \#3) for contribution score  
* Skip mini-rollouts (method \#6) in v1, optimize later if needed

**Stage 3: RL**

* Use dense auxiliary rewards:  
  * Counterfactual retrieval simulation (method \#2)  
  * Distillation training (method \#4) for attention signal  
  * Computational cost penalty for writing  
* Combine all signals with learned weights

**Evaluation**:

* Use memory ablation (method \#7) sparingly for debugging  
* Focus on task success rate, retrieval hit rate, compression quality

---

1. **Explicit credit assignment tracking**:  
   * Log for each memory: when written, when retrieved, impact on agent decision  
   * Compute "memory utility" \= (task success | memory retrieved) \- (task success | memory not retrieved)  
   * Visualize memory lifespan: write time → retrieval time → impact time  
2. **Ablation-based reward decomposition**:  
   * For each candidate memory, run mini-rollouts with/without that memory  
   * Measure: retrieval hit rate, agent decision change, task success delta  
   * This gives ground-truth "future usefulness" for best-of-n  
3. **Retrieval simulation during training**:  
   * At each training step, simulate future retrieval queries  
   * Compute retrieval precision/recall for each candidate memory  
   * Use this as auxiliary loss or reward shaping  
4. **Compression quality metrics**:  
   * ROUGE/BLEU between memory value and original observation  
   * Semantic similarity (embedding distance)  
   * Information retention (can a model answer questions from memory?)  
   * Faithfulness (does memory contain hallucinations?)  
5. **Budget-aware analysis**:  
   * Plot: episode progress vs. memory budget remaining  
   * Identify: when does constructor write most? Early, middle, late?  
   * Compare: budget usage in successful vs. failed episodes  
6. **Failure mode detection**:  
   * Monitor: % of memories never retrieved (wasted writes)  
   * Monitor: % of queries with no retrieval hits (missing memories)  
   * Monitor: % of episodes hitting budget limit  
   * Monitor: average memory value length (detect copy-paste behavior)

---

## **Step 2: System Architecture**

### **Project Structure**

memory\_constructor/  
├── configs/  
│   ├── model.yaml              \# Model hyperparameters  
│   ├── training.yaml           \# Training hyperparameters  
│   ├── data.yaml               \# Data paths and preprocessing  
│   └── evaluation.yaml         \# Evaluation settings  
├── data/  
│   ├── webshop/  
│   │   ├── raw/                \# Raw WebShop trajectories  
│   │   ├── processed/          \# Processed trajectories  
│   │   └── splits/             \# Train/val/test splits  
│   └── schemas.py              \# Data schema definitions  
├── memory\_constructor/  
│   ├── \_\_init\_\_.py  
│   ├── models/  
│   │   ├── \_\_init\_\_.py  
│   │   ├── constructor.py      \# Memory constructor model  
│   │   ├── retriever.py        \# Fixed retriever interface  
│   │   └── agent.py            \# Fixed downstream agent wrapper  
│   ├── data/  
│   │   ├── \_\_init\_\_.py  
│   │   ├── webshop\_parser.py   \# Parse raw WebShop data  
│   │   ├── sft\_dataset.py      \# SFT training dataset  
│   │   ├── candidate\_dataset.py \# Best-of-n dataset  
│   │   └── rl\_dataset.py       \# RL rollout dataset  
│   ├── training/  
│   │   ├── \_\_init\_\_.py  
│   │   ├── sft\_trainer.py      \# SFT training loop  
│   │   ├── candidate\_sampler.py \# Candidate generation  
│   │   ├── candidate\_scorer.py  \# Candidate scoring  
│   │   ├── bestofn\_trainer.py  \# Best-of-n training  
│   │   └── rl\_trainer.py       \# RL training loop  
│   ├── environment/  
│   │   ├── \_\_init\_\_.py  
│   │   ├── webshop\_env.py      \# WebShop environment wrapper  
│   │   ├── memory\_store.py     \# Append-only memory store  
│   │   └── trajectory\_generator.py \# Dynamic trajectory generation  
│   ├── evaluation/  
│   │   ├── \_\_init\_\_.py  
│   │   ├── metrics.py          \# Evaluation metrics  
│   │   ├── evaluator.py        \# Main evaluation loop  
│   │   └── baselines.py        \# Baseline implementations  
│   └── utils/  
│       ├── \_\_init\_\_.py  
│       ├── json\_utils.py       \# JSON parsing/validation  
│       ├── logging\_utils.py    \# Logging and visualization  
│       └── openrlhf\_utils.py   \# OpenRLHF integration helpers  
├── scripts/  
│   ├── 01\_preprocess\_webshop.py  
│   ├── 02\_train\_sft.py  
│   ├── 03\_generate\_candidates.py  
│   ├── 04\_score\_candidates.py  
│   ├── 05\_train\_bestofn.py  
│   ├── 06\_train\_rl.py  
│   └── 07\_evaluate.py  
├── experiments/  
│   └── README.md               \# Experiment tracking  
├── tests/  
│   └── test\_\*.py  
├── requirements.txt  
└── README.md

### **Component Design**

#### **1\. Memory Constructor Agent Instance (`memory_constructor/environment/agent_instance.py`)**

**Core component that handles both static and dynamic trajectories**

class MemoryConstructorAgentInstance(AgentInstanceBase):  
    """  
    Agent instance for memory constructor training.  
    Unified interface for:  
    \- Static trajectory replay (SFT, best-of-n)  
    \- Dynamic trajectory generation (RL)  
    """

    def \_\_init\_\_(self, config):  
        super().\_\_init\_\_()  
        self.config \= config  
        self.fixed\_agent \= GPT5Agent(config.api\_key)  \# Load from .env  
        self.retriever \= HybridRetriever(config)  
        self.memory\_store \= None  
        self.mode \= None  \# "static" or "dynamic"

    async def reset(self, states: dict, \*\*kwargs):  
        """Initialize episode \- handles both static and dynamic modes"""  
        pass

    async def step(self, states: dict, \*\*kwargs):  
        """Execute one step: constructor \-\> retriever \-\> agent \-\> reward"""  
        pass

#### **2\. Hybrid Retriever (`memory_constructor/models/retriever.py`)**

**Input**: Query string \+ memory store

**Output**: Top-k retrieved memories

class HybridRetriever:  
    def \_\_init\_\_(self, config):  
        self.bm25 \= BM25Retriever()  
        self.dense \= DenseRetriever(model\_name="sentence-transformers/all-MiniLM-L6-v2")  
        self.k \= config.retrieval\_k  
        self.bm25\_weight \= config.bm25\_weight  \# e.g., 0.5  
        self.dense\_weight \= config.dense\_weight  \# e.g., 0.5

    def retrieve(self, query: str, memory\_store: MemoryStore):  
        \# BM25 retrieval on keys  
        bm25\_results \= self.bm25.retrieve(query, memory\_store, k=self.k)

        \# Dense retrieval on keys \+ value preview  
        dense\_results \= self.dense.retrieve(query, memory\_store, k=self.k)

        \# Combine scores  
        combined \= self.\_combine\_scores(bm25\_results, dense\_results)

        return combined\[:self.k\]

#### **3\. GPT-5 Agent Wrapper (`memory_constructor/models/agent.py`)**

**Input**: Current observation \+ retrieved memories

**Output**: Action

class GPT5Agent:  
    def \_\_init\_\_(self, api\_key: str):  
        self.client \= OpenAI(api\_key=api\_key)  
        self.model \= "gpt-5"

    def act(self, observation: str, retrieved\_memories: List\[dict\]):  
        \# Format input with memories  
        prompt \= self.\_format\_prompt(observation, retrieved\_memories)

        \# Call GPT-5  
        response \= self.client.chat.completions.create(  
            model=self.model,  
            messages=\[  
                {"role": "system", "content": "You are a WebShop agent..."},  
                {"role": "user", "content": prompt}  
            \],  
            temperature=0.7,  
        )

        action \= response.choices\[0\].message.content  
        return action

    def \_format\_prompt(self, observation, retrieved\_memories):  
        \# Include retrieved memories in prompt  
        memory\_text \= "\\n".join(\[  
            f"Memory: {m\['value'\]}" for m in retrieved\_memories  
        \])  
        return f"Observation: {observation}\\n\\nRelevant memories:\\n{memory\_text}\\n\\nWhat action should you take?"

#### **4\. Memory Store (`memory_constructor/environment/memory_store.py`)**

class MemoryStore:  
    def \_\_init\_\_(self, max\_capacity: int):  
        self.memories \= \[\]  
        self.max\_capacity \= max\_capacity

    def append(self, memory\_item: dict):  
        if len(self.memories) \>= self.max\_capacity:  
            raise MemoryBudgetExceeded()  
        self.memories.append(memory\_item)

    def get\_all(self):  
        return self.memories

    def get\_budget\_remaining(self):  
        return self.max\_capacity \- len(self.memories)

    def to\_dict(self):  
        return {  
            "memories": self.memories,  
            "budget\_remaining": self.get\_budget\_remaining(),  
        }

---

## **Step 3: Data Schemas**

### **1\. Raw WebShop Trajectory (from trajectory generation repo)**

@dataclass  
class WebShopTrajectory:  
    episode\_id: str  
    task\_id: str  
    task\_idx: int  
    instruction: str                 \# Task description  
    steps: int                       \# Number of steps  
    final\_reward: float              \# Terminal reward (0.0 or 1.0)  
    is\_success: bool  
    is\_partial: bool  
    termination: str                 \# "done", "max\_steps", etc.  
    format\_errors: int  
    final\_ctx\_tokens: int  
    trajectory: List\[WebShopNode\]    \# Sequence of nodes  
    metadata: dict

@dataclass  
class WebShopNode:  
    node\_type: str                   \# "OBS\_TOOL", "THOUGHT", "ACT\_TOOL"  
    payload: dict                    \# Contains raw\_text, action\_str, tool\_name, etc.  
    call\_id: str                     \# Unique ID for this node  
    context\_tokens: int  
    t: float                         \# Timestamp  
    failure\_flag: Optional\[bool\]     \# For OBS\_TOOL nodes  
    failure\_reason: Optional\[str\]  
    ref\_ids: Optional\[List\[str\]\]     \# For THOUGHT nodes

### **2\. Memory Item Schema (with temporal ordering)**

@dataclass  
class MemoryItem:  
    write: bool  
    keys: List\[str\]                  \# Multiple keys for retrieval  
    value: str                       \# Compressed memory content  
    timestamp: float                 \# When this memory was written  
    step\_id: int                     \# Which step it was written at  
    metadata: dict                   \# Additional info for debugging

### **3\. SFT Training Sample**

@dataclass  
class SFTSample:  
    sample\_id: str  
    trajectory\_id: str  
    step\_id: int

    \# Input context  
    observation: str                 \# Current observation (from OBS\_TOOL node)  
    local\_history: List\[str\]         \# Last N steps (THOUGHT \+ ACT\_TOOL \+ OBS\_TOOL)  
    memory\_store: List\[MemoryItem\]   \# Current memories with timestamps  
    budget\_remaining: int  
    episode\_progress: float          \# step\_id / total\_steps

    \# Target output  
    target\_memory: MemoryItem        \# {write: bool, keys: \[...\], value: "...", timestamp: ...}

    \# Metadata  
    future\_task: str                 \# What the agent needs to do next  
    hindsight\_label: str             \# Why this memory is useful (from reward model)  
    contribution\_score: float        \# How much this memory contributes to final result  
    metadata: dict

### **4\. Candidate Memory List (Best-of-N)**

@dataclass  
class CandidateMemoryList:  
    sample\_id: str  
    trajectory\_id: str  
    step\_id: int

    \# Input context (same as SFT)  
    observation: str  
    local\_history: List\[str\]  
    memory\_store: List\[MemoryItem\]  
    budget\_remaining: int  
    episode\_progress: float

    \# Sampled candidates  
    candidates: List\[CandidateMemory\]

    \# Best candidate selection  
    best\_candidate\_idx: int  
    selection\_method: str            \# "hindsight", "counterfactual", "weighted\_merge"

    metadata: dict

@dataclass  
class CandidateMemory:  
    candidate\_id: int  
    memory\_item: MemoryItem          \# {write: bool, keys: \[...\], value: "...", timestamp: ...}

    \# Scores (weighted merge of hindsight and counterfactual)  
    hindsight\_score: float           \# Based on what was actually retrieved  
    counterfactual\_score: float      \# Based on what would have been retrieved  
    task\_success\_score: float        \# Impact on downstream task  
    retrieval\_usefulness: float      \# Retrieval hit rate  
    compactness\_score: float         \# Token efficiency  
    redundancy\_score: float          \# Overlap with existing memories  
    faithfulness\_score: float        \# Compression quality

    \# Combined score (weighted merge)  
    total\_score: float

    \# Debugging info  
    future\_retrieval\_hits: int       \# How many times retrieved  
    future\_impact\_steps: List\[int\]   \# Which steps it helped  
    sampled\_perspectives: List\[str\]  \# Different perspectives used for sampling

### **5\. Online RL Rollout Sample**

@dataclass  
class RLRolloutSample:  
    rollout\_id: str  
    episode\_id: str

    \# Trajectory  
    steps: List\[RLStep\]

    \# Episode-level info  
    episode\_success: bool  
    episode\_reward: float  
    total\_memories\_written: int  
    budget\_exhausted: bool

    metadata: dict

@dataclass  
class RLStep:  
    step\_id: int  
    timestamp: float

    \# State  
    observation: str  
    local\_history: List\[str\]  
    memory\_store: List\[MemoryItem\]   \# With timestamps  
    budget\_remaining: int

    \# Constructor action  
    constructor\_output: MemoryItem   \# {write: bool, keys: \[...\], value: "...", timestamp: ...}  
    constructor\_log\_prob: float

    \# Agent query generation  
    agent\_query: str                 \# Compact sentence formatted by GPT-5  
    retrieved\_memories: List\[MemoryItem\]  \# Top-k retrieved (k=3-4)  
    retrieval\_scores: List\[float\]    \# Hybrid scores (BM25 \+ dense)

    \# Agent action  
    agent\_action: str  
    agent\_log\_prob: float

    \# Reward  
    step\_reward: float  
    auxiliary\_rewards: dict          \# {retrieval\_hit: ..., budget\_penalty: ..., compactness: ...}

    \# Value estimates  
    value: float  
    advantage: float

### **6\. Evaluation Record**

@dataclass  
class EvaluationRecord:  
    eval\_id: str  
    model\_checkpoint: str  
    dataset: str

    \# Episode-level metrics  
    episodes: List\[EpisodeEvaluation\]

    \# Aggregate metrics  
    task\_success\_rate: float  
    avg\_memories\_written: float  
    avg\_retrieval\_hit\_rate: float  
    avg\_memory\_redundancy: float  
    avg\_value\_length: float  
    budget\_exhaustion\_rate: float

    \# Baseline comparisons  
    baseline\_comparisons: dict

    metadata: dict

@dataclass  
class EpisodeEvaluation:  
    episode\_id: str  
    task: str  
    success: bool  
    total\_reward: float

    \# Memory statistics  
    num\_memories\_written: int  
    num\_no\_writes: int  
    write\_ratio: float

    \# Retrieval statistics  
    num\_retrievals: int  
    num\_retrieval\_hits: int  
    retrieval\_hit\_rate: float  
    avg\_retrieval\_score: float       \# Hybrid score

    \# Memory quality  
    avg\_num\_keys\_per\_memory: float  
    avg\_value\_length: float  
    memory\_redundancy: float  
    temporal\_distribution: dict      \# When memories were written (early/mid/late)

    \# Detailed trace  
    steps: List\[dict\]

---

## **Step 4: Offline Data Construction Pipeline**

### **Pipeline Overview**

The offline pipeline transforms static WebShop trajectories into training data for SFT and best-of-n stages.

**Input**: Raw WebShop trajectories (successful episodes)

**Output**: SFT training samples \+ best-of-n candidate lists

### **4.1 Hindsight Labeling Strategy (with Reward Model)**

For each step in a trajectory, we need to determine:

1. Should a memory be written? (write=true/false)  
2. If yes, what keys and value?  
3. How much does this memory contribute to the final result?

**Reward Model Approach** (recommended):

* Design a reward model to evaluate how much contribution one memory may contribute to the final result  
* **Option 1: LLM as a judge**:  
  * Prompt GPT-5 to evaluate: "Given this observation and the final task success, how useful would it be to remember this information?"  
  * Score: 0-1 (0 \= not useful, 1 \= critical)  
* **Option 2: Counterfactual evaluation**:  
  * Replay trajectory with and without this memory  
  * Contribution \= (success with memory) \- (success without memory)  
* **Option 3: Retrieval-based heuristic**:  
  * If information in observation is retrieved later → high contribution  
  * If information is never retrieved → low contribution

**Hindsight Heuristic** (simpler fallback):

* Look ahead in the trajectory to identify "future-relevant information"  
* If the current observation contains information that will be needed later, write a memory  
* Keys: Extract entities/attributes mentioned in current observation that appear in future observations  
* Value: Compress the relevant information from current observation

**Example** (from WebShop trajectory):

* Step 2: Observation \= "B09LSKQF8C \[SEP\] Superbox S3 Pro Dual Band Wi-Fi 2.4Ghz 5Ghz Supports 6K Video \[SEP\] $329.0"  
* Step 5: Agent needs to recall the price and features → Memory should have been written at step 2  
* Hindsight label:

{  
  "write": true,  
  "keys": \["price", "Superbox S3 Pro", "dual band", "6K video"\],  
  "value": "Superbox S3 Pro: dual band Wi-Fi (2.4Ghz/5Ghz), 6K video support, price $329.0",  
  "timestamp": 1773065531.05,  
  "step\_id": 2,  
  "contribution\_score": 0.9  \# High contribution (from reward model)  
}

* 

**Implementation**:

def label\_with\_reward\_model(trajectory, step\_id, llm\_judge):  
    observation \= trajectory\["trajectory"\]\[step\_id\]\["payload"\]\["raw\_text"\]  
    final\_success \= trajectory\["is\_success"\]  
    instruction \= trajectory\["instruction"\]

    \# Use LLM as judge  
    prompt \= f"""  
    Task: {instruction}  
    Current observation: {observation}  
    Final outcome: {"Success" if final\_success else "Failure"}

    Question: How useful would it be to remember information from this observation for completing the task?  
    Rate from 0 (not useful) to 1 (critical).  
    Also suggest what keys and value to write.  
    """

    response \= llm\_judge.generate(prompt)  
    contribution\_score \= parse\_score(response)  
    keys, value \= parse\_memory(response)

    return {  
        "write": contribution\_score \> 0.3,  \# Threshold  
        "keys": keys,  
        "value": value,  
        "contribution\_score": contribution\_score,  
    }

### **4.2 Candidate Sampling**

For best-of-n training, we need to generate multiple candidate memories per step.

**Sampling Strategy**:

1. Use the SFT-trained constructor to generate N candidates (N=4-8)  
2. Use nucleus sampling (top-p=0.9) for diversity  
3. Always include no\_write as one candidate  
4. Ensure all candidates are valid JSON

**Implementation**:

def sample\_candidates(constructor, observation, history, memory\_store, N=4):  
    candidates \= \[\]

    \# Always include no\_write  
    candidates.append({  
        "write": False,  
        "keys": \[\],  
        "value": ""  
    })

    \# Sample N-1 write candidates  
    for i in range(N-1):  
        candidate \= constructor.generate(  
            observation, history, memory\_store,  
            temperature=0.9,  
            top\_p=0.9,  
            do\_sample=True  
        )  
        candidates.append(candidate)

    return candidates

### **4.3 Candidate Scoring (Weighted Merge of Hindsight and Counterfactual)**

For each candidate, compute multiple scores using both hindsight and counterfactual evaluation:

**1\. Hindsight Score** (what was actually retrieved):

* Replay the trajectory with this candidate memory  
* Track when this memory was actually retrieved by the agent  
* Score \= (\# actual retrieval hits) / (\# total queries)

**2\. Counterfactual Score** (what would have been retrieved):

* For each future step, simulate retrieval with this candidate  
* Use the hybrid retriever (BM25 \+ dense) to compute retrieval scores  
* Score \= average retrieval score across all future queries  
* **Key insight**: This makes rewards more dense by evaluating potential usefulness even if not retrieved

**3\. Sampling from Different Perspectives**:

To make counterfactual evaluation more robust, sample memory slots from different perspectives:

* **Temporal perspective**: Early, middle, late in episode  
* **Content perspective**: Different aspects of observation (price, features, constraints)  
* **Granularity perspective**: Detailed vs. compressed representations

**4\. Task Success Score** (expensive, use sparingly or optimize later):

* Run mini-rollout with fixed agent using this candidate  
* Compare task success with/without this memory  
* Score \= success\_delta

**5\. Compactness Score**:

* Score \= 1 \- (value\_length / max\_value\_tokens)  
* Penalize verbose memories

**6\. Redundancy Score**:

* Compute overlap with existing memories in store  
* Use embedding similarity or token overlap  
* Score \= 1 \- max\_similarity\_to\_existing

**7\. Faithfulness Score**:

* Check if value is grounded in observation  
* Use NLI model or embedding similarity  
* Score \= similarity(value, observation)

**Combined Score** (weighted merge):

\# Merge hindsight and counterfactual  
retrieval\_usefulness \= (  
    0.5 \* hindsight\_score \+  
    0.5 \* counterfactual\_score  
)

\# Final score  
total\_score \= (  
    0.4 \* retrieval\_usefulness \+  
    0.3 \* task\_success\_score \+      \# Optimize later if too expensive  
    0.1 \* compactness\_score \+  
    0.1 \* redundancy\_score \+  
    0.1 \* faithfulness\_score  
)

**Special case for no\_write**:

* Use overall reward directly  
* If it saves budget (increases budget-saving reward) and doesn't decrease utility much, it will naturally score higher  
* hindsight\_score \= 0 (no memory added)  
* counterfactual\_score \= 0 (no potential retrieval)  
* task\_success\_score \= baseline (no change)  
* compactness\_score \= 1.0 (saves budget)  
* redundancy\_score \= 1.0 (no redundancy)  
* faithfulness\_score \= 1.0 (no hallucination risk)

### **4.4 Best-of-N Selection**

Select the candidate with highest total\_score as the target for training.

**Important**: If no\_write has the highest score, it should be selected. This teaches the constructor when NOT to write.

### **4.5 Preference Pair Derivation (Optional)**

From best-of-n results, create preference pairs for DPO-style training:

* Chosen: Best candidate  
* Rejected: Worst candidate (or random lower-scored candidate)

This can be used as an alternative to best-of-n SFT.

---

## **Step 5: Minimal v1 Implementation Plan**

### **Phase 1: Core Infrastructure (Week 1\)**

**Goal**: Set up project structure and core components

1. **Project setup**:  
   * Create directory structure  
   * Set up configs (model.yaml, training.yaml, data.yaml)  
   * Install dependencies (OpenRLHF, transformers, datasets, etc.)  
2. **Data schemas** (`data/schemas.py`):  
   * Implement all dataclasses from Step 3  
   * Add validation methods  
   * Add serialization/deserialization  
3. **Memory store** (`memory_constructor/environment/memory_store.py`):  
   * Implement append-only store  
   * Add budget tracking  
   * Add serialization for checkpointing  
4. **JSON utilities** (`memory_constructor/utils/json_utils.py`):  
   * Robust JSON parsing with fallback  
   * Validation against schema  
   * Truncation enforcement

### **Phase 2: Fixed Components (Week 1-2)**

**Goal**: Implement fixed retriever and agent wrappers

1. **Fixed retriever** (`memory_constructor/models/retriever.py`):  
   * Start with simple BM25 retriever (using rank\_bm25 library)  
   * Input: query string \+ memory store  
   * Output: top-k memories with scores  
   * Log retrieval statistics  
2. **Fixed agent wrapper** (`memory_constructor/models/agent.py`):  
   * Load a pre-trained model (e.g., Qwen2-7B fine-tuned on WebShop)  
   * Format input: observation \+ retrieved memories  
   * Generate action  
   * **Important**: Verify agent actually uses memories (run ablation)  
3. **WebShop environment** (`memory_constructor/environment/webshop_env.py`):  
   * Wrapper around WebShop gym environment  
   * Add memory store integration  
   * Add retrieval step before agent action  
   * Log full trajectories

### **Phase 3: Data Pipeline (Week 2\)**

**Goal**: Process WebShop data into training samples

1. **WebShop parser** (`memory_constructor/data/webshop_parser.py`):  
   * Load raw WebShop trajectories  
   * Parse into WebShopTrajectory dataclass  
   * Filter for successful episodes only  
   * Create train/val/test splits  
2. **Hindsight labeling** (`memory_constructor/data/webshop_parser.py`):  
   * Implement simple heuristic: identify future-relevant info  
   * Generate target memories for each step  
   * Create SFTSample dataclass instances  
3. **SFT dataset** (`memory_constructor/data/sft_dataset.py`):  
   * Implement PyTorch Dataset  
   * Format prompts for constructor  
   * Tokenize inputs and targets  
   * Collate function for batching

### **Phase 4: SFT Training (Week 2-3)**

**Goal**: Train initial constructor with supervised learning

1. **Constructor model** (`memory_constructor/models/constructor.py`):  
   * Wrap Qwen2.5-9B with OpenRLHF Actor  
   * Implement prompt formatting  
   * Implement JSON parsing and validation  
   * Add truncation enforcement  
2. **SFT trainer** (`memory_constructor/training/sft_trainer.py`):  
   * Use OpenRLHF's SFTTrainer  
   * Standard language modeling loss on target JSON  
   * Log training metrics  
   * Save checkpoints  
3. **Training script** (`scripts/02_train_sft.py`):  
   * Load config  
   * Initialize model, dataset, trainer  
   * Run training loop  
   * Evaluate on validation set

### **Phase 5: Candidate Sampling & Scoring (Week 3-4)**

**Goal**: Generate and score candidate memories

1. **Candidate sampler** (`memory_constructor/training/candidate_sampler.py`):  
   * Load SFT-trained constructor  
   * Generate N candidates per sample  
   * Use vLLM for fast batch inference  
   * Always include no\_write candidate  
2. **Candidate scorer** (`memory_constructor/training/candidate_scorer.py`):  
   * Implement retrieval simulation  
   * Compute all 5 scores (retrieval, task success, compactness, redundancy, faithfulness)  
   * Combine scores with weights  
   * Select best candidate  
3. **Candidate dataset** (`memory_constructor/data/candidate_dataset.py`):  
   * Store candidates with scores  
   * Create dataset for best-of-n training  
   * Target \= best candidate  
4. **Generation script** (`scripts/03_generate_candidates.py`):  
   * Load SFT model  
   * Generate candidates for all training samples  
   * Save to disk  
5. **Scoring script** (`scripts/04_score_candidates.py`):  
   * Load candidates  
   * Score each candidate  
   * Select best-of-n  
   * Save scored dataset

### **Phase 6: Best-of-N Training (Week 4\)**

**Goal**: Train constructor on best-of-n targets

1. **Best-of-N trainer** (`memory_constructor/training/bestofn_trainer.py`):  
   * Similar to SFT trainer  
   * Train on best-of-n selected targets  
   * Monitor improvement over SFT baseline  
2. **Training script** (`scripts/05_train_bestofn.py`):  
   * Load best-of-n dataset  
   * Train constructor  
   * Evaluate

### **Phase 7: Evaluation (Week 4-5)**

**Goal**: Comprehensive evaluation framework

1. **Metrics** (`memory_constructor/evaluation/metrics.py`):  
   * Write/no\_write accuracy  
   * Retrieval hit rate  
   * Task success rate  
   * Memory redundancy  
   * Value length statistics  
   * Budget usage  
2. **Baselines** (`memory_constructor/evaluation/baselines.py`):  
   * No memory baseline  
   * Always write baseline  
   * Random write baseline  
   * Heuristic memory (rule-based)  
3. **Evaluator** (`memory_constructor/evaluation/evaluator.py`):  
   * Run full episodes with constructor  
   * Collect all metrics  
   * Compare against baselines  
   * Generate evaluation report  
4. **Evaluation script** (`scripts/07_evaluate.py`):  
   * Load trained model  
   * Run evaluation  
   * Generate plots and tables

---

## **Step 6: OpenRLHF Integration**

### **Key Insight: Use Multi-Turn Agentic Training**

OpenRLHF provides `AgentInstanceBase` and `MultiTurnAgentExecutor` for multi-turn agent training. This is perfect for our memory constructor because:

1. **Unified interface** for both static and dynamic trajectories  
2. **Multi-turn support** \- constructor makes decisions at each step  
3. **Easy extension** \- same code works for offline (static) and online (dynamic) training  
4. **Async support** \- can generate trajectories asynchronously for better throughput

### **Architecture with AgentInstance**

from openrlhf.utils.agent import AgentInstanceBase, MultiTurnAgentExecutor

class MemoryConstructorAgentInstance(AgentInstanceBase):  
    """  
    Agent instance for memory constructor training.  
    Handles both static (replay) and dynamic (online) trajectories.  
    """

    def \_\_init\_\_(self, config):  
        super().\_\_init\_\_()  
        self.config \= config  
        self.fixed\_agent \= load\_gpt5\_agent(config.api\_key)  
        self.retriever \= HybridRetriever(config)  
        self.memory\_store \= None  
        self.trajectory\_data \= None  \# For static trajectories  
        self.current\_step \= 0

    async def reset(self, states: dict, \*\*kwargs):  
        """  
        Initialize episode.

        For static trajectories:  
            states \= {"trajectory\_id": "...", "trajectory\_data": \[...\]}

        For dynamic trajectories:  
            states \= {"task": "...", "webshop\_env": env}  
        """  
        self.memory\_store \= MemoryStore(self.config.episode\_memory\_budget)  
        self.current\_step \= 0

        \# Check if static or dynamic  
        if "trajectory\_data" in states:  
            \# Static trajectory replay  
            self.mode \= "static"  
            self.trajectory\_data \= states\["trajectory\_data"\]  
            observation \= self.trajectory\_data\[0\]\["observation"\]  
        else:  
            \# Dynamic trajectory generation  
            self.mode \= "dynamic"  
            self.webshop\_env \= states\["webshop\_env"\]  
            observation \= self.webshop\_env.reset()

        return {  
            "observation": self.\_format\_observation(observation),  
            "memory\_store": self.memory\_store.to\_dict(),  
            "budget\_remaining": self.memory\_store.get\_budget\_remaining(),  
            "episode\_progress": 0.0,  
        }

    async def step(self, states: dict, \*\*kwargs):  
        """  
        Execute one step of memory constructor \+ agent.

        states contains:  
            \- observation\_text: Current observation  
            \- action\_text: Constructor output (JSON string)  
        """  
        observation\_text \= states\["observation\_text"\]  
        action\_text \= states\["action\_text"\]

        \# 1\. Parse constructor action  
        try:  
            constructor\_action \= json.loads(action\_text)  
        except:  
            constructor\_action \= {"write": False, "keys": \[\], "value": ""}

        \# 2\. Update memory store  
        if constructor\_action.get("write", False):  
            self.memory\_store.append(constructor\_action)

        \# 3\. Agent queries retriever  
        agent\_query \= self.\_generate\_agent\_query(observation\_text)  
        retrieved\_memories \= self.retriever.retrieve(agent\_query, self.memory\_store)

        \# 4\. Agent acts  
        if self.mode \== "static":  
            \# Use ground-truth action from trajectory  
            agent\_action \= self.trajectory\_data\[self.current\_step\]\["action"\]  
            next\_observation \= self.trajectory\_data\[self.current\_step \+ 1\]\["observation"\]  
            step\_reward \= self.trajectory\_data\[self.current\_step\]\["reward"\]  
            done \= self.trajectory\_data\[self.current\_step\]\["done"\]  
        else:  
            \# Dynamic: agent generates action  
            agent\_action \= self.fixed\_agent.act(observation\_text, retrieved\_memories)  
            next\_observation, step\_reward, done, info \= self.webshop\_env.step(agent\_action)

        \# 5\. Compute rewards  
        task\_reward \= step\_reward  
        auxiliary\_rewards \= self.\_compute\_auxiliary\_rewards(  
            constructor\_action, retrieved\_memories, agent\_query  
        )

        total\_reward \= (  
            task\_reward \+  
            self.config.retrieval\_hit\_weight \* auxiliary\_rewards\["retrieval\_hit"\] \+  
            self.config.budget\_penalty\_weight \* auxiliary\_rewards\["budget\_penalty"\] \+  
            self.config.compactness\_weight \* auxiliary\_rewards\["compactness"\]  
        )

        \# 6\. Update state  
        self.current\_step \+= 1  
        episode\_progress \= self.current\_step / len(self.trajectory\_data) if self.mode \== "static" else 0.0

        \# 7\. Prepare next observation  
        next\_states \= {  
            "observation": self.\_format\_observation(next\_observation),  
            "memory\_store": self.memory\_store.to\_dict(),  
            "budget\_remaining": self.memory\_store.get\_budget\_remaining(),  
            "episode\_progress": episode\_progress,  
        }

        return {  
            "rewards": total\_reward,  
            "scores": torch.sigmoid(torch.tensor(total\_reward)).item(),  
            "environment\_feedback": next\_states\["observation"\],  
            "done": done,  
            "sampling\_params": None,  \# Use default  
            "extra\_logs": {  
                "task\_reward": task\_reward,  
                "retrieval\_hit": auxiliary\_rewards\["retrieval\_hit"\],  
                "budget\_penalty": auxiliary\_rewards\["budget\_penalty"\],  
                "compactness": auxiliary\_rewards\["compactness"\],  
                "write\_action": constructor\_action.get("write", False),  
                "num\_memories": len(self.memory\_store.memories),  
            },  
        }

    def \_compute\_auxiliary\_rewards(self, constructor\_action, retrieved\_memories, agent\_query):  
        rewards \= {}

        \# Retrieval hit reward  
        if constructor\_action.get("write", False):  
            rewards\["retrieval\_hit"\] \= 0.1 if len(retrieved\_memories) \> 0 else 0.0  
        else:  
            rewards\["retrieval\_hit"\] \= 0.0

        \# Budget penalty  
        budget\_remaining \= self.memory\_store.get\_budget\_remaining()  
        if budget\_remaining \== 0:  
            rewards\["budget\_penalty"\] \= \-0.5  
        else:  
            rewards\["budget\_penalty"\] \= 0.0

        \# Compactness reward  
        if constructor\_action.get("write", False):  
            value\_length \= len(constructor\_action\["value"\].split())  
            rewards\["compactness"\] \= 0.05 \* (1 \- value\_length / self.config.max\_value\_tokens)  
        else:  
            rewards\["compactness"\] \= 0.0

        return rewards

class MemoryConstructorExecutor(MultiTurnAgentExecutor):  
    """Executor for memory constructor agent."""

    def \_\_init\_\_(self, config):  
        super().\_\_init\_\_(MemoryConstructorAgentInstance, config)

### **SFT Training with AgentInstance**

For SFT, we use static trajectories with hindsight labels:

from openrlhf.trainer import SFTTrainer  
from openrlhf.models import Actor  
from openrlhf.utils import get\_strategy, get\_tokenizer

\# Setup  
args \= parse\_args()  
strategy \= get\_strategy(args)  
strategy.setup\_distributed()

\# Model  
model \= Actor(  
    pretrain\_or\_model="Qwen/Qwen2.5-9B",  
    attn\_implementation="flash\_attention\_2",  
    param\_dtype="bf16",  
    lora\_rank=0,  \# Full fine-tuning  
    ds\_config=strategy.get\_ds\_train\_config(is\_actor=True),  
)

\# Tokenizer  
tokenizer \= get\_tokenizer("Qwen/Qwen2.5-9B", model.model, "right", strategy)

\# Dataset: Format as multi-turn conversations  
\# Each sample is a trajectory with multiple turns  
train\_dataset \= MemoryConstructorSFTDataset(  
    dataset=load\_hindsight\_labeled\_trajectories(),  
    tokenizer=tokenizer,  
    max\_length=2048,  
    strategy=strategy,  
    multiturn=True,  \# Enable multi-turn mode  
)

\# Dataloader  
train\_dataloader \= strategy.setup\_dataloader(  
    train\_dataset,  
    args.micro\_train\_batch\_size,  
    True,  \# shuffle  
    True,  \# drop\_last  
    train\_dataset.collate\_fn,  
)

\# Optimizer & Scheduler  
optim \= strategy.create\_optimizer(model, lr=args.learning\_rate)  
scheduler \= get\_scheduler("cosine", optim, num\_warmup\_steps=100, num\_training\_steps=1000)

\# Trainer  
trainer \= SFTTrainer(  
    model=model,  
    strategy=strategy,  
    optim=optim,  
    train\_dataloader=train\_dataloader,  
    eval\_dataloader=eval\_dataloader,  
    scheduler=scheduler,  
    max\_epochs=args.max\_epochs,  
)

\# Train  
trainer.fit(args, consumed\_samples=0, num\_update\_steps\_per\_epoch=len(train\_dataloader))

### **Best-of-N Training with AgentInstance**

For best-of-n, we still use static trajectories but with candidate sampling:

**Step 1: Generate candidates using vLLM**

from vllm import LLM, SamplingParams

\# Load SFT model  
llm \= LLM(  
    model="path/to/sft\_checkpoint",  
    tensor\_parallel\_size=2,  
    gpu\_memory\_utilization=0.9,  
)

\# For each trajectory step, generate N candidates  
sampling\_params \= SamplingParams(  
    max\_tokens=512,  
    temperature=0.9,  
    top\_p=0.9,  
    n=4,  \# Generate 4 candidates per prompt  
)

\# Generate  
prompts \= \[format\_prompt(sample) for sample in dataset\]  
outputs \= llm.generate(prompts, sampling\_params)

\# Parse candidates  
candidates\_per\_sample \= \[\]  
for output in outputs:  
    candidates \= \[parse\_json(gen.text) for gen in output.outputs\]  
    \# Always add no\_write as a candidate  
    candidates.append({"write": False, "keys": \[\], "value": ""})  
    candidates\_per\_sample.append(candidates)

**Step 2: Score candidates using AgentInstance**

\# For each candidate, replay the trajectory and compute score  
executor \= MemoryConstructorExecutor(config)

for sample\_idx, candidates in enumerate(candidates\_per\_sample):  
    trajectory\_data \= dataset\[sample\_idx\]

    candidate\_scores \= \[\]  
    for candidate in candidates:  
        \# Reset agent instance with static trajectory  
        states \= {  
            "trajectory\_id": trajectory\_data\["trajectory\_id"\],  
            "trajectory\_data": trajectory\_data\["steps"\],  
        }  
        await executor.reset(states)

        \# Replay trajectory with this candidate  
        total\_score \= 0  
        for step in trajectory\_data\["steps"\]:  
            result \= await executor.step({  
                "observation\_text": step\["observation"\],  
                "action\_text": json.dumps(candidate),  
            })  
            total\_score \+= result\["rewards"\]

        candidate\_scores.append(total\_score)

    \# Select best candidate  
    best\_idx \= np.argmax(candidate\_scores)  
    best\_candidate \= candidates\[best\_idx\]

    \# Save for training  
    save\_bestofn\_sample(sample\_idx, best\_candidate, candidate\_scores)

**Step 3: Train on best-of-n targets**

Same as SFT, but use best-of-n selected targets.

### **RL Training with AgentInstance**

For RL, we use the same AgentInstance but with dynamic trajectory generation:

from openrlhf.trainer import PPOTrainer  
from openrlhf.trainer.ray import RayActorGroup

\# Initialize agent executor  
agent\_executor \= MemoryConstructorExecutor(config)

\# Custom reward function that uses AgentInstance  
def memory\_constructor\_reward\_func(queries, prompts, labels, \*\*kwargs):  
    \# This is called by PPO trainer for each rollout  
    \# queries: Full sequences (prompt \+ constructor output)  
    \# prompts: Input prompts

    batch\_size \= len(queries)  
    rewards \= \[\]

    for i in range(batch\_size):  
        \# Parse constructor output  
        constructor\_output \= queries\[i\]\[len(prompts\[i\]):\]

        \# Use AgentInstance to compute reward  
        result \= await agent\_executor.step({  
            "observation\_text": prompts\[i\],  
            "action\_text": constructor\_output,  
        })

        rewards.append(result\["rewards"\])

    return {  
        "rewards": torch.tensor(rewards),  
        "scores": torch.sigmoid(torch.tensor(rewards)),  
        "extra\_logs": {"avg\_reward": np.mean(rewards)},  
    }

\# Initialize Ray actors  
actor\_model \= RayActorGroup(...)  
critic\_model \= RayActorGroup(...)  
reward\_model \= None  \# Use custom reward function  
ref\_model \= RayActorGroup(...)

\# PPO Trainer  
ppo\_trainer \= PPOTrainer.remote(  
    "path/to/bestofn\_checkpoint",  
    strategy,  
    actor\_model,  
    critic\_model,  
    reward\_model,  
    ref\_model,  
    vllm\_engines,  
    reward\_fn=memory\_constructor\_reward\_func,  \# Custom reward using AgentInstance  
    n\_samples\_per\_prompt=1,  
)

\# Training loop  
for epoch in range(args.max\_epochs):  
    for rollout\_samples in rollout\_dataloader:  
        status, global\_step \= ray.get(ppo\_trainer.train\_step.remote(rollout\_samples, global\_step))

### **Benefits of AgentInstance Approach**

1. **Unified interface**: Same code for static and dynamic trajectories  
2. **Easy debugging**: Can replay any trajectory step-by-step  
3. **Flexible reward**: Can easily add/modify auxiliary rewards  
4. **Async support**: Can generate trajectories asynchronously for better throughput  
5. **Future-proof**: Easy to extend to online RL with dynamic trajectory generation

---

## **Step 7: RL Stage Design**

### **7.1 RL Environment**

**Custom WebShop RL Environment** (`memory_constructor/environment/webshop_env.py`):

class MemoryConstructorEnv:  
    def \_\_init\_\_(self, webshop\_env, fixed\_agent, fixed\_retriever, config):  
        self.webshop \= webshop\_env  
        self.agent \= fixed\_agent  
        self.retriever \= fixed\_retriever  
        self.memory\_store \= MemoryStore(config.episode\_memory\_budget)  
        self.config \= config

    def reset(self):  
        obs \= self.webshop.reset()  
        self.memory\_store.clear()  
        self.history \= \[\]  
        return self.\_format\_state(obs)

    def step(self, constructor\_action):  
        \# Constructor action: {write: bool, keys: \[...\], value: "..."}

        \# 1\. Update memory store  
        if constructor\_action\["write"\]:  
            self.memory\_store.append(constructor\_action)

        \# 2\. Agent queries retriever  
        agent\_query \= self.\_generate\_agent\_query()  
        retrieved\_memories \= self.retriever.retrieve(agent\_query, self.memory\_store)

        \# 3\. Agent acts  
        agent\_action \= self.agent.act(self.current\_obs, retrieved\_memories)

        \# 4\. Environment step  
        next\_obs, reward, done, info \= self.webshop.step(agent\_action)

        \# 5\. Compute auxiliary rewards  
        auxiliary\_rewards \= self.\_compute\_auxiliary\_rewards(  
            constructor\_action, retrieved\_memories, reward  
        )

        \# 6\. Update history  
        self.history.append({  
            "obs": self.current\_obs,  
            "constructor\_action": constructor\_action,  
            "agent\_action": agent\_action,  
            "reward": reward,  
        })

        self.current\_obs \= next\_obs

        return self.\_format\_state(next\_obs), reward, done, {  
            "auxiliary\_rewards": auxiliary\_rewards,  
            \*\*info  
        }

    def \_compute\_auxiliary\_rewards(self, constructor\_action, retrieved\_memories, task\_reward):  
        rewards \= {}

        \# Retrieval hit reward  
        if constructor\_action\["write"\]:  
            \# Check if any memory was retrieved  
            rewards\["retrieval\_hit"\] \= 0.1 if len(retrieved\_memories) \> 0 else 0.0  
        else:  
            rewards\["retrieval\_hit"\] \= 0.0

        \# Computational cost penalty (for writing)  
        if constructor\_action\["write"\]:  
            rewards\["write\_cost"\] \= \-0.05  \# Cost penalty  
        else:  
            rewards\["write\_cost"\] \= 0.0

        \# Compactness reward  
        if constructor\_action\["write"\]:  
            value\_length \= len(constructor\_action\["value"\].split())  
            rewards\["compactness"\] \= 0.02 \* (1 \- value\_length / self.config.max\_value\_tokens)  
        else:  
            rewards\["compactness"\] \= 0.0

        return rewards

### **7.2 Reward Function (with Computational Cost Penalty)**

**Total Reward** \= Task Reward \+ Auxiliary Rewards \- Computational Cost

def compute\_total\_reward(task\_reward, constructor\_action, retrieved\_memories, config):  
    total \= task\_reward  \# Main signal (0.0 or 1.0 for WebShop)

    \# Auxiliary rewards  
    auxiliary \= 0.0

    \# 1\. Retrieval hit reward (if memory was useful)  
    if constructor\_action.get("write", False):  
        \# Check if any memory was retrieved in this or future steps  
        if len(retrieved\_memories) \> 0:  
            auxiliary \+= config.retrieval\_hit\_weight \* 0.1  
        else:  
            auxiliary \+= 0.0  
    else:  
        auxiliary \+= 0.0

    \# 2\. Computational cost penalty (for writing)  
    if constructor\_action.get("write", False):  
        \# Writing a memory has a computational cost  
        cost\_penalty \= config.write\_cost\_penalty \* 0.05  
        auxiliary \-= cost\_penalty  
    else:  
        \# No cost for no\_write  
        auxiliary \+= 0.0

    \# 3\. Compactness reward (encourage compression)  
    if constructor\_action.get("write", False):  
        value\_length \= len(constructor\_action\["value"\].split())  
        compactness \= 1 \- (value\_length / config.max\_value\_tokens)  
        auxiliary \+= config.compactness\_weight \* 0.02 \* compactness  
    else:  
        auxiliary \+= 0.0

    \# 4\. Distillation signal (from LLM-as-judge)  
    if "distillation\_score" in constructor\_action:  
        auxiliary \+= config.distillation\_weight \* 0.1 \* constructor\_action\["distillation\_score"\]

    total \+= auxiliary

    return total, {  
        "task\_reward": task\_reward,  
        "retrieval\_hit": auxiliary if constructor\_action.get("write") else 0.0,  
        "write\_cost": \-cost\_penalty if constructor\_action.get("write") else 0.0,  
        "compactness": compactness if constructor\_action.get("write") else 0.0,  
        "distillation": constructor\_action.get("distillation\_score", 0.0),  
    }

**Recommended weights** (v1):

* retrieval\_hit\_weight: 0.1 (reward for useful memories)  
* write\_cost\_penalty: 0.05 (cost for writing)  
* compactness\_weight: 0.02 (encourage compression)  
* distillation\_weight: 0.1 (LLM-as-judge signal)

**Key insight**: If a memory is useful (high retrieval hit), the reward (0.1) is much larger than the cost (0.05), so the constructor learns to write useful memories. If a memory is not useful, the cost penalty dominates, so the constructor learns to skip writing.

### **7.3 RL Training with OpenRLHF**

**Use PPO with custom reward function**:

from openrlhf.trainer import PPOTrainer  
from openrlhf.trainer.ray import RayActorGroup

\# Custom reward function  
def memory\_constructor\_reward\_func(queries, prompts, labels, \*\*kwargs):  
    \# queries: Full sequences (prompt \+ constructor output)  
    \# prompts: Input prompts  
    \# labels: Not used for RL  
    \# kwargs: Contains auxiliary\_rewards from environment

    batch\_size \= len(queries)  
    rewards \= \[\]

    for i in range(batch\_size):  
        \# Parse constructor output  
        constructor\_output \= parse\_json(queries\[i\]\[len(prompts\[i\]):\])

        \# Run environment step  
        env\_reward, auxiliary\_rewards \= run\_env\_step(constructor\_output)

        \# Compute total reward  
        total\_reward \= compute\_total\_reward(env\_reward, auxiliary\_rewards, config)  
        rewards.append(total\_reward)

    return {  
        "rewards": torch.tensor(rewards),  
        "scores": torch.sigmoid(torch.tensor(rewards)),  
        "extra\_logs": {"avg\_reward": np.mean(rewards)},  
    }

\# Initialize Ray actors  
actor\_model \= RayActorGroup(...)  
critic\_model \= RayActorGroup(...)  
reward\_model \= None  \# Use custom reward function instead  
ref\_model \= RayActorGroup(...)

\# PPO Trainer  
ppo\_trainer \= PPOTrainer.remote(  
    "path/to/bestofn\_checkpoint",  
    strategy,  
    actor\_model,  
    critic\_model,  
    reward\_model,  
    ref\_model,  
    vllm\_engines,  
    reward\_fn=memory\_constructor\_reward\_func,  \# Custom reward  
    n\_samples\_per\_prompt=1,  \# No best-of-n during RL  
)

\# Training loop  
for epoch in range(args.max\_epochs):  
    for rollout\_samples in rollout\_dataloader:  
        status, global\_step \= ray.get(ppo\_trainer.train\_step.remote(rollout\_samples, global\_step))

### **7.4 Debugging RL Training**

**Key metrics to monitor**:

1. Average episode reward  
2. Constructor write rate (% of steps with write=true)  
3. Retrieval hit rate  
4. Budget exhaustion rate  
5. Policy entropy (ensure exploration)  
6. KL divergence from reference policy  
7. Value function loss

**Common failure modes**:

* Constructor always writes → increase budget penalty  
* Constructor never writes → increase retrieval hit reward  
* Policy collapse → increase KL penalty, reduce learning rate  
* Reward hacking → check auxiliary reward weights

---

## **Step 8: Evaluation Design**

### **8.1 Evaluation Metrics**

**Task-Level Metrics**:

* Task success rate  
* Average episode reward  
* Average episode length

**Memory-Level Metrics**:

* Write rate (% steps with write=true)  
* Average memories per episode  
* Budget exhaustion rate  
* Average value length  
* Average number of keys per memory

**Retrieval-Level Metrics**:

* Retrieval hit rate (% queries with ≥1 retrieved memory)  
* Retrieval precision (% retrieved memories actually used by agent)  
* Average retrieval score

**Compression-Level Metrics**:

* Compression ratio (value length / observation length)  
* ROUGE-L between value and observation  
* Semantic similarity (embedding distance)

**Redundancy Metrics**:

* Average pairwise similarity between memories  
* % of memories never retrieved

### **8.2 Baselines**

1. **No Memory**: Agent acts without any memory  
2. **Always Write**: Write full observation at every step  
3. **Random Write**: Randomly decide to write with p=0.5  
4. **Heuristic Memory**: Rule-based (e.g., write when seeing product info)  
5. **Oracle Hindsight**: Use ground-truth hindsight labels

### **8.3 Ablations**

1. **max\_value\_tokens**: \[64, 128, 256, 512, 1024, 2048\]  
2. **max\_num\_keys**: \[1, 2, 4, 8\]  
3. **episode\_memory\_budget**: \[5, 10, 20, 50, unlimited\]  
4. **Number of candidates N**: \[1, 2, 4, 8, 16\]  
5. **Retrieval method**: \[BM25, dense, hybrid\]  
6. **Auxiliary reward weights**: Grid search

### **8.4 Evaluation Protocol**

**Test Set**: Hold out 20% of WebShop trajectories

**Evaluation Procedure**:

1. Load trained constructor  
2. For each test episode:  
   * Reset environment  
   * At each step:  
     * Constructor generates memory action  
     * Agent queries retriever  
     * Agent acts  
     * Log all statistics  
   * Compute episode metrics  
3. Aggregate across all episodes  
4. Compare against baselines  
5. Generate plots and tables

**Qualitative Analysis**:

* Inspect top-10 best episodes: What memories were written?  
* Inspect top-10 worst episodes: What went wrong?  
* Visualize memory timeline: When were memories written/retrieved?  
* Check for failure modes: Copy-paste, never-retrieved memories, budget exhaustion

---

## **Implementation Timeline**

### **Week 1: Infrastructure**

* Project setup  
* Data schemas  
* Memory store  
* JSON utilities  
* Fixed retriever (BM25)  
* Fixed agent wrapper

### **Week 2: Data Pipeline**

* WebShop parser  
* Hindsight labeling  
* SFT dataset  
* Data preprocessing script

### **Week 3: SFT Training**

* Constructor model  
* SFT trainer  
* Training script  
* Initial evaluation

### **Week 4: Best-of-N**

* Candidate sampler  
* Candidate scorer  
* Best-of-N dataset  
* Best-of-N training  
* Evaluation

### **Week 5: RL (if time permits)**

* RL environment  
* Custom reward function  
* RL training script  
* Final evaluation

---

## **Critical Files to Modify/Create**

### **New Files to Create:**

**Core AgentInstance Architecture**:

1. `memory_constructor/environment/agent_instance.py` \- MemoryConstructorAgentInstance (unified static/dynamic interface)  
2. `memory_constructor/environment/memory_store.py` \- Append-only memory store  
3. `memory_constructor/models/retriever.py` \- Hybrid retriever (BM25 \+ dense)  
4. `memory_constructor/models/agent.py` \- GPT-5 agent wrapper

**Data Processing**:  
5\. `memory_constructor/data/webshop_parser.py` \- Parse WebShop trajectories  
6\. `memory_constructor/data/sft_dataset.py` \- SFT dataset (multi-turn format)  
7\. `memory_constructor/data/candidate_dataset.py` \- Best-of-n dataset

8\. `data/schemas.py` \- All dataclass definitions

**Training**:  
9\. `memory_constructor/training/candidate_sampler.py` \- Generate candidates with vLLM  
10\. `memory_constructor/training/candidate_scorer.py` \- Score candidates using AgentInstance  
11\. `memory_constructor/training/sft_trainer.py` \- SFT training wrapper

12\. `memory_constructor/training/rl_trainer.py` \- RL training wrapper

**Evaluation**:  
13\. `memory_constructor/evaluation/metrics.py` \- All evaluation metrics  
14\. `memory_constructor/evaluation/evaluator.py` \- Evaluation framework using AgentInstance

15\. `memory_constructor/evaluation/baselines.py` \- Baseline implementations

**Scripts**:  
16\. `scripts/01_preprocess_webshop.py` \- Parse and preprocess trajectories  
17\. `scripts/02_train_sft.py` \- SFT training script  
18\. `scripts/03_generate_candidates.py` \- Generate candidates with vLLM  
19\. `scripts/04_score_candidates.py` \- Score candidates using AgentInstance  
20\. `scripts/05_train_bestofn.py` \- Best-of-N training script  
21\. `scripts/06_train_rl.py` \- RL training script (uses AgentInstance)

22\. `scripts/07_evaluate.py` \- Evaluation script

**Utilities**:  
23\. `memory_constructor/utils/json_utils.py` \- JSON parsing/validation  
24\. `memory_constructor/utils/logging_utils.py` \- Logging and visualization

25\. `memory_constructor/utils/openrlhf_utils.py` \- OpenRLHF integration helpers

### **Existing Files to Inspect (Read-Only):**

* `/home/yuhao/code/WebShop_VanillaReactAgent_TrajectoryGeneration/` \- Understand trajectory format  
* `/home/yuhao/code/.env` \- Load GPT-5 API key  
* OpenRLHF's `AgentInstanceBase`, `MultiTurnAgentExecutor` \- Use as base classes

### **Existing OpenRLHF Files (No Modification Needed):**

* Use OpenRLHF's `SFTTrainer`, `PPOTrainer`, `Actor`, `get_strategy`, etc. as-is  
* Integrate via Python imports, not code modification

---

## **Verification Plan**

### **After SFT Training:**

1. Generate samples from constructor \- check JSON validity  
2. Check write rate \- should be reasonable (20-50%)  
3. Check value lengths \- should respect max\_value\_tokens  
4. Run evaluation \- compare against no-memory baseline

### **After Best-of-N Training:**

1. Compare task success vs. SFT baseline  
2. Check retrieval hit rate improvement  
3. Analyze candidate diversity  
4. Verify best-of-n selection improves over random selection

### **After RL Training:**

1. Check policy doesn't collapse (entropy \> threshold)  
2. Verify reward increases over training  
3. Compare against best-of-n baseline  
4. Run full ablation suite

---

## **User-Specific Configuration**

Based on your setup:

1. **WebShop Data**: Use trajectory generation repo at `/home/yuhao/code/WebShop_VanillaReactAgent_TrajectoryGeneration`  
   * First step: Inspect this repo to understand trajectory format  
   * Generate or use existing trajectories for training  
2. **Fixed Agent**: Use GPT-5 API as the base agent  
   * Load API key from `/home/yuhao/code/.env`  
   * Implement agent wrapper that calls GPT-5 with observation \+ retrieved memories  
   * This is actually ideal \- no need to train an agent, and GPT-5 is strong enough to benefit from memory  
3. **Retriever**: Hybrid (BM25 \+ dense)  
   * BM25 for keyword matching  
   * Dense retriever (e.g., sentence-transformers) for semantic matching  
   * Combine scores with weighted sum  
4. **Compute**: 4-8 GPUs available  
   * Use full fine-tuning (no LoRA needed)  
   * Batch size: 128 (micro batch size: 4-8 per GPU)  
   * Can train Qwen2.5-9B comfortably  
   * Use DeepSpeed ZeRO-2 for memory efficiency

---

## **Next Steps**

1. **Inspect trajectory generation repo** to understand data format  
2. **Create project structure** following the architecture above  
3. **Implement core components** (memory store, retriever, agent wrapper)  
4. **Build data pipeline** (parser, hindsight labeling, SFT dataset)  
5. **Train SFT model** on hindsight-labeled data  
6. **Generate and score candidates** for best-of-n  
7. **Train best-of-n model** on selected candidates  
8. **Evaluate** against baselines  
9. **(Optional) RL training** if SFT \+ best-of-n results are promising
