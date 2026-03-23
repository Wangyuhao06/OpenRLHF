# Demand-Aware Hindsight Labeling

## Context

当前 `HindsightLabeler` 逐步独立标注（每步只看3个未来observation），存在三个核心问题：
1. **无跨步优化**：同一信息可能被多个future step需要，但当前逐步决策无法合并需求
2. **无retrieval保证**：生成的keys未考虑HybridRetriever的BM25+Dense匹配机制，导致memory写了但检索不到
3. **memory_store为空**：SFT样本中memory_store始终为空，训练时模型看不到累积的memory上下文

新方案通过 **反向需求分析 → 全局最简memory规划 → 检索命中验证** 三阶段流水线解决上述问题。

## Design Assessment

这个方案非常合理。核心优势：
- 反向遍历天然识别信息流的critical path
- 全局规划确保memory slot最少化 + 需求merge
- 检索验证闭环保证生成的memory在推理时可被检索到
- 累积memory_store让SFT训练更接近推理时的真实场景

## Implementation Plan

### Phase 1: Backward Demand Analysis

**文件**: [hindsight_labeling.py](memory_constructor/memory_constructor/data/hindsight_labeling.py)

新增 `DemandAwareHindsightLabeler` 类，与现有 `HindsightLabeler`、`HeuristicLabeler` 并列。

**方法**: `_analyze_demands(trajectory, steps) -> Dict[int, Dict[int, str]]`

- **自适应策略**：
  - **短trajectory（<= 10步）**：单次LLM调用分析整个trajectory的信息依赖
  - **长trajectory（> 10步）**：分chunk/逐步backward分析，每次处理一个future step，向前查找其依赖的source steps
- 输出 `demand_dict`: `{source_step: {future_step: demand_description}}`
- Prompt设计要点：
  - 输入完整trajectory（每步的observation摘要 + action）或当前chunk
  - 要求LLM识别所有 `(source_step, future_step, demand_info)` 三元组
  - 仅标注真正的信息依赖（不含trivial导航步骤）

```
Prompt核心结构:
- Task instruction
- 完整trajectory摘要（每步obs[:300] + action）或当前分析的step及其前序steps
- 要求输出: {"demands": [{"source_step": int, "future_step": int, "demand": str}, ...]}
```

### Phase 2: Global Memory Plan Generation

**方法**: `_generate_memory_plan(trajectory, steps, demand_dict) -> List[Dict]`

- **单次LLM调用**，输入完整trajectory + Phase 1的demand_dict
- 约束条件在prompt中明确：
  1. 每步最多1个memory
  2. 多个demand指向同一source_step时merge为1个memory
  3. 不同source_step有重叠信息时合并到更少的slot
  4. 无demand的step不生成memory
- **检索感知key生成**：prompt中包含future step的observation上下文，引导LLM选择与future query有token overlap的keys
- 输出: `[{"step_id": int, "keys": [...], "value": str, "demanded_by": [int,...], "reasoning": str}]`

```
Prompt核心结构:
- Task instruction + trajectory
- Demand分析结果（每个source step被哪些future steps需要，需要什么）
- 明确约束（max 1 memory/step, minimize total, merge demands）
- 检索提示：keys应包含future step observation中可能出现的搜索词
- 格式约束（max_num_keys, max_value_tokens）
```

### Phase 3: Retrieval Hit Verification

**方法**: `_verify_and_refine_plan(steps, memory_plan, demand_dict) -> List[Dict]`

利用现有 `HybridRetriever`（[retriever.py](memory_constructor/memory_constructor/models/retriever.py)）做闭环验证：

1. 对plan中每个memory，构建模拟MemoryStore（包含所有已写入的memory）
2. 用每个demanding step的observation作为proxy query
3. 调用 `HybridRetriever.retrieve(query, store, k=3, return_scores=True)`
4. 检查目标memory是否在top-k结果中
5. 若miss，收集失败信息，**调用LLM refinement**：
   - 告知LLM当前keys、失败的query、检索器机制（BM25 tokenize=lowercase+split, Dense=sentence-transformers cosine）
   - LLM输出refined keys
   - 重新验证（最多5轮refinement）

**依赖**（只读使用，不修改）:
- [retriever.py](memory_constructor/memory_constructor/models/retriever.py) - `HybridRetriever`
- [memory_store.py](memory_constructor/memory_constructor/data/memory_store.py) - `MemoryStore`（API: `.add()`, `.get_all()`）

### Phase 4: SFT Sample Assembly + Input Signal Enhancement

**方法**: `_assemble_sft_samples(trajectory, steps, memory_plan) -> List[SFTSample]`

#### 4a. 累积memory_store
- 按step_id顺序遍历，维护 `accumulated_memories` 列表
- 每步的 `memory_store` = 该步之前所有已写入的memory（模拟推理时的真实状态）
- `budget_remaining` = `memory_budget` - `len(memory_store)`
  - `memory_budget = int(len(steps) * budget_ratio)`，按episode步数比例计算
- `target_memory` = plan中该步的memory（或write=False）
- 写入的memory加入accumulated列表，供后续步使用

#### 4b. 增强输入信号（新增）

**Retrieval context**: 每步生成时，用当前observation查询accumulated memory store：
- 调用 `HybridRetriever.retrieve(observation, memory_store, k=3, return_scores=True)`
- 将检索结果格式化加入prompt，让模型看到哪些信息已可检索
- 存入SFTSample的metadata中（`retrieval_context` 字段）

**Agent task context**: 让memory constructor理解它服务的agent在做什么任务：
- 包含task instruction（已有）
- 新增 **task overview / purpose framing**：从instruction中提取核心需求要点（产品类型、约束条件、预算等），以结构化方式呈现
- 提取方式：在Phase 1 demand analysis时一并让LLM提取（每条trajectory一次，复用同一LLM调用），存入trajectory-level metadata

#### 4c. 更新后的Prompt Template

```
You are a memory constructor for a shopping agent. You decide what to store
in the agent's memory to help it complete its task.

Agent Task:
{instruction}

Key Requirements:
{task_requirements}    ← 新增：结构化的任务需求要点

Current Observation:
{observation}

Local History (last 3 steps):
{local_history}

Current Memory Store:
{memory_store}

Currently Retrievable (top matches for this observation):
{retrieval_context}    ← 新增：当前observation的检索结果

Budget Remaining: {budget_remaining}
Episode Progress: {episode_progress:.1%}

Decide: should you write a memory now? Consider:
- Does this observation contain NEW information not already in memory?
- Would this information help the agent in future steps?
- Is this information retrievable from existing memory?

Respond in JSON:
{"write": true/false, "keys": ["key1", ...], "value": "..."}
```

### Integration

**文件**: [01_preprocess_webshop.py](memory_constructor/scripts/01_preprocess_webshop.py)

- `--labeling_method` 添加 `"demand_aware"` 选项
- 新增 `--budget_ratio` 参数（default=2.0，即budget = steps * 2.0）
- 实例化 `DemandAwareHindsightLabeler` 并调用 `label_trajectories()`

## Files to Modify

| File | Change |
|------|--------|
| [hindsight_labeling.py](memory_constructor/memory_constructor/data/hindsight_labeling.py) | 新增 `DemandAwareHindsightLabeler` 类（~350行） |
| [sft_dataset.py](memory_constructor/memory_constructor/data/sft_dataset.py) | 更新prompt template，增加retrieval_context和task_requirements字段 |
| [01_preprocess_webshop.py](memory_constructor/scripts/01_preprocess_webshop.py) | 添加 `demand_aware` labeling method + `--budget_ratio` arg |

## Reuse Existing Code

- `extract_observations_and_actions()`, `extract_local_history()` from [webshop_parser.py](memory_constructor/memory_constructor/data/webshop_parser.py)
- `MemoryItem`, `SFTSample` from [schemas.py](memory_constructor/memory_constructor/data/schemas.py)
- `parse_json_robust`, `validate_memory_item` from [json_utils.py](memory_constructor/memory_constructor/utils/json_utils.py)
- `HybridRetriever` from [retriever.py](memory_constructor/memory_constructor/models/retriever.py)
- `MemoryStore` from [memory_store.py](memory_constructor/memory_constructor/data/memory_store.py)
- OpenAI client pattern from existing `HindsightLabeler.__init__`（API key从 `/home/yuhao/work/code/.env` 加载，支持OpenAI/DeepSeek）

## LLM Call Budget

每条trajectory（典型3-10步）：
- Phase 1: 1次（demand analysis）
- Phase 2: 1次（memory plan）
- Phase 3: 0~N次refinement（N = 失败memory数 x 最多5轮）
- **典型case**: 2-4次LLM调用，与当前逐步标注（3-10次/trajectory）相当或更少

## Error Handling

| 阶段 | 失败 | Fallback |
|------|------|----------|
| Phase 1 LLM调用失败 | JSON parse error / API error | 用heuristic: 含product details的step作为source，后续comparison/purchase step作为demander |
| Phase 2 LLM调用失败 | 同上 | 退化为现有 `HindsightLabeler.label_step()` 对有demand的step逐步标注 |
| Phase 2 plan超budget | memory数 > budget | 按demand_count从小到大裁剪（budget = steps * budget_ratio） |
| Phase 3 retriever加载失败 | sentence-transformers下载问题 | 跳过验证，直接使用plan |
| Phase 3 refinement后仍miss | max attempts exhausted | 保留原keys，log warning |
| 全部失败 | 3个phase都失败 | 退化为 `HeuristicLabeler.label_trajectory()` |

## Verification

1. 在现有test trajectories上运行: `python scripts/01_preprocess_webshop.py --labeling_method demand_aware`
2. 检查输出JSONL中：
   - 每条trajectory的memory写入数 <= demand analysis识别的source step数
   - 每步的memory_store正确累积（step N的store包含step 0..N-1的写入）
   - budget_remaining = int(len(steps) * budget_ratio) - len(memory_store)
   - target_memory的keys能被HybridRetriever检索到
3. 与现有heuristic标注结果对比memory数量和coverage
