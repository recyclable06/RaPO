# RaPO 论文复现工程当前 Handoff

> **历史快照边界（2026-08-01）：** 本文保留 2026-07-31 handoff 的事实、失败记录和当时的下一步，不再作为现役状态入口。当前检查点以仓库根 `AGENTS.md`、`docs/audit/2026-07-31-independent-audit.md` 和对应 `docs/remediation/batch*/` 为准。本文中的旧命令输出和实验记录在新角色未现场重跑时只能按 `recorded-but-not-rerun` 使用，不得改写成当前现场结果。
>
> 批次 1 后续已完成独立 CPU 验收，但这不代表 GPU-ready、训练重跑、论文级完整复现通过或正式实验放行。历史临时路径也不构成永久 artifact。

> 快照日期：2026-07-31<br>
> 文档状态：事实型 handoff<br>
> 当前分支：`main`<br>
> 当前仓库 commit：`77ed06c1da1f35368136675cc6b7569b07cb186d`<br>
> GitHub：https://github.com/recyclable06/RaPO<br>
> 当前实验状态：Task 6 后暂停；不得直接继续 Task 7<br>
> 当前工作范围：只复现 RaPO 论文，不开展 OPD follow 工作

## 事实标签

本文使用以下标签区分信息来源：

- **论文明确**：直接来自论文正文、公式、附录或实验设置。
- **仓库实现**：当前 commit 中实际存在的代码行为。
- **用户要求**：用户在完整协作过程中明确提出、目前仍有效的目标或边界。
- **2080 Ti 适配**：为使实验能在 8×RTX 2080 Ti 上运行而采用的下降配置。
- **工程假设**：论文未公开，当前复现根据 Visual-RFT 或工程判断暂定的选择。
- **已确认问题**：已由代码、日志或实验复现确认。
- **未解决风险**：有合理证据，但尚未通过独立审查或受控实验定论。
- **需要实验**：无法仅靠代码阅读判断。

本文记录当前事实，不替代新 agent 的独立实现审计。

# 1. 工程概览

## 1.1 论文和研究问题

复现论文为：

> *Overcoming Catastrophic Forgetting in Visual Continual Learning with Reinforcement Fine-Tuning*

论文研究视觉持续学习中的灾难性遗忘：模型按顺序学习多个任务，每个新任务只能访问当前任务训练数据，同时需要保留旧任务能力。

论文首先观察到，GRPO 等 Reinforcement Fine-Tuning（RFT）方法虽然比 SFT 更不容易遗忘，但仍存在明显遗忘。论文将其中一个原因概括为 **trajectory-level drift agnosticism**：同一输入的多个 rollout 即使获得相同任务奖励，相对于上一任务策略的分布漂移也可能差异很大，而普通 GRPO 不会利用这种差异进行 credit assignment。

## 1.2 RaPO 的目标与两个组件

RaPO，即 Retention-aware Policy Optimization，试图在学习新任务时偏向“完成当前任务且相对上一任务策略漂移较小”的轨迹。

它包含两个核心组件：

1. **Retention Reward**

   对当前策略生成的每条轨迹，计算其 token 平均 log-probability 相对上一任务冻结策略的正向漂移，再将漂移映射为 `(0, 1]` 的保留奖励。漂移越小，保留奖励越高。

2. **Cross-Task Advantage Normalization（CTAN）**

   不再只使用当前 batch 的奖励标准差进行 advantage normalization，而是维护跨 optimizer step、跨任务持久化的标准差 EMA，以降低任务切换时奖励尺度突变造成的不稳定。

Retention Reward 主要改变组内轨迹排序；CTAN 主要稳定 advantage 的尺度。

## 1.3 当前优先复现范围

**论文明确：**

- ImageNet-R；
- 200 个类别；
- 10 个不重叠任务；
- 每任务 20 类；
- 每类 5 个训练样本；
- 无历史训练数据 replay；
- Qwen2-VL-2B；
- 每任务训练 2 epochs；
- 结果取 3 个随机 class orders 的均值和方差；
- 报告 Last Accuracy、Forgetting；
- 论文所有实验使用 8×NVIDIA H100。

**用户要求：**

当前首先完成上述 class-incremental image classification 复现，暂不扩展：

- OPD follow；
- object detection；
- video classification；
- domain-incremental learning；
- 其他数据集或模型尺寸。

这样做是为了先验证 RaPO 两个核心组件、真实 Trainer 数据流和持续学习串链，再讨论后续研究。

## 1.4 与 Visual-RFT、GRPO 和官方仓库的关系

- **GRPO** 是比较基线，也是 Visual-RFT 已提供的训练方法。
- **Visual-RFT** 提供 Qwen2-VL GRPO Trainer、视觉数据处理和分类 reward。当前工程固定使用：
  `Liuziyu77/Visual-RFT@2ffad63b25ddd79bfe25d3e046645401201c89d6`。
- 本工程没有 vendor 整个 Visual-RFT，而是：
  - 独立实现 RaPO 核心模块；
  - 用单独 patch 接入固定的 Visual-RFT commit；
  - 保留可单元测试、可审阅的数学实现。
- **官方 RaPO 仓库**是 `LMMMEng/RaPO`。截至 2026-07-31 的在线核验，该仓库仍只有 README，并声明正在整理代码、之后开源，没有可用训练实现或配置：[LMMMEng/RaPO](https://github.com/LMMMEng/RaPO)。
- 因此，当前仓库是**独立复现**，不是官方代码的修改版，也不能声称与作者内部实现完全相同。

## 1.5 最终论文级目标

最终希望：

1. 冻结一套经过独立审计、适合高端 GPU/BF16 的正式配置；
2. 从 Task 1 重新开始 GRPO 与 RaPO 两条完整链；
3. 每任务严格训练 2 epochs；
4. 完成 10 个任务；
5. 对全部已见类别进行正式评估；
6. 对 3 个冻结 class orders 重复；
7. 输出：
   - 每个 order 的完整准确率矩阵；
   - Last Accuracy；
   - Forgetting；
   - 三个 order 的均值和方差；
   - 代码、模型、数据、配置、状态和结果 provenance；
8. 判断当前独立实现是否能复现论文所报告的“RaPO 相比 GRPO 降低遗忘”的趋势。

论文 ImageNet-R 10-task 表中报告：

- GRPO：Last Accuracy `74.67±1.27`，Forgetting `20.02±4.91`；
- RaPO：Last Accuracy `85.92±1.82`，Forgetting `4.69±1.71`。

这些数值是参考目标，不是当前 Task 1～6 bounded 实验的直接验收线，因为官方 class orders、5-shot 划分和多个训练超参数尚未公开。

# 2. 用户在完整对话中最终汇集的需求

## 2.1 研究范围

| 最终需求 | 来源背景 | 当前状态 | 说明 |
|---|---|---|---|
| 先忠实复现 RaPO，不做 OPD follow | 初始范围经用户再次收紧 | 已满足范围控制 | OPD 仍明确 out of scope |
| 以论文 PDF 为主要方法依据 | 公式实现与 Task 5 审计阶段 | 部分满足 | 已核对公式与附录；仍需新 agent 独立复审 |
| 独立实现 Retention Reward 和 CTAN | 初始工程目标 | 已实现 | 位于 `src/rapo/` |
| 审查真实 Trainer 数据流 | Task 4～5 差距触发 | 部分满足 | 已做一次审计；不能视为最终正确性证明 |
| 对比同配置 GRPO 与 RaPO | 全程实验要求 | bounded 范围满足 | Task 1～6 已有对照；正式配置未开始 |
| ImageNet-R、5-shot、10-task、20 classes/task、无 replay | 论文复现规格 | 部分满足 | 数据构造已实现；只运行到 Task 6、order 0 |
| 每任务严格 2 epochs | 论文设置 | 尚未满足 | 当前使用 `max_steps=20`，约 2.92 epochs |
| 完成 3 个 class orders | 论文设置 | 尚未满足 | 只使用 provisional order 0 |
| 输出 Last Accuracy、Forgetting、完整矩阵 | 用户与论文要求 | 部分满足 | bounded Task 1～6 已输出；正式全量未完成 |
| 区分论文事实与未公开配置 | 用户明确要求 | 已在文档中执行 | 仍需在正式 run manifest 中机器化 |
| 固定上游 commit、模型 revision、seed、类别顺序和样本划分 | 可复现性要求 | 部分满足 | Git/模型/seed 已固定；划分是自定义 provisional |
| 防止错误 checkpoint、错误状态、错误任务串链 | 多任务运行要求 | 部分满足 | launcher 做基本存在性检查，但缺 provenance 强校验 |
| 支持恢复、审计和结果追踪 | 工程要求 | 部分满足 | 有状态文件、日志和哈希；缺完整可靠 resume/run manifest |
| 先利用现有 2080 Ti 做小规模验证 | 硬件阶段决策 | 已完成 | Task 1～6 为该阶段结果 |
| 高端 GPU 可用后重新判断下降配置 | 集群信息变化 | 尚未完成 | 4090 仅完成推理复评，没有 BF16 训练 |
| 正式高端 GPU 路径与 2080 Ti legacy 路径分离 | 硬件变化后的最终要求 | 尚未完成 | 当前只有通用 smoke 脚本和 2080 wrapper |
| 配置冻结后从 Task 1 重启正式链 | Task 6 暂停决策 | 尚未执行 | 是后续硬约束 |
| 不把 Task 1～6 当论文正式结果 | 用户最终要求 | 已明确记录 | 当前均为 engineering/diagnostic evidence |
| 新 agent 独立审查并分阶段整改 | 当前 handoff 目标 | 待新 agent 执行 | 不应继承此前审计结论而跳过复审 |
| 使用 GitHub 管理并保持服务器同步 | 仓库协作要求 | 已满足当前快照 | 本地、GitHub、服务器均在当前 SHA |
| 使用 Conda、SSH、tmux 和显式 GPU 选择 | 服务器工作方式 | 已执行 | 遵守实验室服务器规范 |
| 无特殊情况可自主推进，但特殊风险应暂停 | 长期协作约定 | 已执行 | Task 6 数值风险触发暂停 |

## 2.2 硬件变化后的重新解释

最初实验主要受限于 8×RTX 2080 Ti。用户后来确认实验室集群还存在 RTX 3090、RTX 4090、NVIDIA L40 和 RTX 5090 等资源，并允许按需申请。

这意味着：

- “只能采用 FP16/SDPA/小图像/短 completion”已经不是永久约束；
- 但“集群有高级卡”不等于当前有一组空闲、同质、可连续使用的 8 卡；
- 4090 上完成过推理，不等于 BF16 训练链已经兼容；
- 高端 GPU 正式路径必须重新探测、设计和冻结；
- 当前 2080 Ti Task 1～6 不能直接续接成新的正式 BF16 Task 7～10。

# 3. 当前仓库状态

## 3.1 Git 状态

- GitHub：`https://github.com/recyclable06/RaPO.git`
- 分支：`main`
- 最新完整 SHA：
  `77ed06c1da1f35368136675cc6b7569b07cb186d`
- 本地与服务器：
  `main...origin/main`
- 当前 tracked 工作区：除本 handoff 外无其他改动
- tag：无
- CI：当前没有 `.github/workflows` 或其他可验证 CI
- 当前 commit 的服务器单元测试：`29 passed`

## 3.2 上游固定信息

- Visual-RFT：
  `2ffad63b25ddd79bfe25d3e046645401201c89d6`
- Qwen 模型：
  `Qwen/Qwen2-VL-2B-Instruct`
- 模型 revision：
  `895c3a49bc3fa70a340399125c650a463535e71c`
- Transformers：
  `336dc69d63d56f232a183a3e7f52790429b871ef`
- TRL：`0.14.0`
- PyTorch：`2.5.1+cu124`
- DeepSpeed：`0.15.4`
- datasets：`3.6.0`

## 3.3 目录和职责

```text
RaPO/
├── README.md
├── pyproject.toml
├── environment.yml
├── environment-train.yml
├── src/rapo/
│   ├── core.py
│   ├── integration.py
│   ├── runtime.py
│   ├── data.py
│   └── evaluation.py
├── patches/
│   └── visual_rft_2ffad63_rapo.patch
├── scripts/
│   ├── apply_visual_rft_patch.sh
│   ├── bootstrap_env.sh
│   ├── bootstrap_train_env.sh
│   ├── run_imagenet_r_smoke.sh
│   ├── run_imagenet_r_2080ti_smoke.sh
│   ├── evaluate_qwen2_vl.py
│   ├── probe_qwen_gpu.py
│   └── download/install helpers
├── configs/
│   ├── deepspeed_zero3_cpu_offload.json
│   ├── deepspeed_zero3_cpu_offload_scale10.json
│   └── deepspeed_zero3_cpu_offload_scale9.json
├── tests/
│   ├── test_core.py
│   ├── test_integration.py
│   ├── test_runtime.py
│   ├── test_data.py
│   └── test_evaluation.py
└── docs/
    ├── reproduction_spec.md
    ├── upstream.md
    ├── trainer_integration.md
    ├── data_and_evaluation.md
    ├── smoke_test.md
    └── experiments/
```

## 3.4 核心模块

- `src/rapo/core.py`
  - trajectory drift；
  - one-sided truncation；
  - detached retention reward；
  - reward composition；
  - CTAN EMA 和 advantage。

- `src/rapo/integration.py`
  - Trainer 配置；
  - Task 1 与 Task 2+ reward 行为；
  - CTAN 状态保存和恢复；
  - `rapo_state.json` 格式。

- `src/rapo/runtime.py`
  - 显式 `RAPO_DISABLE_CUDNN=1` 兼容开关。

- `src/rapo/data.py`
  - deterministic ImageNet-R 类别顺序和 5-shot 划分；
  - cumulative seen-class prompt；
  - Visual-RFT task export。

- `src/rapo/evaluation.py`
  - `<answer>` 提取和 exact-match；
  - prediction 聚合；
  - Last Accuracy 与 Forgetting；
  - 小图像 padding。

## 3.5 Trainer patch

`patches/visual_rft_2ffad63_rapo.patch` 只修改固定 Visual-RFT checkout 中：

- `src/virft/src/open_r1/grpo_classification.py`
- `src/virft/src/open_r1/trainer/grpo_trainer.py`

应用方式：

```bash
bash scripts/apply_visual_rft_patch.sh /path/to/Visual-RFT
```

脚本要求：

- checkout HEAD 必须等于固定 commit；
- checkout 应在应用 patch 前干净；
- `git apply --check` 必须通过。

当前服务器 checkout 保持在固定 commit，只有上述两个文件因 patch 而 modified。

## 3.6 当前 README 声明

README 当前声明：

- 核心公式实现、状态持久化、patch、数据与 evaluator 已完成；
- 2080 Ti reduced gate 已通过；
- bounded comparison 已推进到 Task 6；
- Task 6 BF16 paired stability gate 未完成；
- full 10-task experiment 未完成。

# 4. 论文到代码的当前实现映射

| 论文操作 | 论文位置 | 当前实现 | 测试/证据 | 当前判断 |
|---|---|---|---|---|
| trajectory drift | Eq. (2) | `core.trajectory_drift` | mask、长度归一化测试 | 已实现；需复审 Trainer 输入是否完全对应 |
| actor-anchor token log-ratio | Eq. (2) | actor/ref per-token logps 相减 | Task 2+ 运行日志 | 已实现 |
| one-sided truncation | Eq. (2) | `clamp_min(0)` | 负漂移归零测试 | 已实现 |
| stop-gradient | Eq. (3) 前文字 | drift 与 retention 均 detach | `requires_grad=False` 测试 | 已实现 |
| retention reward | Eq. (3) | `exp(-alpha*drift)` | bounded reward 测试 | 已实现 |
| reward composition | Eq. (4) | `task + weight*retention` | reward 加法测试 | 已实现 |
| 在 advantage 前注入 reward | Eq. (4) 说明 | patch 在 normalization 前构造 total reward | Trainer patch | 已实现 |
| CTAN EMA | Eq. (5) | `CrossTaskAdvantageNormalizer` | 持久化和配置校验测试 | 已实现，但更新时机有歧义 |
| group-relative centering | Eq. (6) | 本地连续 rollout group mean | group mean 测试 | 已实现 |
| CTAN denominator | Eq. (6) | 全局 gathered rewards 的 sample std 进入 EMA | global std 测试 | 工程假设，论文未公开分布式口径 |
| Task 1 不使用 retention | Section 4 | `task_index==1` 时 total=task reward | Task 1 测试 | 已实现 |
| Task 1 仍使用 CTAN | 论文 CTAN 跨任务描述 | RaPO controller 从 Task 1 起启用 | state updates 实验 | 当前实现选择 |
| Task 2+ 使用上一任务 anchor | Section 3.2.1 | ZeRO-3 ref model 从本次输入模型目录加载 | Task 2～6 实验 | 已实现；缺 provenance 强校验 |
| classification exact-match | 附录 reward | `classification_answer_is_correct` | cat/catfish 等测试 | 已实现 |
| format reward | 附录 reward | 保留 Visual-RFT format verifier | Task 5 审计 | 当前认为匹配，需独立复查 |
| cumulative seen-class prompt | 附录 prompt | `data.make_classification_prompt` | data tests | 已实现 |
| 无 future classes | CIL prompt 协议 | task export 使用 `seen_class_names` | task rows 测试 | 已实现 |
| CTAN 跨任务持久化 | Eq. (5)/(6) 描述 | checkpoint/final output 写 `rapo_state.json` | round-trip 测试与 Task 1～6 state | 已实现 |
| CTAN task continuity | 论文序列语义 | state 保存 `task_index` | 无连续性验证 | **未完整实现** |

此前审计结论是“截至 Task 5 未发现 material RaPO 公式接入错误”。这只能说明当时检查未发现明显错位，不能替代新 agent 的独立审查，也不能证明分布式和 optimizer-step 语义完全正确。

# 5. 完整实验历史

## 5.1 CPU 与无 GPU 阶段

### 核心实现

最初建立：

- `trajectory_drift`
- `retention_reward`
- `combine_rewards`
- `CrossTaskAdvantageNormalizer`
- 状态序列化
- deterministic data builder
- evaluation metrics

当前共有 29 个测试，覆盖：

- masking；
- 每条轨迹长度归一化；
- one-sided clamp；
- detach；
- reward shaping；
- CTAN EMA；
- group centering；
- global reward std；
- state round trip；
- exact-match；
- deterministic data；
- prompt；
- lower-triangle metrics；
- cuDNN 开关。

服务器 `rapo` Conda 环境在当前 commit 上通过 `29 passed`。这不是 Trainer 端到端正确性的证明。

### Visual-RFT patch 验证

固定 Visual-RFT commit 后：

- patch 可通过 `git apply --check`；
- GRPO-disabled/RaPO-disabled 路径保持原 GRPO normalization；
- core 模块可独立导入；
- patch 后入口可导入；
- 当前外部 checkout 只修改预期的两个文件。

## 5.2 2080 Ti 单卡模型加载

硬件：

- 1×RTX 2080 Ti；
- FP16；
- SDPA。

结果：

- Qwen2-VL-2B 成功加载；
- 2,208,985,600 参数；
- base model 约 4,215 MiB peak allocated；
- 约 4,528 MiB peak reserved。

这证明模型可在单张 2080 Ti 上加载，不证明训练可行。

## 5.3 早期 optimizer 与 8-rollout gate

### 初始失败

使用上游动态 FP16 initial scale `65,536`：

- 进程无 OOM；
- 日志显示 `grad_norm=0`；
- DeepSpeed optimizer state 显示 `overflow=True`；
- Adam state 为空；
- 说明 optimizer update 被跳过。

关闭 gradient checkpointing 没有解决。

### scale 4,096

- 两 rollout 的单步/五步 gate 可运行；
- 八 rollout 单步可运行；
- 八 rollout 五步在部分步骤出现重复 learning rate 和 stale gradient norm；
- 说明进程成功退出并不等于每次 optimizer update 都执行。

### scale 2,048

八 rollout 五步重跑：

- 完整 learning-rate transitions；
- finite、nonzero gradient norms；
- 无 OOM、NaN 或 skipped update；
- observed memory 最高约 10,806 MiB；
- 对 11 GiB 2080 Ti 几乎无余量。

该阶段固定了后续 Task 1～4 的 2080 Ti 基础配置。

## 5.4 Task 1～2 五步 reduced run

配置：

- 8×RTX 2080 Ti；
- FP16 + SDPA；
- 8 rollouts；
- prompt/completion：`512/32`；
- max pixels：`100352`；
- gradient checkpointing；
- ZeRO-3；
- CPU optimizer offload；
- gradient accumulation 2；
- initial loss scale `2048`；
- 每任务 5 optimizer steps；
- 评估 5 samples/class。

验证：

- RaPO Task 1 CTAN state 保存；
- Task 2 从 Task 1 final model 启动；
- Task 2 加载 Task 1 `rapo_state.json`；
- Task 2 retention drift 从第一步的 0 变为非零；
- final model 可重新加载；
- GRPO control 完成。

结果：

| Method | Last Accuracy | Forgetting |
|---|---:|---:|
| GRPO | 92.5% | 0.0% |
| RaPO | 92.0% | 1.0% |

差异只有一个样本，不能排名。

### 已确认集成问题

Task 2 首次启动失败：Visual-RFT 根据 checkpoint 路径中是否包含 `Qwen2-VL` 推断模型类型。序列输出目录不一定包含该字符串。

修复后改用 `AutoConfig.model_type`。这是已修复的 confirmed bug。

## 5.5 Task 1～2 二十步链

相同 2080 Ti 配置，每任务 20 optimizer steps。

训练结果：

| Method | Task | Runtime (s) | Mean loss | Final grad norm |
|---|---:|---:|---:|---:|
| GRPO | 1 | 741.00 | 0.005732 | 0.9381 |
| GRPO | 2 | 740.29 | 0.000350 | 3.7970 |
| RaPO | 1 | 766.71 | 0.005636 | 0.5864 |
| RaPO | 2 | 738.05 | 0.000315 | 1.4342 |

20 samples/class 的评估：

| Method | Last Accuracy | Forgetting |
|---|---:|---:|
| GRPO | 93.875% | 2.0% |
| RaPO | 94.125% | 1.5% |

注意：

- 每任务 100 个训练样本；
- world size 8；
- per-device batch 1；
- gradient accumulation 2；
- 20 optimizer steps 实际跨约 2.92 epochs；
- CTAN 每任务更新 38 次，而不是 20 次；
- 这揭示了 CTAN 按 forward microbatch 更新的语义。

## 5.6 Task 3

训练 commit：

- RaPO：`1818a98...`
- evaluator：`e4da23d...`

配置保持 Task 1～2 不变。

训练结果：

| Method | Runtime (s) | Mean loss | Final grad norm |
|---|---:|---:|---:|
| GRPO | 750.37 | 0.000231 | 0.0366 |
| RaPO | 890.54 | 0.000303 | 1.1808 |

评估：

| Method | Last Accuracy | Forgetting |
|---|---:|---:|
| GRPO | 92.9167% | 2.0% |
| RaPO | 92.5% | 2.625% |

### 已确认问题与修复

1. **Dataset.map 并发冲突**

   8 个 rank 同时向共享文件系统写 Hugging Face cache，首次启动失败。随后用 `training_args.main_process_first` 串行化 rank 0 preprocessing。

2. **过小图像**

   评估在 `n07873807/deviantart_12.jpg` 处失败，该图像宽度为 27，小于 Qwen2-VL 的 28-pixel spatial factor。随后只对不足维度 padding 到 28。

3. evaluator 使用临时文件后原子替换，失败时没有留下可误认为完整结果的 partial JSONL。

## 5.7 Task 4

配置仍为：

- FP16；
- SDPA；
- scale 2048；
- cuDNN 默认开启；
- prompt 512；
- completion 32；
- 20 steps；
- 8 rollouts；
- 20 eval samples/class。

训练：

| Method | Runtime (s) | Mean loss | Final grad norm |
|---|---:|---:|---:|
| GRPO | 750.79 | 0.000349 | 2.3555 |
| RaPO | 859.79 | 0.000474 | 1.8057 |

结果：

| Method | Last Accuracy | Forgetting |
|---|---:|---:|
| GRPO | 90.5% | 4.0% |
| RaPO | 86.625% | 10.0% |

Task 4 开始出现明显差距：

- RaPO 在新 Task 4 上比 GRPO 多 10 个正确样本；
- 但 Task 1 少 64 个；
- paired audit 没有发现 key mismatch、duplicate、missing answer、模型保存或 anchor 路径错误。

这说明现象更像旧任务遗忘，但不能证明原因是 RaPO 算法。

## 5.8 Task 5

### 关键失败

RaPO Task 5 首次尝试在 backward 触发：

- `CUDNN_STATUS_INTERNAL_ERROR`；
- 两次 clean launch 可复现；
- cuDNN v7 compatibility path 也失败；
- 禁用 cuDNN 后训练可继续；
- 但 initial scale 2048 出现 early skipped optimizer update；
- partial run 被停止，没有用于比较。

### 最终共享配置

GRPO 和 RaPO 都使用：

- `RAPO_DISABLE_CUDNN=1`；
- initial loss scale 1024；
- 其他配置保持相同。

训练：

| Method | Runtime (s) | Mean loss | Final grad norm |
|---|---:|---:|---:|
| GRPO | 762.26 | 0.000279 | 0.9527 |
| RaPO | 770.01 | 0.001250 | 0.2119 |

结果：

| Method | Final row correct counts | Last Accuracy | Forgetting |
|---|---|---:|---:|
| GRPO | `[344,375,346,361,366]` | 89.60% | 4.375% |
| RaPO | `[281,363,336,366,363]` | 85.45% | 10.0% |

RaPO 仍明显落后，但新任务学习没有整体崩溃。

### Task 5 实现审计

差距重复后进行公式、Trainer、anchor、CTAN、数据、prompt、evaluation 审计。

审计结论：

- 没发现 material 公式接入错位；
- 发现 prompt truncation 边界；
- 确认 Task 1～5 实际 prompt token 最大值均未超过 512；
- 记录了 CTAN 更新频率、全局 std 和未公开配置等不确定性。

## 5.9 Task 6

### 数据和 prompt

- train：100；
- cumulative test：17,391；
- 实测训练 prompt 最大 567 tokens；
- 512 不再安全；
- 两个方法统一使用 prompt limit 1024。

### scale 1024 路径

- RaPO 一步 gate 通过；
- RaPO 20 steps 完成并保存；
- 对应 GRPO 在 step 8 出现 skipped optimizer update；
- 因此不能把 scale-1024 RaPO 和失败的 GRPO 作为公平 pair。

### 正式 bounded pair：scale 512

两个方法都从各自 Task 5 final model 重新启动，并统一使用：

- FP16；
- SDPA；
- ZeRO-3 CPU optimizer offload；
- gradient accumulation 2；
- gradient checkpointing；
- prompt/completion `1024/32`；
- 8 rollouts；
- cuDNN disabled；
- initial loss scale 512；
- 20 steps。

训练：

| Method | Runtime (s) | Mean loss | Final grad norm |
|---|---:|---:|---:|
| GRPO | 755.41 | 0.000424 | 0.1823 |
| RaPO | 769.82 | 0.000547 | 0.7709 |

两者完成完整 learning-rate sequence，无 OOM、NaN、traceback 或 skipped update。

结果：

| Method | Last Accuracy | Forgetting |
|---|---:|---:|
| GRPO scale-512 | 85.00% | 8.95% |
| RaPO scale-512 | 65.25% | 33.75% |

RaPO scale-1024 诊断结果：

- Last Accuracy `72.7083%`
- Forgetting `24.50%`

这证明不同 FP16 loss-scale 路径对应的最终模型行为差异很大，但没有成功的 scale-1024 GRPO 对照，所以不能建立方法或因果结论。

## 5.10 4090 推理复评

为了检查“大幅差距是否由 2080 Ti 推理造成”，同一个 RaPO scale-512 Task 6 模型在单张 RTX 4090 上重新评估。

设置仍为：

- FP16；
- SDPA；
- greedy generation；
- 32 new tokens；
- 20 samples/class。

结果：

- 2080 Ti final row：`[93,187,240,342,332,372]`
- RTX 4090 final row：`[92,187,240,343,332,372]`
- 两者总正确数均为 `1566/2400`
- Last Accuracy 均为 `65.25%`
- Forgetting 均为 `33.75%`
- 2,400 个样本中只有 4 个 correctness decisions 不同

因此，大幅差距基本不能归因于 2080 Ti 推理硬件。这个控制实验没有验证 4090/BF16 训练。

## 5.11 当前暂停点

Task 7 未启动。

暂停原因：

1. formal bounded scale-512 pair 虽可执行，但 RaPO 结果明显不利；
2. RaPO 对 FP16 loss scale 高度敏感；
3. 4090 推理复评没有消除差距；
4. 当前链已经混合多项 2080 Ti 兼容设置；
5. 继续 Task 7 不能回答数值稳定性和正式配置问题；
6. 需要先进行 BF16 paired stability gate 和独立审查。

# 6. 2080 Ti 下降适配清单

| 设置 | 引入原因/阶段 | 是否共享 | 语义影响 | 后续建议 |
|---|---|---|---|---|
| FP16 | 2080 Ti 不支持 BF16 | GRPO/RaPO | 有数值稳定影响 | 仅 legacy；正式路径优先 BF16 |
| SDPA | Turing 不支持当前 FlashAttention-2 路径 | 共享 | 主要是 kernel/性能变化，仍可能影响数值 | legacy 保留；正式路径重新验证 FA2/SDPA |
| gradient checkpointing | 降低显存 | 共享 | 通常不改目标函数，但影响运行和数值顺序 | 是否保留由正式显存实测决定 |
| ZeRO-3 | 模型/参考模型分布式显存压力 | 共享 | 改变分布式执行；理论目标不变 | 4090 路径重新测，不自动删除 |
| CPU optimizer offload | 11 GiB 显存不足 | 共享 | 主要性能/数值执行路径差异 | 不应默认进入正式路径 |
| max pixels 100352 | 控制图像 token 和显存 | 共享 | 会改变视觉输入分辨率，属于实验语义偏差 | 正式路径必须重新冻结 |
| completion length 32 | 控制生成显存和时间 | 共享 | 可能截断 reasoning/answer | 仅 bounded profile |
| prompt limit 512 | 早期 2080 path | 共享 | 超限时存在已确认 Trainer bug | Task 1～5 未触发；不再安全 |
| prompt limit 1024 | Task 6 规避截断问题 | 共享 | 当前是规避，不是根治 | 正式路径需 assert/修复 |
| max steps 20 | 控制成本 | 共享 | 实际约 2.92 epochs，不符合论文 2 epochs | 仅 bounded profile |
| 5 eval samples/class | 早期快速 gate | 共享 | 样本过少，不可排名 | 已由 20/class 取代 |
| 20 eval samples/class | Task 1～6 bounded 评估 | 共享 | 不是全 test set | 仅诊断 |
| `save_strategy=no` | 避免约 26 GiB optimizer checkpoint | 共享 | final model 可保存，但无法可靠 task 内恢复 | bounded 使用；正式链需恢复策略 |
| `RAPO_DISABLE_CUDNN=1` | Task 5 可复现 cuDNN backward error | 共享 | 改变 convolution kernel 路径 | 仅 legacy；正式路径默认不应开启 |
| loss scale 65536 | 上游初始路径 | 共享诊断 | optimizer update 溢出并跳过 | 不使用 |
| loss scale 4096 | 初期 reduced gate | 共享诊断 | 8-rollout 多步仍会 skip | 不用于稳定链 |
| loss scale 2048 | Task 1～4 | 共享 | 当时稳定，但 Task 5 no-cuDNN 下失效 | legacy 历史配置 |
| loss scale 1024 | Task 5 formal、Task 6 diagnostic | 共享于 Task 5 | Task 6 GRPO 仍 skip；RaPO 结果与 scale512 差异大 | 不应成为正式默认 |
| loss scale 512 | Task 6 formal bounded pair | 共享 | 技术上稳定，但最终行为明显变化 | 仅 Task 6 诊断 |
| 2 early rollouts | 最初可行性 gate | 仅早期 | 不符合论文 `n=8` | 已退役 |
| 8 rollouts | 稳定 gate 起 | 共享 | 与论文一致 | 正式路径应保留 |
| per-device batch 1、GA=2 | 显存/上游默认 | 共享 | 改变 effective batch 和 CTAN update schedule | 正式冻结前重审 |
| cuDNN allocator/cache flush tolerance | 高显存压力 | 共享 | 主要吞吐影响 | 正式路径重新测 |

结论：

- FP16 loss-scale、禁用 cuDNN、短 completion、小 max_pixels、bounded eval 和 `max_steps=20` 应明确留在 `legacy/smoke` profile。
- ZeRO-3、gradient checkpointing 和 CPU offload 是否仍需使用，必须通过 4090/L40 等目标硬件的实测决定。
- BF16 正式 profile 不应继承 FP16 dynamic loss-scale 配置。

# 7. 当前实验结果与异常

## 7.1 Task 6 完整 bounded 矩阵

每个 cell 为 400 个固定样本。

### GRPO scale-512

| After task | T1 | T2 | T3 | T4 | T5 | T6 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 380 | - | - | - | - | - |
| 2 | 372 | 379 | - | - | - | - |
| 3 | 364 | 380 | 371 | - | - | - |
| 4 | 345 | 375 | 363 | 365 | - | - |
| 5 | 344 | 375 | 346 | 361 | 366 | - |
| 6 | 288 | 357 | 330 | 355 | 353 | 357 |

- Last Accuracy：`2040/2400 = 85.00%`
- Forgetting：`8.95%`

### RaPO scale-512

| After task | T1 | T2 | T3 | T4 | T5 | T6 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 380 | - | - | - | - | - |
| 2 | 374 | 379 | - | - | - | - |
| 3 | 359 | 381 | 370 | - | - | - |
| 4 | 281 | 370 | 360 | 375 | - | - |
| 5 | 281 | 363 | 336 | 366 | 363 | - |
| 6 | 93 | 187 | 240 | 342 | 332 | 372 |

- Last Accuracy：`1566/2400 = 65.25%`
- Forgetting：`33.75%`

## 7.2 客观观察

- 差距从 Task 4 开始明显。
- Task 5 差距没有消失。
- Task 6 scale-512 下，RaPO 旧任务准确率大幅下降。
- RaPO Task 6 新任务准确率为 `372/400`，高于 GRPO 的 `357/400`。
- 因此 RaPO 并非没有学会新任务，主要异常是旧任务保留。
- Task 6 paired audit 显示：
  - T1：GRPO-only 196，RaPO-only 1；
  - T2：GRPO-only 171，RaPO-only 1；
  - T3：GRPO-only 91，RaPO-only 1；
  - T6：GRPO-only 1，RaPO-only 16。
- 代表性旧任务错误包括：
  - `axolotl → newt`
  - `stingray → mantis`
  - `pelican → bird`

## 7.3 loss-scale 敏感性

同一 Task 5 RaPO 起点：

- scale 1024 RaPO：
  - Last Accuracy 72.7083%
  - Forgetting 24.50%
- scale 512 RaPO：
  - Last Accuracy 65.25%
  - Forgetting 33.75%

这说明结果对 FP16 执行路径敏感。

但现有证据不能证明：

- loss scale 是差距的唯一原因；
- scale 1024 更接近论文；
- RaPO 算法本身必然不稳定；
- GRPO 在 scale 1024 下会得到何种公平结果，因为对应运行 skipped update。

## 7.4 当前证据能排除的解释

现有证据降低了以下解释的优先级：

- Task 4～6 不是 evaluation key 错配；
- 不是 GRPO/RaPO 使用不同样本顺序；
- 不是 duplicate prediction 造成；
- 不是缺失 `<answer>` 造成；
- 不是 Task 6 final model 未保存；
- 不是 2080 Ti 推理相对 4090 导致的大幅差距；
- Task 1～5 没有触发 512 prompt limit；
- Task 6 使用 1024 后没有触发已知截断边界；
- bounded 32-token Task 4～6 audit 没发现 missing answer。

## 7.5 当前证据不能证明的结论

- 不能把 Task 6 结果当成论文反例；
- 不能断言当前 Trainer 集成完全正确；
- 不能断言 Retention Reward 或 CTAN 本身导致遗忘；
- 不能断言 BF16 正式路径会复现同一现象；
- 不能断言当前 provisional class order 具有代表性；
- 不能断言论文结果可在 4090 上按当前配置重现；
- 不能将单次 order-0 bounded 结果推广到 3-order 正式结果。

# 8. 已确认问题和未解决风险

## 8.1 `confirmed_bug`

### P-01 Prompt 截断与 generate 输入不一致

Visual-RFT Trainer：

1. 对局部 `prompt_ids` 和 `prompt_mask` 做 `[-max_prompt_length:]`；
2. 但调用 `generate(**prompt_inputs)` 时仍传原始、未截断 inputs；
3. 随后使用截断后的 `prompt_length` 切分 generation output。

超长 prompt 时可能导致 prompt/completion 边界错位和 log-probability 计算错误。

当前状态：

- Task 1～5 prompt 未超过 512，未触发；
- Task 6 改为 1024，最大 567，暂时绕开；
- bug 本身没有修复；
- 正式路径必须增加一致截断或 fail-fast assertion。

### P-02 旧 checkpoint 的 model type dispatch

已修复。当前使用 `AutoConfig.model_type`，不再依赖目录名称。

### P-03 分布式 Dataset.map cache 并发

已通过 `main_process_first` 缓解并在 Task 3 重跑验证。

### P-04 过小图像评估崩溃

已通过最小 spatial-factor padding 缓解。

## 8.2 `already_mitigated_not_fixed`

- prompt=1024 只绕开截断 bug；
- `RAPO_DISABLE_CUDNN=1` 绕开 2080 Ti cuDNN 错误，但没有修复根因；
- lowering loss scale 避免 skipped update，但造成明显行为敏感性；
- atomic JSONL 防止 partial result 冒充完成，但不验证 expected manifest count；
- fixed sorted subset 提供可比性，但不等于正式全量评估。

## 8.3 `unresolved_risk`

### 状态连续性

`rapo_state.json` 保存 `task_index`，但加载时只校验：

- format version；
- retention/CTAN settings；
- normalizer state。

它不验证：

- saved task 是否恰好为当前 task-1；
- state 与 model 是否来自同一 run；
- model 与 manifest/class order 是否一致；
- GRPO/RaPO chain 是否串错；
- Git commit、Visual-RFT commit、DeepSpeed config 是否匹配。

### CTAN 更新语义

当前 CTAN 在每次 Trainer forward/microbatch 更新。Task 1～6 每任务 20 optimizer steps，但 CTAN 每任务增加 38 updates。

论文文字写“at every optimization step”，当前实现可能与作者预期不同。必须独立判断：

- 每 microbatch 更新；
- 每 gradient accumulation window 更新一次；
- 或使用其他全局 batch 口径。

### world size 与分布式语义

当前 denominator 使用全局 gather 后的 reward std，而 numerator 使用本地 rollout group mean。变化以下设置会改变 CTAN：

- world size；
- per-device batch；
- gradient accumulation；
- rank 上的 rollout group 排列；
- CTAN 更新次数。

因此不能把 8 卡链静默改成 3～4 卡而仍称为同配置。

### 训练预算

20 optimizer steps 约为 2.92 epochs，不是论文 2 epochs。当前 scheduler 也按 20 steps 运行。正式实验必须改为 epoch-based flow 并验证 dataloader 与 partial accumulation 行为。

## 8.4 `paper_ambiguity`

论文未公开：

- 学习率；
- optimizer 细节；
- per-device batch；
- gradient accumulation；
- max prompt/completion length；
- max image pixels；
- attention implementation；
- precision；
- DeepSpeed/FSDP 细节；
- 3 个具体 class orders；
- 每类具体 5-shot 样本；
- CTAN 第一个 batch 的初始化方式；
- distributed `sigma_batch` 口径；
- Last Accuracy 在不等 task test size 下的 micro/macro 细节。

## 8.5 `experiment_protocol_deviation`

- 2080 Ti 而非 8×H100；
- FP16 而非已确认的作者 dtype；
- SDPA；
- reduced max pixels；
- completion 32；
- max steps 20；
- order 0 单次运行；
- bounded 20 samples/class；
- Task 5 起 cuDNN disabled；
- Task 1～4、Task 5、Task 6 使用不同 initial loss scales；
- Task 6 prompt limit 与前五任务不同；
- 无正式 2 epochs；
- 无三顺序均值/方差。

## 8.6 `missing_infrastructure`

- 没有完整 10-task orchestration runner；
- 没有正式 full-test inference runner；
- 没有统一 machine-readable run manifest；
- 没有自动 task/model/state/provenance chain validation；
- 没有经过验证的 task 内断点续跑；
- 没有 prediction sample-key deduplication；
- 没有 manifest expected count 校验；
- evaluator 将 FP16/SDPA 写死；
- 没有分布式端到端自动集成测试；
- 没有 CI；
- 没有正式 4090/BF16 profile；
- 没有明确分离 legacy 2080 profile。

# 9. 关键历史决策及其原因

## 9.1 固定 Visual-RFT commit

Visual-RFT 上游安装脚本依赖 moving branches，并存在冲突的 vLLM 依赖。固定 commit 可以：

- 避免上游漂移；
- 使 patch 可审阅；
- 让 Trainer 行为和依赖可追踪；
- 防止后续代码变化被误认为 RaPO 效果。

## 9.2 核心独立实现、patch 接入

Retention Reward 和 CTAN 先在 `src/rapo` 独立实现，是为了：

- 直接对照公式；
- 在 CPU 上测试；
- 不把数学实现埋进大型 Trainer；
- 便于未来替换 upstream；
- 避免 vendor 整个 Visual-RFT。

## 9.3 先做 reduced engineering gates

在未知显存、kernel 和分布式兼容性时，直接启动 10-task 长实验风险过高。先验证：

- 单卡模型加载；
- 一步 optimizer；
- 8 rollouts；
- save/reload；
- Task 2 anchor/state；
- evaluator；
- 小规模持续学习矩阵。

## 9.4 使用 2080 Ti

当时最明确可调度的是 8×RTX 2080 Ti。用户要求先充分利用现有资源，并只在 gate 失败或资源不足时申请高级卡。

## 9.5 禁用 cuDNN

Task 5 RaPO backward 在 clean launches 中重复触发 `CUDNN_STATUS_INTERNAL_ERROR`，cuDNN v7 兼容模式也失败。禁用 cuDNN 是明确记录的硬件兼容 workaround，不是算法要求。

## 9.6 降低 loss scale

多次出现“进程成功但 optimizer update 被跳过”。判断依据包括：

- repeated learning rate；
- stale/zero gradient norm；
- DeepSpeed `overflow=True`；
- 空 Adam state。

因此依次尝试 65536、4096、2048、1024、512。每次公平比较均要求两个方法使用相同 scale；Task 6 scale1024 因 GRPO 失败，最终正式 bounded pair 重跑为 scale512。

## 9.7 Task 6 prompt 改为 1024

Task 1～5 最大 prompt 为 508；Task 6 最大为 567。已知 Trainer 截断路径不安全，因此把两个方法统一改为 1024，以避免进入错误路径。

## 9.8 Task 7 暂停

继续 Task 7 不能回答：

- FP16 数值敏感性；
- BF16 是否稳定；
- Task 4～6 差距来自实现、配置还是随机性；
- 正式硬件 profile 如何设置。

因此暂停比继续累积混合配置结果更合理。

## 9.9 需要 BF16 gate

BF16：

- 避免 FP16 dynamic loss-scale 机制；
- 可以检查 Task 6 大幅差距是否依赖 FP16 数值路径；
- 更接近 H100 时代的常见训练精度；
- 但仍需证明当前模型、DeepSpeed、attention 和硬件组合可运行。

## 9.10 正式实验必须从 Task 1 重跑

当前 Task 1～6 权重已经受到以下历史配置影响：

- FP16；
- SDPA；
- max pixels 100352；
- completion 32；
- `max_steps=20`；
- 不同 loss scales；
- Task 5～6 cuDNN disabled；
- Task 6 prompt limit 改变；
- CTAN state 在当前 world size/microbatch schedule 下累计。

若从 Task 6 checkpoint 切换 BF16/high-end profile 再继续 Task 7，会产生不可解释的混合链。现有 Task 5 checkpoint 最多用于 Task 6 BF16 诊断；正式结果必须在配置冻结后从 base model、Task 1 重新运行。

# 10. 当前可用硬件和硬件变化

## 10.1 RTX 2080 Ti 主节点

当前只读探测：

- 8×NVIDIA GeForce RTX 2080 Ti；
- 每卡 `11264 MiB`；
- compute capability `7.5`；
- 双 NUMA；
- `nvidia-smi topo -m` 显示 PIX/NODE/SYS PCIe 路径；
- 没有显示 NVLink 连接。

它适合：

- CPU/GPU plumbing；
- legacy smoke；
- 小规模兼容诊断。

不适合默认承担：

- BF16；
- 当前 FlashAttention-2；
- 高分辨率、长 completion 的正式 10-task run。

GPU 占用是动态状态，每次运行前必须重新检查。

## 10.2 集群高级卡

用户已确认集群存在：

- RTX 3090；
- RTX 4090；
- NVIDIA L40；
- RTX 5090；
- 多台 RTX 2080 Ti 节点。

但以下信息仍待重新探测：

- 当前空闲卡数量；
- 是否存在连续可用的同质 8 卡；
- 每节点卡数；
- GPU 间拓扑；
- 可使用时长；
- 是否允许跨节点；
- NCCL 状态；
- 目标节点驱动和 CUDA；
- 高端节点是否已安装对应训练依赖。

不得仅凭“集群有 4090”假设已有完整正式资源。

## 10.3 4090 已验证和未验证内容

已验证：

- 单卡 RTX 4090 可以加载 Task 6 RaPO model；
- FP16+SDPA bounded inference 成功；
- 输出与 2080 Ti 推理的 aggregate metrics 基本一致。

未验证：

- BF16 training；
- 8×4090 distributed training；
- FlashAttention-2；
- ZeRO-3 是否仍必要；
- CPU offload 是否仍必要；
- 正式 image/completion budget；
- 长时间稳定性。

## 10.4 4090 不等于 H100

即使使用 8×4090，也仍与论文的 8×H100 存在：

- 显存容量；
- Tensor Core/吞吐；
- GPU 互联；
- 通信拓扑；
- kernel；
- 训练稳定性；
- batch/sequence 可承载范围

等差异。它可以成为更可靠的独立复现硬件，但不能自动称为 paper-identical hardware。

## 10.5 world size 约束

减小 world size 会改变：

- effective batch；
- gradient accumulation；
- 每 epoch optimizer-step 数量；
- global reward std；
- CTAN update 次数；
- rollout group/rank 分布；
- 通信和数值顺序。

因此 BF16 gate 应优先保持 homogeneous world size 8。若只能使用更少 GPU，必须把它作为新实验配置重新定义，而不是当前链的直接延续。

# 11. 远程环境和 artifact 清单

## 11.1 执行约定

- 主仓库：
  `/home/zhenglifeng/projects/RaPO`
- patched Visual-RFT：
  `/home/zhenglifeng/projects/Visual-RFT-RaPO-2b5561a`
- Conda：
  `/home/zhenglifeng/miniforge3`
- 环境：
  - `rapo`
  - `rapo-train`
- 模型：
  `/home/zhenglifeng/models/Qwen2-VL-2B-Instruct-895c3a4`
- 数据：
  - `/home/zhenglifeng/data/imagenet-r`
  - `/home/zhenglifeng/data/rapo-imagenet-r/order_0`
- 输出：
  `/home/zhenglifeng/outputs/rapo-smoke`
- 日志：
  `/home/zhenglifeng/logs/rapo-smoke`
- 结果：
  `/home/zhenglifeng/results/rapo-smoke`

实验室机器的 `/home/zhenglifeng` 跨相关节点共享。用户也被允许在 `/mnt/Datasets` 下创建个人数据目录，但当前实际数据位于个人 home。

远程任务约定：

1. 运行前 `nvidia-smi`；
2. 显式设置 `CUDA_VISIBLE_DEVICES`；
3. 长任务使用 tmux；
4. 合理设置 CPU workers；
5. 不修改公共系统环境；
6. 需要集群代理时使用实验室提供的网络脚本，结束后恢复；
7. 数据、模型和日志不进 Git。

## 11.2 当前环境

`rapo-train` 当前实测：

- PyTorch `2.5.1+cu124`
- Transformers `4.49.0.dev0`
- TRL `0.14.0`
- DeepSpeed `0.15.4`
- datasets `3.6.0`
- `flash_attn`：当前未安装

因此 README/runbook 中固定的 FlashAttention `2.7.4.post1` 是可选目标版本，不是当前环境已安装事实。

## 11.3 数据与模型 provenance

ImageNet-R archive：

- bytes：`2191079936`
- SHA-256：
  `18c6bf493b39a0d975d48e587437f562caab9c52ae6327dcfa9dd8eb54aa1b52`

当前 order-0 manifest SHA-256：

`711f193ad1cc9864bb8dc0c1299c02d3ebc5ab88e434689d6cc2ab652ff1d977`

当前磁盘规模：

- source ImageNet-R：约 2.1 GiB
- order-0 exports：约 11 GiB

模型文件来自 ModelScope mirror，但逐文件哈希与固定 Hugging Face revision 对齐。完整 model revision 已记录在 `PROVENANCE.txt`。

## 11.4 Task 6 三个关键模型

每个目录约 4.2 GiB：

```text
/home/zhenglifeng/outputs/rapo-smoke/
├── 2080ti-grpo-task06-step20-n8-scale9-nocudnn-prompt1024-nosave
├── 2080ti-rapo-task06-step20-n8-scale9-nocudnn-prompt1024-nosave
└── 2080ti-rapo-task06-step20-n8-scale10-nocudnn-prompt1024-nosave
```

另外保留：

- Task 6 RaPO scale1024 one-step gate；
- Task 1～5 两条历史 final model 链；
- 若干 superseded/diagnostic 目录。

不能仅凭目录存在把 superseded 输出当成 formal artifact。

## 11.5 Task 6 结果哈希

目录：

`/home/zhenglifeng/results/rapo-smoke/eval-20-per-class-step20/`

| Artifact | SHA-256 |
|---|---|
| `grpo-after6.jsonl` | `552588ce72e5f604f7821059e553bb1269d1dff67025607413e17d342782ac05` |
| `rapo-after6.jsonl` | `7d6b458daa4c1e38b6c28a7ba16f7a2319d681cdc598e18db700a23ee0a49a94` |
| `grpo-metrics-6task.json` | `6bdeb2e4016686a9a33e1a408986867469650fef72b3995c606a21b247e38665` |
| `rapo-metrics-6task.json` | `abecebce6fc9e557a04919efd63888101828a679d0300aa2f5081bae68689728` |
| `task06-paired-audit.json` | `2caa42323d040b6b386bc53ca6de25cbf8411c0f612ab8ac549b436111e1571f` |
| `rapo-scale10-after6.jsonl` | `8ff2a65983b08a4d60b6fe537900b38eb90597edf260c55e5e7c471968b0f895` |
| `rapo-scale10-metrics-6task.json` | `33013e5ce3660e8556bf082c3cf98fb5f5807e67f20ef59aeb5817dabaac7699` |
| `rapo-scale9-4090-after6.jsonl` | `d36586fffb4e81ee6d0b5f12a3f508b66ac51a184b3ff2dc4077ccc7be4bf281` |
| `rapo-scale9-4090-metrics-6task.json` | `dd7e5d19db5261c1d43846e0e6df534026778e063533eda4af52c94963756b2e` |
| `task06-scale9-inference-hardware-audit.json` | `878a5fe1b15c065cc25388d8aacb88c463ecb5ef2673cb4669228e38b8dd10eb` |

## 11.6 关键失败日志

Task 5 日志目录保留：

- repeated cuDNN failures；
- cuDNN v7 failure；
- disabled-cuDNN FP16 skipped-update failure；
- formal scale1024 success logs。

Task 6 明确保留：

```text
2080ti-grpo-task06-step20-n8-scale10-nocudnn-prompt1024-nosave.failed-fp16-skip.log
2080ti-grpo-task06-step20-n8-scale9-nocudnn-prompt1024-nosave.log
2080ti-rapo-task06-step1-n8-scale10-nocudnn-prompt1024.log
2080ti-rapo-task06-step20-n8-scale10-nocudnn-prompt1024-nosave.log
2080ti-rapo-task06-step20-n8-scale9-nocudnn-prompt1024-nosave.log
```

另有三份 Task 6 RaPO scale512 tmux 启动命令错误日志：

- `failed-launch-no-cd`
- `failed-launch-cwd2`
- `failed-launch-sendkeys`

这些失败没有形成训练结果，属于执行历史，不应混入方法比较。

## 11.7 Git 中没有的 artifacts

以下只存在本机 ignored 或服务器，不在公开 Git：

- 论文 PDF；
- 实验室服务器规范；
- ImageNet-R；
- exported Arrow datasets；
- Qwen 模型；
- final models；
- optimizer checkpoints；
- full logs；
- predictions；
- metrics JSON；
- audit JSON；
- tmux launcher helpers。

Git 只保存：

- 代码；
- 配置；
- patch；
- 测试；
- 小型事实型实验文档；
- artifact 路径和哈希。

# 12. 尚未完成的工作

以下只列优先级，不授权立即实施。

## P0：必须先处理

1. **新 agent 独立审查**
   - 重新从论文公式到 Trainer 数据流核对；
   - 不直接接受“未发现 material mismatch”的旧结论；
   - 特别关注 CTAN update semantics、global std、anchor logps 和 prompt/completion slicing。

2. **BF16 数值稳定性验证**
   - 探测一组同质 8 GPU；
   - 确认 BF16、attention 和 DeepSpeed 兼容；
   - 对 GRPO/RaPO 做公平的 Task 6 paired diagnostic；
   - 可以从 Task 5 checkpoint 做诊断，但不得把它当正式链。

3. **正式配置冻结**
   - dtype；
   - attention；
   - GPU/world size；
   - ZeRO/offload；
   - batch/GA；
   - prompt/completion；
   - max pixels；
   - epoch scheduler；
   - checkpoint；
   - evaluation；
   - seed；
   - run manifest。

4. **判断 Task 4～6 差距来源**
   - 数值路径；
   - CTAN 更新频率；
   - 短预算；
   - 数据顺序；
   - Trainer 语义；
   - 随机性；
   - 算法行为。

## P1：正式链前完成

1. 修复或 fail-fast 防止 prompt 截断不一致；
2. model/state/manifest/class-order/config provenance 校验；
3. 严格 2-epoch runner；
4. 明确 CTAN update semantics；
5. full-test evaluation；
6. task 内可恢复 checkpoint；
7. expected prediction count 和去重；
8. 可配置 evaluator dtype/attention；
9. 建立 4090/BF16 formal profile；
10. 保留独立 2080 Ti legacy profile；
11. 配置冻结后从 Task 1 重启。

## P2：工程完善与最终实验

1. machine-readable run manifests；
2. CI；
3. patch apply/integration regression test；
4. distributed end-to-end integration test；
5. micro/macro metric 口径确认；
6. docs 同步；
7. 三个 class orders；
8. full 10-task GRPO/RaPO；
9. mean/std；
10. 最终 reproduction report。

# 13. 给新 agent 的建议阅读顺序

1. 本地论文 `2605.09640v1.pdf`
   - Section 3.1～3.3；
   - Eq. (2)～(6)；
   - Section 4.1；
   - prompt/reward appendix；
   - ablations。

2. `docs/reproduction_spec.md`

3. `src/rapo/core.py`

4. `src/rapo/integration.py`

5. `patches/visual_rft_2ffad63_rapo.patch`

6. 固定 Visual-RFT 的：
   `grpo_trainer.py` 与 `grpo_classification.py`

7. `src/rapo/data.py`

8. `src/rapo/evaluation.py`

9. `scripts/run_imagenet_r_smoke.sh`

10. `scripts/run_imagenet_r_2080ti_smoke.sh`

11. `scripts/evaluate_qwen2_vl.py`

12. `docs/experiments/2026-07-30-2080ti-four-task-step20.md`

13. `docs/experiments/2026-07-30-2080ti-five-task-step20.md`

14. `docs/experiments/2026-07-30-five-task-implementation-audit.md`

15. `docs/experiments/2026-07-31-2080ti-six-task-stability-boundary.md`

16. 本 handoff

17. 独立形成自己的：
   - paper-to-code matrix；
   - confirmed bugs；
   - unresolved risks；
   - BF16 gate 方案；
   - 分阶段整改计划。

# 14. 新 agent 不应默认接受的结论

- 不要因为 29 个测试通过就认定 Trainer 集成正确。
- 不要因为公式逐行相似就认定分布式实验语义一致。
- 不要把一次“未发现 material mismatch”视为正确性证明。
- 不要把 2080 Ti compatibility settings 当正式默认。
- 不要把 Task 6 的差距直接归因于 RaPO 算法。
- 不要把 loss-scale 敏感性视为已证明的因果关系。
- 不要认为 scale512 比 scale1024 更“正确”。
- 不要直接继续 Task 7。
- 不要把当前 Task 1～6 混合配置链作为最终结果。
- 不要从 Task 6 checkpoint 切 BF16 后继续并称为正式实验。
- 不要假设目前有空闲同质 8×4090。
- 不要假设 4090 推理成功等于 BF16 训练成功。
- 不要假设 FlashAttention 已安装。
- 不要假设官方代码仍未发布；每次新会话应在线重查。
- 不要假设论文未公开超参数的值。
- 不要在 formal config 冻结前启动 10-task run。
- 不要静默改变 world size、gradient accumulation 或 rollout grouping。
- 不要把 provisional seeds `0/1/2` 冒充作者官方 class orders。
- 不要把 bounded 20 samples/class 冒充 full test evaluation。
- 不要忽略失败日志和 skipped optimizer updates。

# 15. Open questions

## 硬件与环境

1. 当前确切可用的 RTX 4090/L40 数量是多少？
2. 是否存在同一节点、同质、可连续占用的 8 GPU？
3. GPU 间拓扑和 NCCL 是否稳定？
4. 可以使用多久？
5. 目标节点能否稳定运行 BF16？
6. 当前 PyTorch 2.5.1/CUDA 12.4 是否适合目标卡？
7. 是否安装和支持 FlashAttention 2？
8. RTX 5090 是否需要单独升级 PyTorch/CUDA/FlashAttention？
9. 是否仍需 ZeRO-3？
10. 是否仍需 CPU optimizer offload？
11. 是否仍需 gradient checkpointing？

## 算法与 Trainer

12. CTAN 应按 microbatch 还是 optimizer step 更新？
13. `sigma_batch` 应是 per-rank、global samples 还是 optimizer batch？
14. task boundary 的 CTAN state 是否需要额外 bias correction？
15. Task 1 是否应从第一 microbatch 起使用 CTAN？
16. frozen anchor 的构造和 token logps 是否与作者一致？
17. format reward 的实际权重和 aggregation 是否与官方实现一致？
18. prompt 正确截断应该如何与 multimodal image tokens 对齐？
19. standard GRPO KL 配置是否与论文一致？

## 正式配置

20. 正式 max pixels 是多少？
21. 正式 prompt limit 是多少？
22. 正式 completion length 是多少？
23. 正式 learning rate、batch、GA、warmup/scheduler 是什么？
24. 是否使用 FlashAttention 2 还是 SDPA？
25. 正式 checkpoint 频率和保留策略是什么？
26. 需要多少重复 run 才能判断数值稳定性？
27. Task 6 BF16 gate 是否先从现有 Task 5 checkpoint 做诊断？
28. 正式 3-order 实验的总 GPU 时预算是多少？
29. metrics 应使用何种 micro/macro 口径？
30. 作者官方代码和数据划分何时发布？

---

# 附录 A：一页式项目摘要

本工程独立复现 RaPO 论文在 ImageNet-R 上的 rehearsal-free、5-shot、10-task class-incremental image classification。当前已实现 Retention Reward、CTAN、状态持久化、deterministic 数据构造、exact-match evaluator 和固定 Visual-RFT Trainer patch，并在 8×RTX 2080 Ti 上完成 Task 1～6 的 bounded GRPO/RaPO 链。

Task 1～3 两者接近；Task 4 起 RaPO 旧任务遗忘明显增大。Task 6 的 scale512 对照中，GRPO 为 `85.00% / 8.95%`，RaPO 为 `65.25% / 33.75%`；RaPO scale1024 诊断为 `72.71% / 24.50%`。同一 scale512 模型在 2080 Ti 与 4090 上评估指标一致，基本排除了推理 GPU 是主要原因，但结果对 FP16 loss scale 敏感。

当前结果不是论文级结果：只覆盖一个 provisional class order、20 steps、20 samples/class，并混用了 FP16、SDPA、低图像预算、cuDNN workaround 和不同 loss scales。Task 7 已暂停。下一步应先进行独立审查和 8 卡 BF16 Task 6 paired stability gate，冻结正式高端 GPU 配置，然后从 base model 的 Task 1 重新运行完整 10-task、3-order 实验。

# 附录 B：当前需求检查表

| Requirement | Status | Evidence | Remaining work |
|---|---|---|---|
| 独立复现 RaPO | 部分满足 | core、patch、实验链 | 需正式 10-task |
| 只聚焦 RaPO，不做 OPD | 满足 | README scope | 保持边界 |
| 论文公式实现 | 已实现待复审 | `core.py`、tests | 新 agent 独立审计 |
| 真实 Trainer 接入 | 部分满足 | Visual-RFT patch、Task 1～6 | 分布式语义复核 |
| Retention Reward | 已实现 | Eq.2～4 mapping | BF16 validation |
| CTAN | 已实现有风险 | state + global std | update semantics |
| Task 2+ anchor | 部分满足 | ref model 路径 | provenance 强校验 |
| ImageNet-R 200 类 | 满足 | manifest builder | 官方划分待发布 |
| 10×20 classes | 数据支持 | manifest | 只运行到 T6 |
| 5-shot | provisional 满足 | deterministic split | 非官方 5-shot |
| 无 replay | 满足 | current-task train export | 正式验证 |
| Qwen2-VL-2B | 满足 | pinned model | 正式 profile |
| 每任务 2 epochs | 未满足 | 当前 20 steps≈2.92 epochs | epoch runner |
| GRPO/RaPO 公平对照 | bounded 满足 | paired configs | formal BF16 pair |
| 8 rollouts | 满足 | Task 1～6 | 正式保持 |
| α=20、λ=0.5、β=0.999 | 满足 | configs/state | 继续固定 |
| cumulative prompt | 满足 | data.py | 截断防线 |
| exact-match reward | 满足 | evaluator/patch/tests | format path 复核 |
| Last Accuracy | 部分满足 | bounded metrics | full test/3 orders |
| Forgetting | 部分满足 | bounded metrics | 口径确认 |
| 完整矩阵 | 部分满足 | Task 1～6 | 10 tasks |
| 三个 class orders | 未满足 | only order0 | orders 0/1/2 provisional |
| 随机种子记录 | 部分满足 | seed0/data_seed0 | run manifest |
| 上游 commit 固定 | 满足 | patch/launcher | 持续校验 |
| checkpoint/state 防错 | 部分满足 | existence/settings checks | chain hash validation |
| 可恢复训练 | 未满足 | resume flag 存在但未完整验证 | formal resume test |
| 结果追踪 | 部分满足 | docs、hashes、logs | unified manifest |
| 2080 legacy path | 已存在但未正式分层 | wrapper | profile separation |
| 4090/BF16 formal path | 未完成 | 仅 4090 inference | 训练 gate |
| CI | 未满足 | 无 workflow | 建立 CI |
| 正式 Task 1 重启 | 未执行 | pause decision | config freeze 后执行 |
| 新 agent 无损接手 | 满足 | 本 handoff | 后续随工程状态维护 |

# 附录 C：关键配置演变表

| 阶段 | GPU | dtype/attention | Rollouts | Prompt/Completion | Pixels | Steps | cuDNN | FP16 initial scale | 解释 |
|---|---|---:|---:|---|---:|---:|---|---:|---|
| 单卡 load | 1×2080 Ti | FP16/SDPA | - | - | - | 0 | default | - | 模型加载 gate |
| 初始 optimizer | 8×2080 Ti | FP16/SDPA | 2 | 512/32 | 100352 | 1～5 | default | 65536 | overflow、skip |
| 两 rollout gate | 8×2080 Ti | FP16/SDPA | 2 | 512/32 | 100352 | 1～5 | default | 4096 | 可运行 |
| 八 rollout gate | 8×2080 Ti | FP16/SDPA | 8 | 512/32 | 100352 | 5 | default | 4096 | 部分 update skip |
| 稳定 gate | 8×2080 Ti | FP16/SDPA | 8 | 512/32 | 100352 | 5 | default | 2048 | 通过 |
| Task 1～2 5-step | 8×2080 Ti | FP16/SDPA | 8 | 512/32 | 100352 | 5/task | default | 2048 | sequential gate |
| Task 1～4 20-step | 8×2080 Ti | FP16/SDPA | 8 | 512/32 | 100352 | 20/task | default | 2048 | bounded chain |
| Task 5 失败 | 8×2080 Ti | FP16/SDPA | 8 | 512/32 | 100352 | 未完成 | error/v7/off | 2048 | cuDNN error 后 skip |
| Task 5 formal | 8×2080 Ti | FP16/SDPA | 8 | 512/32 | 100352 | 20 | disabled | 1024 | 两方法完成 |
| Task 6 scale1024 | 8×2080 Ti | FP16/SDPA | 8 | 1024/32 | 100352 | 20 | disabled | 1024 | RaPO 成功，GRPO step8 skip |
| Task 6 formal | 8×2080 Ti | FP16/SDPA | 8 | 1024/32 | 100352 | 20 | disabled | 512 | paired 完成 |
| Task 1～6 eval | 1×2080 Ti | FP16/SDPA | greedy | dataset prompt/32 new | 100352 | - | - | - | 20 samples/class |
| Task 6 control eval | 1×4090 | FP16/SDPA | greedy | dataset prompt/32 new | 100352 | - | - | - | 同模型硬件控制 |
| 未来正式路径 | 待确认同质 8 GPU | BF16，attention 待定 | 8 | 待冻结 | 待冻结 | 2 epochs | default 优先 | 不适用 | 必须从 Task 1 重启 |

共同 bounded training 参数：

- learning rate `1e-6`
- per-device train batch `1`
- gradient accumulation `2`
- GRPO KL beta `0.04`
- ZeRO-3
- CPU optimizer offload
- gradient checkpointing
- seed/data seed `0`
- `save_strategy=no` 的 20-step 链仍保存 final model，但不保存完整 optimizer checkpoint

# 附录 D：已知问题清单

| ID | 状态 | 严重度 | 证据 | 建议下一步 |
|---|---|---:|---|---|
| P-01 | confirmed_bug | 高 | Trainer 截断局部 ids，却向 generate 传原 inputs | 正确截断或 fail fast |
| P-02 | fixed | 中 | checkpoint 路径 model dispatch 失败 | 保留回归测试 |
| P-03 | mitigated | 中 | 多 rank Dataset.map cache 冲突 | 增加 distributed test |
| P-04 | mitigated | 低 | 27px image 评估失败 | 保留 padding test |
| P-05 | unresolved_risk | 高 | state task_index 不做 continuity check | 验证 `saved=t-1` |
| P-06 | reproducibility_gap | 高 | model/state/manifest 无共同 run ID | run manifest + hashes |
| P-07 | paper_ambiguity | 高 | CTAN 按 38 microbatches 更新而非 20 steps | 独立审计/对照实验 |
| P-08 | unresolved_risk | 高 | CTAN global std 依赖 world size | 冻结分布式语义 |
| P-09 | protocol_deviation | 高 | 20 steps≈2.92 epochs | 实现 2-epoch runner |
| P-10 | missing_infrastructure | 高 | 无 10-task orchestrator | 正式 runner |
| P-11 | missing_infrastructure | 高 | 无 full-test inference orchestration | 全量 evaluator |
| P-12 | missing_infrastructure | 高 | resume 未端到端验证 | interruption/resume test |
| P-13 | unresolved_risk | 中 | prediction 不按 sample key 去重 | key validation |
| P-14 | unresolved_risk | 中 | 不校验 manifest expected counts | count contract |
| P-15 | confirmed limitation | 中 | evaluator FP16/SDPA 写死 | CLI/config 化 |
| P-16 | paper_ambiguity | 中 | Last Accuracy micro/macro 未公开 | 向作者/论文实现核对 |
| P-17 | reproducibility_gap | 高 | class orders、5-shot 非官方 | 保留 manifest，等待官方 |
| P-18 | unresolved_risk | 中 | selective export 会重写 manifest | compare-before-write |
| P-19 | missing_infrastructure | 高 | 无 machine-readable run manifest | 设计 schema |
| P-20 | missing_infrastructure | 中 | 无 CI | 添加 CPU/patch checks |
| P-21 | hardware_compatibility | 高 | Task 6 FP16 scale 敏感 | BF16 paired gate |
| P-22 | protocol_deviation | 高 | Task 1～6 配置混合 | 正式链从 T1 重启 |
| P-23 | missing_dependency | 高 | 当前无 flash_attn | 正式节点验证安装 |
| P-24 | external_blocker | 高 | 官方代码未发布 | 定期重查/等待作者 |
| P-25 | governance_gap | 低 | 仓库无项目级 AGENTS.md | handoff 后再决定是否建立 |

# 附录 E：术语表

| 术语 | 含义 |
|---|---|
| RaPO | Retention-aware Policy Optimization；论文提出的持续学习 RFT 方法 |
| GRPO | Group Relative Policy Optimization；用组内相对奖励计算 advantage 的 RL 方法 |
| Retention Reward | 根据当前策略相对上一任务策略的轨迹漂移生成的保留奖励 |
| CTAN | Cross-Task Advantage Normalization；跨 batch、跨任务持久化奖励标准差 EMA |
| Anchor policy | 上一任务结束时冻结的策略，用作 Task `t` 的 retention reference |
| Reference model | Trainer 中的冻结模型；当前同时服务 GRPO KL 和 RaPO anchor logps |
| Trajectory drift | 当前 actor 与 anchor 在已生成 token 上的平均 log-probability ratio，经单侧截断 |
| One-sided truncation | 将负的平均 drift 截为 0，避免通过低置信输出提高 retention reward |
| Stop-gradient | drift/retention 只作为 scalar reward，不直接反向传播到 actor logits |
| Class-incremental learning | 每个任务引入不重叠新类别，最终需识别所有已见类别 |
| Rehearsal-free | 新任务训练时不使用旧任务训练样本 |
| Rollout group | 同一输入从当前策略采样的多条输出；论文 `n=8` |
| Group-relative centering | 每条 reward 减去同 prompt rollout group 的均值 |
| Microbatch | gradient accumulation 中的一次 forward/backward 输入 |
| Optimizer step | 累积一个或多个 microbatches 后真正更新参数的一步 |
| Loss scale | FP16 为避免 underflow 而放大 loss/grad 的尺度；overflow 时会降低并跳过更新 |
| BF16 | bfloat16；指数范围更大，通常无需 FP16 dynamic loss scaling |
| FP16 | IEEE half precision；本项目 2080 Ti path 使用，存在 overflow/skip 风险 |
| SDPA | PyTorch scaled dot-product attention；2080 Ti path 使用 |
| FlashAttention 2 | 高性能 attention kernel；当前服务器环境未安装，2080 Ti 不支持当前目标版本 |
| ZeRO-3 | DeepSpeed 将 model states 在 ranks 间分片的显存优化 |
| CPU optimizer offload | 将 optimizer state/计算部分放 CPU，以降低 GPU 显存 |
| Last Accuracy | 最终任务训练后，对所有已见类别测试集的总体准确率 |
| Forgetting | 旧任务历史最佳准确率到最终准确率的平均下降 |
| Provenance | 将代码、模型、数据、配置、状态、硬件和结果关联起来的来源链 |
| Bounded run | 缩小训练/评估预算的工程验证，不能自动视为正式论文结果 |
