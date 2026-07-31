# RaPO 独立审查（2026-07-31）

## 裁决摘要

- **当前不具备论文级正式实验放行条件。** 可以在人工批准后进入“整改批次 1”，但不得直接继续 Task 7、BF16 gate 或正式 10-task 实验。
- 本审查覆盖 A（论文/算法）、B（Trainer/分布式/状态链）、C（数据/评估/协议/provenance）、D（硬件/数值/artifact）四域。共列 22 条 finding（1 个 P0，18 个 P1，3 个 P2；其中 14 corroborated、7 deferred、1 rejected）；不为数量凑 finding。
- 最关键的已证实问题是：超长 prompt 的生成/切片边界不一致；CTAN 按 `compute_loss` microbatch 而非 successful optimizer step 更新；checkpoint/state/dataset/config 缺统一血缘；没有可证明等价的 resume；2-epoch、full-test、3-order、10-task 正式协议和评估完整性契约尚未落地。
- 现有 Task 1–6、2080 Ti/4090 结果只可作为 bounded、recorded-and-reread 诊断证据；没有重跑训练或推理，不应据此归因 RaPO 算法优劣。

状态含义：`corroborated` 为本轮证据支持；`rejected` 为旧结论被反证（单列于后）；`deferred` 为证据不足，不能当缺陷事实。证据标签严格区分 `paper fact`、`repo fact`、`live-read artifact/command`、`recorded-but-not-rerun`、`assumption`。

## 1. 任务 0：快照与范围基线

实际输出：

```text
> git status --short --branch
## main...origin/main [ahead 1]

> git rev-parse HEAD
158ee72f880addd6294907bc695213f5aa43f0f1

> git diff --name-status 77ed06c1da1f35368136675cc6b7569b07cb186d..158ee72f880addd6294907bc695213f5aa43f0f1
A       docs/handoff/current_project_handoff.md

> git hash-object docs/handoff/current_project_handoff.md
ffe4f3f7e7ae98d90d123bdef57e651980a2ee9f

> Get-FileHash -Algorithm SHA256 .\2605.09640v1.pdf
FC48B658FD3D96980F4733CBD9672AA21BE2F3B04E73958DE54BF4E0E15965E0
```

论文文件存在于 `C:\Users\Administrator\Desktop\RaPO\2605.09640v1.pdf`，SHA-256 与目标给定值一致。HEAD 与基线 diff 均匹配，因此没有因快照漂移而停止的审查域。

`rg -n "^def test_" tests` 实际列出 29 个测试函数：

```text
tests\test_core.py:14:def test_trajectory_drift_masks_normalizes_clamps_and_detaches():
tests\test_core.py:28:def test_trajectory_drift_rejects_empty_trajectory():
tests\test_core.py:34:def test_retention_reward_is_bounded_and_combines_with_task_reward():
tests\test_core.py:43:def test_ctan_persists_ema_across_task_boundary():
tests\test_core.py:56:def test_ctan_advantages_preserve_group_centering():
tests\test_core.py:72:def test_ctan_can_use_global_std_without_updating_during_evaluation():
tests\test_core.py:87:def test_ctan_rejects_incompatible_saved_configuration():
tests\test_evaluation.py:13:def test_classification_matching_follows_paper_normalization_and_exactness():
tests\test_evaluation.py:27:def test_tiny_images_are_padded_to_the_model_spatial_factor():
tests\test_evaluation.py:39:def test_prediction_aggregation_counts_each_accuracy_cell():
tests\test_evaluation.py:65:def test_continual_metrics_use_micro_last_accuracy_and_macro_forgetting():
tests\test_evaluation.py:86:def test_continual_metrics_require_complete_lower_triangle():
tests\test_integration.py:18:def test_task_one_uses_task_reward_without_retention():
tests\test_integration.py:35:def test_task_two_adds_detached_retention_reward():
tests\test_integration.py:58:def test_ctan_uses_global_reward_std_but_local_group_means():
tests\test_integration.py:80:def test_state_round_trip_across_task_boundary(tmp_path):
tests\test_integration.py:93:def test_state_rejects_changed_paper_settings(tmp_path):
tests\test_integration.py:109:def test_state_paths_require_rapo_to_be_enabled():
tests\test_data.py:41:def test_manifest_is_deterministic_disjoint_and_exhaustive(tmp_path):
tests\test_data.py:57:def test_class_order_and_sample_seeds_are_independent(tmp_path):
tests\test_data.py:76:def test_visual_rft_rows_use_current_training_data_and_cumulative_vocabulary(
tests\test_data.py:106:def test_class_map_supports_standard_imagenet_index_layout(tmp_path):
tests\test_data.py:124:def test_class_map_supports_official_imagenet_r_readme(tmp_path):
tests\test_data.py:141:def test_manifest_rejects_class_count_mismatch(tmp_path):
tests\test_data.py:154:def test_prompt_matches_closed_set_contract():
tests\test_data.py:163:def test_cli_can_export_only_one_visual_rft_task(tmp_path, monkeypatch):
tests\test_runtime.py:7:def test_cudnn_is_unchanged_by_default(monkeypatch):
tests\test_runtime.py:14:def test_cudnn_can_be_explicitly_disabled(monkeypatch):
tests\test_runtime.py:21:def test_invalid_cudnn_setting_is_rejected():
```

本机 `python -m pytest -q` 的既有收集失败（缺 `torch` 且 `rapo` 未安装）未重试为“通过”；handoff 的 “29 passed” 仍只是服务器记录。

## 2. 来源与现场证据核验

- `paper fact`：直接阅读上述 PDF 的 Eq.(2)–(6)、§4.1 与 Appendix A/B。论文明确：10×20 类、5-shot/no replay、3 个随机 class orders、全部已见类 test、每任务 2 epochs、8×H100；`n=8, α=20, λ=0.5, β=0.999`。Eq.(5) 明写 CTAN 在每个 optimization step 更新。
- 2026-07-31 在线核验：[arXiv 2605.09640](https://arxiv.org/abs/2605.09640) 仍只有 v1（2026-05-10）；[作者 RaPO 仓库](https://github.com/LMMMEng/RaPO) 仍只发布 README，并称代码待整理发布；其 `main` 为 [8367961e536810917b8f2dede3265380813c2841](https://github.com/LMMMEng/RaPO/commit/8367961e536810917b8f2dede3265380813c2841)。因此作者实现、官方 split 和未披露超参数均不能猜。
- 固定上游 [Visual-RFT 2ffad63b25ddd79bfe25d3e046645401201c89d6](https://github.com/Liuziyu77/Visual-RFT/commit/2ffad63b25ddd79bfe25d3e046645401201c89d6) 可读；远程 checkout HEAD 精确匹配，只有 `grpo_classification.py`、`grpo_trainer.py` 两个文件含本项目 patch，reverse-check 成功。
- 远程只读核验可用：主机时间 `2026-07-31T23:39:50+08:00`；训练 checkout 为 `77ed06c1da1f35368136675cc6b7569b07cb186d`；8 张 GPU 均为 RTX 2080 Ti、11264 MiB、compute capability 7.5、driver 580.95.05。
- 数据 tar 本轮重算 SHA-256：`18c6bf493b39a0d975d48e587437f562caab9c52ae6327dcfa9dd8eb54aa1b52`。三个 manifest hash 分别为 `711f193ad1cc9864bb8dc0c1299c02d3ebc5ab88e434689d6cc2ab652ff1d977`、`c7b44ce71ad7d6be9bc1e136cee63b843d97a9415b39b6838e62dbed5188d523`、`2b8669ef21824aecaa49f4e8b78f6c8bf49ebbde88720c4204185b109c029c5d`。
- 模型 provenance 指向 `Qwen/Qwen2-VL-2B-Instruct`、HF revision `895c3a49bc3fa70a340399125c650a463535e71c`，远程记录称逐文件 SHA-256 匹配；该“匹配”本轮只重读 provenance 文件，属于 `recorded-but-not-rerun`。
- 当前六任务预测 artifact 本轮只读统计均为 8400 行且 key 唯一；三个 after-6 文件均为 2400 行/2400 unique keys。它证明当前 bounded 文件未重复，不证明 evaluator 具备完整性防线。

## 3. Eq.(2)–(6) 与实现矩阵

| 论文项 | repo/固定上游映射 | 裁决 |
|---|---|---|
| Eq.(2) actor-anchor completion-token log-ratio与单侧截断 | `src/rapo/core.py:19-40`；远程 `grpo_trainer.py:436-446,496-500` | 非截断输入静态匹配；超长 prompt 被 AUD-006 破坏 |
| Eq.(3) detached `exp(-αd)` | `src/rapo/core.py:43-50` | corroborated |
| Eq.(4) `Rtask + λRret` | `src/rapo/core.py:53-59`; `src/rapo/integration.py:94-104` | corroborated |
| Eq.(5) EMA 与跨任务持久 | `src/rapo/core.py:90-106,150-169` | 持久化结构匹配；时钟 AUD-007；初始化 AUD-005 |
| Eq.(6) group mean / CTAN denominator | `src/rapo/core.py:130-148`; `src/rapo/integration.py:120-132` | numerator 匹配；distributed scope AUD-003 |
| Task 1 无 retention、Task 2 起启用 | `src/rapo/integration.py:86-104`; launcher `:127-143` | corroborated |
| Appendix prompt、`Racc+Rfmt`、exact normalization | `src/rapo/data.py:18-24`; `src/rapo/evaluation.py:13-43`; patched classification reward | 静态匹配；不等于端到端已证 |

论文没有编号 pseudocode 框；叙述算法链为：以前一任务最终权重初始化 actor 并冻结 anchor → 当前策略生成 rollout group → task 与 retention reward 合成 → CTAN 归一化 → 更新 actor → 跨任务保存 CTAN。

## 4. Findings

`AUD-001/002` 是 A 域对 prompt/CTAN 的候选重复项，去重后并入证据更完整的 `AUD-006/007`；为保持审查期间引用稳定，不重用编号。

### AUD-003 — deferred · P1 · distributed CTAN

- **结论：** repo 选择“跨 rank 全量 total rewards 的 sample std”，但论文未说明 `σbatch` 是 per-rank、global samples、prompt groups 还是 accumulation-window batch，也未说明 Bessel correction。
- **论文证据：** PDF p.5 Eq.(5) 仅称当前 batch rollout rewards 的 batch-level std；Eq.(6) 只明确 numerator 的同 prompt group mean。
- **代码证据：** `src/rapo/integration.py:120-132` 使用 `std(correction=1)`；`src/rapo/core.py:130-148`；远程 Trainer `:502-508` 使用 `accelerator.gather(rewards)`。
- **实验/命令证据：** 官方代码未发布；A、B 两域独立复核均只能确认工程选择，不能确认作者口径。
- **影响/置信度：** world size 与 batch 组织会改变 advantage scale；“披露缺口”高置信，作者口径未知。
- **整改批次/测试：** 批次 1 冻结决策；用同一逻辑 reward multiset 做 1/2/8-rank local/global、sample/population std 对照，要求各 rank EMA 一致。

### AUD-004 — deferred · P1 · GRPO surrogate / clip / KL

- **结论：** 论文称 practical RaPO 使用 clipped GRPO；固定 Trainer 没有 old-policy logps/clip，使用一次性 `exp(logp-logp.detach())`，launcher 另设 KL `beta=0.04`、LR `1e-6`。单次 rollout 使用时可在采样点局部等价，但作者 reuse、clip、KL 配置未披露。
- **论文证据：** PDF §3.1 与 Appendix A 明称 clipped GRPO；§3.3 未公开实验 KL beta。
- **代码证据：** 远程 `grpo_trainer.py:516-519`；全文件无 `clip|old_per_token|num_iterations|importance`；`scripts/run_imagenet_r_smoke.sh:188-189`。
- **实验/命令证据：** 固定上游与 patch 静态复查；官方代码缺失。
- **影响/置信度：** 不能宣称与作者 optimizer objective 等价；中高置信披露缺口，故不判 mismatch。
- **整改批次/测试：** 批次 1；声明 reuse，证明单次梯度等价，并用多次 reuse/ratio 越界测试 clip，写入 run manifest。

### AUD-005 — deferred · P2 · CTAN initialization

- **结论：** 论文未定义首批前的 `σ̂0`；repo 首批直接令 `σ̂=σbatch`，是合理但未获作者证实的假设。
- **论文证据：** Eq.(5) 给递推但无初值；Appendix A 仅要求非负初始化。
- **代码证据：** `src/rapo/core.py:101-105`。
- **实验/命令证据：** 无官方实现可对照。
- **影响/置信度：** `β=0.999` 下初期差异会跨任务延续；披露缺口高置信，作者初值未知。
- **整改批次/测试：** 批次 2；对 first-batch、zero-init、warm-start 三种轨迹做固定 reward 测试并冻结规则。

### AUD-006 — corroborated · P1 · Trainer token boundary

- **结论：** prompt 超过 `max_prompt_length` 时，只截断局部 `prompt_ids/mask`，却以未截断 `prompt_inputs` 生成，再按截断长度切分；prompt 尾部会被误作 completion 并污染 reward 与 actor/anchor logps。
- **论文证据：** PDF p.4 Eq.(2) 的 `yi` 是生成 rollout，平均只覆盖 generated tokens；Appendix B 将 prompt/output 分离。
- **代码证据：** 固定远程 `grpo_trainer.py:395-420,423-446,496-500`；A、B 两 agent 独立定位，编排器复读同一行段。
- **实验/命令证据：** 纯长度复现：`original_prompt=567 max_prompt=512 true_new_tokens=32 trainer_completion_slice=87 prompt_tail_misclassified=55`；567 仅为 recorded-but-not-rerun，缺陷不依赖该数值。
- **影响/置信度：** completion、drift、reward 与 loss 语义均错；高。
- **整改批次/测试：** 批次 1；多模态超限 prompt 必须验证 generate 输入、视觉 token、mask 与 completion 边界一致，或 fail-fast。

### AUD-007 — corroborated · P1 · CTAN update clock

- **结论：** 论文要求每个 optimization step 更新；当前在每次训练态 `compute_loss`/microbatch 更新，时钟受 gradient accumulation 与 skipped update 影响。
- **论文证据：** PDF p.5 Eq.(5) 明写 “At every optimization step”。
- **代码证据：** `src/rapo/core.py:90-106`、`src/rapo/integration.py:112-133`；远程 Trainer `:388,503-508` 无 optimizer-step gate；launcher `:19,186-187` 默认 GA=2。
- **实验/命令证据：** Task 1–6 state 本轮重读 updates 为 `38,76,114,152,190,228`，而记录为每任务 20 optimizer steps；A、B 独立复核，编排器复读 artifact。
- **影响/置信度：** `β` 的时间常数与论文定义不一致，数值差异不能归因算法；高。
- **整改批次/测试：** 批次 1；GA=2/4 分别统计 forward、successful/skipped optimizer step，要求每个成功 step 恰更新一次、各 rank 与 resume 后连续。

### AUD-008 — corroborated · P1 · state/model lineage

- **结论：** state 保存 `task_index`，加载时却不校验 saved/current task 或前序 checkpoint；错误 state 可被当前 task 接受。
- **论文证据：** 论文算法链要求 task `t` 的 anchor/CTAN 从正确的 `t-1` 状态延续。
- **代码证据：** `src/rapo/integration.py:139-160`；launcher 只查路径存在。
- **实验/命令证据：** 编排器只读内存实验输出：`ACCEPTED_SAVED_TASK_INDEX 999 CURRENT_TASK 2`。
- **影响/置信度：** 错 task/model/state 链可静默训练；高。
- **整改批次/测试：** 批次 1；交换 task/model/state/hash 必须 fail-fast，并验证 task 1–10 状态单调链。

### AUD-009 — deferred · P1 · save/resume equivalence

- **结论：** patch 有 state callback 和底层 resume 入口，但 versioned runner 不暴露完整 resume；没有 optimizer/scheduler/RNG/global-step/CTAN 的 interrupted-vs-uninterrupted 等价证明。静态缺口已证，但因禁止启动训练且无第二 agent 独立动态反证，按双证据规则降为 deferred。
- **论文证据：** 非论文显式条款；属于正式长链可复查性要求。
- **代码证据：** resume 只见于 patch/docs/integration，`scripts/run_imagenet_r_smoke.sh` 无 resume 参数；bounded runner 最终仅保存模型。
- **实验/命令证据：** B 域全仓搜索与编排器复核；没有可接受的等价 artifact。
- **影响/置信度：** 10-task 中断后无法证明续跑仍属同一实验；对“未证明”高置信，对实际底层能否等价未知。
- **整改批次/测试：** 批次 1/2；在固定 step 中断，比较权重、optimizer/scheduler、RNG、CTAN、global step 与后续 loss/token 序列。

### AUD-010 — deferred · P2 · world-size sensitivity

- **结论：** 当前实现是 local group mean + gathered global std；论文未定义 distributed scope。因此代码行为已证，但“world-size sensitivity 本身就是论文 mismatch”的旧说法证据不足。
- **论文证据：** PDF Eq.(5)–(6) 无 rank/accumulation 口径。
- **代码证据：** `src/rapo/integration.py:120-132`；远程 Trainer `:502-508`。
- **实验/命令证据：** 未运行 1/2/8-rank 对照；这是明确禁跑范围内的后续短实验。
- **影响/置信度：** 迁移 world size 前不可静默沿用；对风险高置信，对作者口径未知。
- **整改批次/测试：** 批次 2；与 AUD-003 合并执行 rank-invariance/manifest 测试。

### AUD-011 — corroborated · P1 · 2-epoch contract

- **结论：** 当前 launcher 以 `max_steps` 驱动，不能强制论文每任务 2 epochs；Task 6 实际日志结束在 `epoch: 2.92`。
- **论文证据：** PDF p.8 明确所有 classification models 每任务 2 epochs。
- **代码证据：** `scripts/run_imagenet_r_smoke.sh:11,198-203` 使用 `--max_steps`。
- **实验/命令证据：** 远程 Task 6 日志本轮重读 `epoch: 2.92`；训练未重跑。
- **影响/置信度：** 训练预算与论文不一致，绝对数值不可直接比较；高。
- **整改批次/测试：** 批次 2；formal runner 必须严格 2 epochs，并用样本/生成次数、optimizer steps 与日志三重断言。

### AUD-012 — corroborated · P1 · 10-task state chain

- **结论：** repo 仅有单 task launcher与手工传递 model/state，没有 10-task 有向链、失败恢复或配对方法编排。
- **论文证据：** PDF p.7 的 ImageNet-R 为 10 个连续任务。
- **代码证据：** `scripts/run_imagenet_r_smoke.sh:4-12,91-143`；无 task-loop orchestrator。
- **实验/命令证据：** 远程仅 order0 Task 1–6 有导出/运行；Task 7 被明确禁止。
- **影响/置信度：** 容易漏 task、错 anchor/state 或混方法配置；高。
- **整改批次/测试：** 批次 3；无 GPU dry-run 验证 10-task DAG、前序 hash、失败重启与 GRPO/RaPO 配对。

### AUD-013 — corroborated · P1 · evaluation integrity

- **结论：** evaluator 不按样本键去重，也不按 manifest 校验期望样本集合/数量；完整下三角只证明 metric cell 存在，不证明每个测试样本恰出现一次。
- **论文证据：** PDF p.7 的 Last Accuracy/Forgetting 以真实、全部已见类 test population 为输入。
- **代码证据：** `src/rapo/evaluation.py:95-130` 忽略 `relative_path` 并逐行计数；`:145-189` 只查 cell；`scripts/evaluate_qwen2_vl.py:143-151` 虽输出 key，聚合不使用。
- **实验/命令证据：** 编排器无写入内存复现把同一行输入两次，输出 `correct=2,total=2` 且指标成功；本轮远程统计当前 bounded artifact 则确为 8400/8400 unique。
- **影响/置信度：** 缺行、重复、外来样本可静默改变 A/F；高。
- **整改批次/测试：** 批次 2；duplicate/missing/unknown/target mismatch 均 fail-fast，每 cell key 集与 manifest 精确相等。

### AUD-014 — corroborated · P1 · unified provenance

- **结论：** 模型、stage dataset、manifest/order、配置、repo/upstream、前序 checkpoint/state 和结果没有统一机器可读 run ID/hash 链；现有零散 hash 未绑定。
- **论文证据：** 3 orders、固定协议、2 epochs 的结果必须可关联；统一 provenance 是复现要求，不是论文算法公式。
- **代码证据：** `src/rapo/data.py:185-203,273-282` 无 source/run hash；launcher `:91-123` 只检查路径/commit 标志；evaluator `:53-71,143-151` 不绑定 stage/model/manifest/config。
- **实验/命令证据：** 远程 outputs/results/logs 搜索 `*manifest*/*provenance*/*run*.json` 无输出；编排器复核 metrics JSON 只有矩阵/A/F。
- **实验/命令证据补充：** 三个 Task 6 模型目录顶层均无 `manifest|provenance|commit|sha` 文件；10 个结果 hash 虽与 handoff 匹配，但只能证明现存文件未变，不能绑定生产链。
- **影响/置信度：** 错模型、order、stage、配置或前序链可被标成合法结果；高。
- **整改批次/测试：** 批次 1；交换任一输入应拒绝，所有 task/result 继承同一 run ID 和不可变输入 hash。

### AUD-015 — corroborated · P1 · full-test / 3-order formal protocol

- **结论：** 当前没有强制 3 orders、full-test、GRPO/RaPO 配对与最终 mean/std 的正式路径；2-epoch 和 10-task 分别由 AUD-011/012 记录，不在本条重复计数。
- **论文证据：** PDF p.7 要求三随机 class orders、测试全部已见类并报告跨 order 结果。
- **代码证据：** evaluator `scripts/evaluate_qwen2_vl.py:23-48` 默认每类 bounded 子集，无显式 all-samples/count contract；无 order-loop/mean-std orchestrator。
- **实验/命令证据：** order0 仅导出/运行 Task 1–6；order1/2 只有 manifest。Task 6 full seen test 应为 17391，当前 cell 仅 2400；Task 10 full 为 29000。
- **影响/置信度：** 当前数值不是论文级 protocol 输出；高。
- **整改批次/测试：** 批次 3；每 order 55 cells、full expected counts、两方法 key/config/seed 配对，最终三 order mean/std。

### AUD-016 — deferred · P1 · official split identity

- **结论：** repo 的 seeds 0/1/2 与 `sample_seed=0` 是可复查的 provisional independent split；论文未公开实际 class orders/5-shot 样本，官方代码未发布，不能称同源。
- **论文证据：** PDF p.7 只说 3 random orders 和每类 5-shot，无 seed/identity。
- **代码证据：** `src/rapo/data.py:76-78,105-110,139-150`；`docs/data_and_evaluation.md:19-29` 明示 provisional。
- **实验/命令证据：** 三 manifest 均 200 类、1000 unique train、29000 unique test，hash 已在 §2 现场核验。
- **影响/置信度：** 绝对 A/F 可能因 split 不同而偏移；披露缺口高置信，作者 identity 未知。
- **整改批次/测试：** 批次 1 冻结并标 `independent reproduction`；官方 metadata 发布后逐项 diff，禁止静默替换。

### AUD-017 — deferred · P2 · metric weighting

- **结论：** 论文未说明 task test size 不等时 Last Accuracy 用 micro 还是 task-macro；repo 选择 micro Last、task-macro Forgetting。
- **论文证据：** PDF p.7 只有自然语言，未给 micro/macro 公式。
- **代码证据：** `src/rapo/evaluation.py:133-143,213-227`。
- **实验/命令证据：** full task sizes 不等；当前 bounded cell 等大，不能区分两种 Last 口径。
- **影响/置信度：** full-test A 会随加权口径变化；歧义高置信，作者口径未知。
- **整改批次/测试：** 批次 1/3；同一 full matrix 同时输出 micro/task-macro 并记录差值，后续对齐官方口径。

### AUD-018 — corroborated · P1 · selective export integrity

- **结论：** selective export 无条件重写顶层 manifest 后只导出指定 task；旧 task 目录不校验新 manifest，可形成跨 seed/order 混合 artifact。没有证据表明当前 artifact 已混配。
- **论文证据：** PDF p.7 的每个 order 是固定任务序列，不允许训练/测试混用 split。
- **代码证据：** `src/rapo/data.py:311-318,342-375`；`docs/data_and_evaluation.md:139-141`。
- **实验/命令证据：** 远程 order0 是一个 manifest 加 Task 1–6 目录，各目录不携 manifest hash；C 域与编排器静态复核一致。
- **影响/置信度：** 后续增量导出可使历史 stage 与当前 manifest 不一致；代码路径高置信。
- **整改批次/测试：** 批次 1；既有目录上改变 seed/split 必须 fail-fast 或新版本，每个 export 绑定 manifest SHA。

### AUD-019 — corroborated · P0 · BF16 readiness gate

- **结论：** BF16 paired stability gate 尚未完成；当前可达节点与环境不能构成 formal BF16 readiness 证据。
- **论文证据：** PDF p.6 为 8×H100，p.8 为每任务 2 epochs。
- **代码证据：** `README.md:31-34` 仍列 BF16 gate 未通过；`scripts/run_imagenet_r_smoke.sh:15-23,150-153` 只有可选 BF16/FA2 开关；DeepSpeed configs 同时保留 FP16 dynamic scale/CPU offload。
- **实验/命令证据：** 编排器与 D 域现场只见 8×2080 Ti CC7.5、无 `flash-attn`、无 BF16 training artifact；Task 6 成功日志是 FP16/SDPA。该条满足 paper+code+live environment 三重证据。
- **影响/置信度：** 不得把 scale512 结果解释为 formal 算法结果，也不得放行正式 10-task；高。
- **整改批次/测试：** 批次 4；先 one-step load/backward/update/save/reload，再从各方法相同 Task 5 前态做 paired 20-step Task 6，要求 BF16 on、FP16/loss-scale off、无 overflow/skip/NaN/OOM/NCCL hang。只作诊断。

### AUD-020 — rejected · P1 · “没有 legacy profile”

- **结论：** 驳回 handoff §8.6 “没有明确分离 legacy 2080 Ti profile”的字面结论；已有专用 wrapper。残余事实是分层不完整、formal profile 不对称且环境变量仍可能泄漏。
- **论文证据：** PDF p.6 为 8×H100；论文未指定 FP16/SDPA/cudnn 等细节。
- **代码证据：** `scripts/run_imagenet_r_2080ti_smoke.sh:6-18` 明确固定 FP16、SDPA、512/32、100352、GC、GA2、默认 n=2；`src/rapo/runtime.py:11-23` 有显式 cuDNN workaround。
- **实验/命令证据：** Task 6 日志现场证实 wrapper 派生的 legacy 设置实际使用；D 域与编排器静态复核一致。
- **影响/置信度：** 不能继续声称“完全没有分层”；高。
- **整改批次/测试：** 批次 2；dry-run 必须证明 legacy 降级项不流入 formal，两者生成独立 machine-readable manifest。

### AUD-021 — corroborated · P1 · missing formal hardware profile

- **结论：** 没有冻结的 high-end/BF16 formal profile；generic smoke defaults 不是 formal profile。当前高端 8 卡可用性、拓扑、NCCL readiness 仍 `deferred`，不把未知硬件状态写成缺陷事实。
- **论文证据：** PDF p.6 为 8×H100，不能把单卡 4090 推理自动视为 paper-identical。
- **代码证据：** `scripts/run_imagenet_r_smoke.sh:15-24` 是环境变量 smoke defaults；现有三个 DeepSpeed 配置均为 `zero3_cpu_offload` FP16 scale 11/10/9，无 formal 命名/冻结 manifest。
- **实验/命令证据：** 当前 alias 只展示 8×2080 Ti；4090 证据仅为单卡 FP16/SDPA inference artifact，未见 BF16/FA2/8-card training artifact。
- **影响/置信度：** 显存、attention、ZeRO/offload、world size 与稳定性假设无法冻结；profile 缺失高置信，高端资源现状未知。
- **整改批次/测试：** 批次 2；目标节点核验 driver/CUDA/package、BF16 ops、attention forward/backward、8-rank NCCL、ZeRO/offload A/B、显存余量与短稳定性。

### AUD-022 — corroborated · P1 · evaluator dtype/attention

- **结论：** evaluator 写死 FP16/SDPA，不能忠实记录或复用 formal BF16/attention profile。
- **论文证据：** PDF pp.6–8 未公开 evaluator dtype/attention，故这里是 repo/provenance 缺陷，不宣称 paper mismatch。
- **代码证据：** `scripts/evaluate_qwen2_vl.py:80-85` 固定 `torch.float16` 与 `attn_implementation="sdpa"`。
- **实验/命令证据：** 4090 control artifact 来自该 FP16/SDPA evaluator；它只控制 inference hardware，不能验证 BF16 training/formal evaluator。
- **影响/置信度：** formal model 会被静默切回 legacy inference 路径，结果 manifest 也无法表达选择；高。
- **整改批次/测试：** 批次 2；dtype/attention 显式配置与 manifest，legacy 输出兼容，formal 组合实际加载目标 dtype/kernel，不支持时 fail-fast。

### AUD-023 — corroborated · P1 · legacy numerical stability

- **结论：** cuDNN disable 与降低 FP16 loss scale 是有日志支持的 legacy workaround，但根因未修复且结果对 scale 敏感，不能升格为 formal default。
- **论文证据：** PDF p.6 为 8×H100；论文未公开 cuDNN、precision、loss scale。
- **代码证据：** `src/rapo/runtime.py:11-23` 只关闭 cuDNN；`configs/deepspeed_zero3_cpu_offload*.json:2-8` 的 initial scale power 为 11/10/9。
- **实验/命令证据：** 三份日志现场命中 `CUDNN_STATUS_INTERNAL_ERROR`；Task 5 scale2048 step1/2 与 Task 6 GRPO scale1024 step7/8 分别出现重复 LR/stale grad norm；scale512 两方法完成 20 steps但 epoch 2.92。没有 paired scale1024 GRPO/RaPO。
- **影响/置信度：** Task 6 差距混有 legacy 数值路径，不能隔离算法效应；高。
- **整改批次/测试：** 批次 4 BF16 pair；legacy 若保留，应自动识别 skipped update/重复 LR/stale grad，读取 DeepSpeed overflow/optimizer state，禁止不同 scale 配对。

### AUD-024 — corroborated · P1 · formal-chain restart

- **结论：** formal profile 冻结后必须从 pinned base model 的 Task 1 重启；由 legacy Task 6 checkpoint 切 BF16 后继续 Task 7 只能算混合诊断链。
- **论文证据：** PDF p.6/p.8 是统一硬件协议与 2 epochs。
- **代码证据：** legacy wrapper 固定 FP16/SDPA/小预算；当前 Task 6 为 scale512、cuDNN off、`max_steps=20`。
- **实验/命令证据：** Task 6 日志现场为 FP16/SDPA、cuDNN off、20 steps、epoch 2.92；目录/hash 只证明 legacy 链存在。
- **影响/置信度：** Task 7 切换 precision/kernel/budget 会混合 CTAN 历史与数值路径，无法归因；高。
- **整改批次/测试：** 批次 5；Task 1 manifest 必须指向 pinned base revision、无 prior state、formal profile hash，随后每 task 校验 parent model/state/data/config 链。

## 5. 编排器交叉复核

编排器亲自复读所有 P0/P1 的论文段落、versioned code/patch 或固定上游行段，并复查相关命令/artifact。下表给出双证据门槛；不满足者已经降为 `deferred`。

| Finding | 亲自复核的双证据 | 结果 |
|---|---|---|
| AUD-003 | paper Eq.(5)/(6) + integration/Trainer global std code；A/B 两域独立意见 | deferred（作者 scope 未知） |
| AUD-004 | paper clipped GRPO + 固定 Trainer 无 clip/old-policy code | deferred（一次性梯度等价尚未实验） |
| AUD-006 | paper generated-token 定义 + fixed Trainer slicing code；A/B 独立定位 | corroborated |
| AUD-007 | paper optimization-step 原文 + core/integration/Trainer code + state counts | corroborated |
| AUD-008 | paper task chain + integration code + saved-task=999 内存实验 | corroborated |
| AUD-009 | runner/patch 静态搜索；没有动态 interrupted/resume 实验或第二 agent 反证 | **降为 deferred** |
| AUD-011 | paper 2 epochs + launcher max_steps + Task 6 `epoch:2.92` | corroborated |
| AUD-012 | paper 10 tasks + launcher 单 task code + 远程只到 Task 6 | corroborated |
| AUD-013 | evaluator code + duplicate-row 可复现实验 + current artifact unique 反向检查 | corroborated |
| AUD-014 | paper protocol + provenance code gaps + 远程 run-manifest 搜索/目录检查 | corroborated |
| AUD-015 | paper full-test/3 orders + evaluator/runner code + remote counts | corroborated |
| AUD-016 | paper 未给 identities + repo provisional seed algorithm + official code absence | deferred |
| AUD-018 | paper fixed order + exporter overwrite code；当前已混配无证据 | corroborated 风险，不宣称已发生 |
| AUD-019 | paper H100/2 epochs + repo gate/config + live 2080/BF16 artifact absence | corroborated |
| AUD-020 | paper hardware fact + 2080 wrapper/runtime code + Task 6 log | rejected 旧字面结论 |
| AUD-021 | paper H100 + smoke/DeepSpeed code + live hardware/artifact inventory | corroborated profile 缺口；资源可用性 deferred |
| AUD-022 | evaluator code + 4090 FP16/SDPA artifact | corroborated |
| AUD-023 | paper环境差异 + runtime/config code + cuDNN/scale logs | corroborated |
| AUD-024 | paper统一协议 + legacy runner/config + Task 6 log | corroborated |

当前结果 artifact hash 本轮逐个重算并与 handoff 相符（只证明文件身份，不是 rerun）：

| 文件 | SHA-256 |
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

该 hardware audit 现场确认：2080 Ti/4090 各 2400 行、ordered keys 相同、A/F 都为 `65.25%/33.75%`，但有 4 个 correctness decisions 不同。因此“相同 aggregate metrics”成立，“逐样本输出完全相同”被驳回。

## 6. handoff §8 高优先问题裁决

| 旧项 | 裁决 | 本审查依据/调整 |
|---|---|---|
| P-01 prompt slicing | corroborated | AUD-006，保留 P1 |
| P-05 state continuity | corroborated | AUD-008 |
| P-06 model/state/manifest run ID | corroborated | AUD-014；与 P-19 去重并提升为 formal 前 P1 |
| P-07 CTAN 38 microbatches vs 20 steps | corroborated | AUD-007；从“ambiguity”提升为已证时钟 mismatch |
| P-08 global std/world size | deferred | AUD-003/010；代码行为已证，作者分布式 scope 未披露 |
| P-09 20 steps≈2.92 epochs | corroborated | AUD-011，paper+launcher+log |
| P-10 10-task orchestrator | corroborated | AUD-012 |
| P-11 full-test orchestration | corroborated | AUD-015 |
| P-12 resume | deferred | AUD-009；静态缺口成立，但动态等价未测，按双证据降级 |
| P-17 unofficial split | deferred | AUD-016；“provisional independent split”成立，与作者是否不同未知 |
| P-19 run manifest | corroborated | AUD-014；旧 P2 排序过低，formal chain 前 P1 |
| P-21 FP16 scale sensitivity | corroborated | AUD-023；不能据此归因算法 |
| P-22 mixed Task 1–6 chain | corroborated | AUD-024；formal 必须从 Task 1 重启 |
| P-23 no flash-attn | corroborated repo/env fact；优先级降为条件项 | 当前环境未安装；formal 是否必须 FA2 取决于 BF16 gate，不能预设 |
| P-24 official code absent | corroborated external blocker | 2026-07-31 在线重查仍未发布；相关作者配置全部 deferred |

§8 其余逐项：P-02 `AutoConfig` 修复、P-03 `main_process_first`、P-04 tiny-image padding 均静态 `corroborated`，但 handoff 的历史运行通过仍为 recorded-not-rerun；prompt=1024、cuDNN disable、lower loss-scale、atomic JSONL、fixed subset 只属 mitigation，不能当 correctness/full protocol；prediction key/count 分别由 AUD-013 证实缺防线；selective export 由 AUD-018 证实代码风险；evaluator 写死由 AUD-022 证实。§8.4 未披露的 LR、batch、GA、lengths、pixels、attention、precision、DeepSpeed、splits、CTAN init/scope、metric weighting全部是 `paper ambiguity`，不得改写成作者配置。§8.5 中唯一明确 paper deviations 是硬件型号、2 epochs、3 orders、full test；FP16/SDPA/pixels/completion/cuDNN/loss scale 是 repo facts + 作者未知选择。§8.6 “无明确 legacy profile”被 AUD-020 驳回为字面错误；准确说法是“已有 wrapper，但 formal/legacy 分层不完整”。

## 7. handoff §12 裁决

| §12 项 | 裁决 |
|---|---|
| P0-1 独立审查 | **本报告已完成**；发现 AUD-006/007 等反证，不接受旧的“未发现 mismatch” |
| P0-2 BF16 stability | corroborated outstanding；实际 BF16 结果 deferred，见 AUD-019 |
| P0-3 formal config freeze | corroborated outstanding；profile、provenance、protocol均未闭合 |
| P0-4 Task 4–6 差距来源 | deferred；必须先修语义/冻结配置，再用 paired diagnostics 分离数值、时钟、预算、顺序与随机性 |
| P1-1 prompt | corroborated outstanding，AUD-006 |
| P1-2 provenance | corroborated outstanding，AUD-008/014/018 |
| P1-3 strict 2 epochs | corroborated outstanding，AUD-011 |
| P1-4 CTAN semantics | clock mismatch corroborated；distributed scope deferred，AUD-003/007 |
| P1-5 full-test | corroborated outstanding，AUD-015 |
| P1-6 resume | deferred pending dynamic equivalence，AUD-009 |
| P1-7 prediction count/dedup | corroborated outstanding，AUD-013 |
| P1-8 evaluator dtype/attention | corroborated outstanding，AUD-022 |
| P1-9 high-end/BF16 formal profile | corroborated outstanding，AUD-019/021 |
| P1-10 2080 legacy profile | partially satisfied；wrapper 已有，完整隔离未完成，AUD-020 |
| P1-11 从 Task 1 重启 | corroborated necessary and not done，AUD-024 |

§12 P2 中，machine-readable run manifest 必须提升到 formal chain 前 P1；CI、patch regression、distributed e2e 仍是工程项；metric weighting deferred；三个 orders、full 10-task、mean/std 是明确未完成的 paper protocol。docs 同步不在本审查授权的修改范围。

## 8. 被驳回或降级的旧结论

- “CTAN update semantics 只是 ambiguity”被驳回：当前确实按 microbatch/forward 更新，与 paper optimization-step 文字不一致。
- “world-size sensitivity 已证明算法 bug”降为 deferred：repo choice 已证，作者 scope 未知。
- 非截断 prompt 下“anchor logps 已发现 mismatch”被驳回；actor/anchor token/mask 静态映射一致。作者完整实现等价仍 deferred。
- classification prompt、format/accuracy reward 的静态 mismatch 被驳回；它们与 Appendix 一致，但不等于端到端正确。
- “当前 prediction artifact 已重复/缺行”被驳回；本轮 key audit 未发现。真正问题是标准 pipeline 无 guard。
- “FP16/SDPA/pixels/completion/cuDNN 是已确认作者配置 deviation”降级：只有 repo 设置已证，作者设置未披露。
- “2080 Ti 与 4090 输出完全相同”被驳回；aggregate 相同但 4 个 correctness decisions 不同。
- “完全没有 legacy 2080 profile”被驳回；wrapper 已存在，缺的是完整隔离和 formal 对称层。
- Eq.(5) 中示例 `β=0.99` 与实验默认 `β=0.999` 不是冲突。
- “未发现 material mismatch”被 AUD-006/007 直接反证。

## 9. 整改与实验放行建议（只规划，不实施）

### 是否允许进入整改

**允许人类决定后进入受控整改；不允许进入 BF16 gate 或正式训练。** 整改仅处理本报告有证据的 finding，并应另行授权；本审查没有修改业务代码、测试、配置、handoff 或判卷标准。

### 按依赖排序的批次

1. **批次 1：语义与血缘。** 修/fail-fast prompt 边界；把 CTAN 迁到 successful optimizer-step 时钟；冻结或显式标注 `σbatch`、initialization、clip/reuse/KL 的独立复现选择；建立 task/model/state/data/config/run hash chain；安全化 selective export。
2. **批次 2：正式契约。** strict 2-epoch runner；动态 resume equivalence；prediction dedup/expected set/count；可配置 evaluator dtype/attention；完整 formal/legacy profile 隔离。
3. **批次 3：无 GPU orchestration 验收。** 10-task×2 methods×3 orders DAG dry-run；full-test 55 cells/order；paired keys/config/seeds；失败恢复、mean/std、artifact schema。
4. **批次 4：短 GPU 诊断。** 在已通过兼容性检查的同质 8 卡环境做 BF16 one-step 与 paired Task 6 20-step gate；该结果只诊断数值路径，不延续 formal chain。
5. **批次 5：正式链。** 全部门槛通过、配置与 manifest 冻结后，从 pinned base model Task 1 开始 10 tasks×3 orders；不得接续 legacy Task 6。

### 必须先实验再定的问题

- 1/2/8 rank 下 `σbatch` 的 local/global、sample/population 口径与 CTAN update/EMA rank consistency。
- clipped GRPO 与当前一次性 surrogate 的梯度等价边界、rollout reuse、KL/clip 设置。
- CTAN first-batch initialization 的短轨迹影响。
- interrupted-vs-uninterrupted resume 的模型、optimizer、scheduler、RNG、global step、CTAN 与 token/loss 等价。
- BF16、attention、ZeRO/offload、gradient checkpointing、显存余量、NCCL 与 skipped/nonfinite 行为。
- full-test 同一预测矩阵的 micro 与 task-macro Last Accuracy 差值。
- Task 4–6 差距来源；在上述语义/数值变量隔离前不得归因 RaPO 算法。

### BF16 gate 前置条件

1. AUD-006/007/008/014 等语义与 provenance P1 已整改并测试；AUD-003/004/005 的独立复现选择已显式冻结。
2. 取得同节点、同型号、world-size 8 的高端 GPU；记录 driver/CUDA、拓扑、显存/占用与 NCCL collective。当前 8×2080 Ti 不满足。
3. 冻结同一 Task 5 起点、data/order、world size 8、rollout 8、batch/GA、prompt/completion、pixels、seed；两方法唯一差异是 RaPO 开关。
4. 明确 BF16 on、FP16/dynamic loss-scale off；FA2/SDPA 经实际 import + forward/backward 后择一，不因“4090 可推理”预设。
5. one-step load/backward/update/save/reload 后再 paired 20-step；检查 overflow/skip、LR、grad norm、NaN/OOM/NCCL hang。
6. 生成完整 run manifest/hash；gate 只能作为 Task 6 数值诊断，不能直接 Task 7。

### 正式 10-task 放行条件

- 所有 corroborated P0/P1 已关闭并有对应测试；所有 deferred P1 已通过实验/官方信息裁决，或在 independent reproduction protocol 中显式冻结且不冒充官方配置。
- repo、fixed upstream+patch、base model、data tar、3 manifests、formal profile 与 runner 均固定 hash，工作树状态写入 manifest。
- 每 task 恰好 2 epochs；3 个冻结 orders；GRPO/RaPO 同 seed/key/config；正式训练从 base Task 1 启动。
- 每 order 有完整 55-cell lower triangle；每 cell 与 manifest expected key set 精确相等、无重复/缺失/外来样本；使用 full test。
- checkpoint/resume 等价、parent model/state/run ID 连续；legacy 和 formal 输出目录/profile 不可混用。
- 每 task 保存 logs、optimizer/scheduler/RNG/CTAN、prediction、metrics 与 hashes；最终报告三 order mean/std，同时披露 micro/macro 选择。

## 10. Deferred / blocked 边界

- 作者代码仍未发布，因此 exact distributed std、CTAN init、clip/reuse/KL、完整 optimizer、dtype/attention、orders/5-shot identities、metric weighting 均 deferred。
- 没有既有高端节点入口；按授权未扫描网络、未申请权限。高端卡数量/拓扑/空闲、NCCL、BF16 training readiness 均 deferred。
- 本机完整 pytest 因缺 `torch`/未安装 `rapo` 在收集期失败；未安装依赖。服务器旧 “29 passed” 未重跑。
- 正式 full-test/10-task、BF16 training、Task 7 均因硬边界未运行。
- 详情同步于 `docs/audit/BLOCKED.md`。没有证据路线连续失败三次。

## 11. 审查范围证明

完成条件命令在最终收口时执行；预期且实际均无业务路径输出。审查产生的唯一 workspace 写入为本报告、`PROGRESS.md`、`BLOCKED.md`，均位于 `docs/audit/`，未 stage/commit/push。
