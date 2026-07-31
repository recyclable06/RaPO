# Blocked / deferred evidence

1. **作者实现未发布（deferred）**：截至 2026-07-31，官方 RaPO 仓库仍仅有 README。无法核对 distributed `sigma_batch`、CTAN 初值、clip/reuse/KL、完整 optimizer、dtype/attention、具体 class orders/5-shot identities 与 metric weighting；禁止用 Visual-RFT patch 猜作者行为。
2. **高端 BF16 训练环境不可核验（deferred）**：既有 SSH 只到 8×RTX 2080 Ti；无已授权的 8×4090/H100/L40/5090 入口。未扫描网络、未申请权限；GPU 数量/拓扑/空闲、NCCL、BF16/attention/DeepSpeed compatibility 均未知。
3. **本机完整测试不可运行（deferred）**：`python -m pytest -q` 因缺 `torch` 且 `rapo` 未安装在收集期失败；按边界未安装依赖。handoff 的 “29 passed” 仍为 recorded-but-not-rerun。
4. **训练/正式评估被硬边界禁止（deferred）**：未运行 BF16 paired gate、resume equivalence、1/2/8-rank CTAN、full-test、10-task、Task 7 或长时 GPU 作业。
5. **因果归因未排除（deferred）**：Task 4–6 差距仍可能来自 Trainer token/CTAN 语义、FP16/loss-scale、短预算、数据顺序、随机性或算法；整改与受控实验前不得归因。

没有证据路线连续失败三次；其余无。
