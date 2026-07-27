# Upstream provenance

The initial engineering baseline is Visual-RFT:

- repository: `https://github.com/Liuziyu77/Visual-RFT`
- inspected commit: `2ffad63b25ddd79bfe25d3e046645401201c89d6`
- license: Apache-2.0

The baseline is used because it provides a Qwen2-VL GRPO trainer and visual
classification rewards. It is not evidence that the unreleased RaPO code used
the same implementation.

At the inspected commit, reward aggregation and group-relative normalization
are implemented in:

```text
src/virft/src/open_r1/trainer/grpo_trainer.py
```

The integration seam is immediately after task reward aggregation and before
advantage computation. The initial RaPO core remains independent so its
mathematics can be tested before modifying the trainer.
