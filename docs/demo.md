# 五分钟演示

首次运行先在仓库根目录执行 `uv sync --locked`。

## 1. 生成演示报告和恢复记录

```bash
uv run --locked python scripts/public_demo.py
```

命令打印 `output_dir`。这个目录存有 `summary.json`、正常运行的报告、恢复后的报告、证据 JSON 和断点前后的轨迹。每次运行使用一个新目录。

`summary.json` 中可以检查这些结果。

| 字段 | 预期 |
|---|---|
| `normal.review_status`、`recovery.review_status` | 固定审查场景完成一次修订后为 `passed` |
| `citation_integrity_verified` | 检查结论所引用的证据区间可追溯到原文 |
| `recovery.original_deadline_preserved` | 原截止时间保留 |
| `recovery.checkpoint_reuse` | 有已保存规划与抓取被复用，抓取数量随取消发生时的并发进度变化 |
| `coalescing.upstream` | 16 个相同并发查询共用 1 个上游请求 |

## 2. 展示网页轨迹

把 `<RUN_DIR>` 换成刚才命令的 `output_dir`。

```bash
uv run --locked research serve --port 8768 --db "<RUN_DIR>/web.sqlite3" --output-dir "<RUN_DIR>/reports"
```

打开 [执行轨迹](http://127.0.0.1:8768/harness)。选择经历了两次执行的任务，依次查看中断、复用和报告完成。展开研究要点，展示结论、证据编号、原文字符区间与缺口。打开 [研究工作台](http://127.0.0.1:8768/) 可以现场运行一次 Red–Blue 离线演示。

时间紧时，可直接打开 [示例 Markdown](../examples/demo-report.md) 并点击 `E1`，再打开 [评测报告](evaluation.md) 对照恢复收益和真实研究对照中的失败。

## 3. 手动触发断点

```bash
uv run --locked research harness-start --stop-after-fetch 1
uv run --locked research harness-resume "<HARNESS_ID>"
uv run --locked research harness-show "<HARNESS_ID>"
```

第一条故意中断并返回退出码 1。把它输出的 Harness ID 填到后两条命令中，在原截止时间内恢复。强制退出与租约接管的复现实验可运行 `uv run --locked python scripts/benchmark_research_harness.py`。
