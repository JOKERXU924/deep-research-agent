# 架构与边界

| 层 | 职责 | 代码 |
|---|---|---|
| 接口 | CLI、本机 Web 工作台、任务状态和进度事件 | [CLI](../src/deep_research/cli.py)、[Web](../src/deep_research/web_app.py) |
| 编排 | 按依赖调度检索和抓取，处理并发、超时与取消 | [DAG](../src/deep_research/dag_research.py)、[调度器](../src/deep_research/scheduler.py) |
| 证据 | 保存原文快照及引用区间，按研究要点和字节预算选段 | [选段](../src/deep_research/evidence_selection.py)、[证据映射](../src/deep_research/research_map.py) |
| 审查 | Red 提出支持问题，Blue 提交有界修订，程序校验引用与快照 | [审查](../src/deep_research/review.py) |
| 可恢复执行 | 按任务保存已完成的外部动作，使用租约控制接管，继承期限和动作预算 | [运行层](../src/deep_research/harness.py)、[检查点](../src/deep_research/harness_store.py) |
| 检索复用 | 以命名空间和请求参数区分缓存，相同并发请求共用同实例内的任务 | [检索缓存](../src/deep_research/retrieval_cache.py) |

恢复重新进入研究流程，通过动作身份与完整性校验重放已保存响应。动作输出和完成事件在同一 SQLite 事务提交。远端模型和本地数据库之间存在提交间隙，结果未知的模型调用会阻止自动续跑。租约只能约束后续本地提交。

工作台面向本机单用户。请求合并的范围为一个进程中的同一缓存实例。公开演示将真实服务替换为内置资料和固定模型，供读者直接验证执行行为。真实模型研究可使用常规 `run-dag` 入口。

项目内置合成题库。HotpotQA 相关接口按 [HotpotQA 项目](https://hotpotqa.github.io/) 的任务与指标实现，用户可导入本地数据。语义选段支持可选的 [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)，需要另行准备权重并遵守其许可证。
