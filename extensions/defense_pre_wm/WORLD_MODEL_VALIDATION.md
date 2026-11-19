# World Model 冒烟记录（2026-11-19）

依据 `reference/plan02_worldModel/worldmodel.md` 和 `defense_worldmodel_starter` 实现，
正式代码位于 `defense_task/worldmodel_stage.py`、`worldmodel_eval.py`、`integration.py`。
参考文件保持原样。没有运行完整测试集、长训练或新环境轨迹采集。

## 环境和检查

- Python：`/home/pan/miniforge3/envs/robot_defense/bin/python`，3.11。
- 安装 PyTorch 2.9.1+cpu（官方 CPU wheel），不依赖 MuJoCo 渲染。
- `pytest extensions/defense_pre_wm/tests/test_worldmodel_smoke.py -q`：**2 passed**。
- 覆盖真实存储格式的合成轨迹加载、按整局划分、采样与通道隔离、burn-in 后有限梯度、
  一次优化、后验过滤、无未来观测的先验想象、过滤器状态不变、检查点重载一致、短窗口处理、
  终止/截断 continuation、错误 schema 拒绝及事件指标。
- 新增 Python 文件语法检查及 `git diff --check` 通过。

## 真实数据烟测

`inspect --data artifacts/defense_pre_wm/datasets/sine5` 完整读取现有 100 对 NPZ/JSON：

| 集合 | Episode | 转移数 | 最短/最长 |
| --- | ---: | ---: | ---: |
| train | 70 | 28073 | 62/1400 |
| validation | 15 | 5828 | 59/666 |
| test | 15 | 5851 | 93/635 |

维度、字段切片、动作、四元数、奖励/continuation 与本地契约一致。
只对真实训练集拟合归一化参数。烟测训练参数：CPU、seed=42、steps=2、batch=2、length=4、burn=2。
第 1/2 步验证损失约为 2.9283/2.9226；成功保存最佳检查点并重载。
随后对 15 个留出 episode 在 10/20/40 步各抽每局 1 个窗口，成功输出 JSON。

产物保存在 [smoke_20261119](../../artifacts/defense_worldmodel/smoke_20261119)：
`world_model.pt`、`normalization.npz`、`training.jsonl`、`openloop_report.json`。
报告记录的是原始临时目录的检查点路径，保留相同文件内容的副本在上述目录。

## 验收边界

**这些仅证明执行链路可用，不代表模型学会了物块运动。** 2 步模型不是可冻结部署的性能基线。
本次未完成长期训练、正式开环性能验收、阈值调优、速度/轨迹 OOD 验收或 Actor 接入。
只有完成这些后续实验，才能按计划判定模型是否优于简单预测器并冻结首个正式检查点。

本实现提供恒速和恒加速度基线；文档建议的 EKF、物块置换增强、可拦截性筛选、
完整威胁排序和多粒子不确定性校准属于后续研究项，未作为这次起始实现的已验证能力。
检查点选择依据固定窗口上的后验验证损失，正式验收仍需检查开环指标。
OOM/超长序列 padding 尚未实现，当前通过明确拒绝过长序列防止静默漏掉短失败局。
