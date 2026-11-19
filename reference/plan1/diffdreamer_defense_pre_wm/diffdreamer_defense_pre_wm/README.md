# DiffDreamer Defense / Pre-World-Model code

本项目是**World Model 训练之前的完整阶段**，不包含 RSSM、Actor-Critic 或可微分接触模拟。

## 已完成

- `DefenseTask(ManipulationEnv)`：桌面半径 0.20 m 防御圆、n 个自由刚体物块
- `linear / sine / pulsed` 速度模式（每次 episode 的轨迹参数独立随机）
- 进攻驱动力在 MuJoCo 物理子步施加，不强制改物块 `qpos`
- 抓取 `_check_grasp` + 连续抬升判定清除；桌面物块外轮廓进入禁区判定突破
- 独立 `r_disc` / `r_diff` 和事件/终止统计
- OSC_POSE 规则抓取基线（只供数据采集，不保证每次成功）
- 统一固定向量状态数据集，记录 `(o_t,a_t,r_t,c_t,o_{t+1})`

## 环境要求

本包针对 **robosuite 1.5.x 官方 API**。你的 fork 具体提交未能通过公开 URL 读取，首次运行需要检查它是否与此 API 兼容。
示例 Python 3.10/3.11；先安装你 fork 中自带的依赖：

```bash
git clone https://github.com/bright-moon-67/robosuite.git
cd robosuite
pip install -e .
cd /path/to/diffdreamer_defense_pre_wm
pip install -e ".[test]"
```

不要同时再安装另一个 PyPI robosuite 版本覆盖你 fork。

## 运行

```bash
# 无渲染最小烟雾测试
python -m defense_task.cli smoke --blocks 1 --horizon 100
# 图形界面手工检查（需桌面 OpenGL 环境）
python -m defense_task.cli demo --blocks 1 --horizon 400
# 批量评估规则控制策略（在引入 World Model 之前的必要基线）
python -m defense_task.cli evaluate --blocks 5 --episodes 20
# 五物块采样，速度范围按需调
python -m defense_task.cli collect --blocks 5 --episodes 20 --horizon 1400 \
    --out datasets/defense --min-speed .008 --max-speed .016 --stagger 6
pytest -q tests/test_geometry_dataset.py
pytest -q tests/test_sim_smoke.py
```

所有数值为**初始实验值**，不是论文参数。特别需要在真实仿真中检查：
`Panda` 的机械臂可达空间、夹爪默认朝向、实际单次抓取耗时、
物块力控速度是否与目标速度一致。

## 物理与事件约束

- 物块 `qpos` 仅在 `reset()` 时设定；进攻阶段施加 `xfrc_applied` 水平外力
- 接触和自由刚体运动由 MuJoCo 求解；抓取后不再提供外部推进力
- `status=0` active, `status=1` lifted/cleared, `status=2` entered defense circle
- 夹持且物块中心高于桌面 `0.11 m` 并连续保持 3 个策略步才清除
- 非清除物块，只要其**投影外轮廓**触碰防御圆且底部靠近桌面，即判突破
- 默认首次突破即 episode 失败；全部清除成功
- 已清除物块**不会消失**，后续仍可能与环境物体碰撞，但不再进攻、不计突破

## 数据集格式

每个 `episode_XXXXX.npz` 保存：
- `obs`, `action`, `next_obs`: 每一行动严格对应 `o_t`, `a_t`, `o_{t+1}`
- `reward`, `r_disc`, `r_diff`, `done`
- `status`, `cleared_count`, `breached_count`

`defense_robot` = arm q, dq, eef xyz, eef velocity。
`defense_blocks` = 每物块6维：相对末端位置xyz、active、grasped、cleared。
`defense_status` = 每物块 status code；`defense_clock` = 时间归一化。
固定顺序拼接的 obs 的 schema 在 `defense_task/dataset.py`。

`r_diff` 目前只是一个**连续函数形式的奖励值**；
要真正实现 DiffDreamer BPTT，需要随后加入**可微分机械臂动力学**与同步 rollout。
`r_disc` 将来用于 RSSM reward head 和 Critic，但当前并未训练网络。

## 建议的执行顺序

1. 把 `smoke --blocks 1` 跑通，查看 ring/table/box 坐标
2. `demo --blocks 1`，验证抓取不受进攻外力强制牵引
3. 统计几十次单块防御成功/失败，调 OSC 增益和落点
4. 单块线性 → 单块变速 → 五块慢速错峰
5. 收集 Replay，验证所有通道无 NaN、无状态泄漏
6. **下一阶段**才训练 RSSM，并接入可微分 robot dynamics

## 与原始 DiffDreamer 的联系

原文的 Eq. (9)–(11) 指出：
RSSM 负责潜在环境预测，可微分物理负责显式自身动力学，
联合 Actor 同步预测，并分别处理离散任务奖励与连续运动奖励。

本阶段仅完成数据、真实仿真、规则基线。不要把 MuJoCo 普通 step 误称为可微分 rollout。

**速度标签与避免信息泄漏**：`block_pos_gt`、`block_vel_gt` 只在离线数据集
作为诊断/评估标签保存，绝不能拼接进 RSSM/Actor 的输入。规则基线根据前后两帧相对位置
和机器人末端世界位置估计物块速度。仿真器使用隐藏的每物块速度与轨迹参数驱动进攻。
