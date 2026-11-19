# DefenseTask：DiffDreamer 的 World Model 前置阶段

本包根据 `reference/plan1/01_zhunbei.md` 和该目录的 Python 参考实现改写，
已在本地 robosuite 1.5.2 / MuJoCo 3.9.0 / Python 3.11 环境实际运行。
安装位置：`extensions/defense_pre_wm`；包名：`defense_task`。

这是独立扩展包，导入 `defense_task.environment` 时注册 `DefenseTask`。
原 `TableDefense` 是先前的场景预览环境；本阶段使用符合新要求的 `DefenseTask`，
没有覆盖旧环境，也不要求修改 robosuite 核心源码。

## 本阶段交付

- Panda + 平行夹爪、桌面中心半径 0.20 m 的防御圆，默认 5 个自由刚体方块。
- `linear`、`sine`、`pulsed` 三类运动，每次 reset 随机采样位置、速度、启动抖动、相位及频率。
- MuJoCo 每个物理子步施加限幅 `xfrc_applied`，运行中不修改物块 qpos，不焊接到夹爪。
- 真实双指接触且中心高于桌面 0.11 m，连续 3 个控制步后清除。原 TableDefense 的“只要抬高/推出外圈”规则不在本环境使用。
- 旋转/倾斜方块八顶点的水平凸包与防御圆相交即突破；最低点离桌面超过 3.5 cm 时不按桌面突破处理。
- 分离离散事件奖励和连续运动奖励，记录真实终止、时间上限截断和 continuation。
- 仅利用当前/历史观测的紧迫性规则基线，世界坐标 OSC_POSE 增量动作；带超时重试与末端姿态保持。
- 严格对齐的 NPZ 数据集、元信息 JSON、校验器和不跨 episode 的 ReplayBuffer 序列采样器。
- 纯 Python、真实 MuJoCo 接触/抓取/力控/重置测试及批量基线评估。

本阶段没有训练 RSSM、Actor-Critic，也没有实现可微分机器人 BPTT。
`r_diff` 仅表示连续形式奖励，普通 MuJoCo `step()` 不能直接作为 PyTorch 可微 rollout。

## 默认配置及适配修改

| 项目 | 当前值 / 说明 |
| --- | --- |
| 防守圆 | 世界坐标 (0,0)，半径 0.20 m |
| 桌面 | 1.00×1.10 m，表面高度 0.80 m |
| 方块 | 边长 4.4 cm，密度 600 kg/m³，约 51.1 g |
| 出生区域 | 半径 0.32–0.40 m；100°–260°环形扇区，靠近负 x 方向的 Panda 底座 |
| 速度 | 0.008–0.016 m/s；pulsed 径向速度系数为 0.75–1.25，sine 另含横向速度 |
| 启动 | 2 s 准备 + 每物块 6 s 间隔 + 各自 0–0.5 s 抖动 |
| 横向速度 | 每 episode 采样 0.002–0.005 m/s，频率 0.5–1.0 rad/s，随机相位 |
| 桌面/方块滑动摩擦 | 0.025，作为主动滑动目标的初版物理抽象 |
| 外力 | 速度反馈时间常数 0.035 s + 摩擦补偿，水平合力最大 0.12 N |
| 控制频率 | 20 Hz；MuJoCo 子步 0.002 s |
| 默认 horizon | 1400 步（70 s） |

参考代码的 1.2 m 桌面、0.38–0.49 m 出生半径和高摩擦/大推力是未验证的初值。
这里收缩桌面和环形区以改善 Panda 可达性，同时校准低速推进。出生角度仍可设置成全圆，
但默认仅使用可达性较好的环形扇区；全圆抓取性能未验证。
默认场景允许抓取尚未启动但已经出现在桌面上的物块，这会降低慢速错峰任务难度。

当前 fork 的 `load_composite_controller_config(robot="Panda")` 返回已经展平的 `body_parts.right`。
直接加载通用 `BASIC` 会带入不属于 Panda 的左臂/躯干等配置。现在统一使用 Panda 专属配置，
明确设置 `OSC_POSE`、`input_type=delta`、`input_ref_frame=world`。
动作前三维归一化到 ±0.05 m 的末端位置增量，中间三维归一化到 ±0.5 rad 的旋转增量，最后一维控制夹爪。
策略中的比例增益是归一化动作增益，并不等于米单位的直接位置跳变。

推进力在当前子步检测到双指接触，或物块中心高于正常放置高度 8 mm 时暂停。
被抬起但未形成真实夹持的物块不能清除；连续夹持中断会清空保持计数。
已清除物块仍有碰撞和重力，释放后不再进攻或触发突破。默认首次突破/全清除立即终止，
最终一个物块清除后不会继续执行释放；中间物块会由状态机释放。

## 安装与运行

```bash
cd /home/pan/projects/robosuite
source /home/pan/miniforge3/etc/profile.d/conda.sh
conda activate robot_defense
# 使用当前工作区 fork，无需另行下载或覆盖 robosuite
pip install -e .
pip install -e './extensions/defense_pre_wm[test,video]'

# 无渲染运行
python -m defense_task.cli smoke --blocks 1 --horizon 100
python -m defense_task.cli demo --blocks 1

# 无窗口的物理抓取视频（原速，非先前的固定顺序脚本）
MUJOCO_GL=egl python -m defense_task.cli demo --blocks 1 --seed 43 \
  --video artifacts/defense_pre_wm/my_grasp.mp4

# 评估（不加随机动作）
python -m defense_task.cli evaluate --blocks 5 --episodes 20 --seed 200 \
  --report artifacts/defense_pre_wm/my_eval.json

# 采集：每步 15% 概率以随机动作替代规则动作，提供失败和恢复样本
python -m defense_task.cli collect --blocks 5 --episodes 100 --motion sine \
  --seed 1000 --epsilon .15 --out artifacts/defense_pre_wm/my_dataset \
  --report artifacts/defense_pre_wm/my_collect.json
python -m defense_task.cli validate --out artifacts/defense_pre_wm/my_dataset

MUJOCO_GL=egl python -m pytest extensions/defense_pre_wm/tests -q
```

输出文件不会静默覆盖：数据集按现有最大编号继续追加，视频和报告要求新的文件名。
每个 episode 单独设置 `base_seed + episode_index`，元信息记录 seed 和配置。
重新启动采集时如需不同 episode，请更换 `--seed`；文件编号与 seed 独立。
CLI 的 `--horizon` 同时决定环境上限和采集长度，不会悄悄丢弃最后一个转移。
headless 评估不创建渲染上下文；视频需要 EGL（不可用时可选 OSMesa）。

## 观测与无泄漏输入

只调用 `vectorize_observation(obs)`，不要把全部 `obs`、`info` 或元信息直接拼接给模型。
明确允许的顺序如下，N 个物块总宽度为 **25+7N**（五块为 60）：

| 字段 | 维数 | 内容 |
| --- | --- | --- |
| `defense_robot` | 20 | 7 关节位置、7 关节速度、末端世界位置与瞬时线速度 |
| `defense_blocks` | 6N | 每块相对末端 xyz、active、grasped、cleared |
| `defense_status` | N | 0 进攻/待启动、1 清除、2 突破 |
| `defense_clock` | 1 | 当前时间除以 episode 时间上限 |
| `defense_eef_quat` | 4 | 世界坐标末端姿态 xyzw，属于机器人自身感知 |

最后一个姿态字段是为当前 OSC 增量控制补充的本体观测，用来防止接触造成末端姿态逐步偏移。
当前 fork 会把长度为 1 的 Observable 转为标量，向量化入口统一按一维展平。
物块真实速度、运动模式参数、随机启动时间和未来轨迹均不在上述输入中。
规则策略根据 `相对位置 + 末端世界位置` 的连续观测估计物块速度，从来不接收 env、info 或轨迹参数。
机械臂与物块的位置仍是理想化的仿真状态反馈，本阶段没有相机感知噪声模型。

## 奖励和终止

- `r_disc = 3×新清除数 - 12×新突破数 + 8×本步首次全清除`。
- `r_diff = -0.18*tanh(5*d_nearest_active) - 0.0005*sum(action²)`。
- `reward = r_disc + r_diff`；连续部分的最近目标切换和 active mask 也会带来分段性，不能据此声称已完成光滑端到端梯度。
- `terminated`：成功或者默认的首次突破；`truncated`：仅时间耗尽。
- `done = terminated OR truncated`；两者同时发生时任务终止优先。
- `continuation = 1-terminated`，超时最后一步仍为 1，保留价值 bootstrap。
- 几何事件在控制步末检查，当前低速配置经过测试；极高速运动可能需要子步连续碰撞事件检测。

## 数据集与 Replay

每个 `episode_XXXXX.npz` 是一整局，JSON 包含观测切片 schema、动作单位、seed、配置和诊断元信息。
第 t 行严格表示：`obs_t, action_t, reward_{t+1}, continuation_{t+1}, next_obs_{t+1}`。

训练通道：`obs/action/next_obs/reward/r_disc/r_diff/done/terminated/truncated/continuation/is_first/is_last`。
诊断通道：`status/cleared_count/breached_count/block_pos_gt/block_vel_gt/prev_block_pos_gt/prev_block_vel_gt/desired_vel_gt/time/next_time/action_source`。

`block_*_gt` 标签对应 t+1；`prev_block_*_gt` 对应 t。`action_source=1` 表示随机探索动作。
JSON 中的 `hidden_motion_diagnostics` 只用于复现实验，不加载为模型输入。
`desired_vel_gt` 是诊断期望速度，接触或抬起时实际驱动还会被抑制，不能当作实际速度标签。

校验器检查：所有通道形状与有限值、动作范围、相邻 `next_obs[t]==obs[t+1]`、
奖励拆分、结束标记、continuation、状态单向变化、计数、mask、标签位置和时间对齐。

```python
from pathlib import Path
from defense_task import ReplayBuffer

replay = ReplayBuffer(capacity_episodes=1000, seed=0)
for path in sorted(Path('artifacts/defense_pre_wm/datasets/sine5').glob('episode_*.npz')):
    replay.add(path)
batch = replay.sample(batch_size=16, length=32)
# batch['state']: (16,33,60)，含首个状态及每一步 next_obs
# batch['action']: (16,32,7)
# reward/continuation 等: (16,32)
```

序列不会跨 episode；按合法窗口数量加权采样。随机截取的窗口不一定从 episode 首帧开始，
`is_first` 保留真实重置标记，下一阶段 RSSM 需要自行设计 burn-in 或隐状态初始化。
Replay 仅保存/返回训练通道，隔离全部 GT 诊断；不同物块数量应使用独立 Replay，当前不做 padding。

## 验证报告

实测结果、参考代码修正清单及产物链接见 [VALIDATION.md](VALIDATION.md)。
