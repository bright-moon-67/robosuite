# 桌面防守任务 TableDefense

基于项目现有 `ManipulationEnv`、`TableArena`、`BoxObject` 和 Panda 机械臂搭建。
环境负责来袭物块、碰撞、观测与任务判定；后续策略通过标准 `env.step(action)` 接入。
新增环境已在包入口注册，可以直接 `suite.make("TableDefense", ...)`。

## 默认场景

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `robots` | `Panda` | 现有 Panda 机械臂及双指夹爪 |
| `table_full_size` | `(0.8, 1.0, 0.05)` m | 沿用原桌面结构，仅适当加宽 y 方向 |
| 桌面高度 | `0.8` m | 与 Lift 场景一致 |
| `defense_center` | `(-0.12, 0)` m | 世界坐标下的圆心，靠近机械臂以改善可达性 |
| `defense_radius` | `0.20` m | 青色标线标出的防守圆，标线无碰撞 |
| `num_attackers` | `5` | 5 种颜色的物块 |
| `cube_size` | `0.04` m | 立方体完整边长，适合 Panda 夹爪 |
| `spawn_radius` | `(0.34, 0.38)` m | 初始物块中心距圆心的距离范围 |
| 来袭方向 | `-80°…80°` | 机械臂前方半圆，避免从底座后方来袭 |
| `attack_speed` | `0.008` m/s | 直线径向速度约 0.8 cm/s，可传入逐物块速度列表 |
| `start_delay` | `2.0` s | 首个物块启动前的准备时间 |
| `launch_interval` | `4.0` s | 物块依次启动的时间间隔；设为 0 即同时进攻 |
| `motion` | `linear` | 直线向圆心；还支持 `weave`、`mixed` |
| `removal_height` | `0.12` m | 物块最低点离桌面的移除高度 |
| `removal_hold_time` | `0.15` s | 必须持续高于移除高度，避免跳动误判 |
| `removal_radius` | `0.46` m | 整个物块的平面投影离开此半径也算移除 |
| `control_freq` | `20` Hz | 策略控制频率 |
| `horizon` | `1200` 步 | 默认最长 60 秒 |

初始方块边缘距离防守圆约 11.7–15.7 cm。速度较慢，加上错峰启动，
便于先验证抓取和移除流程；这组参数的固定脚本演示已实现 5/5 移除、0 越界。
这不是训练后策略的性能结论，也不保证任意随机化配置能成功。

## 物理运动和规则

- 物块具有自由关节、质量、重力和碰撞，通过限幅的水平力与速度反馈缓慢移动。
  不在运行中硬设位置，不与夹爪强制绑定。低摩擦桌面加推进力是初版“主动移动目标”的抽象。
- 双指接触形成抓取，或物块被略微抬离桌面时，推进力暂停；放回桌面且尚未移除时可以继续来袭。
- `linear` 沿初始径向直线运动。`weave` 在径向速度上叠加正弦横向速度，
  默认振幅 1.8 cm、周期 8 s；此时总速度可能高于径向 `attack_speed`。`mixed` 交替使用两种运动。
- 越界判定使用旋转后立方体八个顶点在桌面上的凸包，检测整个物块投影是否与圆相交，
  而不是只检测物块中心。物块已被抬高到安全高度时不再触发桌面越界。
- 任何一次越界都会锁定失败，圆圈变红；默认立即结束。
  `terminate_on_breach=False` 可让失败后的演示继续。
- 达到高度并保持足够时间，或整个投影离开外圈，算作永久移除；推进力关闭。
  物块仍真实存在并受重力作用，落下后的已移除物块不再作为来袭威胁。
  因此初期不用设计专门投放箱或精确落点。
- 所有物块移除且从未越界才算成功。`ignore_done=True` 仅用于视频预览，
  允许完整播放抓取、抬升、搬开和松爪过程。
- 奖励采用事件计分：每移除一块 `+1/N`，每块首次越界 `-1`，再乘 `reward_scale`。
  `reward_shaping` 仅为兼容现有环境调用保留，当前不增加稠密奖励。

## 运行与渲染

当前机器可使用已有 conda 环境：

```bash
cd /home/pan/projects/robosuite
source /home/pan/miniforge3/etc/profile.d/conda.sh
conda activate robot_defense
pip install -e .
pip install -r requirements-defense.txt

# 固定脚本的物理抓取演示，生成 MP4、GIF、截图和事件 JSON
MUJOCO_GL=egl python -m robosuite.demos.demo_table_defense --mode scripted --gif

# 机械臂保持静止，观看物块来袭及越界后圆圈变红
MUJOCO_GL=egl python -m robosuite.demos.demo_table_defense --mode incoming

# 调整场景：3 个物块同时、混合路线进攻
MUJOCO_GL=egl python -m robosuite.demos.demo_table_defense \
  --mode incoming --count 3 --speed 0.006 --motion mixed --launch-interval 0

# 仅运行物理验证并保存 JSON，不渲染
MUJOCO_GL=egl python -m robosuite.demos.demo_table_defense --no-video

# 场景回归测试
MUJOCO_GL=egl python -m pytest tests/test_environments/test_table_defense.py -q
```

视频默认保存至 `artifacts/table_defense/`。主画面为斜视，右上角为俯视，
右侧显示防守半径、速度、移除数量和越界数量。默认 **2 倍速播放**，
叠加文字中的时间仍为真实仿真时间；可用 `--playback 1` 输出原速。

`--mode scripted` 是独立的可视化验证脚本：按固定顺序发送现成 OSC 末端位姿控制命令，
实际合拢夹爪抓取，再抬升、移开和释放。没有训练策略，也没有绕过碰撞或瞬移物块。
固定脚本仅验证默认配置；改动速度、数量、运动模式之后，需要独立评估抓取是否仍可行。
如果本机 EGL 不可用，可以改用 `MUJOCO_GL=osmesa` 软件渲染。

## 后续算法接口

```python
import robosuite as suite

env = suite.make(
    "TableDefense",
    robots="Panda",
    num_attackers=5,
    defense_radius=0.20,
    attack_speed=0.008,
    motion="linear",
    use_camera_obs=False,
    has_renderer=False,
    has_offscreen_renderer=False,
)
obs = env.reset()
# action = your_policy(obs)
# obs, reward, done, info = env.step(action)
env.close()
```

沿用 Panda 默认动作空间（默认 OSC_POSE 增量控制 + 夹爪），演示脚本另外配置
为世界坐标下的绝对末端位姿控制。`obs` 在现有机器人状态之外提供：

- `attackers_pos`：N×3 个位置，展平；
- `attackers_vel`：N×3 个线速度，展平；
- `attackers_status`：每块 `[removed, breached]`，展平；
- `defense_geometry`：圆心 x、y 和半径。

`info` 提供 `success`、`failed`、`removed_count`、`breach_count`、`active_count`、
`clearances`、`removed`、`breached`、`time`。其中 `clearances <= 0` 表示投影与防守圆相交，
是否判失败还要结合高度及已移除状态。`env.events` 记录带时间戳的移除与越界事件。

配置会检查初始重叠、物块是否在桌面内、防守圆是否在桌面内及参数范围；
增加物块数量或扩大半径时，可能也需要调整桌面尺寸、初始半径和启动间隔。
