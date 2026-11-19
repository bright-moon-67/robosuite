# 第一阶段验收报告

日期：2026-11-19（Asia/Shanghai）。依据 `reference/plan1/01_zhunbei.md` 和同目录 Python 参考代码。
本报告中的评估均为本机真实 MuJoCo 运行，不是原参考包的“尚未验证”声明。

## 实现范围

独立包 `extensions/defense_pre_wm/defense_task`，已 editable 安装到已有 `robot_defense` 环境。
运行时注册 `DefenseTask`；没有修改 robosuite 核心，也没有覆盖原先的 `TableDefense`。

| 要求 | 落地位置 | 验证 |
| --- | --- | --- |
| 桌面、防守圈、可变数量自由刚体 | `environment.py`, `geometry.py` | 1/3/5/8 个物块初始化及软/硬 reset |
| 随机出生、三类进攻运动 | 同上 | 同 seed 可复现，连续 reset 变化，物块不重叠且在圆外 |
| 物理子步刚体外力 | `_pre_action` | `xfrc_applied` 施力，三模式速度跟踪与合力上限 |
| 真实抓取、持续抬升清除 | `_update_events` | 双指真实接触抓起；无夹持的高空物块不算清除；失去抓取清空保持计数 |
| 物块外轮廓突破 | `circle_intrusion` | 圆边界、中心在圆外但角点侵入、旋转方块、高空免桌面突破 |
| 不可逆状态机与拆分奖励 | `_update_events`, `reward` | 多事件累加、不重复发放、成功奖励、状态即时更新 |
| 无真实速度泄漏的观测 | `_setup_observables`, `vectorize_observation` | hidden motion 改动不改变当前输入，输入白名单 |
| 观测驱动规则基线 | `policy.py` | 连续位置估速、紧迫性选择、闭爪、抬升、释放及恢复 |
| 严格对齐的采集与 replay | `dataset.py`, `cli.py` | 实际轨迹链校验、终止/截断、GT 时标、连续窗口不跨局 |
| CLI、数据、评估入口 | `cli.py` | smoke/demo/evaluate/collect/validate 实际运行 |

## 与参考代码/旧场景的关键差异

1. **Panda 配置与动作坐标系**：通用 `BASIC` 改为 Panda 专属组合控制器；固定世界坐标增量 OSC_POSE，避免将世界坐标误差当成底座坐标动作。保留原有关节索引接口，经本 fork 验证有效，不作无依据替换。
2. **可达范围与推进力**：参考值未经过真实实验，桌面改为 1.0×1.1 m，出生半径 0.32–0.40 m，环形扇区 100°–260°；桌面和方块使用一致摩擦，按实际质量设置速度反馈与 0.12 N 合力限幅。
3. **推进时钟和接触**：使用 MuJoCo 当前子步时间；当前子步双指接触就暂停力，避免使用上一控制步的滞后抓取标记。每次 episode 重新生成隐藏运动参数。
4. **清除规则更严格**：按新任务参考要求，只有真实夹持且达到高度并持续保持才清除。没有沿用旧 TableDefense 的高空自动清除或外圈清除捷径。
5. **末端速度和姿态**：本体速度读取 MuJoCo 瞬时 site 线速度，补充末端姿态四元数用于增量控制姿态保持；单元素 Observable 的标量表现统一在向量化时处理。五块观测维数因此明确为 60。
6. **终止语义补全**：增加 `terminated/truncated/continuation/is_first/is_last`，超时允许 bootstrap，真正成功/失败 continuation=0。原参考保存函数没有 continuation，无法仅凭原 `done` 恢复该区别。
7. **Replay 真正实现**：参考只有 NPZ I/O，新增有容量限制、按窗口采样、拒绝混合观测维数、不跨 reset 的 ReplayBuffer。
8. **对齐与质量校验**：参考仅检查长度/部分 NaN；现在校验动作、所有标签、相邻状态、奖励、时间、事件状态与标签对应，并防止静默覆盖已有数据。
9. **标签时标明确**：原 `block_*_gt` 来自 step 后，现明确为 t+1，并新增 t 时刻 `prev_block_*_gt`。所有 GT、隐藏运动参数严格留在诊断通道/JSON。
10. **文档纠正**：原参考说明的“不可读取 fork、未安装仿真”不适用于本机；当前文档给出真实结果和本地命令。不再引用未编译论文里可能变化的公式编号来定义接口。

## 测试

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 \
/home/pan/miniforge3/envs/robot_defense/bin/python -m pytest \
  extensions/defense_pre_wm/tests tests/test_environments/test_table_defense.py -q
```

结果：**66 passed**，包括本阶段 44 项和原 TableDefense 22 项回归测试，无跳过。
另通过 Black/isort 检查。测试仅覆盖相关部分，未声称运行完整 robosuite 测试集。

## 规则基线实测

每局使用表中区间内的独立 seed；速度 0.008–0.016 m/s，启动间隔 6 s，horizon=1400。
评估不插入探索随机动作。成功是整局所有物块均清除且没有突破。

| 任务 | seeds | 局数 | 成功局数 | 成功率 | 清除物块 |
| --- | --- | ---: | ---: | ---: | ---: |
| 单物块直线 | 100–129 | 30 | 28 | 93.3% | 28/30 |
| 单物块变速 pulsed | 300–319 | 20 | 19 | 95.0% | 19/20 |
| 五物块直线 | 200–219 | 20 | 14 | 70.0% | 82/100 |
| 五物块横向扰动 sine | 400–419 | 20 | 14 | 70.0% | 78/100 |

以上 90 局没有超时，其余局因突破终止。失败仍真实存在，规则基线不是训练好的 DiffDreamer。
允许抓取等待启动的方块，慢速错峰设置有助于完成任务，不能将这些成功率外推到同时高速进攻。

原始报告：
- [单块直线](../../artifacts/defense_pre_wm/eval_single_linear.json)
- [单块变速](../../artifacts/defense_pre_wm/eval_single_pulsed.json)
- [五块直线](../../artifacts/defense_pre_wm/eval_five_linear.json)
- [五块扰动](../../artifacts/defense_pre_wm/eval_five_sine.json)
- [真实单块抓取视频，seed=43，原速6.2秒](../../artifacts/defense_pre_wm/single_grasp.mp4)

在五块评估中，排除接近夹爪/抓持/高空目标后的每局速度 RMSE 的平均值：
直线约 **0.606 mm/s**，sine 约 **0.657 mm/s**。每局有效样本数保存在报告中；
这项诊断不是抓取接触时的速度跟踪指标。单独的三模式自由运动测试要求 RMSE < 2 mm/s，均通过。

## 数据集与序列采样验收

位置：[artifacts/defense_pre_wm/datasets/sine5](../../artifacts/defense_pre_wm/datasets/sine5)。
100 局五物块 sine，seed=1000–1099，探索概率 0.15。

- **39,752 条转移**，实际随机替代动作 5,920 次。
- 38 局成功、61 局突破失败、1 局时间上限截断。
- 共清除 313/500 个物块。
- 99 局真正终止，1 局超时；超时最后一行 continuation=1。
- 文件共约 25.1 MB（NPZ+JSON），均可用 `numpy.load(..., allow_pickle=False)` 读取。
- 所有 100 局通过完整校验；没有只保存成功局或删除失败数据。
- Replay 实测 `batch_size=16,length=32`，输出 `state.shape=(16,33,60)`、`action.shape=(16,32,7)`。
- 返回字典只有训练通道，不含真实速度和未来运动参数；窗口不跨 episode。

[采集统计](../../artifacts/defense_pre_wm/collect_sine100.json) · [独立数据验收统计](../../artifacts/defense_pre_wm/dataset_validation.json)。

## 当前边界

- RSSM、可微机器人模型和双路径 Actor-Critic 尚未实现，这是任务要求的下一阶段。
- 连续奖励含最近目标切换、mask 等分段操作；需要下一阶段重新审查可微训练图。
- 物块推进是模拟主动滑动目标的外力模型；抓住后停止推进。桌面低摩擦和理想状态反馈是明确的建模假设。
- 默认只校准 Panda 与环形扇区；全圆、其他机械臂、高速接触和感知噪声未验证。
- XML/普通 MuJoCo state 不包含全部任务隐藏变量；采集重放应使用提供的 seed/配置和转移数据，不能把普通 XML reset 当作完整任务快照恢复。
