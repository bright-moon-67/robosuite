我已经按照我们之前讨论的架构，完成了第一阶段代码，包括动态防御环境、真实物理交互、抓取与突破判定、奖励设计、规则控制器、数据采集以及 Replay Buffer 数据准备。

代码采用独立 Python 包的形式，不需要直接修改 robosuite 核心源码，后续可以继续接入 RSSM 和 DiffDreamer 的双路径优化模块。

完整项目源码
Python 包、环境、基线、数据采集、测试与 README

下载完整 ZIP 项目

查看全部源码（按文件完整展开）

查看 README 和运行说明

验证情况： 已完成语法检查，7 项纯 Python 测试通过；真实 robosuite/MuJoCo 仿真测试因当前环境没有安装这两个库而跳过。另外，你的 GitHub fork 当前无法通过公开接口读取，所以这份代码针对已核实的 robosuite 1.5.x 接口实现，尚不能声称已经在你的 fork 上完整运行验证。

一、任务设置与实现计划

首版采用以下配置，数值都可以通过环境构造参数调整。

配置项	首版设置
机械臂	Franka Panda + 平行夹爪
仿真平台	robosuite + MuJoCo
防御区域	桌面中心、半径 0.20 m
进攻物块	默认 5 个
初始分布	防御圆外的环形区域
初始运动	向圆心低速移动
扩展运动	变速、正弦横向扰动
抓取成功	真实夹持后抬升到规定高度，并稳定保持
防御失败	未清除物块的桌面投影进入防御圆
控制器	OSC_POSE，末端位姿增量 + 夹爪控制
观测形式	结构化状态，不使用 RGB 图像
完整实现步骤

搭建 robosuite DefenseTask 环境

复用 ManipulationEnv、TableArena 和 BoxObject，生成 Panda、桌面、防御圆和多个物块。

实现可配置的进攻轨迹

支持直线向圆心、变速和横向扰动。轨迹参数每个 episode 随机生成，但不直接交给策略。

实现基于 MuJoCo 的物块运动

使用 xfrc_applied 在物理子步施加驱动力，不逐帧强制修改物块位置，因此夹爪可以真实地与物块发生接触。

实现抓取、清除和突破事件

复用 robosuite 的 _check_grasp()；只有真实抓取并达到高度要求才算清除。使用物块外轮廓与防御圆的几何关系判定突破。

实现任务状态机和奖励

每个物块具有进攻中、已清除、已突破状态；离散成功/失败奖励与连续运动奖励独立保存。

实现观测接口

机械臂提供关节位置、速度、末端位置与速度；环境提供物块相对于末端的位置及有效状态，不直接泄露真实进攻轨迹参数。

实现规则控制器

从连续观测估计进攻趋势，选择紧迫物块，执行接近、下降、闭合夹爪、抬升和释放的状态机。

建立数据采集与 Replay Buffer

保存严格对齐的 \(o_t,a_t,r_t,o_{t+1}\)、episode continuation 信息、离散奖励、连续奖励和离线诊断标签。

实现测试及基线评估

测试防御圈边界、不同物块数量、随机生成、数据对齐、奖励一致性，并提供真实 MuJoCo 烟雾测试入口。

二、已经完成的代码文件
文件	功能

environment.py	完整自定义 robosuite 环境

geometry.py	防御区几何判定及物块初始位置采样

policy.py	规则控制器及紧迫性选择

dataset.py	轨迹保存、加载与一致性验证

cli.py	仿真、演示、评估与数据采集命令

test_geometry_dataset.py	纯 Python 边界与数据测试

test_sim_smoke.py	真实 robosuite 环境烟雾测试

上面的链接均对应完整源文件，不是伪代码或接口占位。

三、关键实现方式
1. 物块运动：保留真实接触动力学

环境的 _pre_action() 在每个 MuJoCo 物理子步执行。

其中最关键的是：根据期望进攻速度和物块当前实际速度计算外力，同时考虑摩擦补偿和力的限幅。

代码中的核心处理如下：

# 物块仍由 MuJoCo 自由刚体动力学求解
measured = self._block_v(i)[:2]

mass = max(
    1e-4,
    float(self.sim.model.body_mass[bid])
)

unit = desired / np.linalg.norm(desired)

friction = (
    1.05
    * self.table_friction[0]
    * mass
    * 9.81
)

force = (
    friction * unit
    + 5.0 * (desired - measured)
)

norm = float(np.linalg.norm(force))

if norm > 1.8:
    force *= 1.8 / norm

self.sim.data.xfrc_applied[bid, :2] = force

抓取建立后，不再向被夹持物块施加进攻驱动力。

因此，物块不会在夹爪抓住它之后仍然被程序强制移动到圆心。

2. 世界模型观测：不泄露未来信息

我们此前特别讨论过 DiffDreamer 的 Gate 预测方式：当前 Gate 的位置是已知的，但未来轨迹没有直接交给 RSSM。

这里采用相同的原则。

物块观测主要为：

$$ o_t^{obj,i} = [ p_t^{obj,i}-p_t^{ee}, m_t^{active}, m_t^{grasp}, m_t^{cleared} ] $$

机械臂状态为：

$$ x_t^{robot} = [q_t,\dot q_t,p_t^{ee},v_t^{ee}] $$

这些状态已经通过 robosuite 的 Observable 接口实现。

物块真实速度和位置另外保存在数据集的 block_vel_gt、block_pos_gt 中，用于后续评估预测效果，但不会拼接进 RSSM 或 Actor 的观测输入。

这能够避免 World Model 因为直接获得仿真器的真实速度而失去从历史信息学习运动趋势的必要性。

3. 奖励：保持 DiffDreamer 的两种优化目标

代码已经独立计算：

$$ r_t=r_t^{disc}+r_t^{diff} $$

其中：

离散奖励用于清除成功、突破失败和全部防守成功事件。

连续奖励目前使用末端与有效物块的距离以及动作惩罚，为后续可微运动优化准备接口。

需要严格说明：现在 r_diff 只是连续形式的奖励，尚未接入可微分机械臂动力学进行 BPTT。这一步应当在 World Model 及自身动力学模型准备好后再完成。

四、如何放入你的项目并运行？

建议不要立即复制到你已有的 robosuite 源码目录，而是把下载的文件夹作为一个独立包放在同一工作环境。

在已经安装你的 robosuite fork 的 Python 环境中，进入下载后的目录：

cd diffdreamer_defense_pre_wm

pip install -e ".[test]"

首先测试纯 Python 部分：

pytest -q tests/test_geometry_dataset.py

然后测试环境创建与物理仿真：

python -m defense_task.cli smoke \
    --blocks 1 \
    --horizon 100

确认不报错后，开启图形化演示：

python -m defense_task.cli demo \
    --blocks 1 \
    --horizon 400

再逐步增加到多个进攻物块：

python -m defense_task.cli demo \
    --blocks 5 \
    --motion linear \
    --min-speed 0.008 \
    --max-speed 0.016

在规则控制器完成基础校准后，进行批量评估：

python -m defense_task.cli evaluate \
    --blocks 5 \
    --episodes 20

最后采集 World Model 所需的轨迹：

python -m defense_task.cli collect \
    --blocks 5 \
    --episodes 100 \
    --motion sine \
    --out datasets/defense

输出的数据可直接用 numpy.load() 读取，已经包含后续 RSSM 所需的时序交互字段。

五、哪些部分还需要真实仿真验证？

这份代码不是在你的 fork 中完成端到端实验后导出的结果，因此有几项需要按顺序确认：

检查项目	当前状态
Python 语法检查	已通过
圆形防御区边界判定	已通过单元测试
多物块随机初始位置	已通过单元测试
奖励分解和轨迹对齐	已通过单元测试
规则控制器基础状态逻辑	已通过单元测试
你的 robosuite fork 环境初始化	尚未运行验证
Panda 实际抓取成功率	尚未测量
进攻外力下的实际速度跟踪	尚未测量
五物块防御是否具有可行时间余量	尚未测量

尤其是规则控制器：它已经提供了完整动作生成逻辑，但 Panda 的默认夹爪姿态、桌面可达范围和 OSC 增益仍可能影响真实抓取成功率，不能把“代码完整”直接等同于“抓取性能已验证”。

六、下一阶段与 DiffDreamer 如何衔接？

当前项目到此为止：

robosuite DefenseTask
动态物块 + 机械臂真实接触

Task Observations + Rewards
外部相对状态 + 自身物理状态

Replay Dataset
连续轨迹、动作、奖励、事件

下一阶段
RSSM World Model Training
Encoder + Prior + Posterior + Prediction Heads

后续应再独立实现可微分机械臂动力学；普通 MuJoCo step() 本身不能直接当作 DiffDreamer 的 PyTorch 可微分物理 Rollout。最终才将 RSSM、可微分自身动力学与共享 Actor-Critic 按照原论文 Eq. (9)–(11) 同步耦合。

我建议你现在先验收三个结果：单物块确实向圆心运动、机械臂确实能够物理抓起它、数据采集记录正确。 这三项稳定后，再开展五物块实验和 RSSM 训练，能够避免 World Model 学到的是环境实现错误而不是真正的动态规律。

目前的完整代码与运行说明都已打包在上方 ZIP 中，可以直接作为你这个动态防御任务的初始工程。