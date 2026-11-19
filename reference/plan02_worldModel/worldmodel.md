DiffDreamer 下一阶段：World Model 构建、训练与验收计划

针对你已经完成的 robosuite 多物块动态防御任务，下一阶段我建议先独立完成 RSSM World Model，而不是立即接入 Actor-Critic。

你现在已经有 100 局 sine5 轨迹、60 维观测、7 维动作以及奖励和终止信息，完全具备启动第一版 World Model 训练的条件。

这一阶段的最终目标不是训练出能够成功防御的策略，而是得到一个可反复调用、能预测物块未来运动，并且可以接入 DiffDreamer 双路径架构的 World Model。

具体需要交付四项成果：

训练好的 RSSM，包括 Encoder、Prior、Posterior 和预测头。
能根据实际观测更新的隐状态 \(s_t^{imp}=(h_t,z_t)\)。
不读取未来真实观测、可以递归预测未来物块运动的 Imagination 接口。
完整的预测评测，包括开环位置误差、突破风险、运动泛化以及与简单预测器的比较。
一、先确定 World Model 与机械臂动力学的职责

这一步必须严格延续我们之前讨论的 DiffDreamer 分工。

原论文 Section III-D 规定，World Model 传播隐式状态，可微分物理模型传播机器人显式状态；同一个 Actor 的动作同时驱动两条路径。

对于你的任务，我建议采用：

World Model Training — Current Stage

Replay Buffer
60-D observations · 7-D actions · discrete rewards · continuation

Observation Encoder + RSSM
\(String.raw`s_t^{imp}=(h_t,z_t)`\)
Inference from observations and action history

Future Prediction
Block positions / contact states

Event Prediction
Clear / breach / continuation

Open-loop Evaluation + Saved Checkpoint

本阶段只训练 World Model；暂时不训练 Actor，也不把普通 MuJoCo step 当作可微分机器人动力学。

两条路径以后分别负责什么？
模块	主要职责	当前阶段
RSSM World Model	多物块运动、隐藏运动模式、接触状态和任务事件	现在构建
Differentiable Robot Dynamics	机械臂关节、末端运动和可微物理梯度	下一阶段
Actor	结合隐状态与自身状态决定抓取、拦截动作	暂不训练
Critic	估计清除和突破等离散事件的长期价值	暂不训练

这里有一个接触场景中特别重要的设计：

World Model 可以使用机械臂自身状态作为观测条件，但不应把精确预测全部机械臂关节动力学作为自己的主要任务。

因此，我建议第一版优先预测物块世界位置、物块状态、离散事件与 continuation，同时保留完整 60 维观测作为 RSSM 的历史状态编码输入。

这比强制让一个 RSSM 精确重建机械臂所有高频运动更符合 DiffDreamer 的分工思想。

二、World Model 的具体结构

原论文采用 DreamerV3 风格的 RSSM，包括观测编码、确定性记忆、随机潜变量、先验转移、后验估计、观测重建以及奖励和 continuation 预测。

对于你的 5 物块防御环境，我建议采用以下结构。

模块	输入	输出	建议实现
Encoder	60 维观测	观测特征 \(e_t\)	MLP
Deterministic Dynamics	\(h_t,z_t,a_t\)	\(h_{t+1}\)	GRU
Prior Network	\(h_{t+1}\)	下一状态的先验分布	MLP
Posterior Network	\(h_{t+1},e_{t+1}\)	后验分布	MLP
Position Head	\(h_t,z_t\)	5 个物块的世界位置	MLP
Status Head	\(h_t,z_t\)	每个物块的进攻、清除、突破状态	分类头
Reward Head	\(h_t,z_t\)	离散任务奖励	回归或分布预测
Continuation Head	\(h_t,z_t\)	Episode 继续概率	二分类头

关键设计：RSSM 先验不能依赖未来真实观测。

在有真实数据时，进行后验更新：

$$ \begin{aligned} h_{t+1}&=f_\phi(h_t,z_t,a_t)\\ z_{t+1}&\sim q_\phi(z_{t+1}\mid h_{t+1},e_{t+1}) \end{aligned} $$

在未来想象时，只使用先验：

$$ \boxed{ \begin{aligned} h_{t+1}&=f_\phi(h_t,z_t,a_t)\\ \hat z_{t+1}&\sim p_\phi(z_{t+1}\mid h_{t+1}) \end{aligned} } $$

这里的 \(a_t\) 是你的 7 维机械臂与夹爪动作。

为什么建议预测物块的世界坐标？

你目前的观测是物块相对于末端的位置：

$$ p_t^{rel,i}=p_t^{block,i}-p_t^{eef} $$

虽然输入使用相对位置非常合理，但在评估外部物块未来运动时，可以将它转换为：

$$ \boxed{ p_t^{block,i}=p_t^{rel,i}+p_t^{eef} } $$

然后让 Position Head 预测世界坐标系下的物块位置。

好处是，开环预测物块轨迹时，不需要偷偷读取未来的真实机械臂末端位置，避免信息泄漏。

不过，后续进入抓取阶段时，物块受到机械臂接触影响。届时仅使用动作历史可能不足，需要将快速动力学路径预测的机械臂状态同步提供给交互预测模块。

这属于后续双路径耦合阶段，当前先不增加复杂度。

三、60 维观测如何分配？

以下索引按照你提供的字段顺序设计。正式运行前，要与你每局 JSON 中保存的 observation slices 再核对一次。

索引（Python 切片）	数据	处理
0:14	机械臂关节位置、速度	连续特征归一化
14:20	末端世界位置、速度	连续特征归一化
20:50	5 个物块，每个 6 维	位置归一化，状态标志保留
50:55	每个物块的状态编码	作为分类标签
55:56	归一化时间	保留
56:60	末端姿态四元数	保留，检查约定与范数

这里建议先使用完整 60 维数据编码历史，但让预测损失更侧重物块运动，而不是平均地要求网络精确重建全部机械臂状态。

四、下一阶段的详细执行步骤
步骤	具体工作	验收结果
1	读取现有 sine5 NPZ 和 JSON，检查字段、切片与状态定义	数据加载与维度一致
2	按完整 Episode 划分训练、验证和测试集	70/15/15 局，互不重叠
3	仅用训练集拟合连续特征归一化参数	保存 mean/std
4	先实现 Constant Velocity 预测器	获得不依赖神经网络的预测基准
5	实现 MLP Encoder 和 RSSM	能计算 \((h_t,z_t)\)
6	实现 Prior / Posterior	能分别运行观测更新和无观测想象
7	实现物块位置与状态预测头	输出未来物块位置与状态
8	实现 r_disc 与 continuation 预测头	能预测任务事件相关信息
9	实现 World Model Loss	能正常计算梯度并训练
10	训练初始 RSSM，保存最优检查点	损失有限、验证指标可重复
11	进行开环轨迹预测测试	与真实未来轨迹比较
12	进行速度、轨迹、接触状态泛化测试	明确适用范围和失效情况
13	将 Encoder + RSSM 封装为隐状态估计器	可用于后续 Actor 输入
14	冻结首个验证通过的检查点	为双路径接入建立稳定基线

其中最需要认真处理的是步骤 9–12。

World Model Loss 怎样设计？

根据 DreamerV3 的思路，采用：

$$ \mathcal L_{WM} = \mathcal L_{pred} + \beta_{dyn}\mathcal L_{dyn} + \beta_{rep}\mathcal L_{rep} $$

预测损失可以拆分为：

$$ \mathcal L_{pred} = w_p\mathcal L_{position} +w_s\mathcal L_{status} +w_r\mathcal L_{reward} +w_c\mathcal L_{continue} $$

其中，位置误差使用回归损失，物块状态使用分类损失，continuation 使用二分类损失。

动力学和表征损失则约束先验与后验分布：

$$ \mathcal L_{dyn} = D_{KL}\left[ \operatorname{sg}(q_\phi)\|p_\phi \right] $$ $$ \mathcal L_{rep} = D_{KL}\left[ q_\phi\|\operatorname{sg}(p_\phi) \right] $$

还可以采用 Free Bits、Symlog 等技巧提升稳定性。这些都是标准 DreamerV3 体系中的方法，PyTorch TorchRL 文档已有相应实现。

特别提醒： DiffDreamer 原论文没有报告完整 RSSM 网络超参数和所有损失权重。因此这里是针对你任务的实施方案，不是对原论文隐藏代码的逐参数复现。

五、参考代码：我已经准备好可运行的起始工程

Defense World Model Starter
PyTorch RSSM、Replay 数据加载、训练、开环评测、部署状态接口、单元测试

下载完整 World Model 参考工程

完整 RSSM 训练与评估源码

部署期隐状态更新代码

完整测试计划

运行说明 README

这份参考代码实现了什么？

采用一个较小的 Gaussian RSSM：

参数	初始配置
输入观测	60
动作维度	7
GRU 确定性状态	128
Gaussian 随机潜变量	24
Encoder 特征	128
初始 Batch Size	16
初始序列长度	32
学习率	3e-4
初始训练步数	5000（实验起点，需按验证结果调整）

需要说明：这是便于你快速验证的 Gaussian RSSM，不是完全复刻 DreamerV3 categorical latent、two-hot 分布头的最终版本。 如果后续严格追求算法结构一致性，建议将这些模块替换为 TorchRL 中的 DreamerV3 实现。

运行方式

以下命令中的数据路径使用你提供的本机路径。在你的机器上运行即可；我这里无法直接读取该本机目录。

首先检查数据：

python worldmodel_stage.py inspect \
  --data /home/pan/projects/robosuite/artifacts/defense_pre_wm/datasets/sine5

然后开始第一版训练：

python worldmodel_stage.py train \
  --data /home/pan/projects/robosuite/artifacts/defense_pre_wm/datasets/sine5 \
  --out runs/defense_rssm_v0 \
  --device cuda \
  --steps 5000 \
  --batch 16 \
  --length 32

训练后执行开环评测：

python worldmodel_stage.py eval \
  --data /home/pan/projects/robosuite/artifacts/defense_pre_wm/datasets/sine5 \
  --out runs/defense_rssm_v0 \
  --device cuda

代码会输出 openloop_report.json，记录 World Model 与恒速预测器的误差，以及突破事件的 Precision、Recall 等基础指标。

我已经使用合成轨迹完整测试了 inspect → train → 保存模型 → eval 流程，并通过 5 项自动化测试，包括张量维度、梯度、序列边界、终止/超时、无未来观测预测等。

这些测试不代表模型在你的 100 局真实数据上已经训练成功。 由于合成烟雾测试只做了极少训练步，得到的模型还明显差于恒速预测基线；这是用于验证执行流程的结果，不能当成预测性能结论。

后续与 Actor 如何衔接？

参考工程中还包括一个完整的部署期隐状态过滤器。

它的使用逻辑为：

import numpy as np
from integration_reference import LatentFilter

# model: 已加载检查点的 RSSM
# stats: 训练阶段保存的归一化参数
filter_state = LatentFilter(model, stats, device="cuda")

# Episode 开始，输入第一帧真实观测
latent = filter_state.reset(first_observation)

# 每个后续控制周期：
latent = filter_state.update(
    previous_action,
    current_observation
)

# latent.shape = [1, deter_dim + latent_dim]
# 该张量可与机械臂的显式物理状态拼接给 Actor
actor_latent = latent.detach()

其中 LatentFilter 的完整实现已包含在下载工程里。这套接口将来可以直接对应原论文的：

$$ a_t=\pi_\theta \left( \operatorname{sg}(h_t,z_t), x_t^{robot} \right) $$

但真正接入 Actor 时，仍需要考虑机械臂与物块发生接触后的双向动力学。

六、还可以直接参考哪些成熟仓库？

我优先推荐下面三个，都是已经核对过的公开资源。

danijar/dreamerv3

DreamerV3 作者维护的 JAX 实现，适合核对 RSSM、Imagination、模型损失和整体训练循环。

TorchRL DreamerV3

对你尤其合适。包含 RSSMPriorV3、RSSMPosteriorV3、RSSMRolloutV3、DreamerV3ModelLoss 等组件，可以减少重复实现。

dreamer-core

较小的非官方 PyTorch 组件实现，适合研究 RSSM 的状态展开、KL Balancing、Two-hot 奖励预测和单元测试。

如果你已经确定后续双路径使用 PyTorch，我更建议采用 TorchRL 的 RSSM 组件 + 自定义物块预测头 + 你已有的 Replay Buffer。这样最接近“复用成熟代码，不重复造轮子”的目标。

七、测试条件：怎样判断 World Model 真的学会了预测？

我建议分成四道验收关卡。

Gate 1：基本训练正确性

首先确保模型不是仅仅能够跑完训练，而是状态与数据对齐正确。

测试	通过条件
数据加载	NPZ 字段与 60/7 维定义一致
序列采样	不跨 Episode
状态估计	Posterior 输出有限值
想象展开	不访问未来真实观测
梯度训练	Loss、梯度无 NaN/Inf
Continuation	真终止为 0，超时截断为 1
Checkpoint	重新加载得到一致推理结果

还要特别注意，你的原始 Episode 最短只有 59 步。

当前参考代码只抽取长度足够的完整序列。如果将序列长度调大到 64 或 128，部分短失败局会被排除，可能导致训练数据分布变化。

如果后面需要更长历史，建议实现 Padding + Valid Mask，而不是简单删除所有短轨迹。

Gate 2：开环未来预测

在测试集中选择一段真实历史观测，利用 RSSM 更新 \((h_t,z_t)\)，然后停止提供新的真实观测。

利用数据中记录的未来动作作为条件，让 Prior 连续预测未来物块位置。

建议的预测时域

短期	10 步	0.5 秒
中期	20 步	1.0 秒
长期	40 步	2.0 秒

按照当前 20 Hz 控制频率计算。开环评估时不能使用未来真实物块位置或未来真实机械臂状态修正预测。

核心位置误差：

$$ E_H= \frac{1}{N} \sum_i \left\| \hat p_{t+H}^{obj,i} - p_{t+H}^{obj,i} \right\|_2 $$

建议报告均值、中位数和高分位误差，并按未接触、夹持中、已清除物块分别统计。

尤其要与 Constant Velocity、Constant Acceleration、EKF 等模型比较。

如果 RSSM 不能在未见过的 sine 测试轨迹上优于简单预测器，应该先分析预测误差、历史窗口长度和潜变量训练，而不是直接进入 Actor 优化。

Gate 3：任务相关的突破风险预测

这是你的任务最重要的独特性。

World Model 不仅应该知道物块未来在哪里，还应该知道哪个物块最有可能先进入防御圈。

建议增加三个指标：

Breakthrough Recall：未来真正发生突破的场景中，模型提前预警的比例。
Breakthrough Precision / PR-AUC：避免把大量安全物块错误判断为突破威胁。
Threat Ranking Accuracy：多个进攻物块中，模型能否正确判断最紧迫的目标。

注意最终突破判定需要结合物块外轮廓和桌面接触情况，而不是只判断中心距离是否小于防御半径。

另外，突破本身是稀有事件，不能只看总体分类 Accuracy。一个永远预测“不突破”的模型，可能 Accuracy 很高，却对防守任务毫无帮助。

Gate 4：跨速度和轨迹泛化

现有数据全部来自 sine5，因此需要一批新的、独立生成的测试 Episode。

测试数据	用途
未见过 seed 的 sine	同分布泛化
新频率、新相位的 sine	运动模式参数泛化
更慢、更快的 sine	速度范围泛化
直线匀速	跨轨迹类型泛化
分段变速、突然转向	非平稳运动预测
不同接触和抓取行为	动作条件化预测
使用学习型 Actor 采集的轨迹	策略分布变化下的预测可靠性

你现在不需要将所有新类型都加入训练集。建议先将它们作为 OOD 测试，确定 RSSM 的弱点，再针对性扩充 Replay Buffer。

八、这阶段什么时候算真正完成？

我建议不要只用 Loss 收敛作为完成标准。

完成条件应当是：

RSSM 能从 60 维观测与历史动作中稳定更新隐状态。
Prior 能在不读取未来观测的情况下完成多步想象。
能独立预测多个物块的世界位置，并量化开环误差。
能够预测清除、突破等任务相关状态，并评估稀有事件漏报。
相比简单的运动预测 Baseline，明确知道哪些场景表现更好、哪些更差。
保存了可部署的 Encoder、RSSM 参数及归一化统计量。
World Model 可通过标准接口与后续机械臂显式动力学模型同步展开。

如果这些条件满足，就可以进入：

$$ \boxed{ \text{World Model} + \text{Differentiable Arm Dynamics} + \text{Joint Actor-Critic} } $$

我的建议是，这一阶段先只训练 sine5 的 World Model，并把开环预测评估做好。 等你确认 RSSM 真正掌握了物块运动，再扩展运动种类并接入 Actor；这样后续遇到防守失败时，能够清楚判断问题来自环境预测、机械臂控制还是双路径优化，而不是三个模块互相干扰、难以定位。