---
title: MicroDuck任务五：MotrixLab球平衡与FastSAC学习笔记
date: 2026-10-24
tags: [MicroDuck, 具身智能, 强化学习, FastSAC, MotrixLab, WSL, 项目复盘]
status: 完成独立训练、导出和评估；未达到20秒稳定平衡
---

# MicroDuck任务五：MotrixLab球平衡与FastSAC学习笔记

> 阅读目标：不只照着命令运行，还能说清楚“谁在模拟物理、谁在决定动作、谁在更新模型”，并区分训练成功启动、参数确实更新和策略已经学会平衡。
>
> 本文把教程原理、实际操作和本机结果分开记录。教程作者的示例成绩不算作本人的实验成绩。

## 1. 先用一句话理解任务

让一只14关节的MicroDuck站在会自由滚动的球上，通过不断调整腿和身体保持平衡。这次使用 **MotrixLab提供的环境与FastSAC训练器，从随机初始化的网络开始训练**。

这不是自己发明强化学习算法，也不只是播放别人训练好的模型；更准确的定位是：**复现现成任务的训练流程，再评估自己得到的模型。**

### 1.1 与Task01有什么不同

| 对比项 | Task01篮球平衡路线 | Task05 MotrixLab路线 |
| --- | --- | --- |
| 主要框架 | microduck-playground / mjlab | MotrixLab / MotrixSim |
| 训练算法 | PPO | FastSAC |
| 策略结构 | 带记忆的LSTM blind actor | 不带LSTM的MLP actor |
| actor观察 | 61维，主要依靠本体感觉 | 54维，直接读取球相对位置与速度 |
| 价值网络观察 | 按原任务配置 | 63维，包含额外仿真状态 |
| 动作 | 14个关节的控制目标 | 14个关节的控制目标 |
| 能否直接交换权重 | 不能 | 不能 |

“blind”不是机器人没有观察，而是策略不直接读取精确球状态；Task05能读球状态，信息条件不同。即使都能站球，也不能把成绩简单归因于“某算法一定更强”。

### 1.2 画面里的西瓜到底是什么

物体有绿色条纹，像西瓜；源码仍叫`basketball.xml`。实际碰撞物体是**半径0.14 m、质量0.45 kg的规则球体**，条纹属于显示材质。

把贴图换成西瓜，并不等于已经模拟真实西瓜的不规则外形、内部结构、质量和摩擦。

## 2. 小白需要先懂的几个词

| 名词 | 通俗解释 | 本任务中的例子 |
| --- | --- | --- |
| 环境 environment | 机器人行动后，计算世界怎样变化的系统 | 鸭子脚碰到球，球会滚，鸭子可能倾倒 |
| 观察 observation | 当前允许模型看到的一串数字 | 重力方向、关节角、球相对位置 |
| 动作 action | 策略发给环境的控制目标 | 14个关节下一步往哪调 |
| 策略 policy / actor | 把观察转换为动作的神经网络 | 输入54个数，输出14个数 |
| 奖励 reward | 对当前表现打分的规则 | 直立、球在脚下加分；抖动和碰撞扣分 |
| critic / Q网络 | 估计“某状态下采取某动作的后续收益”的网络 | 帮助actor比较动作好坏，不负责真机执行 |
| 回合 episode | 从一次初始化到失败或到达时限的一段经历 | 从站上球开始，到掉下去结束 |
| transition | 一条交互经验 | 当前观察、动作、奖励、下一观察和结束标志 |
| replay buffer | 存放旧经验的数据池 | 从过去的交互中抽样，重复用于学习 |
| collector | 负责与环境交互、收集经验的部分 | 并行控制多只鸭子，记录发生了什么 |
| learner | 抽取经验、计算损失并更新参数的部分 | 调整actor、critic与熵温度 |
| iteration | 代码定义的一轮计数，不是固定长度的一回合 | 本配置主要按collector交互轮数计数 |
| checkpoint | 训练状态存档 | 网络、归一化统计、优化器状态和计数 |
| seed | 控制随机数起点的整数 | 便于对照，但不保证跨设备逐比特相同 |
| ONNX | 用于交换和执行模型计算的格式 | 导出actor计算图，不包含整个物理世界 |

两个“环境”不要混淆：**Python虚拟环境**装依赖；**强化学习环境**模拟机器人与球。前者是软件工具箱，后者是实验世界。

## 3. 从观察到学习，整条流程怎么连起来

```text
物理环境产生观察 → actor输出动作 → 关节执行、球滚动
       ↑                         ↓
       └──── 下一观察、奖励、是否结束 ────┘
                         ↓
                   存入经验回放池
                         ↓
               learner抽样，更新Q网络与actor
                         ↓
                 把新actor发回collector
```

“异步”表示收集经验和更新网络由不同进程协作，不要求每一步都串行等对方完成。collector不是另一个算法，仍然执行同一套策略。

### 3.1 FastSAC与之前PPO的直观区别

- **PPO通常是on-policy**：主要用当前策略新采集的一批数据更新，随后继续采新数据。
- **SAC是off-policy**：保留经验回放池，旧数据可以被反复抽样使用；同时兼顾奖励和策略熵，即探索的多样性。
- 本实现还使用双Q网络（预测价值分布）、目标网络、观察归一化和异步采样等工程设计。“Fast”是具体实现与系统优化，不是任何机器都能几分钟训练好。

只需先理解：不是程序提前写好了“球往左滚就抬哪条腿”，而是网络根据奖励和失败经历逐渐调整动作。初始网络可能很快跌落，这是正常起点，不是模型文件必然损坏。

## 4. 机器人到底看到了什么、控制了什么

### 4.1 54维actor观察

| 观察组 | 维数 | 作用 |
| --- | ---: | --- |
| 投影重力 | 3 | 判断身体倾斜方向 |
| 基座角速度 | 3 | 判断身体旋转趋势 |
| 球相对位置 | 3 | 球在身体坐标系的哪里 |
| 球速度的身体坐标表示 | 3 | 球正往哪里滚、多快 |
| 相对默认姿态的关节位置 | 14 | 当前腿与头部姿态 |
| 关节速度 | 14 | 关节变化速度 |
| 上一动作 | 14 | 刚才发出了什么控制目标 |
| 合计 | **54** | 神经网络的一行输入 |

critic还读取基座线速度、球世界位置和球世界速度，共多9维，合计63维。actor输入会加配置噪声，critic观察不加这些噪声。

注意“球相对速度”的命名容易误解：应以当前源码计算为准；这里的观察是球线速度转到身体坐标系后的表示，不应自行假设已经减去了机器人的平移速度。

### 4.2 14维动作不是直接写入真实关节角

环境按下式构造位置目标：

```text
目标关节角 = 默认站立姿态 + 环境动作 × 0.5
```

actor内部还会把`tanh`输出映射到环境声明的动作范围，环境会裁剪动作；因此不要把网络内部的[-1,1]和环境动作范围、真实关节角混为一谈。底层位置执行器根据PD增益、力矩限制和实际物理响应追踪目标，不能保证关节瞬间到位。

### 4.3 20秒回合与50Hz控制

- 控制步长0.02秒：每秒下达50次动作。
- 物理步长0.005秒：每个控制步包含4个物理步。
- 默认回合上限20秒：最多1000个控制步。

这都是**仿真时间**。现实中跑完20秒仿真需要多久，还取决于计算、并行数量与渲染。视频帧率也不等于控制频率。

## 5. 奖励为什么不只是“站得越久越好”

以下来自本次固定版本配置，权重是不同目标的相对尺度，不是成功率：

| 奖励项 | 权重 | 作用 |
| --- | ---: | --- |
| alive | 1.0 | 存活保底，避免靠快速结束逃避其他惩罚 |
| upright | 4.0 | 鼓励身体直立 |
| ball_under_feet | 3.0 | 鼓励球留在双脚中点下方 |
| base_height | 1.5 | 鼓励基座接近0.40 m目标高度 |
| dof_default | 0.5 | 避免关节过度偏离默认姿态 |
| action_rate_l2 | −0.5 | 惩罚相邻动作变化过大 |
| limits_dof_pos | −5.0 | 惩罚靠近关节限位 |
| undesired_contacts | −0.2 | 惩罚非足端的不期望接触 |

失败终止包括：基座低于0.22 m、严重倾斜、球心距双脚中点水平距离超过0.20 m、关节位置/速度异常等。

必须区分：

- `terminated`：失败条件触发。
- `truncated`：到达20秒时限，本身不算失败。
- `reset`：重新初始化。画面一直有鸭子，不代表同一只鸭子连续成功站住，可能一直在失败后自动重置。

当前任务随机化初态和观察噪声，**不随机化质量、惯量、摩擦或执行器增益等物理参数**。因此仿真结果不能直接证明真实机器人也能成功。

## 6. 本次软件版本与目录安排

| 项目 | 本次实际设置 |
| --- | --- |
| 系统与显卡 | WSL2 Ubuntu 24.04，RTX 5070 Ti 16 GB |
| 固定源码commit | `24796cff1c00591bd327378df93461e1c4a9f89c` |
| MotrixLab / MotrixSim | 0.4.0 / 0.10.1 |
| Python | 3.10.22，独立`venv-task05` |
| PyTorch / CUDA运行库 | 2.7.0+cu128 / 12.8 |
| NumPy / Numba | 2.2.6 / 0.61.2 |
| ONNX / ONNX Runtime | 1.20.1 / 1.23.2 |
| CUDA验证 | 实际GPU矩阵乘法通过，结果为有限值 |
| 执行方式 | 异步collector＋learner；关闭torch.compile，使用主机内存传输 |

日志明确记录`collector_env=cpu`、`collector_inference=cuda`、`learner=cuda`：本次环境步进在CPU，策略推理和网络更新在GPU。不能笼统写成“整个物理仿真都在GPU上运行”。

沿用Windows＋WSL2 Ubuntu的电脑，不为Task05重装系统；**新建独立虚拟环境**，不修改Task01～04的依赖。Python环境在WSL文件系统里，项目与实验产物在项目目录下；本机WSL虚拟磁盘此前已迁移到数据盘。

```text
项目父目录/
├── microduck-playground-official/   # Task01，保留不动
├── mjswan_playground/               # Task02，保留不动
├── microduck-stilts/                # Task03/04，保留不动
├── MotrixLab/                      # 本次Task05官方源码与训练run
└── outputs/task05_20261024/         # 本次审计记录、导出文件和汇总
```

### 6.1 为什么教程命令与当前仓库不完全一样

教程使用过`python scripts/train.py`和`python scripts/play.py`；本次固定的上游版本已经提供统一入口：

```bash
mxlab train ...
mxlab play ...
mxlab export-onnx ...
```

不要把旧教程命令、最新配置文件和另一个版本的权重随意拼在一起。本次记录源码commit和`uv.lock`，再运行该版本实际提供的命令。

## 7. 安装：在哪个窗口输入什么

### 7.1 Windows侧：获取源码和模型资产

在PowerShell进入计划存放项目的父目录，再执行：

```powershell
git clone https://github.com/Motphys/MotrixLab.git
cd MotrixLab
git checkout 24796cff1c00591bd327378df93461e1c4a9f89c
git lfs pull --include="motrix_robots/src/motrix_robots/assets/microduck/**,motrix_envs/src/motrix_envs/locomotion/ball_balance/**,motrix_envs/src/motrix_envs/common/motphys-ground.png"
```

逐句解释：

1. `git clone`把源码仓库下载到本地，同时创建`MotrixLab`目录。
2. `cd`只改变当前所在文件夹，不会安装软件。
3. `git checkout <commit>`切到本文核验的代码快照；新下载、无修改的仓库才适合直接照做。
4. `git lfs pull`下载网格、纹理等大文件的真实内容；`--include`只取本任务相关资源，减少无关下载。

若`git lfs version`不可用，需要先安装Git LFS。LFS指针常常只有一百多字节，内容以`version https://git-lfs.github.com/spec/v1`开头；它不是可加载的STL模型。以后运行其他任务，还需要补取对应资产，本文选择性下载不代表全仓资源齐备。

本次实际是在Windows用Git LFS下载资产，在Ubuntu里运行Python，两边访问同一份项目文件。不需要在两个系统重复克隆。

### 7.2 打开Ubuntu，并找到同一个目录

在PowerShell输入：

```powershell
wsl -d Ubuntu-24.04
```

进入Ubuntu后，Windows的数据盘通常对应`/mnt/<盘符小写>/`。用`cd`进入刚才的`MotrixLab`目录，确认：

```bash
pwd
ls pyproject.toml uv.lock configs/task/microduck-ball-balance/motrix.fastsac.yaml
```

`pwd`显示当前位置；`ls`列出目标文件。先看见这几个文件，再进行下一步，避免在系统目录误执行安装命令。

### 7.3 在Ubuntu创建独立CUDA环境

以下均在Ubuntu的MotrixLab仓库根目录执行：

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.local/share/microduck/venv-task05"
uv sync --frozen --no-default-groups --extra cuda --extra tbb --python 3.10
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
```

参数解释：

| 内容 | 意思 |
| --- | --- |
| `export` | 设置当前终端及其子进程使用的变量，不会自动永久写进系统配置 |
| `UV_PROJECT_ENVIRONMENT` | 指定依赖安装目录，避免覆盖此前任务环境 |
| `uv sync` | 根据项目声明安装/同步依赖，不是训练 |
| `--frozen` | 使用现有锁文件，不重新解析生成另一个版本组合 |
| `--no-default-groups` | 不安装默认开发工具组，先满足运行需要 |
| `--extra cuda` | 选择项目声明的CUDA版PyTorch依赖 |
| `--extra tbb` | 安装本版本在Linux推荐的Numba线程支持依赖 |
| `--python 3.10` | 选择项目默认的Python3.10系列；不必改动系统Python |
| `source .../activate` | 让当前终端的`python`、`mxlab`指向本任务环境 |

首次下载Python、PyTorch及CUDA依赖可能很大、很慢。CUDA wheel带有用户态运行库，能否调用GPU仍取决于Windows驱动和WSL支持。**不要在WSL里盲目再装一套Linux显卡驱动。**

关闭终端后再次运行，要重新设置变量并激活；也可直接使用`$HOME/.local/share/microduck/venv-task05/bin/python`或该目录下的`mxlab`。

### 7.4 检查安装，不要只看“CUDA可用”一行

```bash
which python
which mxlab
mxlab --version
python -c 'import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); x=torch.randn(128,128,device="cuda"); print((x@x).mean().item())'
```

前两句确认使用哪个环境；最后一句实际创建GPU张量并做矩阵乘法。只有导入成功但计算报错，仍不能开始训练。

注意不要随手执行不带CUDA extra的裸`uv sync`/`uv run`，它们可能按另一套依赖选择重新同步。环境建好后直接用已激活的`mxlab`；需要uv时可用`uv run --no-sync ...`。

## 8. 实际运行：先验证，再扩大

### 8.1 先用相关单元测试检查网络与经验池

```bash
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" pytest==8.3.5
python -m pytest \
  motrix_rl_builtin/tests/test_fastsac_networks.py \
  motrix_rl_builtin/tests/test_fastsac_buffer.py -q
```

实际结果：**12 passed in 2.90s**。这里安装pytest只增加测试工具，不替换此前选好的CUDA训练依赖。

测试通过只说明所覆盖的软件行为正常，不证明机器人已经学会平衡，也不等于整个框架的全部测试通过。

### 8.2 教程的5轮smoke为什么需要仔细理解

当前配置为`learning_starts=10`：collector先收集经验，之后learner才开始更新。**仅运行5轮，生成checkpoint也可能还没有发生梯度更新。**这比PPO任务中“5轮就会更新”的直觉更容易误导初学者。

本次改用32轮小规模检查，保留原来的10轮预热；最终确实有参数更新，但只发生4次Q更新，不能机械认定“32轮就必然做完32×4次更新”，短任务的异步启动与收尾也影响实际学习量：

```bash
export OMP_NUM_THREADS=4
export NUMBA_NUM_THREADS=4

mxlab train task=microduck-ball-balance/motrix.fastsac \
  num_envs=64 play_num_envs=4 seed=1 \
  algo.trainer.num_learning_iterations=32 \
  algo.agent.compile=false \
  algo.trainer.async_options.collector_compile=false \
  'algo.trainer.async_options.transition_ipc="off"' \
  'algo.trainer.async_options.weight_ipc="off"' \
  logging.interval=10 checkpoint.interval=500 play=false
```

逐项理解：

- `mxlab`是安装项目后提供的命令入口，`train`选择训练模式。
- `task=.../motrix.fastsac`加载球平衡任务与FastSAC的组合配置。
- `num_envs=64`同时收集64份独立仿真经验，不是训练64个互不相干的模型。
- `seed=1`固定这次训练的随机种子。
- `num_learning_iterations=32`设置收集迭代预算，不是32个完整回合。
- `compile=false`关闭PyTorch编译加速，不是关闭CUDA，也不是禁止学习。
- 两个编译开关分别对应learner和collector；本次为减少首次编译成本而关闭。
- 两个`ipc="off"`选择主机内存传输，规避本机WSL的CUDA IPC报错；不是把学习改成CPU。引号要保留，防止Hydra把`off`当成布尔值而不是字符串。
- 两个线程变量限制CPU线程数量，避免不同进程过量争抢CPU；它们也是本次运行设置的一部分，不是通用最优参数。
- `logging.interval=10`设置日志间隔；`checkpoint.interval=500`设置中途存档间隔，正常结束仍会写最终存档。
- `play=false`表示训练结束不自动打开交互窗口，之后单独回放。

Hydra使用`键=值`，不是所有参数前都加`--`。Linux的反斜线`\`表示命令接下一行，必须位于行尾；PowerShell的续行符不同，不能把两种写法混着用。

### 8.3 本次扩大预算的学习运行

本次另外做了两组**独立随机初始化训练**：512环境×2000迭代、512环境×10000迭代；不是在32轮checkpoint上直接续训。2000与10000轮都使用训练seed 1、相同任务配方和兼容开关。

以10000轮为例：

```bash
mxlab train task=microduck-ball-balance/motrix.fastsac \
  num_envs=512 play_num_envs=4 seed=1 \
  algo.trainer.num_learning_iterations=10000 \
  algo.agent.compile=false \
  algo.trainer.async_options.collector_compile=false \
  'algo.trainer.async_options.transition_ipc="off"' \
  'algo.trainer.async_options.weight_ipc="off"' \
  logging.interval=10 checkpoint.interval=500 play=false
```

把`10000`改为`2000`就是中等预算运行。理论交互预算分别为512×2000＝1,024,000条、512×10000＝5,120,000条；最终以实际run与计数核验为准。训练过程中保存的周期checkpoint和最终`latest.pt`都会占空间，不需要把这些大文件推送到笔记仓库。

增加预算前先确认小规模运行没有报错。默认教程任务是2048环境、20000迭代；本次具体预算与结果见第10节，不能把减少预算后的短训称为完整收敛复现。

### 8.4 怎么知道真在训练，而不是只有文件

同时检查：

1. 进程正常结束，而不是异常退出前留下半成品。
2. checkpoint包含actor、Q网络、归一化统计、优化器状态与计数。
3. `update_idx`及优化器`step`大于0，证明已发生优化。
4. 权重和日志中的损失没有NaN/Inf。
5. 在独立评估中观察效果，而不只看训练控制台的return。

`return`是该回合累积奖励；改了奖励权重后分数变高，不能直接说平衡能力更好。训练期间有探索噪声，独立评估通常使用确定性动作，两者成绩也不应直接混作同一指标。

## 9. 保存、回放、导出和评估

### 9.1 结果目录为什么不能只留下一个pt文件

```text
runs/microduck-ball-balance/motrix/torch/fastsac/<本次run>/
├── metadata.json       # 任务、框架、种子等元信息
├── task_config.yaml    # 当次最终配置
├── checkpoints/
│   └── latest.pt       # 最终训练存档
└── ...                 # 日志、TensorBoard及其他产物
```

回放程序需要元信息与配置才能重建匹配的环境和网络。只把`latest.pt`挪到一个空文件夹，可能会找不到`metadata.json`。

### 9.2 指定模型回放，避免误选“最新一次”

```bash
mxlab play env=microduck-ball-balance \
  policy=<本次run目录>/checkpoints/latest.pt num_envs=1 seed=101 \
  +rl.agent.compile=false
```

`<本次run目录>`是说明用占位符，必须替换为实际训练目录，不能原样复制。省略`policy`时程序会自动寻找元信息完整的run，可能不是希望比较的那一个。

录制视频：

```bash
mxlab play env=microduck-ball-balance \
  policy=<本次run目录>/checkpoints/latest.pt num_envs=1 seed=101 \
  record_video=true record_seconds=10 record_width=720 record_height=720 \
  +rl.agent.compile=false +rl.agent.amp=false
```

本版本目标文件为run目录下的`play_video.mp4`。录像可能包含自动reset，不能仅凭“文件长10秒”就声称“连续站稳10秒”。

上面是完整录像的参考写法。本机最终模型实际录制的是**3秒、480×480、60 FPS**短片：把命令中的`record_seconds`改为`3`，宽高改为`480`即可。本机WSL使用软件渲染，短片耗时约92秒；缩短录像不影响第9.4节的20秒上限评估。

这里`rl`是用于覆盖存档算法配置的空映射，Hydra首次追加其内部键时要写`+rl.agent...`。如果看到`Key 'agent' is not in struct`，先检查这个`+`，不要误判成checkpoint损坏。

### 9.3 导出自己的ONNX

```bash
mxlab export-onnx run_dir=<本次run目录> output=<输出目录>/task05_policy.onnx
```

使用框架导出器，保留观察归一化及动作变换，不手工只导actor权重。本版本导出器会执行数值一致性检查；本文另外在仿真真实观察上做对照，检查输入输出维度和动作误差。

导出成功不等于已经把模型部署到开发板，也不代表在ONNX闭环里评估过成功率。Task01的61维模型和本任务54维模型不能互换。

### 9.4 这次怎样评估效果

采用独立评估seed 101、202、303，每个seed并行8个环境，共24个**首回合**。每个环境只统计第一次终止或到达20秒上限，不把自动reset后的新回合接在一起冒充连续成功。

记录首回合持续时间、结束类型、是否有非有限数值，并在相同设置下比较不同训练预算。24回合只是这次初步评估，不是大规模泛化基准，也不是24个独立训练种子。

## 10. 本机实际完成了什么

### 10.1 三组模型的真实对照

| 指标 | 小规模检查 | 中等预算 | 扩大预算 |
| --- | ---: | ---: | ---: |
| 并行环境数 | 64 | 512 | 512 |
| collector迭代数 | 32 | 2000 | 10000 |
| 理论采集transition数 | 2,048 | 1,024,000 | 5,120,000 |
| checkpoint的Q更新计数 | 4 | 7,952 | 39,956 |
| actor优化器step | 1 | 1,988 | 9,989 |
| 独立首回合平均时长 | 0.447秒 | 1.024秒 | **3.473秒** |
| 独立首回合最长时长 | 0.78秒 | 1.42秒 | **6.96秒** |
| 撑满20秒的回合 | 0/24 | 0/24 | **0/24** |
| 权重、所记录训练标量、评估观察/动作/奖励有限值检查 | 通过 | 通过 | 通过 |

结论：这次不仅跑通了程序，还确实更新了自己的模型，且较大预算模型在这批评估中比短训模型持续更久。但**最终仍未达到20秒稳定球平衡，不能写成“已经收敛”**。

2000与10000轮组环境数量相同，便于观察预算差异；32轮组还改变了环境数量，只是流程检查，不是严格单变量消融。每种预算仅训练一次，评估seed不同不等于做过多训练seed复现。

### 10.2 为什么训练日志看起来比评估好

10000轮结束时，训练日志最近窗口的平均return约119.42，平均回合长度约822.95步＝16.46秒；独立评估的首回合平均却只有3.47秒。

这两个数字来自不同随机状态、采样方式和统计窗口。训练collector使用带探索的策略，本文独立评估使用确定性动作、指定seed与首回合统计；不能把训练窗口均值当成独立测试成绩。**本次没有单独做实验归因这个差距，不能直接断言是过拟合或某一个配置错误。**

也不要因为看见训练窗口变好就停止检查：真正准备声明“学会任务”时，应扩大独立评估，并分别核验随机策略与确定性策略的表现、初始状态和失败原因。

### 10.3 ONNX数值验证

| 模型 | 官方导出检查的最大绝对误差（32样本） | 真实仿真观察样本数 | 真实观察上的最大动作误差 |
| --- | ---: | ---: | ---: |
| 32轮 | 1.04×10⁻⁷ | 776 | 2.24×10⁻⁷ |
| 2000轮 | 1.43×10⁻⁶ | 1496 | 2.15×10⁻⁶ |
| 10000轮 | 2.15×10⁻⁶ | 1536 | 2.38×10⁻⁶ |

三组均通过导出检查，接口为`obs: [batch,54] → actions: [batch,14]`。`batch`是一次输入几行观察，不是观察维数变了。

真实观察检查取自PyTorch策略驱动的仿真轨迹，分别送进PyTorch与ONNX比较动作；**没有把ONNX单独接入闭环重测上述24回合**，因此这里只证明计算一致性，不声称已经完成ONNX闭环成功率评估。

### 10.4 本地产物位置

```text
MotrixLab/runs/microduck-ball-balance/motrix/torch/fastsac/
├── 26-10-24_22-47-46-876527/   # 64环境×32轮
├── 26-10-24_22-49-13-928321/   # 512环境×2000轮
└── 26-10-24_22-52-23-992935/   # 512环境×10000轮
    ├── metadata.json
    ├── task_config.yaml
    ├── checkpoints/latest.pt
    └── play_video.mp4         # 最终核验的3秒短片
```

本机另外在`outputs/task05_20261024/`保存三组ONNX、首回合审计JSON、真实观察对照样本与汇总图。失败尝试的日志也保留在本地，没有把含机器路径的原始日志上传到公开笔记仓库。

最终10000轮checkpoint的SHA-256为`3acc2b8a7758041419cf7b69b213af0b99183f724d00f635603ccad4f4f2669c`，用于区分模型版本，不是访问密码。

![本机FastSAC训练曲线与独立首回合评估](Task05实验图表/task05_learning_results.png)

读图方法：上排是训练过程统计，不是测试集成功率；左下每组对应24个首回合，箱线图显示分布、三角形表示均值；右下的20秒虚线是回合时间上限，三组均未达到该标准。

![最终模型本机录像中段抽帧](Task05实验图表/task05_replay_frame.png)

这张图来自10000轮模型、seed 101的本地3秒录像约1.5秒处，已核验机器人、球与地面正常显示。它只证明画面内容，不证明连续稳定20秒。画面使用MotrixLab与Pollen Robotics MicroDuck资产，资产许可仍以各上游项目为准；未再分发模型网格或权重。

教程中约5000迭代、return +68.71等记录属于教程作者的实验，不是本机结果。本文只引用本次审计得到的数值，不以官方海报或示例视频替代自己的训练成果。

## 11. 踩坑与排查顺序

| 现象 | 先检查什么 | 处理原则 |
| --- | --- | --- |
| 提示不在Git仓库、找不到配置文件 | `pwd`与`ls pyproject.toml` | 先进入正确的MotrixLab目录 |
| 旧的`train.py`不存在 | 固定commit、`mxlab --version` | 用匹配版本入口，不从别处随意拷脚本 |
| 模型文件格式错误 | 是否仍是百字节LFS指针 | 下载真实资产，再重试 |
| collector说CUDA不可用 | 当前python路径、CUDA extra、GPU张量运算 | 激活正确环境，不只看系统装没装CUDA Toolkit |
| Windows报告`TritonMissing` | 是否开启torch.compile | 教程Windows兼容路线关闭learner、collector编译；本次实际用Ubuntu |
| 安装后又变CPU版依赖 | 是否运行裸`uv sync`/`uv run` | 保持CUDA extra选择，或激活后直接调用mxlab |
| 只有5轮就保存了模型 | `learning_starts`、`update_idx`、优化器step | checkpoint存在不等于参数已更新 |
| 训练开始很慢 | 首次下载、Numba编译、进程预热 | 与死锁区分，结合日志和进程状态检查，不反复重开训练 |
| 录完视频但鸭子一直跌落 | 训练预算、失败reset、首回合时长 | 如实记录未收敛，不把能渲染解释成任务成功 |
| 回放找不到元信息 | metadata、task_config与checkpoint相对位置 | 保留完整run目录结构 |
| 奖励涨了但看起来更抖 | 奖励分项与独立评估 | 不能只比较一个总分 |

### 本次实际遇到并处理的问题

1. **依赖下载过慢。**代理下载大型CUDA依赖明显较慢，确认PyPI及PyTorch下载服务器直连可用后，只在安装进程里设置`NO_PROXY`例外，保持锁文件不变完成安装；没有修改系统全局代理。网络策略依机器而异，不建议一遇到慢就关闭所有代理。
2. **WSL CUDA IPC失败。**最初默认传输方式在`rebuild_cuda_tensor`时报`invalid resource handle`。改用`transition_ipc="off"`与`weight_ipc="off"`后正常训练；CPU负责主机内存传输，GPU仍执行神经网络计算。
3. **Hydra追加键规则。**回放覆盖项直接写`rl.agent.compile=false`报缺键；改用`+rl.agent.compile=false`。训练配置已经声明的键仍用普通`键=值`。
4. **遗漏公共地面纹理。**只下载机器人和球目录不足以保证完整渲染；补取`common/motphys-ground.png`后地面正常显示。本文第7节已把这个路径补进推荐命令。
5. **软件渲染慢、首帧可能还在加载。**WSL的渲染器报告软件适配器。2000轮模型的10秒720p录像等待较久，主动中断后保留约9.27秒片段，不算完整10秒录像；最终10000轮模型改录3秒480p，正常退出，180帧，H.264。首帧与中段都应检查，不能仅看文件存在。
6. **训练成绩高于独立评估。**保留两套结果，不挑好看的训练窗口替代独立测试。这是本次最重要的结果边界。

ONNX Runtime曾警告无法读取WSL的`/sys/class/drm`显卡信息，但本次ONNX检查使用CPU执行器，验证正常通过；这条警告不等于PyTorch的CUDA训练不可用。

## 12. 我现在能说什么，不能说什么

完成训练流程验证后可以说：使用开源任务与FastSAC，建立独立环境，实际进行采样和参数更新，保存自己的模型，并完成回放、导出或评估中已核验的环节。

不能据此说：自主研发FastSAC算法、已经优于PPO、完成真实机器人部署，或只凭短视频就达到稳定平衡。没有做过的环节应明确写“未完成”，不能用教程附带产物补齐。

若继续提升项目含金量：先取得可重复的训练基线，再只改变一个因素，例如摩擦或某个奖励权重；保持训练预算与评估条件一致，比较平衡时长、失败类型和动作变化。更换框架、观察条件、奖励、训练预算后一次性比较，无法判断到底是什么起作用。

## 13. 学习心得与课程反馈

这次真正需要理解的是“学习”发生在哪里：采样只是积累经历，优化器更新才改变网络，而独立评估才检验控制效果。模型能保存、视频能播放和策略能稳定站球，是三个不同层次。

课程优点是把同一个球平衡问题放在PPO与FastSAC两条路线下，有助于理解观察设计、经验复用与训练工程。建议补充固定版本、统一CLI迁移说明、learning_starts预热机制，以及“自动reset的视频不能证明连续稳定”的提示。

## 14. 隐私与公开范围

公开前对本笔记与原笔记目录的Markdown做了模式扫描，并人工核对正文、命令和配图：未发现个人主目录、机器名、个人邮箱、内网地址、访问令牌、密码或私钥。示例使用相对目录与`$HOME`变量，不公开本机账号。

上传范围仅为学习Markdown和两张实验配图；不上传虚拟环境、checkpoint、原始终端日志、SSH文件、代理配置或浏览器截图。Git新提交使用GitHub隐私邮箱，避免把本机配置的个人邮箱写入这次提交。规则扫描只能辅助检查，不能当作绝对无泄漏的保证；已有历史提交不在本次改写范围内。

## 15. 来源

- [Every Embodied Task05教程](https://github.com/datawhalechina/every-embodied/tree/a04dcfa31cf11a1a1115628ba107d9de0af38d52/16-专题组队学习/05-MicroDuck运动策略与任务开发专题/01-任务资料/05-MotrixLab-MicroDuck球平衡与FastSAC)。
- [本次固定MotrixLab源码](https://github.com/Motphys/MotrixLab/tree/24796cff1c00591bd327378df93461e1c4a9f89c)。
- [固定版本球平衡配置](https://github.com/Motphys/MotrixLab/blob/24796cff1c00591bd327378df93461e1c4a9f89c/motrix_envs/src/motrix_envs/locomotion/ball_balance/microduck.py)。
- [固定版本FastSAC任务配方](https://github.com/Motphys/MotrixLab/blob/24796cff1c00591bd327378df93461e1c4a9f89c/configs/task/microduck-ball-balance/motrix.fastsac.yaml)。
- [官方球平衡文档](https://motrixlab.readthedocs.io/zh-cn/latest/user_guide/envs/ball_balance.html)（在线文档会更新，复现时以固定源码为准）。
