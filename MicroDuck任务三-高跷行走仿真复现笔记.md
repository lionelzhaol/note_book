---
title: MicroDuck任务三：高跷行走仿真复现笔记
date: 2026-10-18
tags: [MicroDuck, 具身智能, 强化学习, MuJoCo, 高跷行走, 课程学习, 项目复盘]
status: 两种高度的策略回放与ONNX导出已核验
---

# MicroDuck任务三：高跷行走仿真复现笔记

> 记录范围：截至 2026-10-18，学习者已反馈完成教程操作；本笔记依据本地源码、模型 manifest、导出文件和视频整理可核验成果。未找到的训练记录不补写指标。

## 1. 做成了什么

在 WSL2 Ubuntu 中，使用固定版本的 MicroDuck 高跷任务，分别载入 **25 cm 与 200 cm 的公开策略和匹配形态**，生成本地 MuJoCo 视频并重新导出 ONNX。

与 [[MicroDuck任务一-篮球平衡仿真复现笔记]]、[[MicroDuck任务二-浏览器物理扰动复现笔记]] 相比，本任务重点是理解：**支撑几何、质量、惯量改变后，策略必须与机器人形态匹配；课程学习通过逐步增加难度训练步态。**

本次视频由策略读取仿真状态、输出关节动作，再经 MuJoCo 计算接触和运动生成，不是播放上游演示。

## 2. 环境与存放位置

| 项目 | 配置 |
| --- | --- |
| 系统 | 沿用 WSL2 Ubuntu 24.04，虚拟磁盘位于 E 盘 |
| GPU / Python | 沿用已验证的 NVIDIA CUDA 环境；Python 3.12 |
| 独立项目 | `<项目目录>/microduck/microduck-stilts` |
| 虚拟环境 | `~/.local/share/microduck/venv-stilts`，与前两个任务隔离 |
| 依赖配置 | 固定版本要求 mjlab 1.3.0、warp-lang 1.12.0、torch 2.9.1 |
| 任务 ID | `Mjlab-Stilt-Flat-MicroDuck` |
| 源码提交 | `c5fcc50219fef01ac9931d0079c583ccbb29b689`，本地已核对 |
| 模型来源 | `HannesVonEssen/microduck-stilts` 的 `25cm/` 与 `200cm/` |

## 3. 每一步做了什么

1. **准备独立源码。**建立 `microduck-stilts` 仓库，拉取教程指定提交，通过 sparse checkout 获取 `scripts/`、`src/` 和根目录配置。
2. **配置环境。**使用独立 `venv-stilts`，按锁文件安装依赖；设置 `MUJOCO_GL=egl` 进行离屏视频渲染。
3. **下载模型。**分别下载两种高度的 `policy.onnx`、`checkpoint.pt`、`manifest.json` 及共享配置，避免误用根目录默认模型。
4. **核对模型与形态。**25 cm 对应 25 cm 形态，200 cm 对应 200 cm 形态；两者均使用 `blend=0.5`。
5. **执行回放。**使用 `scripts/export.py` 加载对应 checkpoint，在仿真中执行策略并录制视频。
6. **重新导出 ONNX。**保留下载的原始策略，另存 `artifacts/` 中的本地导出模型。
7. **整理结果。**核对视频帧数、时长、分辨率及模型哈希；结合两种高度学习交替支撑、接触和形态课程。

注意：`export.py` 的作用是**回放与导出**，不会因为运行它而发生 PPO 参数更新。

## 4. 踩坑与关键注意点

| 问题 / 易错点 | 处理与理解 |
| --- | --- |
| 教程同时给 PowerShell 与 Linux 写法 | 当前在 WSL Ubuntu 中，应选 Linux/macOS shell 写法；环境变量使用 `export`，多行命令用反斜杠 `\` 连接 |
| Windows 电脑是否必须用 CPU | 不必。运行位置是 Ubuntu，CUDA 已验证可用，可以使用 `--device cuda:0`；教程的 `--device cpu` 是另一条可用路径 |
| 高度变了却没换模型 | 高度、质量、接触和策略相互绑定；25 cm 模型不能直接当作 200 cm 模型使用 |
| 混用前两个任务的环境 | 重新设置 `UV_PROJECT_ENVIRONMENT`，避免 uv 使用 `venv-mjswan` 或 `venv-official` |
| 导出视频被覆盖 | 同一 checkpoint 再次导出时，视频可能仍叫 `rl-video-step-0.mp4`；当前 25 cm 文件实际为 4 秒，不能按此前计划写成 10 秒 |
| 把训练迭代号当作本次训练量 | manifest 的 2800、6500 是发布模型记载的迭代，不代表自己训练了这些轮次 |
| 把官方评测当作本机结果 | manifest 中的“10 秒、无重置”属于发布方记录；本机没有逐步终止报告时，不能据此声称本次零摔倒或多种子成功率 |

其中 shell 写法是本次实际确认的问题，其余是运行与整理结果时需要保留的边界，不表示都发生过报错。

## 5. 关键命令与原理

在 Ubuntu 的 `microduck-stilts` 仓库目录使用：

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.local/share/microduck/venv-stilts"
export MUJOCO_GL=egl
export MICRODUCK_STILT_HEIGHT_CM=25
export MICRODUCK_STILT_BLEND=0.5

uv run python scripts/export.py Mjlab-Stilt-Flat-MicroDuck \
  --checkpoint-file policies/stilts/25cm/checkpoint.pt \
  --device cuda:0 --num-envs 1 \
  --video True --video-length 200 \
  --video-height 720 --video-width 1280 \
  --running-speed 0.15 \
  --onnx-file artifacts/stilts_25cm.onnx
```

这是教程回放命令的整理，不是本次终端历史的逐字恢复。2 m 回放需要同时将高度改为 `200`、模型目录改为 `200cm`、视频长度改为 `500`，并另存导出文件。

### 为什么高跷不只是“把腿拉长”

高跷作为新的刚体和接触几何进入 MuJoCo，带有质量、惯量、碰撞面及足端位置；原鞋底的地面碰撞相应关闭。策略的同一个关节动作，在不同高跷形态下会产生不同物理后果。

- 策略接口仍为 **61 维观测 → 14 维关节动作**，控制周期 20 ms，目标 50 Hz。
- 动作是围绕默认姿态的关节位置偏移，不是直接指定机器人在画面中的坐标。
- 两种公开策略的观测归一化均已包含在 ONNX 中。
- `blend=0.5` 的末端约为 17 × 22 mm，控制支撑形状；高度则由独立参数指定。
- 仿真每根高跷质量为 `0.012 + 0.001 × 高度(cm)` kg：25 cm 约 37 g，200 cm 约 212 g。不是实际打印件称重。

### 课程学习解决什么问题

先在低高度、较宽支撑上学习稳定行走，再逐步收窄支撑、增加高度，并在阶段间继承策略。这样比让随机策略直接探索 2 m 高跷更容易获得有效奖励。

PPO 根据采样轨迹与奖励更新 actor；critic 辅助估计长期回报。部署只运行 actor，回放本身没有 PPO 更新。

## 6. 最终成果与证据

### 本机视频（按当前文件实测）

| 项目 | 25 cm | 200 cm |
| --- | --- | --- |
| 支撑 blend / 单根名义质量 | 0.5 / 37 g | 0.5 / 212 g |
| 视频时长 | **4.00 秒** | **10.00 秒** |
| 编码帧数 / 帧率 | 200 / 50 FPS | 500 / 50 FPS |
| 分辨率 / 编码 | 1280 × 720 / H.264 | 1280 × 720 / H.264 |
| 下载 ONNX 哈希对照 | 与对应 manifest 一致 | 与对应 manifest 一致 |
| 下载 checkpoint 哈希对照 | 与对应 manifest 一致 | 与对应 manifest 一致 |

视频帧率是文件编码参数，不等于已测得完整闭环的实时计算速度。本次没有像任务一那样的终止计数 JSON，因此不补造零摔倒或重置率。

本机产物（相对 `microduck-stilts/`）：

```text
policies/stilts/25cm/videos/play/rl-video-step-0.mp4
policies/stilts/200cm/videos/play/rl-video-step-0.mp4
artifacts/stilts_25cm.onnx
artifacts/stilts_25cm_reexport.onnx
artifacts/stilts_200cm_reexport.onnx
```

这组对照使用两套各自匹配的公开策略，比较的是两个“模型 + 形态”系统，不能说同一个策略已经学会自动适应全部高度。

## 7. 训练记录与成果边界

学习者已反馈教程内容完成。当前本地 `microduck-stilts` 目录尚未找到新 `model_*.pt`、TensorBoard 事件或训练参数日志，因此本笔记**只把回放、导出和视频列为已核验成果**，不推断训练失败，也不补写完成轮数或提升幅度。若训练产物保存在其他目录，可后续补录路径和指标。

公开 checkpoint 的 actor 与发布策略匹配，但原始 critic/优化器完整快照未保留；发布包提供新建的兼容 critic、清空的优化器历史，属于 **actor-exact warm start**。继续训练不等于完整恢复作者的原始 PPO 现场。

本次可准确表述为：**完成固定源码下的 25 cm / 2 m 高跷策略本地仿真复现、视频生成、模型哈希核对和 ONNX 重新导出，并理解形态课程的作用。**

这些模型标记为 simulation-only。没有进行真实机器人、开发板或打印高跷的实机验证。

## 8. 学习体会与来源

这次认识到，机器人策略不能脱离物理形态单独理解。高跷高度、支撑端和质量变化会影响接触与平衡；看似相同的模型接口，不代表策略可以任意互换。复现时既要看视频，也要核对形态参数、模型版本和实际产物。

- [官方高跷模型与接口](https://huggingface.co/HannesVonEssen/microduck-stilts)。
- [固定版本源码](https://github.com/Vottivott/microduck-playground/tree/c5fcc50219fef01ac9931d0079c583ccbb29b689)。
- [Every Embodied Task 03 教程](https://github.com/datawhalechina/every-embodied/blob/a04dcfa31cf11a1a1115628ba107d9de0af38d52/16-专题组队学习/05-MicroDuck运动策略与任务开发专题/01-任务资料/03-MicroDuck高跷行走强化学习复现/README.md)。
