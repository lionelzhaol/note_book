---
title: MicroDuck任务一：篮球平衡仿真复现笔记
date: 2026-10-15
tags: [MicroDuck, 具身智能, 强化学习, MuJoCo, WSL2, 项目复盘]
status: 预训练模型复现完成，尚未续训
---

# MicroDuck任务一：篮球平衡仿真复现笔记

> 记录范围：截至 2026-10-15，完成开源篮球平衡策略的本地仿真复现、模型一致性检查与视频录制。依据操作记录和本机 `basketball_baseline.json` 整理。

## 1. 做成了什么

在本地 NVIDIA GPU 上运行公开预训练策略，让 MicroDuck 在 MuJoCo 仿真中踩球保持平衡，生成 **30 秒新视频，1500 个控制步，0 次终止或超时重置**。

视频来自本机实际执行的“读取观测 → 策略推理 → 关节动作 → 物理更新”闭环，并非播放仓库现成视频。技能来自开源模型；本次没有训练新模型，也没有连接真实机器人。

## 2. 环境与存放位置

| 项目 | 最终状态 |
| --- | --- |
| 系统 | Windows + WSL2 Ubuntu 24.04，无需双系统 |
| GPU | NVIDIA GPU，16 GB 显存 |
| 训练框架 | Python 3.12；PyTorch `2.9.1+cu128`，CUDA 实际计算通过 |
| 仿真依赖 | 项目锁定的 mjlab 1.3.0、warp-lang 1.12.0；使用 EGL 渲染 |
| Ubuntu 磁盘 | `<WSL存储目录>\Ubuntu-24.04\ext4.vhdx`，已从系统盘迁移 |
| 项目目录 | Windows：`<项目目录>\microduck`；Ubuntu：`~/microduck` |
| Python 虚拟环境 | `~/.local/share/microduck/venv-official`，随 Ubuntu 虚拟磁盘存放 |

WSL 的 Windows 组件仍有少量内容在系统盘。终端初始目录位于 Windows 系统目录时，只代表当前位置，不代表工具安装在那里。

## 3. 每一步做了什么

1. **限定任务。**只做任务一篮球平衡，采用预训练策略复现路线，不要求 RDK X5、HBM 编译或实机。
2. **准备资料和源码。**下载 Every Embodied 的 MicroDuck 专题；另获取官方 `microduck-playground` 的 `scripts/`、`src/` 与根目录依赖文件。教材提供配套视频工具，官方源码提供篮球任务与模型接口。
3. **安装 WSL2。**执行 `wsl --install -d Ubuntu-24.04`，重启 Windows，创建 Ubuntu 用户；运行 `nvidia-smi` 确认能识别显卡。
4. **安装基础工具。**通过 apt 安装 Git、curl、编译和图形库、ffmpeg；安装 uv 与 Python 3.12。
5. **迁移 Ubuntu。**停止 WSL 后用 `wsl --manage Ubuntu-24.04 --move <WSL存储目录>\Ubuntu-24.04` 迁移，原账号、uv 和显卡识别均保留。
6. **解决代理和 Git 下载问题。**配置 WSL 镜像网络、Windows 代理继承，再处理错误目录与遗留锁文件，完成源码检出。
7. **安装并验证依赖。**设置 `UV_PROJECT_ENVIRONMENT`，执行 `uv sync --locked --python 3.12`；验证 `torch.cuda.is_available()` 为 True，并在 GPU 上实际完成矩阵计算。
8. **下载模型。**通过 Hugging Face `snapshot_download` 获取 `HannesVonEssen/microduck-basketball`，下载到 `artifacts/basketball/`，检查 SHA-256 清单。
9. **验证模型一致性。**运行 `verify_basketball_onnx_parity.py`，比较 checkpoint 与 ONNX 的 40 步输出，包含第 20 步的 LSTM 状态重置。
10. **运行并录制。**调用教材 `gpu_inloop_video.py`，使用篮球 ONNX、seed 42、1500 步，生成 MP4、JSON 和观测/动作轨迹。未实际执行后续 PPO 续训。

## 4. 踩过的坑与处理

| 问题 | 原因与处理 |
| --- | --- |
| WSL 安装后提示重启才生效 | WSL/虚拟机平台安装完成不等于 Ubuntu 初始化完成；重启后再创建 Ubuntu 账号 |
| 默认安装占用 C 盘 | Ubuntu、uv、Python 和缓存位于虚拟磁盘内；通过官方 `--move` 迁到 E 盘 |
| 迁移时报 `ERROR_SHARING_VIOLATION` | 仅停止发行版后磁盘仍被占用；执行 `wsl --shutdown`，等待释放后重试成功 |
| Clash 已开，GitHub 仍超时 | Clash 只监听 Windows 本地代理端口，WSL NAT 模式无法直接使用；改镜像网络并设置代理 |
| `git fetch` 很久没输出 | 最终报 443 连接超时，不代表下载成功；先解决网络，用 `--progress` 查看进度 |
| `not a git repository` | 新终端在 system32 目录；先 `cd` 到官方仓库再执行 Git 命令 |
| `.git/shallow.lock` 已存在 | 上次中断遗留；确认 Windows/Ubuntu 均无 Git 下载进程后，只删除该锁文件再重试 |
| 下载包校验不全通过 | checkpoint、ONNX、配置和源码压缩包通过；README/manifest 不匹配，预览视频缺失。原因尚未确认，保留异常记录，不修改校验清单；核心模型另做 parity 检查 |
| ONNX Runtime GPU 探测警告 | 提示无法读取 `/sys/class/drm/card0/device/vendor`；本次 parity 与 GPU 闭环均完成，未因该警告失败。不能单凭这条警告判定 CUDA 不可用 |
| 首次仿真等待较久 | MuJoCo-Warp 首次编译内核；`Recording video to ...` 只表示开始录制，仍需等待脚本结束 |

代理配置位于 Windows 用户目录的 `.wslconfig`：

```ini
[wsl2]
networkingMode=mirrored
autoProxy=true
```

修改后在 PowerShell 执行 `wsl --shutdown`，再重新进入 Ubuntu。必要时在 Ubuntu 当前终端设置：

```bash
export http_proxy=http://<Windows代理地址>:<端口>
export https_proxy=http://<Windows代理地址>:<端口>
```

## 5. 最终结果与文件

| 本机测量项 | 结果 |
| --- | --- |
| checkpoint ↔ ONNX 一致性 | 40 步，最大动作绝对误差 `1.4305e-6` |
| 仿真控制频率 / 时长 | 50 Hz；1500 步 = 30 秒 |
| 实际推理次数 | 1500 |
| 终止或超时重置 | 0；整段连续运行 |
| 闭环中的实时观测数值对照 | 32 次；最大输出误差 `8.3447e-7` |
| 策略推理耗时 | 平均约 1.77 ms，P95 约 2.70 ms；不包含完整控制闭环总耗时 |

这只是 seed 42 的单次仿真结果，不能当作多种子成功率或真机性能。

- 30 秒仿真视频：`outputs/basketball_baseline.mp4`
- 本次运行报告：`outputs/basketball_baseline.json`
- 观测、动作及重置轨迹：`outputs/basketball_baseline.trace.npz`
- 详细安装与学习指南：`任务一-篮球平衡学习指南.md`

视频运行后端：`GPU ONNX via Torch CUDA`。即配套工具读取 ONNX 策略，用 Torch CUDA 计算动作，MuJoCo 完成物理仿真；不要描述为已验证 ONNX Runtime CUDA 推理。



**已完成：预训练模型的本地仿真复现、模型输出一致性验证和视频录制。**




