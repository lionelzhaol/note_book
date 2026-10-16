---
title: MicroDuck任务二：浏览器物理扰动复现笔记
date: 2026-10-16
tags: [MicroDuck, 具身智能, MuJoCo, mjswan, 浏览器仿真, 项目复盘]
status: 本地构建与浏览器拖拽已完成，对照实验及视频记录待补
---

# MicroDuck任务二：浏览器物理扰动复现笔记

> 记录范围：截至 2026-10-16，完成 mjswan Playground 的 MicroDuck 网页构建、服务启动和人工拖拽体验。依据操作记录、本地源码与构建产物整理；未记录的实验不计入已完成成果。

## 1. 做成了什么

在 WSL2 Ubuntu 中构建 MicroDuck 浏览器应用，通过 Windows 浏览器打开 `http://localhost:8092`，成功运行仿真并拖动鸭子，体验物理扰动。

与 [[MicroDuck任务一-篮球平衡仿真复现笔记]] 的区别：任务一用工作站运行篮球策略并输出视频；任务二把已有场景、策略和物理运行时打包到网页，重点是浏览器交互。本次没有重新训练模型。

## 2. 环境与存放位置

| 项目 | 最终状态 |
| --- | --- |
| 系统 | 沿用 WSL2 Ubuntu 24.04，Ubuntu 虚拟磁盘已迁至 E 盘 |
| 项目 | `<项目目录>/microduck/mjswan_playground` |
| Python 环境 | `~/.local/share/microduck/venv-mjswan`，与任务一的 `venv-official` 隔离 |
| 依赖 | 构建清单记录 mjswan `0.11.9`；项目要求 mjlab `1.6.0`，不与任务一共用依赖环境 |
| 前端工具 | mjswan 自带 `.nodeenv`；已验证 Node `24.19.0`、npm `11.17.0` |
| 构建产物 | `mjswan_playground/dist/microduck/` |
| 本地网页 | `http://localhost:8092`，终端服务需保持运行 |

## 3. 每一步做了什么

1. **重新进入 Ubuntu。**在 PowerShell 执行 `wsl -d Ubuntu-24.04`，再进入 E 盘对应的 `/mnt/e/...` 项目目录。
2. **下载项目。**执行 `git clone https://github.com/ttktjmt/mjswan_playground.git`，用 `cd mjswan_playground` 进入仓库。
3. **隔离依赖环境。**将 `UV_PROJECT_ENVIRONMENT` 改为 `venv-mjswan`，执行 `uv sync --all-extras` 安装项目及可选依赖，避免影响任务一。
4. **查看可用任务。**使用 `uv run msp list`；其中 `uv run` 负责在项目环境中执行命令，`msp` 是项目命令别名，`list` 列出任务。
5. **构建网页。**运行 MicroDuck 构建，收集场景、策略、资源索引和网页运行时，生成 `dist/microduck/`。
6. **排查构建失败。**发现 npm 找不到 `node`，将 mjswan 自带 Node 环境加入 PATH 后重试成功。
7. **启动服务。**使用 `uv run msp run microduck --port 8092` 启动本地 HTTP 服务；Ubuntu 自动打开浏览器失败后，手动在 Windows 浏览器访问地址。
8. **人工交互。**已确认能打开页面、运行鸭子仿真并拖拽。尚未留下两次固定条件的扰动对照记录或录制视频。

## 4. 踩过的坑与处理

| 问题 | 原因与处理 |
| --- | --- |
| 沿用任务一的虚拟环境变量 | `UV_PROJECT_ENVIRONMENT` 决定 uv 使用哪个环境；改为独立 `venv-mjswan`，避免不同项目依赖相互影响 |
| `RuntimeError: Build failed with exit code 127` | 构建器已下载 Node，但没有把它加入 PATH；npm 使用 `#!/usr/bin/env node`，因此找不到 Node。检查时自带 node 可运行，而 npm 报 `node: No such file or directory` |
| 系统里 `node`、`npm` 均找不到 | 本次无需额外安装系统 Node；将已有 `.nodeenv/bin` 加入 PATH 后，二者版本检查通过 |
| `lin_vel_y` 不读取仿真状态的警告 | 核对源码后确认：滑轮 heading 策略中该项通过 `_padded(1)` 故意补零。此槽固定为常量符合设计，不等于 Walk 的侧向控制全部失效 |
| `gio: http://localhost:8092: Operation not supported` | Ubuntu 自动调用浏览器失败；不影响 HTTP 服务，直接在 Windows 浏览器打开地址即可 |
| 终端启动服务后不返回提示符 | 服务持续运行，等待浏览器请求，属于正常状态；Ctrl+C 会停止服务 |

本次修复 Node 路径的命令（Ubuntu）：

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.local/share/microduck/venv-mjswan"
export PATH="$UV_PROJECT_ENVIRONMENT/lib/python3.12/site-packages/mjswan/template/.nodeenv/bin:$PATH"

node --version
npm --version
```

此 PATH 只对当前终端有效，依赖 Python 3.12 和当前包布局；重新开终端后需重新设置，未来环境升级也需检查路径。

## 5. 关键命令与运行原理

在 `mjswan_playground` 仓库目录执行：

```bash
# 安装项目和所有声明的可选功能依赖
uv sync --all-extras

# 列出可用示例，不会启动仿真或训练
uv run msp list

# 只构建静态网页
uv run msp build microduck --output-dir dist/microduck

# 构建并启动网页服务
uv run msp run microduck --port 8092
```

最后一条里，前一个 `run` 属于 uv，表示执行程序；后一个 `run` 属于 msp，表示启动指定任务。

运行分两阶段：

- **构建阶段（Ubuntu）：**Python Builder 收集模型、策略和资源，Node/npm 构建网页，输出静态目录。
- **交互阶段（浏览器）：**MuJoCo WASM 计算物理，ONNX Runtime Web 推理，Three.js 渲染；本地服务提供静态文件，不逐步执行 PPO 更新。

拖拽是引擎施加的外力，不是把鼠标位置直接变成机器人关节动作。拖拽位移转换为力/力矩并进入 MuJoCo 外力计算，松开后继续由动力学与策略控制。观察到倒地、弹起或回到初始位置，也不能直接证明策略学会了摔倒恢复；可能涉及自动重置。

## 6. 最终成果与证据

| 检查项 | 已确认结果 |
| --- | --- |
| 网页构建 | 成功；输出 `Saved mjswan application to .../dist/microduck` |
| HTTP 启动 | 日志显示 `Starting server at http://localhost:8092` |
| 浏览器使用 | 已人工确认能打开、运行仿真、拖动鸭子 |
| 场景文件 | 构建清单含 Duck、Ball (right foot)、Ball (left foot)、Rollers，共 4 个场景 |
| 策略资源 | 构建目录实际存在 26 个 `.onnx` 文件；仅核对文件数量，未逐个做 ONNX checker 或效果评测 |
| 控制步长 | 清单声明 `control_dt=0.02`，对应目标 50 Hz；未测量浏览器实际控制频率 |

主要文件：`dist/microduck/index.html`、`manifest.json`、`assets/` 及 `microduck/` 场景资源目录。用 HTTP 访问，不直接双击 HTML，以免浏览器限制 WASM/模块加载。

本次成果是“可交互的本地网页仿真”，不是导出视频，也不包含新训练权重。尚无本次扰动视频和定量抗扰指标，不引用教程作者的截图、错误数或 benchmark 作为自己的结果。


