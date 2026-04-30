# ZenZero - 中国象棋与五子棋 AI (ZenZero - Chinese Chess & Gomoku AI)

这是一个基于 AlphaZero 思想实现的通用棋类 AI 框架，目前已支持中国象棋（Chinese Chess）和五子棋（Gomoku）。项目采用了高度优化的架构，旨在充分利用多核 CPU 和 GPU 资源，实现高性能的自我对弈训练和对战。

## 🚀 核心特性 (Features)

* **高性能自我对弈:** 采用移除了 GIL 的 `free-threading` Python (`nogil`)，实现了真正并行的自我对弈数据生成。
* **动态批处理推理:** 独立的推理服务器，采用动态批处理（Dynamic Batching）技术，最大化 GPU 利用率。
* **Cython 核心加速:** 走法生成和胜负判断等核心逻辑使用 Cython 进行了 C 级别的优化。
* **AI 竞技场 (Arena):** 自动化的 AI 竞技场，通过对战更新 ELO 等级分，科学评估模型棋力。
* **前后端分离架构:**
    * **训练端:** 采用高效的本地 Unix Socket 进行低延迟通信。
    * **对战端:** 支持 AI 算力与 UI 界面分布式部署（例如：高性能服务器负责 MCTS 搜索，本地轻便设备仅负责图形交互）。
* **图形化对战界面 (GUI):** 基于 Pygame 实现，支持 Human vs. AI、AI vs. AI 等多种模式。
* **模块化设计:** 环境、MCTS、网络、播放器、UI 模块清晰分离，易于扩展。

## ⚙️ 项目结构 (Project Structure)

```
ZenZero/
├── core/               # 核心逻辑 (env, mcts, network, player, utils)
├── services/           # 后端服务 (inference, train)
├── apps/               # 应用层 (ui, assets)
├── scripts/            # 启动脚本
├── logs/               # 日志文件
└── data/               # 存放模型、棋谱等数据 (需手动创建)
```

## 🛠️ 安装与设定 (Setup & Installation)

建议准备两个 Python 环境：一个用于高性能计算的 `nogil` 环境，一个用于 UI 的标准环境。

### 1. 克隆项目
```bash
git clone git@github.com:yadiel-tien/AlphaGomoku.git
cd ZenZero
```

### 2. 创建 nogil 计算环境 (用于训练和推理)
推荐使用 [uv](https://github.com/astral-sh/uv) 管理 free-threading 版本的 Python：
```bash
# 安装 3.14t (free-threading)
uv python install 3.14t
# 创建虚拟环境
uv venv --python 3.14t nogil_venv
# 激活环境
source nogil_venv/bin/activate
# 安装核心依赖 (示例)
uv pip install torch numpy requests
```

### 3. 创建标准 UI 环境 (用于图形界面)
```bash
python3 -m venv standard_venv
source standard_venv/bin/activate
pip install pygame requests numpy
```

### 4. 编译 Cython 模块
必须使用 `nogil` 环境编译以确保线程安全：
```bash
source nogil_venv/bin/activate
cd core/env
python setup.py build_ext --inplace
cd ../..
```

## 🎮 使用方法 (Usage)

### 1. 训练新模型
需要两个终端，均使用 `nogil_venv` 环境：
* **终端 1 (服务器):** `python -m scripts.train_server`
* **终端 2 (自我对弈):** `python -X gil=0 -m scripts.selfplay`

### 2. 图形化对战 (推荐)
1. **终端 1 (Hub 服务):** `python -m scripts.infer_hub` (标准环境)
2. **终端 2 (对战后端):** `python -m scripts.play_server` (标准环境)
3. **终端 3 (GUI 界面):** `python -m scripts.ui_play` (标准环境)

### 3. AI 竞技场
```bash
source nogil_venv/bin/activate
python -m scripts.arena
```

## 📝 配置 (Configuration)
文件路径、模型超参数等均可在 `core/utils/config.py` 中修改。

## 📜 授权 (License)
MIT License.
