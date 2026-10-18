# ZenZero - 高性能 AlphaZero 棋类 AI 框架

ZenZero 是一个基于 AlphaZero 算法实现的通用棋类 AI 框架，专注于中国象棋 (Chinese Chess) 和五子棋 (Gomoku)。项目利用 **Python 3.14 (nogil)** 的 free-threading 特性，配合 Cython 核心加速与分布式推理架构，实现了极致的训练与对战性能。

## 🌟 核心亮点 (Technical Highlights)

### 1. 真正的并行自我对弈 (True Parallelism)
传统 Python 受 GIL 限制。ZenZero 采用 **Python 3.14 free-threading (nogil)**，使数十个 MCTS 线程能真正并行运行在不同 CPU 核心上，自我对弈数据生成速度提升显著。

### 2. 高效推理引擎 (High-Performance Inference)
*   **动态批处理 (Dynamic Batching)**: 推理服务器根据负载动态调整 Batch Size，极大提升 GPU 吞吐量。
*   **多级缓冲与低开销 IPC**: 训练侧 (selfplay + 推理 + 训练) 单进程运行，MCTS 通过进程内队列直接请求推理，无序列化开销；对战侧跨进程使用 Unix Domain Socket 低延迟通信。

### 3. C 级核心优化 (Cython Acceleration)
*   中国象棋的走法生成、合法性检查以及五子棋的五子连珠判定均使用 **Cython** 编写，搜索性能相比纯 Python 提升约 30 倍。

---

## ⚙️ 快速开始 (Quick Start)

### 📥 预训练权重配置
前往 [Releases](https://github.com/yadiel-tien/AlphaGomoku/releases) 下载 `.pt` 模型文件，并按以下结构放置：
```text
data/
├── ChineseChess/  # 放入 checkpoint_498.pt
└── Gomoku/        # 放入 checkpoint_141.pt
```
*系统会自动识别编号最大的 `checkpoint` 作为最强模型。*

### 🛠️ 环境搭建
项目只分**前端 / 后端**两套依赖（需先安装 [uv](https://docs.astral.sh/uv/)）：

| 环境类型 | 适用场景 | 安装指令 |
| :--- | :--- | :--- |
| **后端 (`.venv`)** | 训练 (selfplay + 推理 + 权重更新, 单进程) + 对战服务 (play_server) + CLI / Piskvork / 竞技场。训练与对战共用同一套 torch 依赖。 | `uv venv -p cpython-3.14t .venv` <br> `uv pip install -r requirements-backend.txt --extra-index-url https://download.pytorch.org/whl/cu132` |
| **前端 (`.venv-front`)** | GUI 对战客户端 (`ui_play`)，可通过 HTTP 连接任意后端。可部署在 Mac / Windows 等无 GPU 机器上，**不需要 torch**。 | `uv venv -p cpython-3.14t .venv-front` <br> `uv pip install -r requirements-frontend.txt` |

> **注意**: 后端 `torch` 固定为 `2.14.1+cu132` 构建，要求 NVIDIA 驱动 R595 及以上 (CUDA 13.2)。安装时必须带 `--extra-index-url`，否则会装到 PyPI 默认的 cu12x 构建，与驱动不匹配。
> 前端跨机器使用时，把 `core/utils/config.py` 中的 `base_url` 指向后端地址（如 `http://<GPU服务器IP>:5000/`）。

### 🏗️ 编译 Cython 模块 (可选)
核心逻辑提供了 Cython 优化版，编译后可显著提升搜索速度。仅后端需要编译（前端缺少 `.so` 时自动回退纯 Python 实现）：
```bash
cd core/env
.venv/bin/python setup.py build_ext --inplace
cd ../..
```

---

## 📖 使用指南 (Usage)

### 1. 训练新模型
1. **指定游戏**: 在 `core/utils/config.py` 中修改 `game_name`。
2. **启动训练** (后端环境 `.venv`，单进程，一条命令):
    * `python -X gil=0 -m scripts.train_server`
    * 推理引擎、selfplay、模型训练运行在同一进程中：MCTS 通过队列直接请求推理，
      训练命令为直接方法调用，无 socket 开销。
    * `-X gil=0` 必须保留：首次 forward 会惰性导入 triton 并试图重开 GIL，
      强制关闭后 100 个 MCTS 线程才能真并行。

### 2. 图形化 GUI 对战
支持人机对战、双人对战及 AI 观战。前端与后端可同机，也可跨机器 (前端在 Mac/Windows，后端在 GPU 服务器，HTTP 通信)：
1. **启动后端** (后端环境): `python -m scripts.play_server` (Flask API + 推理引擎 + hub)
2. **启动前端** (前端环境): `python -m scripts.ui_play` (GUI 界面；跨机器时先改 `base_url` 指向后端)

### 3. 本地命令行对战 (CLI)
适用于同一台机器上的轻量级命令行交互模式 (均在后端环境 `.venv`)：
1. **终端 1**: `python -m scripts.infer_hub` (启动推理调度中心)
2. **终端 2**: `python -m scripts.cli_play` (启动命令行对战)

### 4. Piskvork (Gomocup) 对战
适用于与 Piskvork 客户端或 Gomocup 兼容的外部软件进行对战 (后端环境 `.venv`)：
1. **启动服务**: `python -m scripts.priskvork_server` (默认端口 12345)
2. **连接**: 在 Piskvork 客户端中添加该 AI 的 IP 和端口。

### 5. AI 竞技场 (Arena)
自动化评估不同版本模型的棋力，并维护 ELO 等级分 (后端环境 `.venv`)。
1. **配置名单**: 在 `data/<游戏名>/rates/candidates.txt` 中写入模型编号（如 `10,20,141`）。
2. **运行**: `python -m scripts.arena`。

---

## 🏗️ 项目结构 (Project Structure)

```text
ZenZero/
├── core/               # 核心逻辑 (env, mcts, network, player, utils)
├── services/           # 后端服务 (inference, train)
├── apps/               # 应用层 (ui, assets)
├── scripts/            # 启动脚本
└── data/               # 模型、棋谱、ELO 数据及配置名单
```

---

## 🧩 扩展性 (Extensibility)

接入新游戏需三步：
1. **实现环境**: 继承 `core/env/env.py` 中的 `BaseEnv`。
2. **配置参数**: 在 `core/utils/config.py` 中定义 `GameConfig`。
3. **UI 开发**: 在 `apps/ui/` 下参考现有游戏实现图形界面。



## 📜 授权 (License)
本项目采用 [MIT License](LICENSE)。欢迎提交 Issue 或 Pull Request 贡献代码！
