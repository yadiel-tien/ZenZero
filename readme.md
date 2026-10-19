# ZenZero - 高性能 AlphaZero 棋类 AI 框架

ZenZero 是一个基于 AlphaZero 算法实现的通用棋类 AI 框架，专注于中国象棋 (Chinese Chess) 和五子棋 (Gomoku)。项目利用 **Python 3.14 (nogil)** 的 free-threading 特性，配合 Cython 核心加速与分布式推理架构，实现了极致的训练与对战性能。

## 🌟 核心亮点 (Technical Highlights)

### 1. 真正的并行自我对弈 (True Parallelism)
传统 Python 受 GIL 限制。ZenZero 采用 **Python 3.14 free-threading (nogil)**，使数十个 MCTS 线程能真正并行运行在不同 CPU 核心上，自我对弈数据生成速度提升显著。

### 2. 高效推理引擎 (High-Performance Inference)
*   **动态批处理 (Dynamic Batching)**: 推理服务器根据负载动态调整 Batch Size，极大提升 GPU 吞吐量。
*   **CPU 多 worker 并行推理**: 每个引擎默认 `核数/4`（上限 8）个并发 forward worker + 动态 intra-op 线程（单人对战升线程保延迟、多局并发降线程防超额订阅）。实测 i9-14900K 上 8 局并发吞吐较单 worker 提升 ~54%，多局时近似铺满全部核心。worker 数可用 `ZZ_INFER_WORKERS` 覆盖；并发正确性由读写门保证（forward 共享、在线换设备独占）。
*   **设备一致性**: 输入张量始终跟随本引擎模型参数所在设备，多引擎并存时某个模型加载失败触发 CPU 降级不会污染其他驻留 GPU 的引擎（曾因全局配置改写导致 conv 权重/输入分设备崩溃）。
*   **全进程内零拷贝 IPC**: 训练与对战均已统一为进程内队列 (queue) 请求推理，无 socket / pickle 序列化开销；跨机器仅 GUI 走 HTTP。

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
**单一环境**（需先安装 [uv](https://docs.astral.sh/uv/)）——前端已 Web 化，浏览器即客户端，无任何前端依赖：

| 环境 (`.venv`) | 适用场景 | 安装指令 |
| :--- | :--- | :--- |
| **后端** | 训练 (selfplay + 推理 + 权重更新, 单进程) + 对战服务 (play_server, 含 Web 前端托管) + CLI / Piskvork / 竞技场 | `uv venv -p cpython-3.14t .venv` <br> `uv pip install -r requirements-backend.txt --extra-index-url https://download.pytorch.org/whl/cu132` |

> **注意**: `torch` 需按设备选择构建（见下表），requirements 不再固定 torch。
> 网页前端由 `play_server` 绑定 `0.0.0.0` 直接对局域网提供服务，浏览器访问 `http://<服务器IP>:5000/` 即可。

### 🖥️ 设备支持与配置建议

推理/训练设备默认 **`auto` 自动探测**，优先级 `CUDA > XPU(Intel) > MPS(Apple) > DirectML > NPU(昇腾) > CPU`。
手动指定：`core/utils/config.py` 的 `'device'`，或环境变量 `ZZ_DEVICE=cuda:0 / mps / cpu / privateuse1 ...`（显式指定不可用会直接报错，防止静默降速）。

| 设备 | torch 安装命令 (先装 requirements 再执行) | 训练 | Web 对战 | 建议 |
| :--- | :--- | :--- | :--- | :--- |
| **NVIDIA (Linux)** ✅ 已验证 | `uv pip install torch --index-url https://download.pytorch.org/whl/cu132` (驱动 R595+/CUDA 13.2) | ★★★★★ | ★★★★★ | 主力环境；混合精度 + 动态批处理全开 |
| **AMD GPU (Linux ROCm)** | `uv pip install torch --index-url https://download.pytorch.org/whl/rocm6.4` | ★★★★ | ★★★★★ | ROCm 走 CUDA API 兼容层，代码零改动 |
| **Intel Arc (Linux XPU)** | `uv pip install torch --index-url https://download.pytorch.org/whl/xpu` | ★★★ | ★★★★ | 自动识别为 xpu |
| **Apple M 系列 (MPS)** | `uv pip install torch` (PyPI 默认含 MPS) | ★★ | ★★★★ | 适合当对战服务器/体验；训练建议交 NVIDIA |
| **Windows (NVIDIA)** | `uv pip install torch --index-url https://download.pytorch.org/whl/cu132` | ★★★★ | ★★★★★ | 与 Linux 同 |
| **Windows (AMD/Intel 显卡)** | `uv pip install torch-directml` | ★ | ★★★ | 以 `privateuse1` 设备探测 |
| **纯 CPU (任意平台)** | `uv pip install torch` (PyPI 默认) | ★ | ★★★ | 对战可用；多 worker 并发吃满多核 (`ZZ_INFER_WORKERS`)，自动关 pin_memory/autocast |

* 模型 checkpoint 跨设备通用（加载统一先落 CPU 再迁移），NVIDIA 上训好的权重可直接在 Mac/CPU 上对战。
* nogil 并行 (`-X gil=0`) 依赖对应平台的 3.14t wheel；无 3.14t 时用普通 3.14 运行即可，仅多线程 MCTS 吞吐下降。

### 🏗️ 编译 Cython 模块 (可选)
核心逻辑提供了 Cython 优化版，编译后可显著提升搜索速度（缺少 `.so` 时自动回退纯 Python 实现）。浏览器端棋规引擎独立实现，不受此项影响：
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

### 2. 图形化对战（Web 前端，推荐 ⭐）
**零额外环境**：后端服务同时托管网页前端，一条命令 + 浏览器即可开打。

* **模式**：人机对战 / 双人同屏 / AI 观战，支持换先手、认输、**悔棋**（人机模式成对回退，无需后端改动）
* **界面**：程序化绘制的水墨风棋盘（SVG）、中文棋谱（炮八平五式记法）、双胜率条、每步 100s 计时、AI 思考动效、全套音效
* **跨设备**：桌面浏览器、手机（底部抽屉式棋谱/胜率）、Mac / Windows 均可访问
* **实现**：`apps/web/` 纯 HTML/CSS/JS 零构建；浏览器内 JS 棋规引擎与 `core/env` 的动作编码**逐位对齐**（随机对局快照级对拍测试保障）

1. **启动后端** (后端环境): `python -m scripts.play_server`
2. **打开浏览器** 访问 `http://<服务器IP>:5000/` 即可对战（本机用 `http://127.0.0.1:5000/`）。
   * 前后端同源部署，无需任何跨域配置；若网页另行托管，可在侧栏「推理服务 → 设置」填写后端地址（API 已开启 CORS）。
   * 双人同屏模式纯本地运行，即使不启动 AI 服务也可娱乐。


### 3. 本地命令行对战 (CLI)
单进程自含推理引擎，无需启动任何服务 (后端环境 `.venv`)：
`python -m scripts.cli_play`

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
├── apps/               # 应用层 (web 网页前端, assets 图片/音效资源)
├── scripts/            # 启动脚本
└── data/               # 模型、棋谱、ELO 数据及配置名单
```

---

## 🧩 扩展性 (Extensibility)

接入新游戏需三步：
1. **实现环境**: 继承 `core/env/env.py` 中的 `BaseEnv`。
2. **配置参数**: 在 `core/utils/config.py` 中定义 `GameConfig`。
3. **UI 开发**: 在 `apps/web/` 下参考现有游戏实现网页界面（棋规引擎需同步移植为 JS，参考 `apps/web/js/env-chess.js` 的对拍测试约束）。



## 📜 授权 (License)
本项目采用 [MIT License](LICENSE)。欢迎提交 Issue 或 Pull Request 贡献代码！
