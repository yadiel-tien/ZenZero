# ZenZero 文档集

> ⚠️ 本目录下是**六篇 HTML 技术文章**（含内联 SVG 图与编号公式）。GitHub 网页只渲染 Markdown，
> 直接点开 `.html` 会显示源码——请选择下面任一阅读方式：

| 方式 | 入口 |
|---|---|
| **GitHub Pages**（推荐，开启后同站直读，样式完整） | `https://yadiel-tien.github.io/ZenZero/`（站点根即本目录） |
| **本地服务**（起了 play_server 就有） | `http://<主机>:5000/docs/` |
| **离线** | 下载后浏览器打开 `index.html`（零外部依赖、零 CDN；指向服务的入口会自动改跳《如何使用》） |

> 开启 Pages：仓库 Settings → Pages → Build and deployment → Source 选
> `Deploy from a branch`，分支 `main`、目录 `/docs`。（已放 `.nojekyll` 跳过 Jekyll 处理。）

## 六篇目录

1. **如何使用**（`usage.html`）— 三步上手、安装与构建、对战、管理台启停训练、排障、测试验收　**［英文版 `en/usage.html`］**
2. **象棋环境**（`xiangqi-env.html`）— state 与历史窗口、动作 ABI、规则裁决、26 面表示、feature epoch
3. **MCTS 与虚拟损失**（`mcts.html`）— PUCT、虚拟损失账本、无屏障流水线、策略目标与视角符号链
4. **推理引擎**（`inference-engine.html`）— 成本模型、自适应凑批、镜像增强、设备一致性、Hub
5. **训练管线**（`training.html`）— 大闭环、轮与代、sims 爬坡、仓与留出集、CE=KL+熵、停止语义
6. **架构、部署与接口**（`architecture.html`）— 进程拓扑、神经网络注册表、环境变量、对战/控制台 API 契约

英文支持：总览与第 1 篇已有英文版（`en/index.html`、`en/usage.html`），页头「中文 / English」互切并记忆偏好
（`?nolang` 可强制停留）；其余四篇翻译中，先行阅读中文版。

各篇附录引用的 `tests/` 回归套件（13 套，无外部框架）随仓库分发。
