"""版本管理 —— 全项目单一事实源。

四条版本轴各司其职，不要混用，也不要给组件另立版本号（理由见 §5 篇「版本管理」）：

  ① 应用版本 __version__   整个软件的发布版本（semver），随 Git tag 走：
                          主版本=表示/协议级破坏性变更或里程碑；次版本=功能；修订=修复。
                          引擎、前端、控制台不单独编号——同仓同发布，避免兼容矩阵爆炸。
  ② 表示版本 feature_version 棋盘张量布局的 epoch（1=吃王制/20面，2=合法着法制/26面…）。
                          烧进 checkpoint，决定特征转换与网络通道；模型"活在自己的表示版本里"（§1 篇 §6）。
  ③ 档位 profile           网络规模（s/m/l），只影响体量与目录，不影响兼容性。
  ④ 模型代数 iteration      档位内的自博弈世代号（checkpoint_N.pt），守擂晋升机制推进（§4 篇 §8）。

界面/接口的展示位置：/info、控制台标题徽章、快照 meta、启动横幅、main.py --version。
"""
from __future__ import annotations

import subprocess
from pathlib import Path

__version__ = '2.1.0'

ROOT = Path(__file__).resolve().parents[2]


def revision() -> str:
    """当前构建的 git 短哈希；不在 git 环境（pip 安装、导出快照）时返回空串。"""
    try:
        out = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                             cwd=ROOT, capture_output=True, text=True, timeout=2)
        return out.stdout.strip() if out.returncode == 0 else ''
    except (OSError, subprocess.SubprocessError):
        return ''


def full(with_revision: bool = True) -> str:
    rev = revision() if with_revision else ''
    return f'{__version__}+{rev}' if rev else __version__
