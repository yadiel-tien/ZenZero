"""resnet_wdl_v1：resnet_v1 同构，value 头改胜/和/负三分类（WDL）。

动机：标量 value + MSE 把 ±1/0 三种结局压成条件均值——"官和"与"五五开"输出相同，
和棋率高的象棋校准差。WDL 用交叉熵学完整结局分布，推理折成期望分
`v = p_win - p_loss ∈ (-1,1)` 供 MCTS/胜率条使用，尺度与终局奖励 ±1/0 一致。

契约：
- forward(x) -> (policy_logits, value)：与 resnet_v1 完全同形，engine/MCTS 零改动；
- forward_wdl(x) -> (policy_logits, wdl_logits)：训练侧专用；
- logit 顺序 [负, 平, 胜]，与 z∈{-1,0,+1} 的标签映射 wdl_class 一致；
- 权重半迁移：与 resnet_v1 除 value.fc2 外全部同名同形，旧 ckpt 可 filtered load
  （trunk/policy 继承，value 头重初始化），见 train_server 的迁移加载。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from core.utils.config import CONFIG, settings, NetConfig

from ..network import BaseNet, ConvBlock, PolicyHead, ResBlock
from ..registry import register

# z∈{-1,0,+1} -> 类别（与 logit 顺序一致），训练与测试共用
CLASS_LOSS, CLASS_DRAW, CLASS_WIN = 0, 1, 2


def wdl_class(z: Tensor) -> Tensor:
    """z > +0.5   → CLASS_WIN  (2)
    z < -0.5   → CLASS_LOSS (0)
    其余 [-0.5, +0.5] → CLASS_DRAW (1)  """
    return torch.where(z > 0.5, CLASS_WIN, torch.where(z < -0.5, CLASS_LOSS, CLASS_DRAW))


class ValueHeadWdl(nn.Module):
    """胜/和/负三头。[B,in,H,W] -> [B,3] logits（顺序：负、平、胜）"""

    def __init__(self, in_channels: int, n_filters: int, hidden_channels: int, n_cells: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, n_filters, kernel_size=1, stride=1)
        self.relu1 = nn.LeakyReLU()
        self.fc1 = nn.Linear(n_cells * n_filters, hidden_channels)
        self.relu2 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_channels, 3)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu1(self.conv(x))
        x = x.reshape(x.shape[0], -1)
        x = self.relu2(self.fc1(x))
        return self.fc2(x)


def expected_score(wdl_logits: Tensor) -> Tensor:
    """[B,3] -> [B]，E[z] = p_win - p_loss ∈ (-1,1)，与终局奖励同尺度"""
    p = F.softmax(wdl_logits, dim=1)
    return p[:, CLASS_WIN] - p[:, CLASS_LOSS]


@register('resnet_wdl_v1')
class WdlNet(BaseNet):
    is_wdl = True

    def __init__(self, config: NetConfig = settings['default_net'],
                 eval_model: bool = True, device=None) -> None:
        if device is None:
            device = CONFIG['device']
        super().__init__(config, eval_model, device)
        self.conv_block = ConvBlock(config['in_channels'], config['n_filters'])
        self.res_blocks = nn.ModuleList(
            [ResBlock(config['n_filters'], config['use_se'])
             for _ in range(config['n_res_blocks'])])
        self.policy = PolicyHead(config['n_filters'],
                                 config['n_policy_filters'],
                                 config['n_cells'],
                                 config['n_actions'])
        self.value = ValueHeadWdl(config['n_filters'],
                                  config['n_value_filters'],
                                  config['n_value_hidden_channels'],
                                  config['n_cells'])
        self.initialize()

    def _heads(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.conv_block(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        return self.policy(x), self.value(x)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        policy_logits, wdl = self._heads(x)
        return policy_logits, expected_score(wdl)

    def forward_wdl(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """训练侧：返回 (policy_logits, wdl_logits)"""
        return self._heads(x)
