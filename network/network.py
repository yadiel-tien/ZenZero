import glob
import json
import os
import traceback
from typing import Self

import torch
from torch import Tensor, nn

from utils.config import CONFIG, settings, NetConfig


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, n_filters: int) -> None:
        """特征提取，输入[B,C,H,W],输出[B,n_filters,H,W]"""
        super().__init__()
        self.conv = nn.Conv2d(in_channels, n_filters, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(n_filters)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        # 如果输入是[C,H,W],则补上B=1，->[1,C,H,W]
        if x.ndimension() == 3:
            x = x.unsqueeze(0)
        return self.relu(self.bn(self.conv(x)))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block,自调节通道权重，可用于增强网络性能。"""
    """暂时还没用"""

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResBlock(nn.Module):
    def __init__(self, n_filters: int, with_se: bool = False) -> None:
        """残差块处理特征，输入输出结构一样[B,n_filters,H,W]"""
        super().__init__()
        self.conv1 = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(n_filters)
        # 可选SEBlock
        self.with_se = with_se
        if with_se:
            self.se = SEBlock(n_filters)

        self.relu2 = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        y = self.relu1(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y += x  # 残差连接
        if self.with_se:
            y = self.se(y)
        return self.relu2(y)


class PolicyHead(nn.Module):
    def __init__(self, in_channels: int, n_filters: int, n_cells: int, n_actions: int) -> None:
        """
        策略头，输出动作策略概率分布。[B,in_channels,H,W]->[B,H*W]/[H*W]
        :param in_channels: 输入通道数
        :param n_filters: 中间特征通道数量
        :param n_actions: 动作空间大小
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, n_filters, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(n_filters)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(n_cells * n_filters, n_actions)

    def forward(self, x: Tensor) -> Tensor:
        # x:[B,in_channels,H,W]
        x = self.relu(self.bn(self.conv(x)))
        # x:[B,n_filters,H,W]
        x = x.reshape(x.shape[0], -1)  # 展平
        # x:[B,n_filters*H*W]
        x = self.fc(x)
        # x:[B,n_actions]
        # 直接返回原始 Logits
        return x


class ValueHead(nn.Module):
    def __init__(self, in_channels: int, n_filters: int, hidden_channels: int, n_cells: int) -> None:
        """价值头,输出当前盘面价值。【B,in_channels,H,W]->[B]"""
        super().__init__()
        self.conv = nn.Conv2d(in_channels, n_filters, kernel_size=1, stride=1)
        self.relu1 = nn.LeakyReLU()
        self.fc1 = nn.Linear(n_cells * n_filters, hidden_channels)
        self.relu2 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_channels, 1)
        self.tanh = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        # x:[B,in_channels,H,W]
        x = self.conv(x)
        x = self.relu1(x)
        # x:[B,n_filters,H,W]
        x = x.reshape(x.shape[0], -1)
        # x:[B,H*W]
        x = self.fc1(x)
        x = self.fc2(self.relu2(x))
        # x:[B,1]
        value = self.tanh(x).reshape(-1)
        # x:[B]
        return value


class BaseNet(nn.Module):
    def __init__(self, config: NetConfig = settings['default_net'], eval_model: bool = True,
                 device=CONFIG['device']) -> None:
        super().__init__()
        self.config = config
        self.device = device
        self.eval_model = eval_model

    def initialize(self):
        self.to(self.device)
        # 推理模式
        if self.eval_model:
            self.eval()
        mode = 'eval' if self.eval_model else 'training'
        print(f'Network initialized in {mode} mode on {self.device}, configuration as below:')
        # 打印配置
        print(json.dumps(self.config, indent=2))
        print("=" * 60)

    @classmethod
    def load_latest_from_folder(cls, folder: str, eval_model: bool = True, device=CONFIG['device']) -> tuple[Self, int]:
        """从文件夹中读取最新的存档加载,没有的话通过config创建新模型
        :return 加载好的模型和编号"""
        pattern = os.path.join(folder, 'checkpoint_*.pt')
        files = glob.glob(pattern)
        # 编号最大的就是最新的模型
        latest_file = max(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        model, success = cls.load_from_checkpoint(latest_file, eval_model=eval_model, device=device)
        if success:
            return model, int(latest_file.split('_')[-1].split('.')[0])

        return model, -1

    @classmethod
    def load_from_checkpoint(cls, path: str, eval_model: bool = True, device=CONFIG['device']) -> tuple[Self, bool]:
        """从存档读取模型配置创建模型，并加载参数
        :return model和是否成功"""
        success = True
        try:
            checkpoint = torch.load(path, map_location=CONFIG['device'])
            config = checkpoint['config']
            # 创建网络
            model = cls(config, eval_model=eval_model, device=device)
            # 加载参数
            model.load_state_dict(checkpoint['model'])
        except (FileNotFoundError, KeyError):
            success = False
            model = cls(eval_model=eval_model, device=device)
        except Exception as e:
            print(e)
            traceback.print_exc()
            raise
        return model, success


class Net(BaseNet):
    def __init__(self, config: NetConfig = settings['default_net'], eval_model: bool = True,
                 device=CONFIG['device']) -> None:
        """
        类alpha zero结构，双头输出policy和value
        """
        super().__init__(config, eval_model, device)
        self.conv_block = ConvBlock(config['in_channels'], config['n_filters'])
        self.res_blocks = nn.ModuleList(
            [ResBlock(config['n_filters'], config['use_se'])
             for _ in range(config['n_res_blocks'])])
        self.policy = PolicyHead(config['n_filters'],
                                 config['n_policy_filters'],
                                 config['n_cells'],
                                 config['n_actions'])
        self.value = ValueHead(config['n_filters'],
                               config['n_value_filters'],
                               config['n_value_hidden_channels'],
                               config['n_cells'])

        self.initialize()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.conv_block(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        return self.policy(x), self.value(x)


class BiGameNet(BaseNet):
    def __init__(self, config: dict = None, eval_model: bool = True, device=CONFIG['device']):
        game1_config = CONFIG['ChineseChess']['default_net']
        game2_config = CONFIG['Gomoku']['default_net']
        self.config = {
            'game1': game1_config,
            'game2': game2_config
        } if config is None else config
        super().__init__(self.config['game1'], eval_model, device)
        self.conv_blocks = nn.ModuleDict()
        self.policy_heads = nn.ModuleDict()
        self.value_heads = nn.ModuleDict()
        for key, game_config in self.config.items():
            self.conv_blocks[key] = ConvBlock(game_config['in_channels'], game_config['n_filters'])
            self.policy_heads[key] = PolicyHead(
                in_channels=game_config['n_filters'],
                n_filters=game_config['n_policy_filters'],
                n_cells=game_config['n_cells'],
                n_actions=game_config['n_actions'])
            self.value_heads[key] = ValueHead(
                in_channels=game_config['n_filters'],
                n_filters=game_config['n_value_filters'],
                hidden_channels=game_config['n_value_hidden_channels'],
                n_cells=game_config['n_cells']
            )

        self.res_blocks = nn.ModuleList(
            [ResBlock(game1_config['n_filters'], game1_config['use_se']) for _ in range(game1_config['n_res_blocks'])])

        self.initialize()

    def _determine_game_key(self, x: Tensor) -> str:
        c, h, w = x.shape[-3:]
        for key, game_config in self.config.items():
            if c == game_config['in_channels'] and h * w == game_config['n_cells']:
                return key
        raise RuntimeError('Input tensor shape do not match!')

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        key = self._determine_game_key(x)
        x = self.conv_blocks[key](x)
        for res_block in self.res_blocks:
            x = res_block(x)
        return self.policy_heads[key](x), self.value_heads[key](x)
