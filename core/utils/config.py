from pathlib import Path
from typing import TypedDict, Literal

from core.utils.types import EnvName


class ColorTheme(TypedDict):
    blue: list[str]
    green: list[str]
    red: list[str]
    orange: list[str]
    grey: list[str]
    black: list[str]


ROOT_DIR = Path(__file__).parent.parent.parent

# 确保必要的目录存在
(ROOT_DIR / "data").mkdir(exist_ok=True)
(ROOT_DIR / "data/ChineseChess").mkdir(exist_ok=True)
(ROOT_DIR / "data/Gomoku").mkdir(exist_ok=True)
(ROOT_DIR / "logs").mkdir(exist_ok=True)
(ROOT_DIR / "services/inference/socks").mkdir(parents=True, exist_ok=True)

color_key = Literal['blue', 'green', 'red', 'orange', 'grey', 'black']


class NetConfig(TypedDict):
    in_channels: int  # 输入通道数
    n_filters: int  # 卷积层filter数量
    n_res_blocks: int  # 残差网络数量
    n_cells: int  # 输入state的H*W
    n_actions: int  # policy输出动作的数量
    use_se: bool  # 是否使用SEBlock
    n_policy_filters: int  # 策略头卷积层filter数量
    n_value_filters: int  # 价值头卷积层filter数量
    n_value_hidden_channels: int  # 价值头隐藏层fc输出通道


class TemperatureConfig(TypedDict):
    tau_decay_rate: float
    exploration_steps: int


class GameConfig(TypedDict):
    screen_size: tuple[int, int]
    grid_size: float
    img_path: str
    tensor_shape: tuple[int, int, int]
    state_shape: tuple[int, int, int]
    augment_times: int
    max_iters: int
    buffer_size: int
    avg_game_steps: int
    default_net: NetConfig
    selfplay: TemperatureConfig
    evaluation: TemperatureConfig


class AppConfig(TypedDict):
    color_themes: ColorTheme
    ChineseChess: GameConfig
    Gomoku: GameConfig
    dirichlet: float
    base_url: str
    device: str
    game_name: EnvName
    socket_path_prefix: str
    hub_socket_path: str
    data_dir: str
    log_dir: str
    buffer_name: str
    best_index_name: str
    ema_name: str
    training_steps_per_sample: int
    win_threshold: float


CONFIG: AppConfig = {
    'color_themes': {
        'blue': ['#2F4F4F', '#1A2F2F', '#0D1A1A', '#2F4F4F'], # 玄武墨
        'green': ['#556B2F', '#3D4D21', '#242E14', '#556B2F'], # 墨绿
        'red': ['#8B0000', '#660000', '#4D0000', '#8B0000'], # 朱砂红
        'orange': ['#B8860B', '#996F09', '#7A5907', '#B8860B'], # 琥珀金
        'grey': ['#696969', '#4F4F4F', '#363636', '#696969'], # 铅灰
        'black': ['#1C1C1C', '#000000', '#000000', '#1C1C1C'] # 漆黑
    },
    'ChineseChess': {
        'screen_size': (600, 800),
        'grid_size': 54,
        'tensor_shape': (20, 10, 9),
        'state_shape': (7, 10, 9),
        'img_path': './apps/assets/graphics/chess/board.jpeg',
        'augment_times': 2,
        'max_iters': 500,
        'buffer_size': 300_000,
        'avg_game_steps': 80,
        'selfplay': {
            'tau_decay_rate': 0.96,
            'exploration_steps': 30
        },
        'evaluation': {
            'tau_decay_rate': 0.9,
            'exploration_steps': 10
        },
        'default_net': {
            'in_channels': 20,  # 输入通道数
            'n_filters': 256,  # 卷积层filter数量
            'n_cells': 10 * 9,  # 输入H*W
            'n_res_blocks': 15,  # 残差网络数量
            'n_actions': 2086,  # policy输出动作的数量
            'use_se': True,  # 是否使用SEBlock
            'n_policy_filters': 32,  # 策略头卷积层filter数量
            'n_value_filters': 32,  # 价值头卷积层filter数量
            'n_value_hidden_channels': 32  # 价值头隐藏层fc输出通道
        }
    },
    'Gomoku': {
        'screen_size': (600, 800),
        'grid_size': 35.2857,
        'tensor_shape': (2, 15, 15),
        'state_shape': (2, 15, 15),
        'img_path': './apps/assets/graphics/gomoku/board.jpeg',
        'augment_times': 8,
        'max_iters': 100,
        'buffer_size': 150_000,
        'avg_game_steps': 40,
        'selfplay': {
            'tau_decay_rate': 0.8,
            'exploration_steps': 10
        },
        'evaluation': {
            'tau_decay_rate': 0.65,
            'exploration_steps': 4
        },
        'default_net': {
            'in_channels': 2,  # 输入通道数
            'n_filters': 256,  # 卷积层filter数量
            'n_cells': 15 * 15,  # 输入H*W
            'n_res_blocks': 11,  # 残差网络数量
            'n_actions': 15 * 15,  # policy输出动作的数量
            'use_se': True,  # 是否使用SEBlock
            'n_policy_filters': 32,  # 策略头卷积层filter数量
            'n_value_filters': 1,  # 价值头卷积层filter数量
            'n_value_hidden_channels': 256  # 价值头隐藏层fc输出通道
        },
    },

    'data_dir': str(ROOT_DIR / 'data'),
    'dirichlet': 0.2,
    'base_url': 'http://192.168.0.126:5000/',
    'device': 'cuda:0',
    'socket_path_prefix': str(ROOT_DIR / 'services/inference/socks/'),
    'hub_socket_path': str(ROOT_DIR / 'services/inference/socks/hub.sock'),
    'log_dir': str(ROOT_DIR / 'logs'),
    'buffer_name': 'buffer.pkl',
    'best_index_name': 'best_index.pkl',
    'ema_name': 'ema.pkl',
    'training_steps_per_sample': 30,
    'win_threshold': 0.52,
    'game_name': 'Gomoku'
}

game_name = CONFIG['game_name']
settings = CONFIG[game_name]
