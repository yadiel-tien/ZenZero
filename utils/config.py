from typing import TypedDict, Literal

from utils.types import EnvName


class ColorTheme(TypedDict):
    blue: list[str]
    green: list[str]
    red: list[str]
    orange: list[str]
    grey: list[str]
    black: list[str]


cwd = '/home/bigger/projects/five_in_a_row'
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
    train_socket_path: str
    data_dir: str
    log_dir: str
    buffer_name: str
    best_index_name: str
    ema_name: str
    training_steps_per_sample: int
    win_threshold: float


CONFIG: AppConfig = {
    'color_themes': {
        'blue': ['#007bff', '#0056b3', '#004085', '#0056b3'],
        'green': ['#28a745', '#218838', '#1e7e34', '#218838'],
        'red': ['#dc3545', '#c82333', '#bd2130', '#c82333'],
        'orange': ['#fd7e14', '#e67e00', '#d45d02', '#e67e00'],
        'grey': ['#6c757d', '#5a6268', '#343a40', '#5a6268'],
        'black': ['#000000', '#333333', '#555555', '#333333']
    },
    'ChineseChess': {
        'screen_size': (600, 800),
        'grid_size': 54,
        'tensor_shape': (20, 10, 9),
        'state_shape': (7, 10, 9),
        'img_path': './graphics/chess/board.jpeg',
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
        'img_path': './graphics/gomoku/board.jpeg',
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

    'data_dir': './data/',
    'dirichlet': 0.2,
    'base_url': 'http://192.168.0.126:5000/',
    'device': 'cuda:0',
    'socket_path_prefix': './inference/socks/',
    'hub_socket_path': './inference/socks/hub.sock',
    'train_socket_path': cwd + '/inference/socks/train.sock',
    'log_dir': './logs/',
    'buffer_name': 'buffer.pkl',
    'best_index_name': 'best_index.pkl',
    'ema_name': 'ema.pkl',
    'training_steps_per_sample': 30,
    'win_threshold': 0.52,
    'game_name': 'Gomoku'
}

game_name = CONFIG['game_name']
settings = CONFIG[game_name]
