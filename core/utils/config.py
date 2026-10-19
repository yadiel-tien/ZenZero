import os
from pathlib import Path
from typing import Literal, TypedDict

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
    tensor_shape: tuple[int, int, int]
    state_shape: tuple[int, int, int]
    augment_times: int
    max_iters: int
    buffer_size: int
    avg_game_steps: int
    default_net: NetConfig
    selfplay: TemperatureConfig
    evaluation: TemperatureConfig


# 最近一次设备决策说明（如显存不足降级 CPU），供 API/前端展示
DEVICE_NOTE: str = ''
DEVICE: str = ''


class AppConfig(TypedDict):
    color_themes: ColorTheme
    ChineseChess: GameConfig
    Gomoku: GameConfig
    dirichlet: float
    base_url: str
    device: str
    game_name: EnvName
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
    'device': 'auto',  # 'auto' 自动探测 | 'cuda:0'/'mps'/'xpu'/'privateuse1'/'cpu' 显式指定; 环境变量 ZZ_DEVICE 优先
    'log_dir': str(ROOT_DIR / 'logs'),
    'buffer_name': 'buffer.pkl',
    'best_index_name': 'best_index.pkl',
    'ema_name': 'ema.pkl',
    'training_steps_per_sample': 30,
    'win_threshold': 0.52,
    'game_name': 'Gomoku'
}

def resolve_device(preference: str | None = None) -> str:
    """解析推理/训练设备。'auto' 优先级: cuda > xpu > mps > privateuse1(DirectML) > npu > cpu。
    显式指定的设备不可用时抛 RuntimeError（快速失败，避免静默降速）。"""
    pref = (preference or os.environ.get('ZZ_DEVICE') or 'auto').strip().lower()
    import torch

    def has_cuda() -> bool:
        return torch.cuda.is_available()

    def has_xpu() -> bool:
        return hasattr(torch, 'xpu') and torch.xpu.is_available()

    def has_mps() -> bool:
        return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

    def has_dml() -> bool:
        try:
            return hasattr(torch, 'privateuseone') and torch.privateuseone.is_available()
        except Exception:
            return False

    def has_npu() -> bool:
        try:
            import torch_npu  # noqa: F401
            return torch.npu.is_available()
        except Exception:
            return False

    if pref == 'auto':
        for ok, name, fallback in ((has_cuda, 'cuda:0', 'NVIDIA GPU'),
                                   (has_xpu, 'xpu', 'Intel GPU (XPU)'),
                                   (has_mps, 'mps', 'Apple Silicon'),
                                   (has_dml, 'privateuse1', 'DirectML'),
                                   (has_npu, 'npu:0', 'Ascend NPU')):
            if ok():
                return name
        return 'cpu'
    # 显式指定：校验可用性
    backend = pref.split(':')[0]
    avail = {'cuda': has_cuda, 'xpu': has_xpu, 'mps': has_mps,
             'privateuse1': has_dml, 'npu': has_npu, 'cpu': lambda: True}.get(backend)
    if avail is None or not avail():
        raise RuntimeError(f"设备 {pref} 不可用。请检查驱动/torch 构建，或使用 device='auto' / 环境变量 ZZ_DEVICE")
    return pref


import platform

_DEV_NAME_CACHE: dict[str, str] = {}


def _cached_name(key: str, getter) -> str:
    if key not in _DEV_NAME_CACHE:
        try:
            _DEV_NAME_CACHE[key] = getter()
        except Exception:
            _DEV_NAME_CACHE[key] = key
    return _DEV_NAME_CACHE[key]


def enumerate_devices() -> list[dict]:
    """按 resolve 优先级枚举全部候选推理设备。
    status: active(使用中) / available(可用未选) / unavailable(存在但起不来) / absent(无硬件或构建不支持)
    排在使用中设备前面的条目必带 reason，解释为什么没用它。"""
    import torch
    out: list[dict] = []
    cur = str(CONFIG['device'])

    # ---- CUDA（逐卡）----
    try:
        n_cuda = torch.cuda.device_count()
    except Exception:
        n_cuda = 0
    if n_cuda == 0:
        out.append({'id': 'cuda', 'name': 'NVIDIA GPU', 'status': 'absent',
                    'reason': '未检测到 NVIDIA GPU（驱动未加载？）', 'vram': None})
    else:
        cuda_ok = torch.cuda.is_available()
        for i in range(n_cuda):
            name = _cached_name(f'cuda:{i}', lambda i=i: torch.cuda.get_device_name(i))
            vram, why = None, ''
            if cuda_ok:
                try:
                    free, total = torch.cuda.mem_get_info(i)
                    vram = {'free': round(free / 2**30, 2), 'total': round(total / 2**30, 2)}
                except Exception as e:
                    from core.network.network import _brief_cuda_error
                    why = f'显存探测失败：{_brief_cuda_error(e)}（加载时将自动降级）'
            entry = {'id': f'cuda:{i}', 'name': name, 'vram': vram,
                     'status': 'available' if cuda_ok else 'unavailable',
                     'reason': why if cuda_ok else 'CUDA 运行时不可用（驱动异常或显存耗尽）'}
            if entry['id'] == cur:
                entry['status'] = 'active'
            out.append(entry)

    # ---- 其余后端 ----
    def probe_be(be_id: str, label: str, check, name_getter=None, count_getter=None,
                 absent_why: str = '本机无对应硬件或 torch 构建不含该后端') -> None:
        try:
            if not check():
                has = False
                try:
                    has = bool(count_getter and count_getter() > 0)
                except Exception:
                    pass
                out.append({'id': be_id, 'name': label, 'status': 'absent',
                            'reason': '未检测到硬件' if has else absent_why, 'vram': None})
                return
        except Exception as e:
            out.append({'id': be_id, 'name': label, 'status': 'unavailable',
                        'reason': _brief_cuda_error(e) if 'CUDA' in str(e) else str(e)[:80], 'vram': None})
            return
        name = label
        if name_getter:
            name = _cached_name(be_id, name_getter)
        e = {'id': be_id, 'name': name, 'status': 'available', 'reason': '', 'vram': None}
        if be_id == cur:
            e['status'] = 'active'
        out.append(e)

    probe_be('xpu', 'Intel GPU (XPU)',
             lambda: hasattr(torch, 'xpu') and torch.xpu.is_available(),
             lambda: torch.xpu.get_device_name(0),
             lambda: torch.xpu.device_count(),
             '未检测到 Intel GPU 或 torch 无 XPU 构建')
    probe_be('mps', 'Apple GPU (MPS)',
             lambda: hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
             None, None, 'MPS 仅支持 macOS（Apple Silicon）')
    probe_be('privateuse1', 'DirectML (Windows GPU)',
             lambda: hasattr(torch, 'privateuseone') and torch.privateuseone.is_available(),
             None, None, '需要安装 torch-directml（仅 Windows）')
    probe_be('npu', 'Ascend NPU',
             lambda: getattr(torch, 'on_device', None) is not None and _npu_available(),
             None, None, '需要安装 torch_npu')

    # ---- CPU 恒可用 ----
    out.append({'id': 'cpu',
                'name': f'CPU · {platform.machine()} {os.cpu_count() or ""}核'.strip(),
                'status': 'active' if cur == 'cpu' else 'available', 'reason': '', 'vram': None})
    # 排在 active 之前的设备若自身无原因，用全局降级说明兜底
    for e in out:
        if e['status'] in ('available', 'unavailable') and not e['reason'] and DEVICE_NOTE:
            e['reason'] = DEVICE_NOTE
    return out


def _npu_available() -> bool:
    try:
        import torch
        return hasattr(torch, 'npu') and torch.npu.is_available()
    except Exception:
        return False


# 统一在此解析一次，下游所有 CONFIG['device'] 均为具体设备字符串
CONFIG['device'] = resolve_device(CONFIG['device'])

game_name = CONFIG['game_name']
settings = CONFIG[game_name]
