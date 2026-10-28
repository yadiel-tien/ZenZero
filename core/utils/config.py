import json
import os
import re
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
    arch: str  # 网络架构名（core/network/registry 注册表键）
    in_channels: int  # 输入通道数
    n_filters: int  # 卷积层filter数量
    n_res_blocks: int  # 残差网络数量
    n_cells: int  # 输入state的H*W
    n_actions: int  # policy输出动作的数量
    use_se: bool  # 是否使用SEBlock
    n_policy_filters: int  # 策略头卷积层filter数量
    n_value_filters: int  # 价值头卷积层filter数量
    n_value_hidden_channels: int  # 价值头隐藏层fc输出通道
    feature_version: int  # 出生时的棋盘表示版本，决定 convert_to_network 布局


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
    feature_version: int  # 棋盘表示版本（feature_version）：输入面语义变更时 +1，buffer/checkpoint 校验用。
    # 注意与训练语义的 epoch（数据过一遍 buffer）无关，勿混用词。
    c_puct: int  # PUCT 探索常数（整棵树统一，NeuronMCTS 读入并经 get_child 传播）
    presets: dict[str, dict]  # 大中小档位：profile 名 -> default_net 的规模覆写（禁覆写棋规/表示级键）
    default_net: NetConfig
    selfplay: TemperatureConfig
    evaluation: TemperatureConfig
    # MCTS 模拟次数爬坡：按**绝对代**（不是本次启动内的轮次），重启不重置。
    # start 起步值 / max 上限 / ramp 多少代爬到上限
    sims: dict


# 最近一次设备决策说明（如显存不足降级 CPU），供 API/前端展示
DEVICE_NOTE: str = ''
DEVICE: str = ''


class AppConfig(TypedDict):
    color_themes: ColorTheme
    ChineseChess: GameConfig
    Gomoku: GameConfig
    dirichlet: float  # 根节点狄利克雷噪声权重 eps（inject_noise 实际读取处）
    base_url: str
    device: str
    game_name: EnvName
    data_dir: str
    log_dir: str
    runs_dir: str
    buffer_name: str
    best_index_name: str
    ema_name: str
    training_passes_per_buffer: int  # 每轮训练把全仓平均过几遍（在线增强语义；取代旧 training_steps_per_sample）
    selfplay_workers: int  # 自博弈线程池大小（= 同时在跑的局数上限）
    selfplay_games_per_round: int  # 每轮自博弈局数；远大于线程数，靠滚动补位把尾部摊薄
    mcts_batch_leaves: int  # 每手同时挂起的 playout 数（虚拟损失批量）；批大小 ≈ 在跑局数 × 本值
    val_holdout_ratio: float  # 采样时划出验证集的比例（不入训练池）
    val_buffer_name: str
    watch_every_games: int  # 自博弈每 N 局在终端打一行战况（0=关闭）
    watch_board_every_games: int  # 每 M 局打一张 ASCII 棋盘观战（0=关闭）
    win_threshold: float


CONFIG: AppConfig = {
    'color_themes': {
        'blue': ['#2F4F4F', '#1A2F2F', '#0D1A1A', '#2F4F4F'],  # 玄武墨
        'green': ['#556B2F', '#3D4D21', '#242E14', '#556B2F'],  # 墨绿
        'red': ['#8B0000', '#660000', '#4D0000', '#8B0000'],  # 朱砂红
        'orange': ['#B8860B', '#996F09', '#7A5907', '#B8860B'],  # 琥珀金
        'grey': ['#696969', '#4F4F4F', '#363636', '#696969'],  # 铅灰
        'black': ['#1C1C1C', '#000000', '#000000', '#1C1C1C']  # 漆黑
    },
    'ChineseChess': {
        'screen_size': (600, 800),
        'grid_size': 54,
        # 表示版本 2：26 面清洁布局（删 v1 差分面、加兵力/将位），
        # 面定义见 ChineseChess.convert_to_network docstring；
        # 旧模型（498 等 v1/20 面）按各自的 feature_version 转换输入，互不混用
        'feature_version': 2,
        'tensor_shape': (26, 10, 9),
        'state_shape': (7, 10, 9),
        'augment_times': 2,
        'max_iters': 500,
        'buffer_size': 300_000,
        'avg_game_steps': 80,
        'c_puct': 5,  # 历史调参值（notes: 3.5→4→5），原为 deepMcts 硬编码
        # 大中小档位（ZZ_PROFILE=s/m/l 或 set_profile 切换；m=历史行为=根目录）：
        # 只允许覆写"规模键"；in_channels/n_actions/n_cells/feature_version 属棋规/表示层，preset 出现即报错
        'presets': {
            's': {'arch': 'resnet_v1', 'n_filters': 128, 'n_res_blocks': 6, 'n_value_hidden_channels': 64},
            'm': {},
            'l': {'arch': 'resnet_v1', 'n_res_blocks': 19, 'n_value_hidden_channels': 256},
        },
        'selfplay': {
            'tau_decay_rate': 0.96,
            'exploration_steps': 30
        },
        # 200 sims 的目标太软太噪（实测 KL(pi200‖pi800)=0.36，占可学空间的 64%）：
        # 起步抬到 400，第 50 代到 800。按绝对代爬坡 → 重启不会退回 200。
        'sims': {'start': 400, 'max': 800, 'ramp_generations': 50},
        'evaluation': {
            'tau_decay_rate': 0.9,
            'exploration_steps': 10
        },
        'default_net': {
            'arch': 'resnet_v1',
            'in_channels': 26,  # 输入通道数（v1=20；旧模型加载使用其 checkpoint 内 config）
            'n_filters': 256,  # 卷积层filter数量
            'n_cells': 10 * 9,  # 输入H*W
            'n_res_blocks': 15,  # 残差网络数量
            'n_actions': 2086,  # policy输出动作的数量
            'use_se': True,  # 是否使用SEBlock
            'n_policy_filters': 32,  # 策略头卷积层filter数量
            'n_value_filters': 32,  # 价值头卷积层filter数量
            'n_value_hidden_channels': 128,  # 价值头隐藏层fc输出通道（原 32 为历史窄口；498 守擂引擎吃自己 ckpt 的 config 不受影响）
            'feature_version': 2  # 出生时的棋盘表示版本，决定 convert_to_network 布局
        }
    },
    'Gomoku': {
        'screen_size': (600, 800),
        'grid_size': 35.2857,
        'feature_version': 1,
        'tensor_shape': (2, 15, 15),
        'state_shape': (2, 15, 15),
        'augment_times': 8,
        'max_iters': 100,
        'buffer_size': 150_000,
        'avg_game_steps': 40,
        'c_puct': 5,
        'presets': {
            's': {'arch': 'resnet_v1', 'n_filters': 128, 'n_res_blocks': 6},
            'm': {},
            'l': {'arch': 'resnet_v1', 'n_res_blocks': 15},
        },
        'selfplay': {
            'tau_decay_rate': 0.8,
            'exploration_steps': 10
        },
        'sims': {'start': 400, 'max': 800, 'ramp_generations': 50},
        'evaluation': {
            'tau_decay_rate': 0.65,
            'exploration_steps': 4
        },
        'default_net': {
            'arch': 'resnet_v1',
            'in_channels': 2,  # 输入通道数
            'n_filters': 256,  # 卷积层filter数量
            'n_cells': 15 * 15,  # 输入H*W
            'n_res_blocks': 11,  # 残差网络数量
            'n_actions': 15 * 15,  # policy输出动作的数量
            'use_se': True,  # 是否使用SEBlock
            'n_policy_filters': 32,  # 策略头卷积层filter数量
            'n_value_filters': 1,  # 价值头卷积层filter数量
            'n_value_hidden_channels': 256,  # 价值头隐藏层fc输出通道
            'feature_version': 1
        },
    },

    'data_dir': str(ROOT_DIR / 'data'),
    'dirichlet': 0.25,  # 原 0.2 从未被读取（死配置）；对齐 deepMcts 此前硬编码的 0.25 后接通
    'base_url': 'http://127.0.0.1:5000/',  # 前端跨机地址仅作展示兜底（现行前端读 localStorage），勿填内网真实地址
    'device': 'auto',  # 'auto' 自动探测 | 'cuda:0'/'mps'/'xpu'/'privateuse1'/'cpu' 显式指定; 环境变量 ZZ_DEVICE 优先
    'log_dir': str(ROOT_DIR / 'logs'),
    'runs_dir': str(ROOT_DIR / 'runs'),  # TensorBoard 根；测试/演练把此键指到临时目录，避免共享仓库路径互踩
    'buffer_name': 'buffer.pkl',
    'best_index_name': 'best_index.pkl',
    'ema_name': 'ema.pkl',
    # 每轮 fit 的步数 = buffer.size × passes ÷ batch_size。旧值等价 ≈50 次/局面曝光
    # （30 遍 × 写入增强 ×N），导致 notes 里反复出现的 loss→0 过拟合；
    # 在线增强后按"全仓平均 8 遍"配平，后续用 Val/ 曲线微调
    'training_passes_per_buffer': 8,
    # 自博弈并发与批量（2026-10-27 吞吐手术）：旧的"每轮固定局数 + 一手一请求"让
    # 批大小 = 在跑局数，尾段剩几局时推理批塌到 1~3、吞吐掉到峰值 1/6。
    # 现在：局数远大于线程数（滚动补位摊薄尾部）+ 每手 K 条 playout 同时飞
    # （在飞请求 = 局数 × K，批不再受并发局数限制）。
    # 线程数拐点（本机 8P+16E、3.14t free-threading 实测聚合 sims/s）：
    #   24 线程 47.3k / 32 线程 46.9k / 32×K16 49.7k / 40 线程 44.8k / 48 线程 38.0k
    #   → 32 附近是甜点，64 线程反而只有 35.9k（比 32 少 28%，超订后上下文切换吃掉了收益）。
    # 在飞请求 = 局数 × K = 32 × 12 = 384 > max_batch 256，够喂满引擎。
    'selfplay_workers': 32,
    'selfplay_games_per_round': 400,
    'mcts_batch_leaves': 12,
    'val_holdout_ratio': 0.02,
    'val_buffer_name': 'val_buffer.pkl',
    'watch_every_games': 20,  # 终端战况播报：每 20 局一行（0 关）
    'watch_board_every_games': 20,  # 观战棋盘：每 M 局一张（0 关；开成 20 即每 20 局配一张棋盘）
    'win_threshold': 0.52,
    'game_name': 'ChineseChess'
}


# 面板/CLI 的运行期覆盖：同一份 config 用环境变量跑不同并发/批量（训练控制台就这么透传的）
for _env, _key in (('ZZ_WORKERS', 'selfplay_workers'),
                   ('ZZ_GAMES_PER_ROUND', 'selfplay_games_per_round'),
                   ('ZZ_BATCH_LEAVES', 'mcts_batch_leaves')):
    _raw = os.environ.get(_env)
    if _raw:
        try:
            CONFIG[_key] = int(_raw)
        except ValueError:
            pass


def resolve_device(preference: str | None = None) -> str:
    """解析推理/训练设备。'auto' 优先级: cuda > xpu > mps > privateuse1(DirectML) > npu > cpu。
    显式指定的设备不可用时抛 RuntimeError（尽早报错，避免静默降速）。"""
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
                    vram = {'free': round(free / 2 ** 30, 2), 'total': round(total / 2 ** 30, 2)}
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

# 活动棋种：默认取配置，允许 ZZ_GAME 环境变量覆盖（测试/演练里跑另一棋种时
# 不必再手改 config——仓库默认是象棋，五子棋那套测试靠它把自己钉死在五子上）
game_name = (os.environ.get('ZZ_GAME') or CONFIG['game_name']).strip()
settings = CONFIG[game_name]

# ---------- 模型档位 profile（S2）：大/中/小 = 规模覆写 + 独立存储目录 ----------
# 'm' 为默认档，路径行为与历史完全一致（data/<Game>/ 根目录，老文件不搬家）；s/l 用子目录。
PROFILE: str = (os.environ.get('ZZ_PROFILE') or 'm').strip().lower()

# preset 禁止覆写的键：in_channels/feature_version 属表示层（棋盘长什么样），
# n_actions/n_cells 属棋规层（动作空间 ABI）——它们要变是"版本/游戏"的事，不是档位的事
_PRESET_FORBIDDEN = {'in_channels', 'n_actions', 'n_cells', 'feature_version'}
_NET_BASES: dict[str, dict] = {g: dict(CONFIG[g]['default_net']) for g in ('ChineseChess', 'Gomoku')}


# ---------- 命名模型配置（用户自定义档位）----------
# 内置 presets(s/m/l) 之外，用户可在 data/profiles.json 里给"规模键覆写"起名字：
#     {"mini64": {"n_filters": 64, "n_res_blocks": 4}, "wide": {"n_filters": 384, ...}}
# 名字即目录名（data/<Game>/<name>/），档位从"三个固定枚举"变成"命名配置模板"：
# 训练按它起网络，对弈/制品/引擎三身份自动跟着走。护栏不变：只许规模键，表示/棋规键禁入。
_PROFILE_SLUG = re.compile(r'^[a-z][a-z0-9_]{0,23}$')

# 自定义配置允许出现的键与取值范围（防手滑：128 通道打成 128000 会起个天文数字网络）
_PROFILE_LIMITS = {
    'arch': None,                       # None = 不校验值（注册表会拒未知架构）
    'n_filters': (16, 1024),
    'n_res_blocks': (1, 64),
    'n_policy_filters': (4, 256),
    'n_value_filters': (4, 256),
    'n_value_hidden_channels': (8, 2048),
    'use_se': None,                     # bool
}


def profiles_file() -> str:
    return os.path.join(CONFIG['data_dir'], 'profiles.json')


def _read_custom_profiles() -> dict:
    """读 data/profiles.json；文件缺失/损坏返回空（训练与对战绝不被配置文件带崩）。"""
    try:
        with open(profiles_file(), encoding='utf-8') as fp:
            data = json.load(fp)
    except (OSError, ValueError):
        return {}
    out = {}
    for name, net in (data or {}).items():
        name = str(name).strip().lower()
        if not _PROFILE_SLUG.match(name) or not isinstance(net, dict):
            continue
        if set(net) & _PRESET_FORBIDDEN:        # 带禁键的条目整体忽略（写入侧同样拒）
            continue
        out[name] = dict(net)
    return out


def validate_profile_net(net: dict) -> tuple[dict | None, str]:
    """校验规模键与取值范围，返回 (清洗后的配置, 错误信息)。"""
    clean: dict = {}
    for k, v in (net or {}).items():
        if k in _PRESET_FORBIDDEN:
            return None, f'{k} 属表示/棋规层，不能出现在模型配置里'
        if k not in _PROFILE_LIMITS:
            return None, f'未知规模键 {k}（可用：{sorted(_PROFILE_LIMITS)}）'
        lim = _PROFILE_LIMITS[k]
        if k == 'arch':
            if not isinstance(v, str) or not v:
                return None, 'arch 需为非空字符串'
        elif k == 'use_se':
            v = bool(v)
        elif lim is not None:
            if not isinstance(v, int) or isinstance(v, bool) or not (lim[0] <= v <= lim[1]):
                return None, f'{k} 需为 {lim[0]}~{lim[1]} 的整数，收到 {v!r}'
        clean[k] = v
    if not clean:
        return None, '配置为空：至少给一个规模键（n_filters / n_res_blocks …）'
    return clean, ''


def save_custom_profile(name: str, net: dict) -> tuple[bool, str]:
    name = str(name or '').strip().lower()
    if not _PROFILE_SLUG.match(name):
        return False, '名字需匹配 [a-z][a-z0-9_]{0,23}（小写字母开头）'
    builtin = set().union(*(set((CONFIG[g].get('presets') or {})) for g in _NET_BASES))
    if name in builtin:
        return False, f'{name} 是内置档位，不可覆写'
    clean, err = validate_profile_net(net)
    if err:
        return False, err
    cur = {}
    try:
        with open(profiles_file(), encoding='utf-8') as fp:
            cur = json.load(fp) or {}
    except (OSError, ValueError):
        cur = {}
    cur[name] = clean
    os.makedirs(CONFIG['data_dir'], exist_ok=True)
    tmp = profiles_file() + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fp:
        json.dump(cur, fp, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, profiles_file())
    return True, f'已保存配置 {name}'


def delete_custom_profile(name: str) -> tuple[bool, str]:
    name = str(name or '').strip().lower()
    try:
        with open(profiles_file(), encoding='utf-8') as fp:
            cur = json.load(fp) or {}
    except (OSError, ValueError):
        return False, '没有自定义配置文件'
    if name not in cur:
        return False, f'{name} 不在自定义配置里（内置档位不可删）'
    cur.pop(name)
    tmp = profiles_file() + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fp:
        json.dump(cur, fp, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, profiles_file())
    return True, f'已删除配置 {name}（其数据目录保留，删名字不删模型）'


def available_profiles(game: str | None = None) -> dict:
    """内置 presets ∪ data/profiles.json 自定义。同名以自定义为准（可覆写内置的规模，
    但不能改名字含义之外的东西——目录与模型仍按该名字走）。"""
    g = game or game_name
    out = dict(CONFIG[g].get('presets') or {})
    out.update(_read_custom_profiles())
    return out


def set_profile(profile: str) -> str:
    """切换档位：把 preset（内置 ∪ 自定义）就地合并进两个游戏的 default_net
    （保持 dict 引用不变，已按引用取默认参数的调用方同步生效）。返回生效档位。"""
    global PROFILE
    p = (profile or 'm').strip().lower()
    for g, base in _NET_BASES.items():
        presets = available_profiles(g)
        if p not in presets:
            raise ValueError(f"Unknown profile '{p}' for {g}. Available: {sorted(presets)}")
        over = dict(presets[p])
        bad = over.keys() & _PRESET_FORBIDDEN
        if bad:
            raise ValueError(f"profile '{p}' 试图覆写棋规/表示级键 {sorted(bad)}；档位只许调规模")
        dn = CONFIG[g]['default_net']
        dn.clear()
        dn.update({**base, **over})
    PROFILE = p
    return p


def get_profile() -> str:
    return PROFILE


def net_config_for(profile: str | None = None, game: str | None = None) -> dict:
    """给定档位的 default_net 规模（不改全局状态；内置 ∪ 自定义）。
    监控面板靠它显示"实际在跑的档位"——面板进程是按默认档 m 启动的，
    直接读 settings['default_net'] 会把 128x6 的 s 档训练显示成 256x15 的 m 档。"""
    g = game or CONFIG['game_name']
    p = (profile or PROFILE).strip().lower()
    presets = available_profiles(g)
    if p not in presets:
        raise ValueError(f"Unknown profile '{p}' for {g}; available {sorted(presets)}")
    return {**_NET_BASES.get(g, CONFIG[g]['default_net']), **presets[p]}


def net_dir(env_name: str, profile: str | None = None) -> str:
    """模型存储目录：m 档 = 历史根路径（零迁移）；其余 = data/<Game>/<profile>/"""
    p = (profile or PROFILE).strip().lower()
    base = os.path.join(CONFIG['data_dir'], env_name)
    return base if p == 'm' else os.path.join(base, p)


set_profile(PROFILE)  # 应用 ZZ_PROFILE（'m' 时为无操作）
