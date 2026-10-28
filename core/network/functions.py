import glob
import os
import pickle

from core.utils.config import CONFIG, game_name, net_dir
from core.utils.types import EnvName


def feature_version_compatible(model_index: int, env_name: EnvName = game_name) -> bool:
    """检查 checkpoint 的棋盘表示版本（feature version）是否与当前 config 匹配（顶层 feature_version 字段，
    缺失时以 in_channels 兜底）。**必须按 env_name 对应的配置比较，不能用全局 settings
    （其绑定的是当前 game_name，跨游戏检查会拿错尺子）**。
    训练启动闸门用：跨版本 时应从新网络开始训练（旧 best 仍可作为评估对手，
    推理引擎按各自 in_channels 切片输入）。文件缺失/损坏按不兼容处理。"""
    from services.inference.functions import get_checkpoint_path
    env_settings = CONFIG[env_name]
    current_version = int(env_settings.get('feature_version', 1))
    path = get_checkpoint_path(env_name, model_index)
    if not os.path.exists(path):
        return False
    try:
        import torch
        ckpt = torch.load(path, map_location='cpu')
    except Exception:
        return False
    if not isinstance(ckpt, dict):
        return False
    if ckpt.get('feature_version') is not None:
        return int(ckpt['feature_version']) == current_version
    cfg = ckpt.get('config')
    if not isinstance(cfg, dict):
        return False
    if cfg.get('feature_version') is not None:
        return int(cfg['feature_version']) == current_version
    # v1 老 checkpoint 无 feature_version 字段：以面数一致性兜底
    return int(cfg.get('in_channels', -1)) == int(env_settings['default_net']['in_channels'])


def read_best_index(env_name: EnvName = game_name, profile: str | None = None) -> int:
    """读取保存的最佳模型index（按档位），找不到的话返回最新版本，仍找不到返回-1"""
    path = os.path.join(net_dir(env_name, profile), CONFIG['best_index_name'])
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    
    # 自动降级：如果没有best_index.pkl，则寻找最大的checkpoint号
    return read_latest_index(env_name, profile)


def read_latest_index(env_name: EnvName = game_name, profile: str | None = None) -> int:
    """读取保存的最新模型index（按档位），找不到的话返回-1"""
    indices = list_all_indices(env_name, profile)
    return max(indices) if indices else -1


def list_all_indices(env_name: EnvName = game_name, profile: str | None = None) -> list[int]:
    """列出某档位所有已存模型index，支持 model_x.pt 和 checkpoint_x.pt（目录不存在=空列表）"""
    pattern = os.path.join(net_dir(env_name, profile), '*.pt')
    model_files = glob.glob(pattern)
    indices = []
    for f in model_files:
        try:
            # 支持 "model_12.pt" 或 "checkpoint_12.pt"
            base = os.path.basename(f)
            index = int(base.replace('.pt', '').split("_")[-1])
            indices.append(index)
        except (ValueError, IndexError):
            continue
    return sorted(indices)


def save_best_index(best_index: int, env_name: EnvName = game_name, profile: str | None = None) -> None:
    """保存最佳model的index（按档位）"""
    folder = net_dir(env_name, profile)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, CONFIG['best_index_name'])
    with open(path, "wb") as f:
        pickle.dump(best_index, f)  # type: ignore


