import os

from core.utils.config import CONFIG, net_dir, get_profile


def get_model_name(env_name: str, model_id: int, profile: str | None = None) -> str:
    """引擎/模型名：m 档保持历史格式（向后兼容 hub 键与日志），其余档位带前缀"""
    p = (profile or get_profile()).strip().lower()
    return f'{env_name}_{model_id}' if p == 'm' else f'{env_name}_{p}_{model_id}'


def get_checkpoint_path(env_name: str, iteration: int, profile: str | None = None) -> str:
    """checkpoint 存储路径：profile 缺省取进程档位；m 档 = 历史根路径（老文件零迁移）"""
    return os.path.join(net_dir(env_name, profile), f'checkpoint_{iteration}.pt')
