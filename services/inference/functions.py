import os

from core.utils.config import CONFIG


def get_model_name(env_name: str, model_id: int) -> str:
    """根据env_name和model_id生成一致的model name"""
    return f'{env_name}_{model_id}'


def get_checkpoint_path(env_name: str, iteration: int) -> str:
    """根据env_name和iteration生成checkpoint的存储路径"""
    return os.path.join(CONFIG['data_dir'], env_name, f'checkpoint_{iteration}.pt')
