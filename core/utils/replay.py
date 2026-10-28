import json
import os.path
import pickle
import time

import numpy as np
from numpy.typing import NDArray

from core.env.functions import get_class
from .config import game_name, CONFIG, settings, net_dir
from .types import EnvName


class ReplayBuffer:
    def __init__(self, capacity: int, batch_size: int, game: EnvName = game_name,
                 file_name: str | None = None) -> None:
        # file_name 允许同档位再分池（如验证集 val_buffer.pkl）
        self.file_name = file_name or CONFIG['buffer_name']
        # 形状从 **game 参数自身** 推导。旧版绑模块级 settings（=当前进程 game_name），
        # 跨游戏实例化会静默写错形状——测试顺手抓出的潜伏 bug，与 feature_version
        # 检查器"拿错尺子"同族，此类"隐式读全局"要连根拔
        game_settings = CONFIG[game]
        self.state_buffer = np.zeros((capacity,) + game_settings['tensor_shape'], dtype=np.float32)
        self.policy_buffer = np.zeros((capacity, game_settings['default_net']['n_actions']), dtype=np.float32)
        self.value_buffer = np.zeros((capacity,), dtype=np.float32)
        self.batch_size = batch_size
        self.pointer = 0
        self.size = 0
        self.game = game
        self.env = get_class(game)
        self.capacity = capacity
        # 在线增强群成员（get_batch 采样时随机施加，写入只存原始样本）
        self.sym_ids = tuple(self.env.symmetry_indices())

    def _path(self, env_name: EnvName) -> str:
        """仓文件落点：CONFIG['buffer_dir'] 指定目录（命名仓实体，主/验同目录成对）
        优先于历史原位 net_dir(env)。不设置时行为与旧版逐字节一致。"""
        return os.path.join(CONFIG.get('buffer_dir') or net_dir(env_name), self.file_name)

    def add(self, data: tuple[NDArray, NDArray, float]) -> None:
        """只存原始样本一份。
        旧版在此处写入全部对称复制（象棋×2/五子×8），buffer 容量被冗余稀释、
        近重复样本在均匀采样下互相放大曝光导致过拟合——增强改到 get_batch 在线做
        （transform_batch 与 augment_data 有逐位对拍测试），
        同样的格数装 8 倍的不同局面。"""
        state, pi, q = data[:3]
        self.append(state, pi, q)

    def append(self, state: NDArray, pi: NDArray, q: float) -> None:
        self.state_buffer[self.pointer] = state
        self.policy_buffer[self.pointer] = pi
        self.value_buffer[self.pointer] = q
        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_batch(self, augment: bool = True) -> tuple[NDArray, NDArray, NDArray]:
        """随机采样一个 batch；augment=True 时对每条样本随机施加一个对称变换
        （每次都是新视角——同一样本多次入批也不再是重复样本）。
        验证集评估请传 augment=False 保证 loss 可比。
            :return (state,pi,q)
                - state:[B,C,H,W]
                - pi:[B,n_actions]
                - q:[B]"""
        if self.size < self.batch_size:
            raise ValueError(f'No enough data! Current size is {self.size},required size is {self.batch_size}.')
        indices = np.random.choice(self.size, size=self.batch_size, replace=False)
        states = self.state_buffer[indices].copy()
        probs = self.policy_buffer[indices].copy()
        values = self.value_buffer[indices].copy()
        if augment and len(self.sym_ids) > 1:
            picks = np.random.randint(len(self.sym_ids), size=self.batch_size)
            out_s = np.empty_like(states)
            out_p = np.empty_like(probs)
            for g, sym in enumerate(self.sym_ids):
                m = picks == g
                if not m.any():
                    continue
                # transform_batch 返回新数组/视图，赋值给 fancy 目标即拷贝
                out_s[m], out_p[m] = self.env.transform_batch(states[m], probs[m], sym)
            states, probs = out_s, out_p
        return states, probs, values

    def save(self, env_name: EnvName = game_name) -> None:
        """保存buffer"""
        path = self._path(env_name)
        data = {
            "states": self.state_buffer[:self.size],
            "pis": self.policy_buffer[:self.size],
            "values": self.value_buffer[:self.size],
            "size": self.size,
            "pointer": self.pointer,
            "game": self.game,
            # 棋盘表示版本：load 端校验，跨版本 数据静默混入比报错严重
            "feature_version": int(CONFIG[self.game].get('feature_version', 1))
        }
        # 档位子目录（data/<Game>/s/ 等）首次写入前不存在——与 m 档根目录不同，必须自建，
        # 否则小模型档第一次 save 就 FileNotFoundError（实测踩中）
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            # 指定protocol=4，否则可能出现load时卡死
            pickle.dump(data, f, protocol=4)  # type: ignore
        # 侧车元数据：监控面板要显示"已用/总容量"，而 buffer.pkl 可能是几 GB，
        # 为了一个数字去反序列化它不划算（也吃不消并发）。这里落一个小 JSON 供只读读取。
        # 写失败绝不能影响训练（磁盘满/权限）→ 静默。
        try:
            stats = {'size': int(self.size), 'capacity': int(self.capacity),
                     'pointer': int(self.pointer), 'game': self.game,
                     'batch_size': int(self.batch_size),
                     'feature_version': int(CONFIG[self.game].get('feature_version', 1)),
                     'saved_at': time.strftime('%Y-%m-%dT%H:%M:%S')}
            tmp = path + '.stats.json.tmp'
            with open(tmp, 'w', encoding='utf-8') as fp:
                json.dump(stats, fp, ensure_ascii=False)
            os.replace(tmp, path + '.stats.json')
        except OSError:
            pass

    def load(self, env_name: EnvName = game_name) -> None:
        """加载buffer，支持修改容量"""
        path = self._path(env_name)
        if not os.path.exists(path):
            print(f"Buffer not found at '{path}', current length: {self.size}")
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
            if data['game'] != self.game:
                raise ValueError('Game mismatch! Replay buffer loading Failed!')
            cur_version = int(CONFIG[self.game].get('feature_version', 1))
            if int(data.get('feature_version', 1)) != cur_version:
                # 跨版本 旧数据不兼容（输入面不同）：保持空 buffer，本次运行重新采集覆盖
                print(f"Buffer at '{path}' is from feature version "
                      f"{data.get('feature_version', 1)}, current is {cur_version}. Discarded (empty buffer).")
                return

            count = min(data['size'], self.capacity)
            src_capacity = data['states'].shape[0]
            pointer = data['pointer']
            if data['size'] < src_capacity:  # 尚未存满
                indices = np.arange(data['size'] - count, data['size'])
            else:
                # 提取逻辑上的最后 count 条数据。
                # 示例：容量10，当前指针2 (最新数据在1,0,9,8...)，需要取5条。
                # 1. 逻辑起点：pointer(2) - count(5) = -3
                # 2. 生成序列：[-3, -2, -1, 0, 1]
                # 3. 对10取模：[7, 8, 9, 0, 1] -> 正确对应了环形存储的物理索引
                indices = (np.arange(count) + (pointer - count)) % src_capacity
            self.size = count
            self.pointer = count % self.capacity

            self.state_buffer[:count] = data['states'][indices]
            self.policy_buffer[:count] = data['pis'][indices]
            self.value_buffer[:count] = data['values'][indices]
            print(f"Replay buffer loaded successfully. Current length: {self.size},Capacity: {self.capacity}")

    def clear(self) -> None:
        self.size = 0
        self.pointer = 0

    def __len__(self):
        return self.size
