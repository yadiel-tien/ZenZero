import os
import pickle
import random
import threading

from tqdm import tqdm

from env.functions import get_class
from player.ai_server import AIServer
from utils.config import CONFIG
from utils.elo import Elo
from utils.logger import get_logger
from utils.types import EnvName
from concurrent.futures import ThreadPoolExecutor, as_completed


class Arena:
    def __init__(self, env_name: EnvName) -> None:
        self.rates: dict[int, Elo] = {}
        self.logger = get_logger('arena')
        self.env_name = env_name
        self.load()
        self.stop_event = threading.Event()

    def run(self, n_games: int) -> None:
        """随机选择对手进行n局对弈"""
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(self.vs_worker) for _ in range(n_games)]
            for future in tqdm(as_completed(futures), total=n_games):
                future.result()

    def run_gauntlet(self, games_per_model=15) -> None:
        """
        【新增方法】车轮战模式：
        遍历每一个模型当'擂主'，让它去打 N 局随机对手。
        这样能保证每个模型都有足够的曝光度。
        """
        # 获取所有模型ID并排序
        all_models = sorted(list(self.rates.keys()))

        print(f"开始车轮战！共 {len(all_models)} 个模型，每人打 {games_per_model} 局...")

        for hero_idx in all_models:
            if hero_idx < 56:
                continue
            self.logger.info(f"=== 当前擂主: Model {hero_idx} ===")
            # 专门为这个擂主跑 N 局
            self.run_for_specific_hero(hero_idx, n_games=games_per_model)

            # 每测完一个擂主，显示一次排名，反馈感很强
            self.show_rank(limit=10, ascend=True)
            self.show_rank(limit=10, ascend=False)

    def run_for_specific_hero(self, hero_idx: int, n_games: int) -> None:
        """为指定擂主启动多线程对战"""
        with ThreadPoolExecutor(max_workers=10) as executor:
            # 提交任务时，把 hero_idx 传进去
            futures = [executor.submit(self.vs_worker, hero_idx) for _ in range(n_games)]
            for future in tqdm(as_completed(futures), total=n_games, desc=f"Model {hero_idx} vs All"):
                future.result()

    def vs_worker(self, hero_idx: int = None) -> None:
        """
        修改后的工作线程：
        如果传入了 hero_idx，就固定它为一方，另一方随机。
        如果没有传入（None），就还是全随机（兼容旧模式）。
        """
        if hero_idx is not None:
            # 【车轮战逻辑】
            index1 = hero_idx
            # 为擂主挑选一个合适的对手 (排除了擂主自己)
            index2 = self.get_rival_for_hero(hero_idx)
        else:
            # 【旧逻辑】全随机
            index1, index2 = self.get_random_models()

        self.versus(index1, index2)

    def get_rival_for_hero(self, hero_idx: int) -> int:
        """为擂主挑选对手"""
        # 候选人列表（排除自己）
        candidates = list(self.rates.keys())
        if hero_idx in candidates:
            candidates.remove(hero_idx)

        if not candidates:
            raise ValueError("没有足够的对手！")

        # 策略A：完全随机挑选 (简单粗暴，覆盖面广)
        # return random.choice(candidates)

        # 策略B：优先挑分数接近的 (收敛最快，推荐)
        hero_score = self.rates[hero_idx].scores
        weights = [1 / (abs(self.rates[c].scores - hero_score) + 1e-6) for c in candidates]
        return random.choices(candidates, weights=weights, k=1)[0]

    def get_random_models(self) -> tuple[int, int]:
        """选择两个model id 以供比赛"""
        if len(self.rates) < 2:
            raise ValueError(f'No enough candidates,please check {self.list_path}.')
        model_indices = list(self.rates.keys())
        # 根据比赛次数多少随机，次数越多，选中概率越低
        weights = [1 / (self.rates[i].n_games_played + 1) for i in model_indices]
        chosen_index = random.choices(model_indices, weights=weights, k=1)[0]
        model_indices.remove(chosen_index)
        # 选择得分相近的对手，权重为1/abs(a-b)
        diff_weight = [1 / (
                abs(self.rates[idx].scores - self.rates[chosen_index].scores) + 1e-6
        ) for idx in model_indices]
        chosen_rival_index = random.choices(model_indices, weights=diff_weight, k=1)[0]
        return chosen_index, chosen_rival_index

    def versus(self, index1: int, index2: int) -> None:
        """index1与index2进行一局对弈，根据胜负结果调整ELO"""
        if index1 in self.rates and index2 in self.rates:
            self.logger.info(f'{index1} VS {index2}: Pre_game:{self.rates[index1]}  {self.rates[index2]}')
            env = get_class(self.env_name)()
            # 对弈
            with AIServer(self.env_name, index1, n_simulation=300, verbose=False) as p1, AIServer(self.env_name, index2,
                                                                                                  n_simulation=300,
                                                                                                  verbose=False) as p2:
                outcome = env.random_order_play((p1, p2), silent=True)
            # 更新score
            e1, e2 = self.rates[index1], self.rates[index2]
            result = ''
            if outcome == 0:
                e1.defeat(e2)
                result = 'win'
            elif outcome == 1:
                e2.defeat(e1)
                result = 'lose'
            elif outcome == -1:
                e1.draw(e2)
                result = 'draw'

            self.save()
            self.logger.info(f'{index1} VS {index2}: {result}')
            self.logger.info(f'----{e1}')
            self.logger.info(f'----{e2}')
        else:
            self.logger.info(f'{index1} or {index2} not found.')

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.list_path), exist_ok=True)
        with open(self.rates_path, "wb") as f:
            pickle.dump(self.rates, f, protocol=4)  # type: ignore
            self.logger.info(f'Arena scores have been saved at {self.rates_path}.')

    def load(self) -> None:
        # 读取rates
        if os.path.exists(self.rates_path):
            with open(self.rates_path, "rb") as f:
                self.rates = pickle.load(f)
            self.logger.info(f'loaded rates from {self.rates_path}')
        else:
            self.logger.info(f'Failed to load data from "{self.rates_path}",file not exist.')
        # 根据
        if os.path.exists(self.list_path):
            self.logger.info(f'Adjusting candidate list from {self.list_path}.')
            with open(self.list_path, "r") as f:
                line = f.readline().strip()
                lst_set = set(int(x) for x in line.split(',') if x.strip())
                for i in lst_set:
                    if i not in self.rates:
                        self.rates[i] = Elo(i)
                        self.logger.info(f'index:{i} joined arena.')
                for i in set(self.rates) - lst_set:
                    self.rates.pop(i)
                    self.logger.info(f'index:{i} was deleted from arena.')

    @property
    def rates_path(self) -> str:
        return os.path.join(CONFIG['data_dir'], self.env_name, 'rates', 'rates.pkl')

    @property
    def list_path(self) -> str:
        return os.path.join(CONFIG['data_dir'], self.env_name, 'rates', 'candidates.txt')

    def show_rank(self, limit: int = 10, ascend=True) -> None:
        """从高到低显示所有排名情况"""
        if not self.rates:
            print('No candidate rates to show.')
            return
        if ascend:
            print('=' * 15 + f' Top {limit} ' + '=' * 15)
        else:
            print('=' * 15 + f' Bottom {limit} ' + '=' * 15)
        for idx, (_, v) in enumerate(sorted(self.rates.items(), key=lambda x: x[1].scores, reverse=ascend)):
            print(f'{idx + 1:>3} {v}')
            if idx + 1 >= limit:
                break

    def scheduled_show(self, interval: int) -> None:
        while not self.stop_event.wait(interval):
            self.show_rank(10, True)
            self.show_rank(10, False)
