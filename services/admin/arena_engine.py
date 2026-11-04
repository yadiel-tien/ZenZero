"""竞技场执行引擎：admin 服务内的进程级单例（线程跑批 + 内存心跳）。

设计：
- 与训练 runner 不同，arena 直接在服务进程起线程——它复用服务已有的推理 hub（同进程
  AIServer 同代自动共享引擎），没必要再付一次 CUDA 初始化 + 模型加载。
- 心跳双写：内存（本进程 API 直读，零延迟）+ logs/_state/arena.json（服务重启后的
  事后取证，与 trainer.json 同族）。
- 同一时刻只允许一场战役；训练在跑时拒绝启动（GPU 抢占策略，除非显式 allow_busy）。
"""
import json
import os
import threading
import time

from core.utils.config import CONFIG
from core.utils.arena import Arena

_STATE_FILE = os.path.join(CONFIG.get('log_dir', 'logs'), '_state', 'arena.json')


class ArenaEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._arena: Arena | None = None
        self._state: dict = {'running': False}

    # ---------- 控制 ----------
    def start(self, game: str, mode: str = 'smart', keys: list[str] | None = None,
              games: int = 30, sims: int = 300, tables: int = 5,
              pair: list[str] | None = None) -> dict:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return {'ok': False, 'error': '已有战役在打，先停止'}
            arena = self._arena if self._arena and self._arena.env_name == game else Arena(game)
            self._arena = arena
            roster = arena.sync_roster(keys if keys else None)
            if len(roster) < 2:
                return {'ok': False, 'error': f'名册只有 {roster}，至少 2 名选手'}
            pr = None
            if mode == 'duel':
                if not pair or len(pair) != 2 or pair[0] not in roster or pair[1] not in roster:
                    return {'ok': False, 'error': '自定义对决需在名册中选两名选手'}
                pr = (pair[0], pair[1])

            def beat(st: dict) -> None:
                st.update({'started_ts': started, 'mode': mode, 'sims': sims,
                           'tables': tables, 'roster': len(roster)})
                self._state = st
                try:
                    os.makedirs(os.path.dirname(_STATE_FILE), exist_ok=True)
                    tmp = _STATE_FILE + '.tmp'
                    with open(tmp, 'w', encoding='utf-8') as f:
                        json.dump(st, f, ensure_ascii=False)
                    os.replace(tmp, _STATE_FILE)
                except OSError:
                    pass

            started = time.time()

            def _run():
                try:
                    arena.run_batch(games=int(games), n_sim=int(sims), tables=int(tables),
                                    mode=mode, pair=pr, on_state=beat)
                except Exception as exc:                    # noqa: BLE001 线程兜底留痕
                    self._state = {**self._state, 'running': False, 'error': str(exc)[:200]}

            beat({'running': True, 'mode': mode, 'pair': list(pr) if pr else None,
                  'done': 0, 'total': games, 'tally': [0, 0, 0], 'feed': [], 'gpm': 0})
            self._thread = threading.Thread(target=_run, name='arena-batch', daemon=True)
            self._thread.start()
            return {'ok': True, 'roster': roster, 'games': games}

    def stop(self) -> dict:
        if self._arena:
            self._arena.stop_event.set()
        return {'ok': True, 'note': '已请求停止：进行中的桌打完本批即收'}

    # ---------- 查询 ----------
    def state(self) -> dict:
        st = dict(self._state)
        st['age_s'] = round(time.time() - (st.get('started_ts') or 0), 1) if st.get('running') else None
        return st

    def ranks(self, game: str) -> dict:
        arena = self._arena if (self._arena and self._arena.env_name == game) else Arena(game)
        if not arena.rates:
            arena.sync_roster(None)
        players = arena.ranks()
        rival: dict[str, dict] = {}
        for e_key, recs in ((p['id'], p['records']) for p in players):
            for rk, wld in recs.items():
                rival.setdefault(e_key, {})[rk] = wld
        return {'game': game, 'players': players, 'rival': rival,
                'candidates': arena.discover(), 'running': bool(self._thread and self._thread.is_alive())}


_engine: ArenaEngine | None = None
_engine_lock = threading.Lock()


def get_engine() -> ArenaEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = ArenaEngine()
        return _engine
