"""训练进程的运行期心跳：给监控面板一份"现在就怎样"的权威数据。

为什么需要：面板原先只能从**日志尾部**猜状态——phase 靠最后一行文本、进度靠数
"Game end" 条数、buffer 只能取上一轮的 perf 记录。于是整轮自博弈期间 buffer 数字
一动不动（看着像死数据），而且卡住时面板也只能干说一句 stalled。

心跳是训练进程自己每 2 秒（或每局）写一次的小 JSON：
    logs/runtime.json
字段都是训练侧的真实内存值（buffer.size/capacity、已完局数、阶段、本轮 iteration），
写失败一律静默——绝不能因为监控影响训练。

并发：只在主线程（self_play 的收结果循环、run 的阶段切换）写，内部用一把锁 + 原子替换，
读侧（面板）永远看到完整文件。
"""
from __future__ import annotations

import json
import os
import threading
import time

from core.utils.config import CONFIG

# 本进程支持的控制能力：runner 会先看心跳里的 caps 再决定能不能发 SIGUSR1。
# 旧进程没有这条 → 面板会拒绝"等本轮结束"（SIGUSR1 的默认动作是**杀掉进程**，不能盲发）。
CAPS = ['heartbeat', 'after_round']

_LOCK = threading.Lock()
_LAST_WRITE = 0.0
MIN_INTERVAL = 2.0          # 秒；同一阶段内的高频更新按这个节流


def path() -> str:
    return os.path.join(CONFIG['log_dir'], 'runtime.json')


def write(force: bool = False, **fields) -> None:
    """写一次心跳。字段命名稳定，面板按名字取值。"""
    global _LAST_WRITE
    now = time.time()
    if not force and now - _LAST_WRITE < MIN_INTERVAL:
        return
    with _LOCK:
        if not force and now - _LAST_WRITE < MIN_INTERVAL:
            return
        _LAST_WRITE = now
        payload = {'ts': now, 'iso': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'pid': os.getpid(), 'caps': list(CAPS), **fields}
        tmp = path() + '.tmp'
        try:
            os.makedirs(os.path.dirname(path()), exist_ok=True)
            with open(tmp, 'w', encoding='utf-8') as fp:
                json.dump(payload, fp, ensure_ascii=False)
            os.replace(tmp, path())
        except OSError:
            pass


def read() -> dict | None:
    try:
        with open(path(), encoding='utf-8') as fp:
            data = json.load(fp)
        data['age_s'] = round(time.time() - float(data.get('ts') or 0), 1)
        return data
    except (OSError, ValueError, TypeError):
        return None


def clear() -> None:
    try:
        os.remove(path())
    except OSError:
        pass
