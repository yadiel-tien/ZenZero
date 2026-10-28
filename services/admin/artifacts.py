"""制品管理：checkpoint / buffer / best_index / TensorBoard runs / 日志。

两条底线：
1. **删除只能落在白名单路径内**（checkpoint_<int>.pt、runs/<run>、logs/<name>/latest.log*），
   并且必须带明确的 confirm 串——缓冲区和 checkpoint 是几小时的算力，不该被误点掉；
2. best_index 是晋升门（selfplay 拿它对擂），回滚/推广都写成显式动作，不做隐式"最新即最佳"。
"""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import numpy as np

from core.network.functions import list_all_indices, read_best_index, save_best_index, read_latest_index
from core.utils.config import CONFIG, net_dir

ROOT = Path(__file__).resolve().parents[2]
_CKPT_RE = re.compile(r'^checkpoint_(\d+)\.pt$')
_RUN_ALLOW = re.compile(r'^[A-Za-z0-9_.\-]+$')


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _dir_size(path: Path, limit: int = 4000) -> int:
    total = 0
    for i, (dirpath, _dirs, files) in enumerate(os.walk(path)):
        if i > limit:
            break
        for name in files:
            total += _size(Path(dirpath) / name)
    return total


def _artifact(kind: str, path: Path, extra: dict | None = None) -> dict:
    st = path.stat() if path.exists() else None
    nbytes = _size(path) if path.is_file() else _dir_size(path)
    item = {'kind': kind, 'name': path.name, 'path': str(path),
            'size': nbytes, 'file_size': nbytes,     # buffer 的 size 会被改成"样本数"，字节数留这里
            'mtime': st.st_mtime if st else None, 'is_dir': path.is_dir()}
    if extra:
        item.update(extra)
    return item


def _bytes_per_sample(game: str) -> int:
    """一条样本在 pickle 里占的净字节数（states=float32 面 + pis=float32 动作 + value=float32）"""
    gs = CONFIG[game]
    planes = int(np.prod(gs['tensor_shape']))
    n_actions = int(gs['default_net']['n_actions'])
    return planes * 4 + n_actions * 4 + 4


def _buffer_stats(path: Path, game: str, label: str, file_name: str) -> dict:
    """已用/总容量。

    容量来源：优先侧车（ReplayBuffer.save 写的 .stats.json），其次按配置推断
    （训练池 = buffer_size；验证池 = max(8000, buffer_size//16)，与 SelfPlayManager 一致）。
    已用样本数：侧车为准；没有侧车就按文件大小反算（≈），并在 UI 上标出来——
    绝不为了一个数字去反序列化几 GB 的 pickle。
    """
    cfg = CONFIG[game]
    if file_name == CONFIG['val_buffer_name']:
        capacity, source = max(8000, int(cfg['buffer_size']) // 16), 'config'
    else:
        capacity, source = int(cfg['buffer_size']), 'config'
    out = {'label': label, 'capacity': capacity, 'capacity_source': source,
           'size': None, 'estimated': False}
    sidecar = Path(str(path) + '.stats.json')
    if sidecar.exists():
        try:
            st = json.loads(sidecar.read_text(encoding='utf-8'))
            out.update({'size': int(st['size']), 'capacity': int(st.get('capacity') or capacity),
                        'capacity_source': 'sidecar', 'saved_at': st.get('saved_at'),
                        'pointer': st.get('pointer')})
            return out
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            pass
    per = _bytes_per_sample(game)
    try:
        est = max(0, (path.stat().st_size - 512) // per)     # 512 留 pickle 框架开销
    except OSError:
        est = 0
    out.update({'size': est, 'estimated': True})
    return out


def list_all(game: str, profile: str) -> dict:
    data_dir = Path(net_dir(game, profile))
    checkpoints = []
    if data_dir.is_dir():
        for entry in data_dir.iterdir():
            m = _CKPT_RE.match(entry.name)
            if m and entry.is_file():
                checkpoints.append(_artifact('checkpoint', entry, {'iteration': int(m.group(1))}))
    checkpoints.sort(key=lambda x: x['iteration'])

    best = read_best_index(game, profile)
    latest = read_latest_index(game, profile)
    buffers = []
    for fname, label in ((CONFIG['buffer_name'], '训练池'), (CONFIG['val_buffer_name'], '验证池')):
        p = data_dir / fname
        if p.exists():
            buffers.append(_artifact('buffer', p, _buffer_stats(p, game, label, fname)))

    runs = []
    runs_dir = Path(CONFIG['runs_dir'])
    if runs_dir.is_dir():
        for entry in runs_dir.iterdir():
            if entry.is_dir():
                runs.append(_artifact('run', entry))

    logs = []
    log_dir = Path(CONFIG['log_dir'])
    _LOG_F = re.compile(r'^(latest\.log(\.\d{4}-\d{2}-\d{2})?|console\.log)$')
    if log_dir.is_dir():
        for entry in sorted(log_dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith(('_', '.')):
                continue
            for f in sorted(entry.iterdir()):
                try:
                    if f.is_file() and _LOG_F.match(f.name) and f.stat().st_size > 0:
                        a = _artifact('log', f)
                        a['module'] = entry.name      # 前端分组用
                        logs.append(a)
                except OSError:
                    continue
    runs.sort(key=lambda x: x['mtime'] or 0, reverse=True)
    logs.sort(key=lambda x: x['mtime'] or 0, reverse=True)
    return {
        'game': game, 'profile': profile, 'net_dir': str(data_dir),
        'best_index': best, 'latest_index': latest,
        'checkpoints': checkpoints, 'buffers': buffers, 'runs': runs, 'logs': logs,
        'totals': {'data': _dir_size(Path(CONFIG['data_dir'])),
                   'runs': _dir_size(runs_dir)},
        'disk_free_gb': round(shutil.disk_usage(str(ROOT)).free / 2 ** 30, 1),
    }


def delete(kind: str, path: str, confirm: str, game: str, profile: str) -> dict:
    """白名单删除。confirm 必须是 'delete'（buffer 另需 'delete-buffer'）。"""
    target = Path(path).resolve()
    data_dir = Path(net_dir(game, profile)).resolve()
    runs_dir = Path(CONFIG['runs_dir']).resolve()
    log_dir = Path(CONFIG['log_dir']).resolve()

    if kind == 'checkpoint':
        m = _CKPT_RE.match(target.name)
        if target.parent != data_dir or not m:
            return {'ok': False, 'error': '只允许删 data/<game>/<profile>/checkpoint_<int>.pt'}
        if confirm != 'delete':
            return {'ok': False, 'error': "需要 confirm='delete'"}
        if int(m.group(1)) == read_best_index(game, profile):
            return {'ok': False, 'error': '这是当前 best_index 的 checkpoint；要删先 promote 别的代'}
    elif kind == 'run':
        if target.parent != runs_dir or not _RUN_ALLOW.match(target.name):
            return {'ok': False, 'error': '只允许删 runs/<run> 目录'}
        if confirm != 'delete':
            return {'ok': False, 'error': "需要 confirm='delete'"}
    elif kind == 'log':
        in_log_dir = target.parent.parent == log_dir
        allowed_name = bool(re.match(r'^(latest\.log(\.\d{4}-\d{2}-\d{2})?|console\.log)$', target.name))
        if not in_log_dir or not allowed_name:
            return {'ok': False, 'error': '只允许删日志台账文件（latest.log 及轮转、console.log）'}
        if confirm != 'delete':
            return {'ok': False, 'error': "需要 confirm='delete'"}
    elif kind == 'buffer':
        if target.parent != data_dir or target.name not in (CONFIG['buffer_name'], CONFIG['val_buffer_name']):
            return {'ok': False, 'error': '只允许删本档位的 buffer 文件'}
        if confirm != 'delete-buffer':
            return {'ok': False, 'error': "清空缓冲区需要 confirm='delete-buffer'（训练池/验证池会重新积累）"}
    else:
        return {'ok': False, 'error': f'未知类型: {kind}'}

    if not target.exists():
        return {'ok': False, 'error': '目标不存在'}
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    return {'ok': True, 'deleted': str(target)}


def set_best(iteration: int, game: str, profile: str) -> dict:
    """把 best_index 设为指定代（晋升门）。必须该 checkpoint 真实存在。"""
    ckpt = Path(net_dir(game, profile)) / f'checkpoint_{int(iteration)}.pt'
    if not ckpt.exists():
        return {'ok': False, 'error': f'checkpoint_{iteration}.pt 不存在，不能设为 best'}
    save_best_index(int(iteration), game, profile)
    return {'ok': True, 'best_index': read_best_index(game, profile),
            'checkpoints': list_all_indices(game, profile)[-5:]}
