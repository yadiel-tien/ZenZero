"""训练控制台的后端路由（Flask Blueprint，挂在 5000 端口的棋类服务上）。

页面：GET /admin
API ：/api/admin/*

写操作（启动/停止/删除/回滚 best）默认无鉴权——本机局域网自用；如果暴露到更大范围，
设环境变量 ZZ_ADMIN_TOKEN=xxx，之后所有写请求需带 `X-ZZ-Token: xxx`。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from flask import Blueprint, jsonify, request, send_from_directory

from core.utils.config import CONFIG
from services.monitor.collect import Monitor

from . import artifacts
from .runner import TrainingRunner

WEB_DIR = Path(__file__).resolve().parents[2] / 'apps' / 'web'
_TOKEN = os.environ.get('ZZ_ADMIN_TOKEN', '').strip()

bp = Blueprint('admin', __name__)
_monitor: Monitor | None = None
_runner: TrainingRunner | None = None


def _m() -> Monitor:
    global _monitor
    if _monitor is None:
        _monitor = Monitor()
    return _monitor


_tasks = None


def _t():
    global _tasks
    if _tasks is None:
        from services.admin.tasks import TaskRunner
        _tasks = TaskRunner()
    return _tasks


def _r() -> TrainingRunner:
    global _runner
    if _runner is None:
        _runner = TrainingRunner()
    return _runner


def _guard():
    """写操作鉴权（未设 token 时放行）与 JSON 解析。

    返回 (response, None) 表示该直接回给客户端（含状态码），(None, data) 才继续。
    """
    if _TOKEN and request.headers.get('X-ZZ-Token', '') != _TOKEN:
        resp = jsonify({'ok': False, 'error': 'token 不对（X-ZZ-Token）'})
        resp.status_code = 401
        return resp, None
    return None, (request.get_json(silent=True) or {})


def _target_game_profile(data: dict) -> tuple[str, str]:
    """请求未指定棋种/档位时，默认落到"最近真有 checkpoint 的那个"，而不是服务进程自己的档位
    （棋类服务常跑 m 档，训练却在 s 档——不这么兜会让面板一打开就看到 498 代的老制品）。"""
    sug = _suggested_target()
    game = str(data.get('game') or request.args.get('game') or sug['game'])
    profile = str(data.get('profile') or request.args.get('profile') or sug['profile'])
    return game, profile


# ---------------------------------------------------------------- 页面与只读
@bp.get('/admin')
def admin_page():
    return send_from_directory(WEB_DIR, 'admin.html')


@bp.get('/api/admin/snapshot')
def snapshot():
    snap = _m().snapshot()
    snap['trainer'] = _r().status()
    return jsonify(snap)


@bp.get('/api/admin/series')
def series():
    return jsonify(_m().series())


def _suggested_target() -> dict:
    """面板默认棋种/档位：按"哪个档位最近有过 checkpoint"猜，比服务进程自己的 profile 靠谱
    （棋类服务常跑 m 档，而训练可能一直在 s 档上）。"""
    from pathlib import Path
    from core.utils.config import net_dir
    best = None
    for game in ('ChineseChess', 'Gomoku'):
        for profile in ('s', 'm', 'l'):
            try:
                d = Path(net_dir(game, profile))
                files = list(d.glob('checkpoint_*.pt')) if d.is_dir() else []
            except OSError:
                continue
            for f in files:
                m = f.stat().st_mtime
                if best is None or m > best[0]:
                    best = (m, game, profile)
    if best:
        return {'game': best[1], 'profile': best[2]}
    return {'game': CONFIG['game_name'], 'profile': os.environ.get('ZZ_PROFILE', 'm')}


@bp.get('/api/admin/env')
def env_info():
    """表单默认值：当前生效的档位/旋钮（含进程环境里已有的覆盖）。"""
    return jsonify({
        'suggested': _suggested_target(),
        'game': CONFIG['game_name'], 'profile': os.environ.get('ZZ_PROFILE', 'm'),
        'device': CONFIG['device'], 'workers': CONFIG.get('selfplay_workers'),
        'games_per_round': CONFIG.get('selfplay_games_per_round'),
        'batch_leaves': CONFIG.get('mcts_batch_leaves'),
        'max_batch': os.environ.get('ZZ_MAX_BATCH', ''),
        'net': CONFIG[CONFIG['game_name']]['default_net'],
        'knobs': {k: CONFIG.get(k) for k in
                  ('selfplay_workers', 'selfplay_games_per_round', 'mcts_batch_leaves')},
    })


# ---------------------------------------------------------------- 训练进程
@bp.get('/api/admin/train/status')
def train_status():
    return jsonify(_r().status())


@bp.post('/api/admin/train/start')
def train_start():
    err, data = _guard()
    if err:
        return err
    return jsonify(_r().start(data))


@bp.post('/api/admin/train/stop')
def train_stop():
    err, data = _guard()
    if err:
        return err
    # kill=立即结束(SIGKILL) / interrupt=等本轮收尾(SIGINT) / finish_round=跑完整轮(SIGUSR1)
    mode = str(data.get('mode') or 'interrupt')
    if mode not in ('kill', 'interrupt', 'immediate', 'finish_round', 'after_round'):
        return jsonify({'ok': False,
                        'error': "mode 只能是 kill / interrupt / finish_round"}), 400
    # 立即结束要等"手上这波局"打完（32 局并发，最长一局可能几十秒），默认给 90s
    return jsonify(_r().stop(mode=mode, grace=float(data.get('grace') or 150.0)))


# ---------------------------------------------------------------- 制品
@bp.get('/api/admin/artifacts')
def artifacts_list():
    game, profile = _target_game_profile({})
    return jsonify(artifacts.list_all(game, profile))


@bp.post('/api/admin/artifacts/delete')
def artifacts_delete():
    err, data = _guard()
    if err:
        return err
    game, profile = _target_game_profile(data)
    return jsonify(artifacts.delete(str(data.get('kind') or ''), str(data.get('path') or ''),
                                    str(data.get('confirm') or ''), game, profile))


@bp.post('/api/admin/best')
def set_best():
    err, data = _guard()
    if err:
        return err
    game, profile = _target_game_profile(data)
    try:
        iteration = int(data.get('iteration'))
    except (TypeError, ValueError):
        return jsonify({'ok': False, 'error': 'iteration 必须是整数'}), 400
    return jsonify(artifacts.set_best(iteration, game, profile))


# ---------------------------------------------------------------- 日志
@bp.get('/api/admin/logs')
def logs():
    name = request.args.get('name', 'selfplay')
    lines = min(int(request.args.get('lines', 200)), 2000)
    grep = request.args.get('grep', '').strip()
    safe = ''.join(ch for ch in name if ch.isalnum() or ch in '._-')
    path = Path(CONFIG['log_dir']) / safe / 'latest.log'
    if not path.exists():
        return jsonify({'ok': False, 'error': f'没有这个日志: {safe}', 'lines': []})
    try:
        size = path.stat().st_size
        with open(path, 'rb') as fp:
            if size > 512_000:
                fp.seek(-512_000, os.SEEK_END)
                fp.readline()
            rows = fp.read().decode('utf-8', 'replace').splitlines()
    except OSError as e:
        return jsonify({'ok': False, 'error': str(e), 'lines': []})
    if grep:
        rows = [r for r in rows if grep in r]
    return jsonify({'ok': True, 'name': safe, 'path': str(path), 'size': size,
                    'lines': rows[-lines:]})


@bp.get('/api/admin/logs/list')
def logs_list():
    base = Path(CONFIG['log_dir'])
    out = []
    if base.is_dir():
        for entry in sorted(base.iterdir()):
            f = entry / 'latest.log'
            if f.exists():
                st = f.stat()
                out.append({'name': entry.name, 'size': st.st_size, 'mtime': st.st_mtime})
    out.sort(key=lambda x: x['mtime'], reverse=True)
    return jsonify({'logs': out})


# ---------------------------------------------------------------- 一键体检
# ---------------------------------------------------------------- 任务（scripts/ 其余入口）
@bp.get('/api/admin/tasks')
def tasks_list():
    return jsonify(_t().status())


@bp.post('/api/admin/tasks/start')
def tasks_start():
    err, data = _guard()
    if err:
        return err
    name = str(data.get('name') or '')
    params = data.get('params') if isinstance(data.get('params'), dict) else {}
    res = _t().start(name, params)
    return jsonify(res), (200 if res.get('ok') else 400)


@bp.post('/api/admin/tasks/stop')
def tasks_stop():
    err, data = _guard()
    if err:
        return err
    res = _t().stop(str(data.get('name') or ''), grace=float(data.get('grace') or 15.0))
    return jsonify(res), (200 if res.get('ok') else 400)


@bp.get('/api/admin/tasks/log')
def tasks_log():
    tail = min(1000, max(20, int(request.args.get('tail') or 200)))
    return jsonify(_t().log(request.args.get('name') or '', tail))


@bp.get('/api/admin/health')
def health():
    """训练相关的一次性体检：文件/进程/最近 perf/崩溃记录。"""
    snap = _m().snapshot()
    rounds = snap.get('rounds') or []
    last = rounds[-1] if rounds else {}
    from core.utils.version import full as _full
    return jsonify({
        'version': _full(),
        'trainer_running': _r().status()['running'],
        'log_age_s': (snap.get('live') or {}).get('age_s'),
        'phase': (snap.get('live') or {}).get('phase'),
        'last_round': {k: last.get(k) for k in ('round', 'iteration', 'round_wall_s', 'buffer_size')},
        'last_perf': (last.get('selfplay') or {}) | {'infer': (last.get('selfplay') or {}).get('infer')},
        'crashes': snap.get('crashes') or [],
        'gpu': (snap.get('hardware') or {}).get('gpu'),
        'cpu': (snap.get('hardware') or {}).get('cpu'),
    })
