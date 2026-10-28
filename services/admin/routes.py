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


_DEVICES: list | None = None


def devices_info() -> list:
    """「推理设备」下拉的选项。不能用 torch 探测——服务进程启动时把
    CUDA_VISIBLE_DEVICES 置空（让 GPU 给 LLM），torch 会谎称没有卡；
    以 nvidia-smi 为准枚举物理卡。hidden 标记本服务当前没授权 CUDA。"""
    global _DEVICES
    if _DEVICES is None:
        out: list = []
        try:
            import subprocess
            r = subprocess.run(['nvidia-smi', '--query-gpu=index,name', '--format=csv,noheader'],
                               capture_output=True, text=True, timeout=8)
            import re as _re
            for line in (r.stdout if r.returncode == 0 else '').splitlines():
                m = _re.match(r'^(\d+),\s*(.+)$', line.strip())
                if not m:
                    continue
                idx, name = m.group(1), m.group(2)
                out.append({'value': f'cuda:{idx.strip()}', 'label': f'cuda:{idx.strip()} · {name.strip()}'})
        except Exception:
            pass
        try:
            import torch
            if out and not torch.cuda.is_available():
                for d in out:
                    d['hidden'] = True
                    d['label'] += '（本服务当前未授权 CUDA，需重启服务时不带隔离变量）'
            mps = getattr(torch.backends, 'mps', None)
            if mps and mps.is_available():
                out.append({'value': 'mps', 'label': 'mps · Apple GPU'})
        except Exception:
            pass
        out.append({'value': 'cpu', 'label': 'cpu'})
        cur = str(CONFIG.get('device') or '')
        if cur and not any(d['value'] == cur or cur.startswith(d['value'].split(':')[0] + ':') for d in out):
            out.insert(0, {'value': cur, 'label': f'{cur} · 当前生效（本服务探测不到该设备）'})
        _DEVICES = out
    return _DEVICES



@bp.get('/api/admin/env')
def env_info():
    """表单默认值：当前生效的档位/旋钮（含进程环境里已有的覆盖）。"""
    return jsonify({
        'suggested': _suggested_target(),
        'game': CONFIG['game_name'], 'profile': os.environ.get('ZZ_PROFILE', 'm'),
        'device': CONFIG['device'], 'devices': devices_info(), 'workers': CONFIG.get('selfplay_workers'),
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
    res = _r().start(data)
    return jsonify(res), (200 if res.get('ok') else 400)


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


# ---------------------------------------------------------------- 经验池
@bp.get('/api/admin/buffers')
def buffers_list():
    """仓清单（注册表=目录现场扫描）：default@<模型> 原位仓 + buffers/ 命名仓，含元数据。"""
    from core.utils.buffers import list_buffers
    game = str(request.args.get('game') or CONFIG['game_name'])
    return jsonify({'game': game, 'buffers': list_buffers(game)})


@bp.post('/api/admin/buffers/clone')
def buffers_clone():
    err, data = _guard()
    if err:
        return err
    if _r().status().get('running'):
        return jsonify({'ok': False, 'error': '训练进行中，池可能正被写入，稍后再克隆'}), 409
    from core.utils.buffers import clone_pool
    ok, msg, meta = clone_pool(str(data.get('game') or CONFIG['game_name']),
                               str(data.get('from') or '').strip().lower(),
                               str(data.get('name') or '').strip())
    return jsonify({'ok': ok, 'error': '' if ok else msg,
                    'buffer': {'name': meta.get('name'), 'from': str(data.get('from') or '')} if ok else {}}), (200 if ok else 400)


@bp.post('/api/admin/buffers/delete')
def buffers_delete():
    err, data = _guard()
    if err:
        return err
    if _r().status().get('running'):
        return jsonify({'ok': False, 'error': '训练进行中，不能删经验池'}), 409
    if str(data.get('confirm') or '') != 'delete-pool':
        return jsonify({'ok': False, 'error': '需要二次确认（输入池名）'}), 409
    from core.utils.buffers import delete_pool
    ok, msg = delete_pool(str(data.get('game') or CONFIG['game_name']),
                          str(data.get('name') or '').strip().lower())
    return jsonify({'ok': ok, 'error': '' if ok else msg}), (200 if ok else 400)


@bp.post('/api/admin/buffers/create')
def buffers_create():
    err, data = _guard()
    if err:
        return err
    from core.utils.buffers import create_named
    ok, msg, meta = create_named(str(data.get('game') or CONFIG['game_name']),
                                 str(data.get('name') or '').strip(), str(data.get('note') or ''))
    res = {'ok': True, 'message': msg, 'buffer': meta} if ok else {'ok': False, 'error': msg}
    return jsonify(res), (200 if ok else 400)


# ---------------------------------------------------------------- 命名模型配置
@bp.get('/api/admin/profiles')
def profiles_list():
    """内置档位(s/m/l) ∪ data/profiles.json 自定义命名配置，含各自模型数。"""
    from core.utils.config import CONFIG, available_profiles, net_config_for
    game = str(request.args.get('game') or CONFIG['game_name'])
    try:
        builtin = set((CONFIG[game].get('presets') or {}).keys())
    except KeyError:
        return jsonify({'ok': False, 'error': f'未知棋种 {game}'}), 400
    out = []
    for name, over in available_profiles(game).items():
        cfg = {}
        try:
            cfg = net_config_for(name, game)
        except Exception:                                # noqa: BLE001 坏条目不炸列表
            pass
        n_models, best = 0, -1
        try:
            from core.network.functions import list_all_indices, read_best_index
            n_models = len(list_all_indices(game, name))
            best = read_best_index(game, name) if n_models else -1
        except Exception:                                # noqa: BLE001
            pass
        out.append({'name': name, 'builtin': name in builtin,
                    'net': {k: cfg.get(k) for k in ('arch', 'n_filters', 'n_res_blocks',
                                                    'n_value_hidden_channels', 'use_se')},
                    'override': over, 'models': n_models, 'best': best})
    return jsonify({'game': game, 'profiles': out})


@bp.post('/api/admin/profiles/save')
def profiles_save():
    err, data = _guard()
    if err:
        return err
    from core.utils.config import save_custom_profile
    ok, msg = save_custom_profile(str(data.get('name') or ''),
                                  data.get('net') if isinstance(data.get('net'), dict) else {})
    res = {'ok': ok, 'message': msg} if ok else {'ok': False, 'error': msg}
    return jsonify(res), (200 if ok else 400)


@bp.post('/api/admin/profiles/delete')
def profiles_delete():
    err, data = _guard()
    if err:
        return err
    from core.utils.config import CONFIG, delete_custom_profile, net_dir
    name = str(data.get('name') or '')
    # 护栏：该配置名下还有 checkpoint 时，删名字会让模型"失踪"——要求先带走数据目录或显式确认
    game = str(data.get('game') or CONFIG['game_name'])
    try:
        d = net_dir(game, name)
        n_pt = len([f for f in os.listdir(d) if f.endswith('.pt')]) if os.path.isdir(d) else 0
    except OSError:
        n_pt = 0
    if n_pt and str(data.get('confirm') or '') != 'delete-profile-with-models':
        return jsonify({'ok': False,
                        'error': f'{name} 名下还有 {n_pt} 个模型文件；确认删除配置名请输入 confirm='
                                 f'"delete-profile-with-models"（模型文件保留在磁盘）'}), 409
    ok, msg = delete_custom_profile(name)
    return jsonify({'ok': ok, 'message': msg} if ok else {'ok': False, 'error': msg}), (200 if ok else 400)


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
    # path= 相对日志根目录的文件（清单接口回传什么就查什么）；name= 模块目录，旧调用兼容
    lines = min(int(request.args.get('lines', 200)), 2000)
    grep = request.args.get('grep', '').strip()
    base = Path(CONFIG['log_dir']).resolve()
    if request.args.get('path'):
        cand = (base / str(request.args['path'])).resolve()
        if not str(cand).startswith(str(base) + os.sep):
            return jsonify({'ok': False, 'error': '路径越界', 'lines': []}), 403
        path, safe = cand, str(request.args['path'])
    elif request.args.get('name') == 'current':
        lp = str(_r().status().get('log_path') or '')     # 当前训练进程的 console 日志（绝对路径）
        path = Path(lp) if lp else base / 'train' / 'latest.log'
        safe = 'current'
    else:
        safe = ''.join(ch for ch in request.args.get('name', 'selfplay') if ch.isalnum() or ch in '._-')
        path, safe = base / safe / 'latest.log', safe
    if not path.exists():
        return jsonify({'ok': False, 'error': f'没有这个日志: {safe}', 'lines': []})
    try:
        size = path.stat().st_size
        with open(path, 'rb') as fp:
            if size > 2_000_000:
                fp.seek(-2_000_000, os.SEEK_END)
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
    """日志清单：只认统一格式 `<模块>/latest.log*`（logger 流）与 `<模块>/console.log*`
    （受管子进程 stdout），一层目录、不递归；`_` 前缀目录（_state/_archive）不进台账。

    历史不合规文件已由一次性迁移归位或存入 logs/_archive/<原路径>——治理在源头
    （写侧统一走 logger / runner 的 console.log 槽位），展示层不再做形状发明。
    「● 实时」(name=current) 独立跟随 runner 正在写的文件，不受清单形态约束。"""
    base = Path(CONFIG['log_dir'])
    out = []
    if base.is_dir():
        for entry in sorted(base.iterdir()):
            if not entry.is_dir() or entry.name.startswith(('_', '.')):
                continue
            for f in sorted(entry.iterdir()):
                if not f.is_file():
                    continue
                if not (f.name.startswith('latest.log') or f.name.startswith('console.log')):
                    continue
                st = f.stat()
                if st.st_size == 0:
                    continue          # 空壳（logger 初始化或崩溃位未触发）不占台账——用户："没有的 log 清理"
                out.append({'name': f.name, 'path': f'{entry.name}/{f.name}',
                            'cat': entry.name, 'size': st.st_size, 'mtime': st.st_mtime})
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
