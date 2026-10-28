"""训练进程管理：从 HTTP 启动 / 停止 / 查询 `scripts.train_server`。

设计要点：
- 训练是**独立子进程**（start_new_session=True）：本 API 服务重启或退出都不影响它，
  反过来训练崩了也不会带走棋类服务；
- 状态落 `logs/_state/trainer.json`：API 服务自己重启后能"认领"还在跑的训练
  （按 pid 存活 + /proc/<pid>/cmdline 里含 train_server 双重确认，避免杀错进程）；
- 停止是"先礼后兵"：SIGINT（train_server 里 register_sigint 会走 shutdown_all）→
  等优雅退出 → SIGTERM → 再等 → SIGKILL；
- 所有参数走环境变量透传，不改仓库里的 config（可复现、可对比、不动源码）。
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = ROOT / 'logs' / '_state'        # 进程状态（trainer.json），非日志、不进清单
STATE_FILE = STATE_DIR / 'trainer.json'

# 允许从面板覆盖的旋钮 → 环境变量（core/utils/config.py 里有对应读取）
_KNOB_ENV = {
    'game': 'ZZ_GAME',
    'profile': 'ZZ_PROFILE',
    'device': 'ZZ_DEVICE',
    'workers': 'ZZ_WORKERS',
    'games_per_round': 'ZZ_GAMES_PER_ROUND',
    'batch_leaves': 'ZZ_BATCH_LEAVES',
    'max_batch': 'ZZ_MAX_BATCH',
    'buffer': 'ZZ_BUFFER',            # 经验池：空=原位池；名字=棋种级命名池；default@<模型>=兼容定位
    'start_from': 'ZZ_START_FROM',    # 起点：auto|fresh|<代数N>
}
_STR_OK = {
    'buffer': lambda v: v == 'default' or v.startswith('default@') or
                        __import__('re').fullmatch(r'[a-z][a-z0-9_]{0,31}', v) is not None,
    'start_from': lambda v: v in ('auto', 'fresh') or v.isdigit(),
}
_INT_KNOBS = ('workers', 'games_per_round', 'batch_leaves', 'max_batch')
_CHOICES = {
    'game': ('ChineseChess', 'Gomoku'),
}


def _known_profile(v: str) -> bool:
    """档位合法性 = 内置 s/m/l ∪ data/profiles.json 里的命名配置。
    读不了配置就不拦（子进程 set_profile 自会报错，错误会进 console 日志）。"""
    try:
        from core.utils.config import available_profiles
        return v in available_profiles()
    except Exception:                                # noqa: BLE001
        return True


class TrainingRunner:
    def __init__(self, cmd: list[str] | None = None, marker: str = 'scripts.train_server',
                 state_file: Path | None = None, log_file: Path | str | None = None) -> None:
        # cmd/marker/state_file/log_file 可注入：测试用假命令 + 临时文件，绝不写生产 logs/
        self._cmd = cmd or [sys.executable, '-X', 'gil=0', '-m', 'scripts.train_server']
        self.marker = marker
        self.state_file = Path(state_file or STATE_FILE)
        if log_file:
            self.log_file = Path(log_file)
        else:                                        # 局部导入与本模块既有风格一致
            from core.utils.config import CONFIG as _C
            self.log_file = Path(_C['log_dir']) / 'train' / 'console.log'
        self.proc: subprocess.Popen | None = None
        self.meta: dict = {}
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._adopt()

    def _reap(self) -> None:
        """回收子进程（否则 SIGKILL 后会留僵尸，PID 一直被占着）。"""
        if self.proc is None:
            return
        try:
            self.proc.wait(timeout=5)
        except Exception:                                 # noqa: BLE001 已退出/非子进程都无所谓
            pass
        if self.proc.returncode is not None:
            self.meta['exit_code'] = self.proc.returncode

    # ---------- 状态持久化 ----------
    def _save(self) -> None:
        try:
            self.state_file.write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding='utf-8')
        except OSError:
            pass

    def _adopt(self) -> None:
        """API 重启后认领仍在跑的训练进程（不持有 Popen，只能按 pid 操作）。"""
        try:
            meta = json.loads(self.state_file.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            return
        pid = int(meta.get('pid') or 0)
        if pid and _pid_is_trainer(pid, self.marker):
            self.meta = meta
        else:
            meta['running'] = False
            meta['stale'] = True
            self.meta = meta

    # ---------- 查询 ----------
    def status(self) -> dict:
        pid = int(self.meta.get('pid') or 0)
        alive = bool(pid) and _pid_is_trainer(pid, self.marker)
        if self.proc is not None:
            code = self.proc.poll()
            if code is not None:
                self.meta['exit_code'] = code          # 顺带回收到期子进程，别留僵尸
        if not alive and self.meta.get('running'):
            # 训练自己退出了：把退出码补齐（子进程归我们管时能拿到）
            self.meta['running'] = False
            self.meta.pop('stop_after_round', None)
            self._save()
        return {
            'running': alive,
            'pid': pid if alive else None,
            'started_at': self.meta.get('started_at'),
            'uptime_s': round(time.time() - self.meta['started_ts'], 1) if alive and self.meta.get('started_ts') else None,
            'opts': self.meta.get('opts') or {},
            'env': self.meta.get('env') or {},
            'log_path': self.meta.get('log_path'),
            'exit_code': self.meta.get('exit_code'),
            'stop_after_round': bool(self.meta.get('stop_after_round')) if alive else False,
            'cmdline': self.meta.get('cmdline'),
        }

    # ---------- 启动 ----------
    def start(self, opts: dict) -> dict:
        if self.status()['running']:
            return {'ok': False, 'error': '训练已在运行（先 stop）', 'status': self.status()}
        clean, err = _validate(opts)
        if err:
            return {'ok': False, 'error': err}
        if not str(clean.get('buffer') or '').strip():
            # 产品规则：经验池必选（棋种级命名资源）；裸命令行不经此处，仍兼容原位池
            return {'ok': False, 'error': '必须先选择经验池（下拉选择或新建）'}
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        log_path = self.log_file                      # 统一格式：<模块>/console.log（追加，历次会话连贯）
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        for key, env_name in _KNOB_ENV.items():
            if clean.get(key) not in (None, ''):
                env[env_name] = str(clean[key])
        cmd = list(self._cmd)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fp = open(log_path, 'ab', buffering=0)
        if log_fp.tell():
            log_fp.write(b'\n')
        log_fp.write(f'═════ 新训练会话 · {datetime.now().isoformat(timespec="seconds")} '
                     f'· {" ".join(cmd[-2:])} ═════\n'.encode())
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=log_fp, stderr=subprocess.STDOUT,
                                start_new_session=True)
        self.proc = proc
        self.meta = {'pid': proc.pid, 'running': True, 'started_at': datetime.now().isoformat(timespec='seconds'),
                     'started_ts': time.time(), 'opts': clean,
                     'env': {k: env[v] for k, v in _KNOB_ENV.items() if v in env},
                     'log_path': str(log_path), 'cmdline': ' '.join(cmd), 'exit_code': None}
        self._save()
        time.sleep(1.2)                                  # 让"启动即崩"能在返回体里直接看到
        st = self.status()
        if not st['running']:
            return {'ok': False, 'error': '训练进程启动后立刻退出，见日志尾部', 'status': st}
        return {'ok': True, 'status': st}

    # ---------- 停止（两种语义） ----------
    # 停止语义（用户定：立即结束=kill 进程，另一个=发 SIGINT）
    _MODES = {
        'kill': 'kill',              # SIGKILL：立刻杀，不存盘，丢本轮进度
        'interrupt': 'interrupt',    # SIGINT：收完在跑的局→存盘→退出（fit 因未开始则跳过评估）
        'immediate': 'interrupt',    # 兼容旧调用
        'finish_round': 'finish_round',  # SIGUSR1：自博弈+fit+评估+perf 全做完再退
        'after_round': 'finish_round',
    }

    def stop(self, mode: str = 'interrupt', grace: float = 150.0) -> dict:
        mode = self._MODES.get(str(mode or 'interrupt'))
        if mode is None:
            return {'ok': False, 'error': f"未知停止方式: {mode}"}
        st = self.status()
        if not st['running']:
            return {'ok': True, 'message': '训练没在跑', 'status': st}
        pid = st['pid']
        if mode == 'kill':
            # 立即结束：直接 SIGKILL，不做任何收尾（本轮已下完的局也不会存盘）
            self.meta.pop('stop_after_round', None)
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            for _ in range(50):
                if not _pid_is_trainer(pid, self.marker):
                    break
                time.sleep(0.1)
            self.meta['running'] = False
            self.meta['stopped_at'] = datetime.now().isoformat(timespec='seconds')
            self.meta['killed'] = True
            self._save()
            self._reap()
            return {'ok': True, 'message': '已立即结束（SIGKILL，未存盘）', 'status': self.status()}
        if mode == 'finish_round':
            if self.meta.get('stop_after_round'):
                return {'ok': True, 'pending': True, 'message': '已经在等本轮结束了',
                        'status': st}
            caps, why = _trainer_caps(pid)
            if 'after_round' not in caps:
                # SIGUSR1 默认动作是终止进程！旧进程没装 handler，盲发等于直接杀死训练
                return {'ok': False,
                        'error': f'该训练进程不支持"等本轮结束"（{why}）——请用"立即结束"',
                        'status': st}
            try:
                os.kill(pid, signal.SIGUSR1)
            except ProcessLookupError:
                pass
            self.meta['stop_after_round'] = True
            self._save()
            return {'ok': True, 'pending': True,
                    'message': '已请求：本轮（自博弈 + fit + 评估）跑完后自动退出；想立刻停就再点"立即结束"',
                    'status': self.status()}
        if self.meta.pop('stop_after_round', None):
            self._save()
        return self._stop_immediate(pid, grace)

    def _still_draining(self, pid: int) -> bool:
        """训练进程还在"自己收自己"吗？读心跳：带 stopping 标记且 20s 内在跳；
        或正处 fit/评估——这两段最长几分钟不写心跳，给 240s 龄期。
        命中就值得继续等：SIGINT 升级成 SIGKILL 会丢掉正在收尾的整轮数据
        （10/28 演练：4 局在飞没收完就被 150s 宽限期到点杀掉）。"""
        try:
            import json as _json
            from core.utils.config import CONFIG
            from services.train import heartbeat as _hb
            p = next((c for c in _hb.read_candidates() if Path(c).exists()), None)
            if p is None:
                return False
            data = _json.loads(Path(p).read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return False
        if int(data.get('pid') or 0) != int(pid):
            return False
        age = time.time() - float(data.get('ts') or 0)
        if data.get('stopping') and age < 20:
            return True
        return data.get('phase') in ('fit', 'eval') and age < 240

    def _stop_immediate(self, pid: int, grace: float) -> dict:
        for sig, wait in ((signal.SIGINT, grace), (signal.SIGTERM, 10.0), (signal.SIGKILL, 5.0)):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                break
            deadline = time.time() + wait
            cap = time.time() + (600.0 if sig == signal.SIGINT else wait)   # 收尾最多再等 10 分钟
            while True:
                if not _pid_is_trainer(pid, self.marker):
                    self.meta['running'] = False
                    self.meta['stopped_at'] = datetime.now().isoformat(timespec='seconds')
                    self._save()
                    self._reap()
                    return {'ok': True, 'message': f'已停止（{sig.name}）', 'status': self.status()}
                if time.time() >= deadline and not (
                        sig == signal.SIGINT and time.time() < cap and self._still_draining(pid)):
                    break
                time.sleep(0.4)
        self.meta['running'] = _pid_is_trainer(pid, self.marker)
        self._save()
        self._reap()
        return {'ok': False, 'error': f'pid {pid} 无法停止', 'status': self.status()}


def _validate(opts: dict) -> tuple[dict, str | None]:
    clean: dict = {}
    for key, value in (opts or {}).items():
        if key not in _KNOB_ENV:
            continue
        if value in (None, ''):
            continue
        if key in _CHOICES and str(value) not in _CHOICES[key]:
            return {}, f'{key} 只能是 {_CHOICES[key]}'
        if key == 'profile' and not _known_profile(str(value)):
            return {}, f'未知档位 {value}（内置 s/m/l 或 data/profiles.json 命名配置）'
        if key in _STR_OK and not _STR_OK[key](str(value)):
            return {}, f'{key} 取值非法: {value}（仓=名字/default@模型；起点=auto/fresh/代数）'
        if key in _INT_KNOBS:
            try:
                iv = int(value)
            except (TypeError, ValueError):
                return {}, f'{key} 必须是整数'
            if iv <= 0 or iv > 1_000_000:
                return {}, f'{key} 超出合理范围: {iv}'
            clean[key] = iv
        else:
            clean[key] = str(value)
    return clean, None


def _trainer_caps(pid: int) -> tuple[list[str], str]:
    """从训练进程的心跳里读能力声明（心跳里的 pid 必须对得上，防止读到上一轮残留）。"""
    try:
        import json as _json
        from core.utils.config import CONFIG
        from services.train import heartbeat as _hb
        p = next((c for c in _hb.read_candidates() if Path(c).exists()), None)
        if p is None:
            return [], '没有心跳文件，像是旧版本进程'
        data = _json.loads(Path(p).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return [], '没有心跳文件，像是旧版本进程'
    if int(data.get('pid') or 0) != int(pid):
        return [], '心跳不是本进程写的（旧版本或已退出）'
    caps = data.get('caps') or []
    if not isinstance(caps, list):
        return [], '心跳里没有 caps'
    return [str(c) for c in caps], ''


def _pid_is_trainer(pid: int, marker: str = 'scripts.train_server') -> bool:
    """pid 活着且确实是训练进程（防 pid 复用杀错人）。"""
    try:
        cmdline = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode('utf-8', 'replace')
    except OSError:
        return False
    return marker in cmdline or 'scripts/train_server.py' in cmdline


