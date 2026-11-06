"""控制台「任务」页的通用子进程管理器：scripts/ 中除训练外的功能入口。

与 TrainingRunner（runner.py）同一套纪律，但更简单——这里没有旋钮到环境变量的映射，
参数一律以命令行 flags 传下去，且只接受任务注册表里声明过的项（白名单校验）：

- 独立子进程（start_new_session=True）：本服务重启不影响任务，任务崩溃不带走服务；
- 状态落 logs/_state/tasks.json：按 pid + /proc cmdline 特征收养，防误杀；
- 停止 = SIGINT →（超时）SIGTERM →（超时）SIGKILL → wait 回收；
- long 类任务（如 piskvork 协议服）常驻直到手动停止；once 类（bench/build）跑完留退出码与日志；
- 一切输出追加到 logs/<任务名>/console.log（统一格式，历次会话连贯），日志尾读走现有 /api/admin/tasks/log。

注册表（新增任务只改这里）：
    module      以 -m 启动的模块，或 script 以脚本路径启动（二选一）
    kind        long | once
    opts        参数 schema：int / choice / flag（bool）
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
STATE_DIR = ROOT / 'logs' / '_state'        # 任务状态（tasks.json），非日志、不进清单
STATE_FILE = STATE_DIR / 'tasks.json'

TASKS: dict[str, dict] = {
    'piskvork': {
        'module': 'scripts.piskvork_server', 'kind': 'long',
        'desc': '以 Gomocup 协议对外提供五子棋对弈：启动后在 host:port（默认 0.0.0.0:12345）'
                       '就绪，Piskvork 等兼容客户端连入即可让 ZenZero 参赛；网络与代数可选，默认最佳模型，'
                       '思考深度按客户端时限自适应，用毕停止即释放资源。',
        'opts': {'profile': {'flag': '--profile', 'type': 'profile', 'game': 'Gomoku', 'default': ''},
                 'model': {'flag': '--model', 'type': 'model', 'game': 'Gomoku', 'default': -1},
                 'sims': {'flag': '--sims', 'type': 'int', 'default': 800},
                 'host': {'flag': '--host', 'type': 'str', 'default': '0.0.0.0'},
                 'port': {'flag': '--port', 'type': 'int', 'default': 12345}},
    },
    'bench': {
        'script': 'scripts/bench_batch_sweep.py', 'kind': 'once',
        'desc': '引擎批扫描：实测前向成本模型 t(b)，核对凑批上限（占 GPU，勿与训练并发）',
        'opts': {'batches': {'flag': '--batches', 'type': 'str', 'default': ''},
                 'iters': {'flag': '--iters', 'type': 'int', 'default': 0}},
    },
    'build_cython': {
        'script': 'scripts/build_cython.py', 'kind': 'once',
        'desc': 'Cython 扩展安全构建（暂存编译 + 原子替换，训练在跑也安全）',
        'opts': {'check': {'flag': '--check', 'type': 'flag'}},
    },
}


def _alive_cmdline(pid: int, marker: str) -> bool:
    try:
        cmd = Path(f'/proc/{pid}/cmdline').read_bytes().decode('utf-8', 'ignore')
    except OSError:
        return False
    return bool(cmd) and marker in cmd.replace('\x00', ' ')


class TaskRunner:
    """按名字管理多个任务（一个进程内单例即可，状态在磁盘）。"""

    def __init__(self, state_file: Path | None = None, registry: dict | None = None,
                 log_root: Path | str | None = None) -> None:
        # registry/state_file/log_root 可注入：测试用假任务与临时目录，绝不写生产 logs/
        self.state_file = Path(state_file or STATE_FILE)
        self.registry = dict(TASKS if registry is None else registry)
        self.log_root = Path(log_root) if log_root else Path(__import__('core.utils.config',
                                                                       fromlist=['CONFIG']).CONFIG['log_dir'])
        self.procs: dict[str, subprocess.Popen] = {}
        self.meta: dict[str, dict] = {}
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._load()
        self._adopt_all()

    # ---------- 状态 ----------
    def _load(self) -> None:
        try:
            self.meta = json.loads(self.state_file.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            self.meta = {}

    def _save(self) -> None:
        try:
            self.state_file.write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding='utf-8')
        except OSError:
            pass

    def _marker(self, name: str) -> str:
        t = self.registry[name]
        return t.get('marker') or t.get('module') or t.get('script') or name

    def _adopt_all(self) -> None:
        for name, m in list(self.meta.items()):
            pid = int(m.get('pid') or 0)
            if pid and _alive_cmdline(pid, self._marker(name)):
                m['running'] = True
            elif m.get('running'):
                m['running'] = False
                m['stopped_at'] = datetime.now().isoformat(timespec='seconds')
        self._save()

    def _build_cmd(self, name: str, params: dict) -> tuple[list[str], str | None]:
        t = self.registry[name]
        # 参数一律先校验（含 argv 任务）：非法值绝不静默忽略；校验不过直接拒起
        flags: list[str] = []
        for key, spec in (t.get('opts') or {}).items():
            raw = params.get(key)
            if raw in (None, ''):
                continue
            typ = spec.get('type', 'str')
            if typ == 'flag':
                if str(raw).lower() in ('1', 'true', 'on', 'yes'):
                    flags.append(spec['flag'])
                continue
            if typ == 'int':
                try:
                    val = int(raw)
                except (TypeError, ValueError):
                    return [], f'{name}: {key} 需为整数'
                if spec.get('flag') == '--port' and not (1 <= val <= 65535):
                    return [], f'{name}: {key} 越界（1-65535）'
                flags += [spec['flag'], str(val)]
                continue
            val = str(raw)
            if spec.get('flag') == '--host' and not val.replace('.', '').replace(':', '').replace('-', '').isalnum():
                return [], f'{name}: {key} 含非法字符'
            if spec.get('choices') and val not in spec['choices']:
                return [], f'{name}: {key} 只能是 {spec["choices"]}'
            flags += [spec['flag'], val]

        if 'argv' in t:                                # 测试/自定义注入（参数已校验）
            return list(t['argv']), None
        cmd = [sys.executable]
        cmd += ['-m', t['module']] if 'module' in t else [t['script']]
        return cmd + flags, None

    # ---------- 操作 ----------
    def status(self) -> dict:
        out = []
        for name, t in self.registry.items():
            m = dict(self.meta.get(name) or {})
            pid = int(m.get('pid') or 0)
            if pid and _alive_cmdline(pid, self._marker(name)):
                m['running'] = True
            elif m.get('running'):
                m['running'] = False
                m['stopped_at'] = datetime.now().isoformat(timespec='seconds')
                p = self.procs.get(name)
                if p is not None:
                    if p.returncode is None:
                        try:
                            p.wait(timeout=0.2)
                        except Exception:                   # noqa: BLE001 已退/非子进程
                            pass
                    m['exit_code'] = p.returncode
            out.append({'name': name, 'desc': t['desc'], 'kind': t['kind'],
                        'opts': t.get('opts') or {},
                        'last_params': (m.get('params') or {}),
                        **{k: m.get(k) for k in
                        ('running', 'pid', 'started_at', 'stopped_at', 'exit_code', 'log_path', 'cmdline')}})
        self._save()
        return {'tasks': out}

    def start(self, name: str, params: dict | None = None) -> dict:
        if name not in self.registry:
            return {'ok': False, 'error': f'未知任务: {name}'}
        st = next((x for x in self.status()['tasks'] if x['name'] == name), None)
        if st and st.get('running'):
            return {'ok': False, 'error': f'{name} 已在运行（pid {st.get("pid")}）', 'status': st}
        params = params or {}
        cmd, err = self._build_cmd(name, params)
        if err:
            return {'ok': False, 'error': err}
        log_path = self.log_root / name / 'console.log'    # 统一格式：<模块>/console.log
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fp = open(log_path, 'ab', buffering=0)
        if log_fp.tell():
            log_fp.write(b'\n')
        log_fp.write(f'═════ 新会话 {name} · {datetime.now().isoformat(timespec="seconds")} '
                     f'· {" ".join(str(x) for x in cmd[1:])} ═════\n'.encode())
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=log_fp, stderr=subprocess.STDOUT,
                                start_new_session=True)
        self.procs[name] = proc
        self.meta[name] = {'pid': proc.pid, 'running': True, 'params': params,
                           'started_at': datetime.now().isoformat(timespec='seconds'),
                           'log_path': str(log_path), 'cmdline': ' '.join(cmd), 'exit_code': None,
                           'kind': self.registry[name]['kind']}
        self._save()
        time.sleep(0.6)                                     # 秒退可见（once 任务很快跑完是常态）
        rc = proc.poll()
        if rc is not None:
            stt = next(x for x in self.status()['tasks'] if x['name'] == name)
            try:
                tail = log_path.read_text(encoding='utf-8', errors='replace')[-1200:]
            except OSError:
                tail = ''
            res = {'log_tail': tail.splitlines()[-10:], 'status': stt, 'exit_code': rc}
            if rc == 0:
                res.update(ok=True, message=f'{name} 已完成（exit 0）')
            else:
                res.update(ok=False, error=f'{name} 退出码 {rc}')
            return res
        return {'ok': True, 'message': f'已启动 {name}（pid {proc.pid}）',
                'status': next(x for x in self.status()['tasks'] if x['name'] == name)}

    def stop(self, name: str, grace: float = 15.0) -> dict:
        if name not in self.registry:
            return {'ok': False, 'error': f'未知任务: {name}'}
        st = next((x for x in self.status()['tasks'] if x['name'] == name), None)
        if not st or not st.get('running'):
            return {'ok': True, 'message': f'{name} 未在运行'}
        pid = int(st['pid'])
        for sig, wait in ((signal.SIGINT, grace), (signal.SIGTERM, 5.0), (signal.SIGKILL, 3.0)):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                break
            deadline = time.time() + wait
            while time.time() < deadline:
                if not _alive_cmdline(pid, self._marker(name)):
                    break
                time.sleep(0.3)
            if not _alive_cmdline(pid, self._marker(name)):
                m = self.meta.setdefault(name, {})
                m['running'] = False
                m['stopped_at'] = datetime.now().isoformat(timespec='seconds')
                p = self.procs.get(name)
                if p is not None:
                    try:
                        p.wait(timeout=3)
                    except Exception:                       # noqa: BLE001
                        pass
                    m['exit_code'] = p.returncode
                self._save()
                return {'ok': True, 'message': f'已停止 {name}（{sig.name}）'}
        return {'ok': False, 'error': f'{name}（pid {pid}）未能停止', 'status': st}

    def log(self, name: str, tail: int = 200) -> dict:
        if name not in self.registry:
            return {'ok': False, 'error': f'未知任务: {name}'}
        m = self.meta.get(name) or {}
        path = m.get('log_path')
        if not path or not os.path.exists(path):
            return {'ok': True, 'lines': [], 'note': '该任务尚无运行记录'}
        try:
            with open(path, 'rb') as fp:
                fp.seek(max(0, os.path.getsize(path) - tail * 240))
                data = fp.read().decode('utf-8', 'replace')
        except OSError as e:
            return {'ok': False, 'error': str(e)}
        return {'ok': True, 'lines': data.splitlines()[-tail:], 'path': path,
                'running': bool(m.get('running'))}
