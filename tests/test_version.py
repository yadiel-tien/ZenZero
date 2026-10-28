"""版本管理回归：单一事实源格式、展示位接缝、四条轴的分离、CLI。

运行：.venv/bin/python tests/test_version.py
"""
import os
import re
import subprocess
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ['ZZ_DEVICE'] = 'cpu'

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from core.utils.version import __version__, full, revision  # noqa: E402


def test_semver_and_build_stamp():
    assert re.fullmatch(r'\d+\.\d+\.\d+([-+][\w.]+)?', __version__), __version__
    f = full()
    assert f.startswith(__version__), f
    rev = revision()
    if rev:                                   # 在 git 工作区里应带构建号；导出包允许为空
        assert re.fullmatch(r'[0-9a-f]{7,40}', rev), rev
        assert f == f'{__version__}+{rev}', f
    print(f'   ok   版本单一事实源：{f}')


def test_display_seams_present():
    """五处展示位各自的取数点必须存在（重构时防丢：/info、meta、health、两前端徽章）。"""
    seams = {
        'services/inference/play_api.py': "probe_device_info(), 'version'",   # /info
        'services/monitor/collect.py': "'version': __version__",              # snapshot meta
        'services/admin/routes.py': "'version': _full()",                     # /api/admin/health
        'apps/web/admin.html': "id=\"appver\"",                               # 控制台标题徽章
        'apps/web/js/app.js': "d.version",                                    # 对弈页设置侧栏
    }
    for rel, needle in seams.items():
        src = open(os.path.join(ROOT, rel), encoding='utf-8').read()
        assert needle in src, f'{rel} 缺版本展示位：{needle!r}'
    print(f'   ok   展示位接缝齐全（{len(seams)} 处）')


def test_axes_are_distinct():
    """四条版本轴互不越权：应用版本 ≠ 表示版本；档位独立；CLI 报应用版本。"""
    from core.utils.config import CONFIG
    fv = CONFIG['ChineseChess']['default_net']['feature_version']
    assert isinstance(fv, int) and fv >= 1   # 表示轴独立：整数 epoch，与应用 semver 无耦合
    assert set(CONFIG['ChineseChess']['presets']) == {'s', 'm', 'l'}
    r = subprocess.run([sys.executable, 'main.py', '--version'], capture_output=True, text=True,
                       cwd=ROOT, timeout=60)
    assert r.returncode == 0 and __version__ in r.stdout, r.stdout + r.stderr
    print(f'   ok   轴分离：app={__version__} · feature={fv} · 档位 s/m/l · CLI→{r.stdout.strip()}')


def main() -> None:
    for fn in (test_semver_and_build_stamp, test_display_seams_present, test_axes_are_distinct):
        print(f'== {fn.__name__}')
        fn()
    print('\nALL 3 VERSION TESTS PASSED')


if __name__ == '__main__':
    main()
