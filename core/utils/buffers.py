"""数据仓（ReplayBuffer）实体化：统一命名、元数据、注册表。

一个"仓"就是一对 pkl（主仓 + 留出验证集）加一份 meta.json。两种来源：

  default@<模型>   历史原位：data/<Game>[/<profile>]/buffer.pkl + val_buffer.pkl。
                   零迁移零复制，训练不选仓时就用它。元数据从 .stats.json 侧车合成。
  <名字>           命名仓：data/<Game>/buffers/<name>/{buffer.pkl, val_buffer.pkl, meta.json}。
                   新建、跨模型共享（同棋种同表示版本即可引用）都走这里。

meta.json（唯一正式元数据，写失败绝不影响训练）：
  name / game / feature_version（装载闸门）/ created_at / updated_at /
  size / capacity / source: {model, iteration}  ← "最新内容来自哪个模型的第几代"

训练侧接缝：train_server 入口解析 ZZ_BUFFER=<名字|default@x>，把目录塞进
CONFIG['buffer_dir']；replay.ReplayBuffer 存取时优先该目录。留空 = 原行为。
"""
from __future__ import annotations

import json
import os
import shutil
import re
import time
from pathlib import Path

from .config import CONFIG, net_dir

SLUG = re.compile(r'^[a-z][a-z0-9_]{0,31}$')

# 训练起点（ZZ_START_FROM）解析所需，纯函数便于测试
START_MODES = ('auto', 'fresh')


def named_root(game: str) -> Path:
    return Path(CONFIG['data_dir']) / game / 'buffers'


def parse_ref(ref: str, cur_model: str) -> tuple[str, str]:
    """'default' / 'default@s' / 'warm128' → (kind, name)。"""
    ref = (ref or '').strip().lower()
    if ref in ('', 'default'):
        return 'default', cur_model
    if ref.startswith('default@'):
        return 'default', ref.split('@', 1)[1] or cur_model
    return 'named', ref


def buffer_dir(game: str, kind: str, name: str) -> Path:
    if kind == 'default':
        return Path(net_dir(game, name))
    return named_root(game) / name


def read_meta(game: str, kind: str, name: str) -> dict:
    """命名仓读 meta.json；default 仓从两个 .stats.json 侧车合成（可能缺→空 dict）。"""
    d = buffer_dir(game, kind, name)
    if kind == 'named':
        try:
            return json.loads((d / 'meta.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return {}
    meta: dict = {'name': f'default@{name}', 'game': game, 'kind': 'default'}
    for f in ('buffer.pkl', 'val_buffer.pkl'):
        st = d / (f + '.stats.json')
        try:
            s = json.loads(st.read_text(encoding='utf-8'))
            key = 'val' if f.startswith('val') else 'main'
            meta[f'{key}_size'] = s.get('size')
            meta[f'{key}_capacity'] = s.get('capacity')
            meta[f'{key}_saved_at'] = s.get('saved_at')
            meta.setdefault('feature_version', s.get('feature_version'))
        except (OSError, ValueError):
            pass
    return meta


def update_meta(game: str, directory: Path, **updates) -> None:
    """命名仓的 meta 增量更新（训练收尾调；任何失败静默——训练永远优先）。
    default 原位仓直接跳过：它的元数据在 .stats.json 侧车里，语义一致，不写两份。"""
    if Path(directory).resolve().parent != named_root(game).resolve():
        return
    path = Path(directory) / 'meta.json'
    try:
        cur = {}
        if path.exists():
            cur = json.loads(path.read_text(encoding='utf-8'))
        cur.update(updates)
        cur['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        tmp = path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(cur, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
        os.replace(tmp, path)
    except (OSError, ValueError):
        pass


def next_pool_name(game: str) -> str:
    """默认递增编号：扫本棋种已有池，poolN 取 max+1。"""
    root = named_root(game)
    hi = 0
    if root.is_dir():
        for d in root.iterdir():
            m = re.fullmatch(r'pool(\d+)', d.name)
            if m:
                hi = max(hi, int(m.group(1)))
    return f'pool{hi + 1}'


def create_named(game: str, name: str = '', note: str = '') -> tuple[bool, str, dict]:
    name = (name or '').strip().lower() or next_pool_name(game)   # 留空＝自动编号
    if not SLUG.match(name or ''):
        return False, '仓名需匹配 [a-z][a-z0-9_]{0,31}（小写字母开头）', {}
    d = named_root(game) / name
    if d.exists() and (d / 'meta.json').exists():
        return False, f'仓 {name} 已存在', {}
    d.mkdir(parents=True, exist_ok=True)
    meta = {'name': name, 'game': game, 'kind': 'named',
            'feature_version': int(CONFIG[game].get('feature_version', 1)),
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'created_by': CONFIG.get('game_name'), 'note': note,
            'size': 0, 'capacity': None, 'source': None}
    (d / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True),
                                 encoding='utf-8')
    return True, f'已创建仓 {name}', meta


def clone_pool(game: str, src: str, dst: str = '') -> tuple[bool, str, dict]:
    """复制经验池（fork 保护）：重训会环形覆盖写入，先克隆一份在新池上做，原池状态不动。
    全量复制——replay.save 为原位覆盖写，硬链接会一起被撕；先拷 .tmp-<dst> 再整体
    rename 就位，崩溃只留可重入清除的临时目录。目标留空=自动递增编号。"""
    d_src = named_root(game) / str(src or '')
    if not (d_src / 'meta.json').exists() or not (d_src / 'buffer.pkl').exists():
        return False, f'源池 {src} 不存在或无数据', {}
    dst = (dst or '').strip().lower() or next_pool_name(game)
    if not SLUG.match(dst):
        return False, '目标池名需匹配 [a-z][a-z0-9_]{0,31}', {}
    if src == dst:
        return False, '目标与源同名', {}
    d_dst = named_root(game) / dst
    if d_dst.exists():
        return False, f'池 {dst} 已存在', {}
    tmp = named_root(game) / f'.tmp-{dst}'
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        tmp.mkdir(parents=True)
        for f in d_src.iterdir():
            if f.name != 'meta.json':
                shutil.copy2(f, tmp / f.name)
        meta = json.loads((d_src / 'meta.json').read_text(encoding='utf-8'))
        meta.update({'name': dst, 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                     'note': (str(meta.get('note') or '') + f' | 由 {src} 克隆').strip(' |')})
        tmp.joinpath('meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True),
                                             encoding='utf-8')
        os.rename(tmp, d_dst)
        return True, f'已克隆 {src} → {dst}', meta
    except OSError as e:
        shutil.rmtree(tmp, ignore_errors=True)
        return False, f'克隆失败：{e}', {}


def delete_pool(game: str, name: str) -> tuple[bool, str]:
    """删除命名经验池（整个目录，含数 GB 数据）；default 原位池不允许从这里删。"""
    if not SLUG.match(str(name or '')):
        return False, '池名不合法'
    d = named_root(game) / name
    if not (d / 'meta.json').exists():
        return False, f'命名池 {name} 不存在'
    try:
        shutil.rmtree(d)
        return True, f'已删除经验池 {name}'
    except OSError as e:
        return False, f'删除失败：{e}'


def list_buffers(game: str) -> list[dict]:
    """注册表 = 目录现场扫描（不维护第二份索引，杜绝与实际文件漂移）。"""
    from .config import available_profiles
    out = []
    for m in available_profiles(game):
        d = Path(net_dir(game, m))          # 目录未出生也是合法 default@（首训自建）
        meta = read_meta(game, 'default', m)
        meta['ref'] = f'default@{m}'
        meta['path'] = str(d.relative_to(CONFIG['data_dir']))
        meta['models_hint'] = m
        out.append(meta)
    root = named_root(game)
    if root.is_dir():
        for d in sorted(root.iterdir()):
            if d.name.startswith('.'):          # .tmp-clone 半成品/隐藏件不进注册表
                continue
            if d.is_dir() and (d / 'meta.json').exists():
                meta = read_meta(game, 'named', d.name)
                meta['ref'] = d.name
                meta['path'] = str(d.relative_to(CONFIG['data_dir']))
                out.append(meta)
    out.sort(key=lambda x: str(x.get('updated_at') or x.get('created_at') or ''), reverse=True)
    return out


def resolve_for_train(ref: str, game: str) -> tuple[Path | None, str, dict]:
    """训练启动闸门：把 ZZ_BUFFER 解析成目录（None=保持原 default 行为）。
    返回 (目录|None, 错误信息, meta)。命名仓不存在则当场创建（新建即用）。"""
    from .config import PROFILE
    kind, name = parse_ref(ref, PROFILE)
    if kind == 'default' and name == PROFILE:
        return None, '', {}
    if kind == 'default':
        d = Path(net_dir(game, name))
        meta = read_meta(game, 'default', name)
        fv = meta.get('feature_version')
        cur_fv = int(CONFIG[game].get('feature_version', 1))
        if fv is not None and int(fv) != cur_fv:
            return None, f'仓 {ref} 是表示版本 {fv}，当前为 {cur_fv}，不能混用', {}
        return d, '', meta
    ok, msg, meta = create_named(game, name)
    if not ok and '已存在' not in msg:
        return None, msg, {}
    d = named_root(game) / name
    meta = read_meta(game, 'named', name)
    fv = meta.get('feature_version')
    cur_fv = int(CONFIG[game].get('feature_version', 1))
    if fv is not None and int(fv) != cur_fv:
        return None, f'仓 {name} 是表示版本 {fv}，当前为 {cur_fv}，不能混用', {}
    return d, '', meta


def resolve_start_from(mode: str, best: int, latest: int, exists_fn, compat_fn) -> tuple[int, str, str]:
    """训练起点决策（④），纯函数便于测试。返回 (喂给 TrainServer 的 model_idx, 说明, 错误)。

    auto   现状：best 兼容则续 best，否则新网络（守擂逻辑不变）
    fresh  新网络（随机初始化），代数天然接着 latest+1（调用方语义）
    <N>    从第 N 代权重分叉：新代数仍从 max(latest, N)+1 起，永不覆盖既有 checkpoint
    """
    mode = (mode or 'auto').strip().lower()
    if mode in ('', 'auto'):
        if best != -1 and compat_fn(best):
            return best, f'续训：从最强权重 {best} 接续', ''
        return -1, '新网络（无兼容的已有权重）', ''
    if mode == 'fresh':
        base = max(best, latest)
        return -1, (f'从头训练：随机初始化新网络，代数从 {base + 1} 起（不覆盖既有 checkpoint'
                    f'{"，守擂者仍为 " + str(best) if best != -1 else ""}）'), ''
    if mode.isdigit():
        n = int(mode)
        if not exists_fn(n):
            return -1, '', f'指定的起始权重不存在：第 {n} 代'
        if not compat_fn(n):
            return -1, '', f'第 {n} 代权重与当前棋盘表示版本不兼容，不能作起点'
        return n, f'分叉训练：权重起点第 {n} 代，新代数从 {max(latest, n) + 1} 起（不覆盖）', ''
    return -1, '', f'未知起点模式 {mode!r}（可用 auto / fresh / 代数）'
