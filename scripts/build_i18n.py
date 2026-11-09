#!/usr/bin/env python3
"""生成 apps/web/js/i18n.js —— ZenZero UI 运行时英文垫片。

词典来源：.audit/dict_out_t1/t2/t3.json（并行翻译产出）+ 手工模板 + 品牌裁决；
垃圾防线：源码残片 key（含 ' " ( ) => :: 的模板切片）永不入典——它们不会出现在 DOM，
只会引入误匹配。数字 key 编译成捕获组正则，value 里 {n} 或第 n 个数字位回填。
"""
import io, json, os, re, sys

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'i18n_data')
BATCHES = [os.path.join(DATA, n) for n in ('dict_out_t1.json','dict_out_t2.json','dict_out_t3.json','dict_out_t5.json')]

# —— 手工裁决层（优先级最高，覆盖批次里的同名 key）——
OVERRIDE = {
    '第 0 代': 'Gen 0',                       # 兜底：修复后理论不再出现
    '禅缘棋境': 'ZenZero',                    # 品牌：英文名即 ZenZero（Daniel 可改）
    '棋道 · 心境': 'The Way of Chess · The Calm of Mind',   # 品牌副题（低赌注，可改）
    '前世': 'Ancestor',
    '王座': 'Throne',
    '引擎吞吐': 'Engine Throughput', '训练步数': 'Training Steps', 'GPU 算力': 'GPU Compute',
    'GPU 显存': 'GPU VRAM', '等待任务启动': 'waiting for task', '自动（优先 GPU）': 'auto (GPU preferred)',
    '训练集': 'Train set', '验证集': 'Validation set', '本进程': 'this process',
    '滚轮缩放 · 拖动平移 · 双击复位': 'wheel zoom · drag pan · double-click resets',
    '拖动平移 · 双指缩放 · 双击复位': 'drag pan · pinch zoom · double-tap resets',
    '展开': 'expand', '收起': 'collapse', '样本': ' samples', '得': '', '分': '',
    '引擎空闲 · 无推理流量': 'engine idle · no inference traffic',
    '等待推理返回': 'waiting for task', '当前自动值': 'current auto values',
    '自适应': 'adaptive', '设备自动': 'device auto', '优先': 'preferred',
    '三分类 logit [负|和|胜]；MCTS 用的期望分 v=p胜−p负 在头外折算':
        'Three-class logits [loss|draw|win]; the expected score v = p(win) − p(loss) used by MCTS is folded outside the head',
    '第一段全连接：把展平向量压到 hidden 个神经元——梯形右缘高度即 ③':
        'First fully-connected layer: flatten vector compressed to hidden neurons — the trapezoid right edge marks ③',
    '1×1 卷积＝逐格通道混合：每个格子的 F 通道线性压到几个通道（本棋类 32），不看邻格、只降维，让后面的 fc 参数省一个数量级':
        "1×1 conv = per-cell channel mixing: each cell's F channels are linearly compressed to a few (32 here); no neighbor view, pure dimensionality reduction — an order of magnitude fewer fc parameters",
    'CE = KL + 熵': 'CE = KL + entropy', 'KL + 熵': 'KL + entropy',
    'Training集': 'Training set', '批 大小': 'Batch size', '大小': ' size',
    '本进程': 'this process', '权重': 'weights', '按需': 'on demand',
    '起→止': 'start→end', 'Custom · 象棋': 'Custom · Chinese Chess',
    '内置 · 双棋种': 'Built-in · both games',
    '当前自动值': 'current auto values', '个历史': ' history runs',
    'ZenZero · 禅缘棋境': 'ZenZero',
    '秒': 's', '纯最强': 'argmax only', '按设备': 'per device',
    '跨机器对战时填写后端地址，例如': 'For cross-machine play, enter the backend address, e.g.',
    '游戏': 'game',
    '池': 'Buffer', '批': 'batch', '桌': 'table', '局': 'games',
    '交换位置': 'Swap sides', '对局操作': 'Game controls',
    '应用版本（神经网络与模型代数见控制台）': 'App version (network & model iteration: see console)',
    '背景音乐': 'Background music', '音乐音量': 'Music volume', '音效音量': 'Sound FX volume', '音效': 'SFX',
    '训练控制台：监控 / 启动停止 / 资产管理': 'Training console: monitor / start-stop / assets',
    '留空 = 与网页同源（推荐：由 play_server 直接托管）': 'Leave blank = same origin as this page (recommended: served by play_server)',
    '托管': 'served',

    '对': 'vs', '值': ' value', '构建': 'build', '版本': 'version', '文档集': 'docs',
    '环境': 'Environment', '训练管线': 'Training Pipeline', '推理引擎': 'Inference Engine',
    '架构与': 'Architecture &', '接口': 'APIs', '应用': 'App', '外观': 'Appearance',
    '跟随系统': 'follow system', '暗色': 'Dark', '明亮': 'Light', '切换': 'switch',
    '永不参与训练': 'never trained on', '永不参与': 'excluded from training', '损失': 'loss',
    '名选手': ' players', '局对局记录': ' rated games',
    '内存': 'Memory',
    '吞吐': 'throughput',
    '搜索效率': 'Search Efficiency',
    '推理耗时': 'Inference',
    '等推理': 'wait inference',
    '选叶': 'pick',
    '步数': 'steps',
    '数据集': 'set',
    '熵': 'entropy',
    '满': 'full',
    'ms/手': 'ms/move',
    '上次': 'last run',
    '上轮': 'last round',
    '最新': 'latest',
    '历史': 'history',
    '收尾后停': 'Stop after round',
    '立即结束': 'Kill now',
    '单次': 'Run once',
    '开': 'On',
    '关': 'Off',
    '删名': 'Remove',
    '新增': 'Add',
    '其它进程': 'Other processes',
    '自动': 'auto',
    '攻': 'vs',
    '暗色': 'Dark',
    '明亮': 'Light',
    '名': '',
    '个历史': 'history runs',
    '王座（冷启动）': 'Throne (cold start)',
    '等待': 'waiting',
    '开始': 'start',
    '进程': 'process',
    '玩家': 'Player',
    '局棋': 'game',
    '个模型': 'models',
    '无推理流量': 'no inference traffic',
    '磁盘余': 'disk free',
    '权重': 'weights',
    '缓存': 'cache',
    '待评估': 'awaiting eval',
    '已排定': 'armed',
    '双棋种': 'both games',
    '每轮': 'per round',
    '线程': 'threads',
    '设备': 'Device',
    '默认': 'default',
    '未选': 'none',
    '当前': 'current',
    '建议': 'suggested',
    '全部': 'all',
    '实时': 'live',
    '尾部': 'tail',
    '目录': 'dir',
    '占用': 'in use',
    '空闲': 'idle',
    '对局中': 'to date',
    '共': 'total',
    '已用': 'used',
    '容量': 'capacity',
    '来源': 'source',
    '用时': 'elapsed',
    '预计': 'est',
    '剩余': 'left',
    '档位': 'network',
    '棋种': 'game',
    '代数': 'iteration',
    '冠军': 'champion',
    '代王座': 'Throne',
}
# 模板（动态插值句，key 用 {N} 占位表达；编译成通用正则后置）
TEMPLATES = [
    (r'第\s*(\d+)\s*代\s*·\s*最佳\s*第\s*(\d+)\s*代', r'Gen $1 · Best $2'),
    (r'上次\s*第\s*(\d+)\s*代\s*·\s*最佳\s*第\s*(\d+)\s*代', r'Last run: Gen $1 · Best $2'),
    (r'第\s*(\d+)\s*代\s*攻\s*(?:第\s*)?(\d+)\s*代王座', r'Gen $1 challenges Throne $2'),
    (r'第\s*(\d+)\s*代\s*攻\s*(?:第\s*)?(\d+)\s*代', r'Gen $1 vs $2'),
    (r'第\s*(\d+)\s*代\s*·\s*王座\s*第\s*(\d+)\s*代', r'Gen $1 · Throne $2'),
    (r'第\s*(\d+)\s*代', r'Gen $1'),
    (r'(\d+)\s*核\s*·\s*load\s*([\d.]+)', r'$1 cores · load $2'),
    (r'(\d+)名\s*·\s*(\d+)\s*局', r'$1 entries · $2 games'),
    (r'(\d+)\s*名选手\s*·\s*(\d+)\s*局对局记录', r'$1 players · $2 rated games'),
    (r'最佳（当前\s*(\d+)）', r'Best (now $1)'),
    (r'holdout\s*(\d+)\s*样本\s*·\s*(\d+)min\s*前', r'holdout $1 samples · $2min ago'),
    (r'（当前自动值：(\d+)\s*线程\s*·\s*每轮\s*(\d+)\s*局\s*K=(\d+)', r'(current auto: $1 threads · $2 games/round K=$3'),
    (r'(\d+)\s*个\s*checkpoint\s*·\s*data\s*([\d.]+)\s*GB\s*·\s*runs\s*([\d.]+)\s*(MB|GB)\s*·\s*磁盘余\s*([\d.]+)\s*GB',
     r'$1 checkpoints · data $2 GB · runs $3 $4 · $5 GB free'),
    (r'胜\s*(\d+)\s*负\s*(\d+)\s*和\s*(\d+)', r'W$1 / L$2 / D$3'),
    (r'v(\d+)：(\d+)局', r'v$1: $2 games'),
    (r'(\d+)\s*局\s*·\s*(\d+)%', r'$1 games · $2%'),
    (r'得分\s*([\d./]+)', r'Score $1'),
    (r'(\d+)min前', r'$1min ago'),
    (r'(\d+)s前', r'$1s ago'),
    (r'(\d+) 秒', r'\1s'),
    (r'(\d+)秒', r'\1s'),
    (r'(\d+)%\s*满', r'$1% full'),
    (r'(\d+)\s*名\s*·\s*(\d+)\s*局', r'\1 players · \2 games'),
    (r'checkpoint\s*(\d+)\s*个', r'$1 checkpoints'),
    (r'(?<![\w])(\d+)\s*个(?=[\s·])', r'\1'),
    (r'(\d+)\s*个\s*checkpoint', r'\1 checkpoints'),
    (r'约\s*(\d+)[~至](\d+)\s*?min', r'~$1-$2 min'),
    (r'应用版本（构建\s*([0-9a-zA-Z.\-+]+)）', r'App version (build $1)'),
    (r'项目文档集：环境\s*/\s*MCTS\s*/\s*推理引擎\s*/\s*训练管线\s*/\s*架构与\s*API',
     r'ZenZero docs: environment / MCTS / inference engine / training pipeline / architecture & APIs'),
    (r'项目文档集：环境\s*/\s*MCTS\s*/\s*推理引擎\s*/\s*训练管线\s*/\s*架构与接口',
     r'ZenZero docs: environment / MCTS / inference engine / training pipeline / architecture & APIs'),
]

def junk_key(k):
    return bool(re.search(r'''["'`]|{ | }|\(\)|=>|::|\bfmt\(|\bconst\b|function''', k))

TOOL_DESC_EN = {
    'piskvork': 'Serve Gomoku over the Gomocup protocol so Piskvork-compatible GUIs and external engines can join.\nPick network & model, hit Start, then point the opponent at <host-IP>:<port> (default 12345); games auto-match the opponent think limit; Stop releases VRAM.',
    'bench': 'Measures engine latency across batch sizes to separate CPU-search from GPU-inference bottlenecks; run it after throughput anomalies or batching tweaks.\nStart with defaults or set batches (e.g. 1,8,32,128) and iters; ~1-2 min; avoid training/game windows; the comparison table lands in logs/perf/batch_sweep.json.',
    'build_cython': 'Rebuilds the C++ acceleration extensions (move generation, etc.); needed after editing Cython sources under core/env.\nJust hit Start: it compiles to staging and swaps atomically on success, keeps the previous build on failure, and is safe to run while training is active. Check mode only verifies existing extensions.',
}


def load():
    d = {}
    for f in BATCHES:
        try:
            d.update(json.load(io.open(f, encoding='utf-8')))
        except FileNotFoundError:
            print('skip missing', f)
        except Exception as e:
            print('BAD JSON', f, e); sys.exit(1)
    drop = [k for k in d if junk_key(k)]
    for k in drop:
        del d[k]
    d.update(OVERRIDE)
    # 工具页长描述：key 直接取任务注册表原文，永不手抄
    try:
        import importlib.util as _il
        _spec = _il.spec_from_file_location('_zz_tasks', os.path.join(
            os.path.dirname(__file__), '..', 'services', 'admin', 'tasks.py'))
        _m = _il.module_from_spec(_spec); _spec.loader.exec_module(_m)
        for _nm, _t in _m.TASKS.items():
            _ds = _t.get('desc')
            if _nm in TOOL_DESC_EN and isinstance(_ds, str):
                d[_ds] = TOOL_DESC_EN[_nm]
    except Exception as _e:
        print('tool desc inject skipped:', _e)

    return d, len(drop)

JS = """/* ZenZero UI i18n shim — 自动生成的词典 + 引擎（勿手改；重新生成：python3 scripts/build_i18n.py）
   默认中文（DOM 原样，零成本）；?lang=en 或右下角药丸切英文：词典整串最长优先替换
   文本节点 / title / placeholder / aria-label / document.title / canvas fillText。
   棋盘美术字与棋谱记谱属"数据"永不翻译（#board-wrap / #moves / .note 子树跳过）。 */
(function () {
  var RAW = __DICT__;
  var SKIP = '#board-wrap,#moves,.note,#task-log,#log_view,pre';
  var isNumKey = /[0-9]/;
  function esc(s){ return s.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&'); }
  var pats = [];
  var EXACT = {};
  Object.keys(RAW).forEach(function (k) {
    var v = RAW[k];
    // 短中文条目（≤2 字且无 ASCII）只做整节点精确匹配——否则单字"对"会污染"对半退避"这类词
    if (k.length === 1 && !/[A-Za-z0-9]/.test(k)) { EXACT[k] = v; return; }
    if (isNumKey.test(k)) {
      var n = (k.match(/\\d+(?:[.,]\\d+)?/g) || []).length;
      var src = esc(k).replace(/\\d+(?:[.,]\\d+)?/g, '(\\\\d+(?:[.,]\\\\d+)?)');
      var rep;
      if (/\\{\\d+\\}/.test(v)) { rep = v.replace(/\\{(\\d+)\\}/g, function(_, x){ return '$' + x; }); }
      else {
        var j = 0;
        rep = v.replace(/\\d+(?:[.,]\\d+)?/g, function(){ return '$' + (++j); });
        if (j !== n) { pats.push({ lit: k, rep: v, len: k.length }); return; }
      }
      pats.push({ re: new RegExp(src, 'g'), rep: rep, len: k.length, lit: null }); return;
    }
    pats.push({ lit: k, rep: v, len: k.length });
  });
  pats.sort(function (a, b) { return b.len - a.len; });
  var TPL = __TPL__.slice().sort(function(a,b){ return b[0].length - a[0].length; });

  function tr(s) {
    if (!s || !/[\\u4e00-\\u9fff]/.test(s)) return s;
    var st = s.trim();
    if (Object.prototype.hasOwnProperty.call(EXACT, st)) return s.replace(st, EXACT[st]);
    var out = s;
    // 先整句模板（长句优先），再词条级替换——避免"第 N 代"词条抢先把复合句嚼碎
    for (var j = 0; j < TPL.length; j++) out = out.replace(new RegExp(TPL[j][0], 'g'), TPL[j][1].replace(/\\\\(\\\\d)/g, '$$$1'));
    for (var i = 0; i < pats.length; i++) {
      var p = pats[i];
      if (p.lit) { if (out.indexOf(p.lit) >= 0) out = out.split(p.lit).join(p.rep); }
      else if (p.re.test(out)) { p.re.lastIndex = 0; out = out.replace(p.re, p.rep); }
    }
    return out;
  }

  var lang = new URLSearchParams(location.search).get('lang')
          || localStorage.getItem('zz_lang_ui') || 'zh';
  if (new URLSearchParams(location.search).get('lang')) localStorage.setItem('zz_lang_ui', lang);
  var pill = null;

  function walk(root) {
    if (lang !== 'en') return;
    try { window.__zzStat && (window.__zzStat.walks++); } catch (e) {}
    var w = document.createTreeWalker(root || document.body, NodeFilter.SHOW_TEXT);
    var n, edits = 0;
    while ((n = w.nextNode())) {
      if (!/[\\u4e00-\\u9fff]/.test(n.nodeValue)) continue;
      var par = n.parentElement;
      if (!par || par.closest(SKIP) || par.closest('[data-zzui]') || /SCRIPT|STYLE/.test(par.tagName)) continue;
      var t = tr(n.nodeValue);
      if (t !== n.nodeValue) { editing = true; n.nodeValue = t; editing = false; edits++; }
    }
    if (!root) {
      var els = document.querySelectorAll('[title],[placeholder],[aria-label]');
      for (var i = 0; i < els.length; i++) {
        var el = els[i];
        if (el.closest && (el.closest(SKIP) || el.closest('[data-zzui]'))) continue;
        ['title', 'placeholder', 'aria-label'].forEach(function (a) {
          var v = el.getAttribute(a);
          if (v && /[\\u4e00-\\u9fff]/.test(v)) { var t2 = tr(v); if (t2 !== v) el.setAttribute(a, t2); }
        });
      }
      var dt = tr(document.title);
      if (dt !== document.title) document.title = dt;
    }
    try { if (window.__zzStat) window.__zzStat.edits += edits || 0; } catch (e) {}
    return edits;
  }

  var editing = false, queued = false;
  function schedule() {
    if (queued || editing) return;
    queued = true;
    requestAnimationFrame(function () { queued = false; walk(); });   // rAF 必到帧：图表密集重绘下也不留中文闪现
  }

  function mkPill() {
    pill = document.createElement('div');
    pill.setAttribute('data-zzui', '1');
    pill.style.cssText = 'position:fixed;right:10px;bottom:12px;z-index:9999;font:12px/1 system-ui;display:flex;gap:0;border:1px solid rgba(127,127,127,.5);border-radius:99px;overflow:hidden;background:rgba(0,0,0,.35);backdrop-filter:blur(3px)';
    pill.innerHTML = '<a data-l="zh" style="padding:5px 10px;color:#eee;cursor:pointer;text-decoration:none">中</a>'
                   + '<a data-l="en" style="padding:5px 10px;color:#eee;cursor:pointer;text-decoration:none">EN</a>';
    function paint() {
      pill.querySelectorAll('a').forEach(function (a) {
        var on = a.dataset.l === lang;
        a.style.background = on ? '#2f4f4f' : 'transparent';
        a.style.color = on ? '#f0d9a8' : '#bbb';
      });
    }
    pill.addEventListener('click', function (e) {
      var l = e.target.dataset && e.target.dataset.l;
      if (!l || l === lang) return;
      localStorage.setItem('zz_lang_ui', l);
      location.reload();
    });
    document.body.appendChild(pill);
    paint();
  }

  // canvas 文字（图表轴/徽标为 JS 绘制，DOM 走不到——包一层 fillText）
  if (lang === 'en' && window.CanvasRenderingContext2D) {
    var proto = CanvasRenderingContext2D.prototype, ft = proto.fillText;
    proto.fillText = function (t) {
      arguments[0] = (typeof t === 'string') ? tr(t) : t;
      return ft.apply(this, arguments);
    };
    var st = proto.strokeText;
    proto.strokeText = function (t) {
      arguments[0] = (typeof t === 'string') ? tr(t) : t;
      return st.apply(this, arguments);
    };
  }

  function boot() {
    window.__zzStat = { walks: 0, edits: 0, errs: [] };
    window.__tr = tr;
    mkPill();
    if (lang !== 'en') return;
    walk();
    new MutationObserver(function (ms) {
      for (var i = 0; i < ms.length; i++) {
        var m = ms[i];
        if (editing) return;
        if (m.type === 'characterData' && m.target && m.target.parentElement && m.target.parentElement.closest('[data-zzui]')) continue;
        if (m.type === 'attributes' && m.target && m.target.closest && m.target.closest('[data-zzui]')) continue;
        schedule(); return;
      }
    }).observe(document.body, { subtree: true, childList: true, characterData: true,
                                attributes: true, attributeFilter: ['title', 'placeholder', 'aria-label'] });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
"""

def main():
    d, dropped = load()
    dict_json = json.dumps(d, ensure_ascii=False, separators=(',', ':'))
    tpl_json = json.dumps(TEMPLATES, ensure_ascii=False)
    out = JS.replace('__DICT__', dict_json).replace('__TPL__', tpl_json)
    io.open('apps/web/js/i18n.js', 'w', encoding='utf-8').write(out)
    print(f'i18n.js written: {len(d)} entries ({dropped} junk keys filtered), {len(out)//1024} KB')

if __name__ == '__main__':
    main()
