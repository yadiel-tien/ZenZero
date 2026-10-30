"""训练监控 HTTP 服务：一个自包含页面，不依赖 CDN。

端点：
  GET /            面板（内联 HTML/CSS/JS，Canvas 手绘曲线）
  GET /api/snapshot 当前快照：轮次记录 / 实时指标 / 硬件
  GET /api/series   曲线数据：每轮指标 + TensorBoard 标量 + 硬件历史
  GET /healthz      存活探针

只读：数据来自 logs/、runs/、/proc、nvidia-smi，绝不写训练产物。
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from core.utils.logger import get_logger

from .collect import Monitor

PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>ZenZero 训练监控</title>
<style>
  :root{--bg:#0f1216;--card:#171c23;--line:#2a323d;--fg:#d7dee8;--dim:#8b97a8;
        --blue:#4aa8ff;--green:#3ddc84;--orange:#ffb454;--red:#ff6b6b;--purple:#b48cff}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
       font:13px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif}
  header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;align-items:baseline;gap:16px;flex-wrap:wrap}
  h1{font-size:15px;margin:0;font-weight:600;letter-spacing:.3px}
  .tag{font-size:11px;color:var(--dim);border:1px solid var(--line);border-radius:99px;padding:1px 8px}
  .phase{font-weight:600}
  .phase.selfplay{color:var(--blue)} .phase.fill{color:var(--orange)}
  .phase.fit{color:var(--purple)} .phase.eval{color:var(--green)} .phase.idle{color:var(--dim)}
  .phase.stalled{color:var(--red)}
  body.stalled .cards{opacity:.45}
  #err{color:var(--red);font-size:12px}
  main{padding:16px 20px 32px}
  .cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin-bottom:16px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
  .card .k{font-size:11px;color:var(--dim);margin-bottom:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .card .v{font-size:19px;font-weight:600;font-variant-numeric:tabular-nums}
  .card .s{font-size:11px;color:var(--dim)}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}
  .panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px 6px}
  .panel h2{font-size:12px;margin:0 0 6px;font-weight:600;color:var(--fg);display:flex;gap:10px;align-items:center}
  .panel h2 .lg{font-weight:400;font-size:11px;color:var(--dim)}
  .legend{display:flex;gap:12px;flex-wrap:wrap;font-size:11px;color:var(--dim);margin:-2px 0 5px}
  .legend span{display:inline-flex;align-items:center;gap:5px}
  .legend i{width:11px;height:3px;border-radius:2px;display:inline-block}
  .legend b{color:var(--fg);font-weight:600;font-variant-numeric:tabular-nums}
  .runfilter{display:flex;gap:4px 14px;align-items:center;flex-wrap:wrap;margin:2px 0 10px;font-size:12px;color:var(--dim)}
  .runfilter .rf-lb{font-weight:600;color:var(--fg)}
  .runfilter label{display:inline-flex;align-items:center;gap:5px;cursor:pointer;user-select:none}
  .runfilter input{accent-color:var(--blue);margin:0;cursor:pointer}
  .runfilter .sw{width:10px;height:10px;border-radius:3px;display:inline-block;flex:none}
  .runfilter .rf-now{color:var(--green);font-weight:600;font-size:10px}
  .runfilter .rf-all{color:var(--blue);cursor:pointer;font-size:11px;text-decoration:underline}
  canvas{width:100%;height:170px;display:block}
  .barwrap{height:6px;background:#0b0e12;border-radius:99px;overflow:hidden;margin-top:6px}
  .barwrap i{display:block;height:100%;background:var(--blue);width:0}
  .crashes{margin-top:14px;border:1px solid #4a2b2b;background:#1d1414;border-radius:10px;padding:10px 12px;display:none}
  .crashes pre{margin:6px 0 0;color:#ffb0b0;font-size:11px;white-space:pre-wrap}
</style></head><body>
<header>
  <h1>ZenZero 训练监控</h1>
  <span class="tag" id="meta">—</span>
  <span class="tag">阶段 <b class="phase" id="phase">—</b></span>
  <span class="tag" id="progress">—</span>
  <span class="tag" id="updated">—</span>
  <span id="err"></span>
</header>
<main>
  <div class="cards" id="cards"></div>
  <div class="runfilter" id="runfilter"></div>
  <div class="grid">
    <div class="panel"><h2>对弈吞吐 <span class="lg">局/分钟（每轮）</span></h2><canvas id="c_gpm"></canvas></div>
    <div class="panel"><h2>引擎批大小 <span class="lg">每轮均值</span></h2><canvas id="c_batch"></canvas></div>
    <div class="panel"><h2>MCTS 每模拟耗时 <span class="lg">选叶子 vs 等引擎 (ms)</span></h2><canvas id="c_mcts"></canvas></div>
    <div class="panel"><h2>Loss <span class="lg">train 与 val</span></h2><canvas id="c_loss"></canvas></div>
    <div class="panel"><h2>评估胜率 <span class="lg">晋升阈值 52%</span></h2><canvas id="c_win"></canvas></div>
    <div class="panel"><h2>硬件占用 <span class="lg">CPU / GPU / 内存 %</span></h2><canvas id="c_hw"></canvas></div>
  </div>
  <div class="crashes" id="crashes"><b>线程崩溃记录（logs/crash）</b><pre id="crash_txt"></pre></div>
</main>
<script>
const $ = id => document.getElementById(id);
const fmt = (v, d=1) => (v===null||v===undefined||isNaN(v)) ? '—' : (+v).toFixed(d);
let seriesCache = null;
/* 训练批次（run）：一个进程一次启动 = 一个批次。新后端 /api/series 按批次嵌套返回；
   每个批次一种颜色 + 图例，筛选条可只看某一个批次。旧后端（扁平结构）合成"累计"伪批次。 */
let RUN_STATE = {vis: null};      // null = 显示全部批次；否则为要显示的批次名集合
let RUN_COLOR = {};               // 批次名 → 颜色
function runColors(){ return ['#4aa8ff','#3ddc84','#ffb454','#ff6b6b','#b48cff','#ffd166','#2dd4bf','#f472b6']; }
function normSeries(d){
  if (d && Array.isArray(d.runs)) return d;
  const name = '__legacy__';
  return {runs: [{name, start: 0, running: false, legacy: true}],
          rounds: {[name]: (d || {}).rounds || {}}, tb: {[name]: (d || {}).tb || {}}};
}
function runLabel(r){
  if (r.legacy) return '累计';
  const dte = new Date(r.start * 1000), p = n => String(n).padStart(2, '0');
  return `${p(dte.getMonth()+1)}-${p(dte.getDate())} ${p(dte.getHours())}:${p(dte.getMinutes())}`;
}
function renderRunFilter(runs){
  const el = $('runfilter'); if(!el) return;
  if(!runs || runs.length < 2){ el.innerHTML = ''; return; }
  const vis = RUN_STATE.vis;
  const boxes = runs.map(r => {
    const on = !vis || vis.has(r.name);
    return `<label><input type="checkbox" ${on?'checked':''} data-run="${r.name}">
      <span class="sw" style="background:${RUN_COLOR[r.name]||'#4aa8ff'}"></span>${runLabel(r)}
      ${r.running?'<b class="rf-now">运行中</b>':''}</label>`;
  }).join('');
  el.innerHTML = `<span class="rf-lb">批次</span>${boxes}
    ${vis!==null?'<a class="rf-all" href="javascript:void(0)">全部</a>':''}`;
  el.querySelectorAll('input').forEach(inp => inp.addEventListener('change', () => {
    const all = new Set(runs.map(r => r.name));
    const names = new Set([...el.querySelectorAll('input[data-run]')]
      .filter(i => i.dataset.run !== '__all__' && i.checked).map(i => i.dataset.run));
    RUN_STATE.vis = names.size === all.size ? null : names;
    tickSeries();
  }));
  const allA = el.querySelector('.rf-all');
  if(allA) allA.addEventListener('click', () => { RUN_STATE.vis = null; tickSeries(); });
}

function card(k, v, s){ return `<div class="card"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s||''}</div></div>`; }

/* ---------- 极简 Canvas 折线图 ---------- */
function draw(id, cfg){
  const cv = $(id); if(!cv) return;
  const dpr = window.devicePixelRatio||1;
  const w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w*dpr; cv.height = h*dpr;
  const g = cv.getContext('2d'); g.scale(dpr,dpr);
  g.clearRect(0,0,w,h);
  const P = {l:44, r:10, t:10, b:18};
  const series = (cfg.series||[]).filter(s => s.pts && s.pts.length);
  if(!series.length){ g.fillStyle='#5b6577'; g.font='12px sans-serif'; g.fillText('暂无数据', P.l, h/2); return; }
  let xs = [], ys = [];
  series.forEach(s => s.pts.forEach(p => { xs.push(+p[0]); ys.push(+p[1]); }));
  let x0 = Math.min(...xs), x1 = Math.max(...xs);
  let y0 = cfg.ymin!==undefined ? cfg.ymin : Math.min(...ys);
  let y1 = cfg.ymax!==undefined ? cfg.ymax : Math.max(...ys);
  if(x1===x0){ x0-=1; x1+=1; }
  if(y1===y0){ y1=y0+1; }
  const pad = (y1-y0)*0.08; y0-=pad; y1+=pad;
  const X = v => P.l + (v-x0)/(x1-x0)*(w-P.l-P.r);
  const Y = v => h-P.b - (v-y0)/(y1-y0)*(h-P.t-P.b);
  // 网格 + y 轴刻度
  g.strokeStyle='#232a34'; g.fillStyle='#7b8698'; g.font='10px sans-serif'; g.lineWidth=1;
  for(let i=0;i<=4;i++){
    const v = y0 + (y1-y0)*i/4, y = Y(v);
    g.beginPath(); g.moveTo(P.l,y); g.lineTo(w-P.r,y); g.stroke();
    g.fillText(cfg.fmtY? cfg.fmtY(v) : v.toFixed(Math.abs(y1-y0)<3?2:0), 4, y+3);
  }
  // x 轴首尾
  g.fillText(String(Math.round(x0)), P.l, h-5);
  const xe = String(Math.round(x1)); g.fillText(xe, w-P.r-g.measureText(xe).width, h-5);
  // 线
  series.forEach(s => {
    g.strokeStyle = s.color; g.lineWidth = 1.6; g.setLineDash(s.dash||[]); g.beginPath();
    s.pts.forEach((p,i) => { const x=X(+p[0]), y=Y(+p[1]); i? g.lineTo(x,y) : g.moveTo(x,y); });
    g.stroke();
    const last = s.pts[s.pts.length-1];
    g.fillStyle = s.color; g.beginPath(); g.arc(X(+last[0]), Y(+last[1]), 2.6, 0, 7); g.fill();
  });
  g.setLineDash([]);
  // 图例（≥2 条线才显示；批次/指标都在名字里）
  const lg = document.getElementById(id + '-legend');
  if(series.length >= 2){
    let el = lg;
    if(!el){ el = document.createElement('div'); el.id = id + '-legend'; el.className = 'legend';
             cv.parentNode.insertBefore(el, cv); }
    el.style.display = 'flex';
    el.innerHTML = series.map(s => {
      const last = s.pts[s.pts.length-1];
      const sw = s.dash ? `border:1px dashed ${s.color};background:transparent` : `background:${s.color}`;
      return `<span><i style="${sw}"></i>${s.name||''}<b>${fmt(last[1])}</b></span>`;
    }).join('');
  } else if(lg){ lg.style.display = 'none'; }
  if(cfg.hline!==undefined){
    g.strokeStyle='#3a4352'; g.setLineDash([4,4]); g.beginPath();
    g.moveTo(P.l, Y(cfg.hline)); g.lineTo(w-P.r, Y(cfg.hline)); g.stroke(); g.setLineDash([]);
  }
}

/* ---------- 更新 ---------- */
function renderCards(s){
  const L = s.live||{}, inf = L.infer||{}, fit = L.fit||{}, ev = L.eval||{}, hw = s.hardware||{};
  const last = (s.rounds||[]).slice(-1)[0] || {};
  const sp = last.selfplay || {};
  const gpu = hw.gpu;
  const cards = [
    card('引擎吞吐', fmt(inf.req_s,0)+' <span style="font-size:11px">req/s</span>', '均批 '+fmt(inf.batch,1)+' · pending '+fmt(inf.pending,0)),
    card('每批前向', fmt(inf.fwd,2)+' ms', '收批窗口 '+fmt(inf.window,2)+' ms'),
    card('本轮局/分', fmt(sp.games_per_min,1), 'sims/s '+fmt(sp.sims_per_s,0)),
    card('每模拟耗时', fmt((sp.mcts||{}).select_ms_per_sim,2)+' ms', '等引擎 '+fmt((sp.mcts||{}).wait_ms_per_sim,2)+' ms'),
    card('训练 step', fmt(fit.step,0), fit.loss!==undefined? 'loss '+fmt(fit.loss,4) : '—'),
    card('验证集', fit.val? fmt(fit.val.policy,3) : '—', fit.val? 'value '+fmt(fit.val.value,3) : '—'),
    card('评估胜率', ev.win_rate!==undefined? (ev.win_rate*100).toFixed(1)+'%' : '—',
         ev.games? `${ev.win}-${ev.lose}-${ev.draw} / ${ev.games} 局` : '—'),
    card('CPU', fmt(hw.cpu,1)+'%', (hw.cores||'—')+' 核 · load '+((hw.load||[])[0]||'—')),
    card('GPU', gpu? fmt(gpu.util,0)+'%' : '—',
         gpu? `${fmt(gpu.mem_used_mb/1024,1)}/${fmt(gpu.mem_total_mb/1024,0)} GB · ${fmt(gpu.temp_c,0)}°C · ${fmt(gpu.power_w,0)}W` : '不可用'),
    card('内存', fmt((hw.mem||{}).pct,1)+'%', `${fmt((hw.mem||{}).used_gb,1)}/${fmt((hw.mem||{}).total_gb,1)} GB`),
    card('Buffer', fmt(s.rounds.slice(-1)[0] && s.rounds.slice(-1)[0].buffer_size,0), '样本'),
    card('轮次', fmt((s.rounds||[]).length,0)+' 轮', '最近 it'+fmt(last.iteration,0)),
  ];
  $('cards').innerHTML = cards.join('');
  const ph = (L.phase||'idle');
  document.body.classList.toggle('stalled', ph==='stalled');
  const phEl = $('phase'); phEl.textContent = ph; phEl.className = 'phase '+ph;
  const p = L.progress;
  $('progress').innerHTML = p? `进度 ${p.done}/${p.total||'?'}<div class="barwrap"><i style="width:${p.total?Math.min(100,100*p.done/p.total):0}%"></i></div>` : '—';
  if(ph==='stalled'){ $('progress').innerHTML = `日志已 ${fmt(L.age_s,0)} 秒无更新（进程可能已退出，下列实时指标是残留）`; }
  $('meta').textContent = `${s.meta.game} · profile ${s.meta.profile} · ${s.meta.device} · ${s.meta.net.n_filters}x${s.meta.net.n_res_blocks} · K=${s.meta.batch_leaves}`;
  $('updated').textContent = '更新 '+new Date().toLocaleTimeString();
  const cr = s.crashes||[];
  $('crashes').style.display = cr.length? 'block':'none';
  $('crash_txt').textContent = cr.join('\n');
}

function renderCharts(d){
  const d0 = normSeries(d);
  const runs0 = d0.runs || [];                    // 后端顺序：新 → 旧
  RUN_COLOR = {};
  const pal = runColors();
  runs0.forEach((r, i) => RUN_COLOR[r.name] = pal[i % pal.length]);
  renderRunFilter(runs0);
  const vis = runs0.filter(r => !RUN_STATE.vis || RUN_STATE.vis.has(r.name))
                   .sort((a, b) => a.start - b.start);   // 旧批次在下、最新批次压在上面
  const colorOf = r => RUN_COLOR[r.name];
  const ptsR = (r, m) => ((d0.rounds[r.name]) || {})[m] || [];
  const ptsT = (r, m) => ((d0.tb[r.name]) || {})[m] || [];
  const perRun = (kind, m) => vis.map(r =>
    ({name: runLabel(r) + (r.running ? '（运行中）' : ''), pts: kind === 'r' ? ptsR(r, m) : ptsT(r, m), color: colorOf(r)}));
  const perRun2 = (kind, m1, m2, n1, n2) => vis.flatMap(r => [
    {name: `${runLabel(r)} · ${n1}`, pts: kind === 'r' ? ptsR(r, m1) : ptsT(r, m1), color: colorOf(r)},
    {name: `${runLabel(r)} · ${n2}`, pts: kind === 'r' ? ptsR(r, m2) : ptsT(r, m2), color: colorOf(r), dash: [5,4]}]);
  draw('c_gpm', {series:perRun('r','games_per_min'), fmtY:v=>v.toFixed(0)});
  draw('c_batch', {series:perRun('r','batch_mean'), ymin:0, fmtY:v=>v.toFixed(0)});
  draw('c_mcts', {series:perRun2('r','select_ms','wait_ms','选叶子','等引擎'), ymin:0, fmtY:v=>v.toFixed(1)});
  draw('c_loss', {series:perRun2('t','Loss/total','Val/total','train','val'), fmtY:v=>v.toFixed(2)});
  draw('c_win', {series:perRun('r','eval_win_rate'), ymin:0, ymax:1, hline:0.52, fmtY:v=>(v*100).toFixed(0)+'%'});
  const hwPts = k => (d0.hw||[]).filter(s=>s[k]!==null&&s[k]!==undefined).map(s=>[s.t, s[k]]);
  draw('c_hw', {series:[{name:'CPU',pts:hwPts('cpu'), color:'#4aa8ff'},{name:'GPU',pts:hwPts('gpu'), color:'#3ddc84'},{name:'内存',pts:hwPts('mem'), color:'#ffb454'}],
                ymin:0, ymax:100, fmtY:v=>v.toFixed(0)});
}

async function tick(){
  try{
    const s = await (await fetch('/api/snapshot',{cache:'no-store'})).json();
    renderCards(s); $('err').textContent = '';
    seriesCache = s.hw_history;
  }catch(e){ $('err').textContent = '拉取失败: '+e; }
}
async function tickSeries(){
  try{
    const d = await (await fetch('/api/series',{cache:'no-store'})).json();
    renderCharts(d);
  }catch(e){ $('err').textContent = '曲线拉取失败: '+e; }
}
window.addEventListener('resize', () => seriesCache && tickSeries());
tick(); tickSeries();
setInterval(tick, 2000);
setInterval(tickSeries, 8000);
</script></body></html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    monitor: Monitor = None          # 由 make_server 注入

    def log_message(self, fmt, *args):          # 静音默认的每请求 stderr 日志
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:                    # noqa: N802  (BaseHTTPRequestHandler 约定)
        path = self.path.split('?')[0]
        try:
            if path in ('/', '/index.html'):
                self._send(200, PAGE.encode('utf-8'), 'text/html; charset=utf-8')
            elif path == '/api/snapshot':
                self._send(200, json.dumps(self.monitor.snapshot(), ensure_ascii=False).encode(),
                           'application/json; charset=utf-8')
            elif path == '/api/series':
                self._send(200, json.dumps(self.monitor.series(), ensure_ascii=False).encode(),
                           'application/json; charset=utf-8')
            elif path == '/healthz':
                self._send(200, b'ok', 'text/plain; charset=utf-8')
            else:
                self._send(404, b'not found', 'text/plain; charset=utf-8')
        except Exception as e:                   # noqa: BLE001  面板故障不该拖垮训练
            body = json.dumps({'error': f'{type(e).__name__}: {e}'}, ensure_ascii=False).encode()
            self._send(500, body, 'application/json; charset=utf-8')


def make_server(host: str = '0.0.0.0', port: int = 8099) -> ThreadingHTTPServer:
    monitor = Monitor()
    handler = type('BoundDashboardHandler', (DashboardHandler,), {'monitor': monitor})
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    httpd.monitor = monitor                       # 供调用方 shutdown 时停采样线程
    return httpd


def serve(host: str = '0.0.0.0', port: int = 8099) -> None:
    logger = get_logger('monitor')
    httpd = make_server(host, port)
    logger.info(f'训练监控面板已启动: http://{host}:{port}/  (Ctrl-C 停止)')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info('收到中断，关闭监控面板')
    finally:
        httpd.monitor.stop()
        httpd.shutdown()


def _selftest() -> None:
    """冒烟：起服务、打三个端点、校验 JSON 结构（CI/本地都能跑）。"""
    import urllib.request
    httpd = make_server('127.0.0.1', 0)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        base = f'http://127.0.0.1:{port}'
        assert urllib.request.urlopen(base + '/healthz', timeout=5).read() == b'ok'
        page = urllib.request.urlopen(base + '/', timeout=5).read().decode()
        assert 'ZenZero 训练监控' in page and '/api/snapshot' in page
        snap = json.loads(urllib.request.urlopen(base + '/api/snapshot', timeout=10).read())
        for key in ('ts', 'meta', 'rounds', 'live', 'hardware', 'hw_history', 'crashes'):
            assert key in snap, f'snapshot 缺 {key}'
        assert isinstance(snap['rounds'], list)
        ser = json.loads(urllib.request.urlopen(base + '/api/series', timeout=20).read())
        for key in ('runs', 'rounds', 'tb', 'hw'):
            assert key in ser, f'series 缺 {key}'
        assert isinstance(ser['runs'], list)
        print(f'selftest OK: rounds={len(snap["rounds"])} hw={snap["hardware"] is not None} '
              f'tb_tags={len(ser["tb"])} 端口={port}')
    finally:
        httpd.monitor.stop()
        httpd.shutdown()
