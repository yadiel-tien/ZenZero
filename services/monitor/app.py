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
    g.strokeStyle = s.color; g.lineWidth = 1.6; g.beginPath();
    s.pts.forEach((p,i) => { const x=X(+p[0]), y=Y(+p[1]); i? g.lineTo(x,y) : g.moveTo(x,y); });
    g.stroke();
    const last = s.pts[s.pts.length-1];
    g.fillStyle = s.color; g.beginPath(); g.arc(X(+last[0]), Y(+last[1]), 2.6, 0, 7); g.fill();
  });
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
  const R = d.rounds, tb = d.tb||{}, hw = d.hw||[];
  draw('c_gpm', {series:[{pts:R.games_per_min, color:'#4aa8ff'}], fmtY:v=>v.toFixed(0)});
  draw('c_batch', {series:[{pts:R.batch_mean, color:'#3ddc84'}], ymin:0, fmtY:v=>v.toFixed(0)});
  draw('c_mcts', {series:[{pts:R.select_ms, color:'#ffb454'},{pts:R.wait_ms, color:'#b48cff'}], ymin:0, fmtY:v=>v.toFixed(1)});
  const lossPts = t => (tb[t]||[]).map(([s,v])=>[s,v]);
  draw('c_loss', {series:[{pts:lossPts('Loss/total'), color:'#4aa8ff'},{pts:lossPts('Val/total'), color:'#ff6b6b'}], fmtY:v=>v.toFixed(2)});
  draw('c_win', {series:[{pts:R.eval_win_rate, color:'#3ddc84'}], ymin:0, ymax:1, hline:0.52, fmtY:v=>(v*100).toFixed(0)+'%'});
  const hwPts = k => hw.filter(s=>s[k]!==null&&s[k]!==undefined).map(s=>[s.t, s[k]]);
  draw('c_hw', {series:[{pts:hwPts('cpu'), color:'#4aa8ff'},{pts:hwPts('gpu'), color:'#3ddc84'},{pts:hwPts('mem'), color:'#ffb454'}],
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
        for key in ('rounds', 'tb', 'hw'):
            assert key in ser, f'series 缺 {key}'
        print(f'selftest OK: rounds={len(snap["rounds"])} hw={snap["hardware"] is not None} '
              f'tb_tags={len(ser["tb"])} 端口={port}')
    finally:
        httpd.monitor.stop()
        httpd.shutdown()
