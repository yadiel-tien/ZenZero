
const $ = id => document.getElementById(id);
const fmt = (v,d=1) => (v===null||v===undefined||isNaN(v)) ? '—' : (+v).toFixed(d);
const gb = b => b>=2**30 ? (b/2**30).toFixed(2)+' GB' : b>=2**20 ? (b/2**20).toFixed(1)+' MB' : (b/1024).toFixed(0)+' KB';
const secs = s => s===null||s===undefined ? '—' : s<90 ? `${s.toFixed(0)}s` : s<5400 ? `${(s/60).toFixed(1)}min` : `${(s/3600).toFixed(1)}h`;
const esc = s => String(s??'').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

function toast(msg, ok=false){
  const t = $('toast'); t.innerHTML = ansiHtml(msg); t.className = 'toast' + (ok?' ok':'');
  t.style.display = 'block'; clearTimeout(t._h); t._h = setTimeout(()=>t.style.display='none', ok?3500:9000);
}
async function api(path, opt={}){
  const headers = {'Content-Type':'application/json'};
  const tk = localStorage.getItem('zz_admin_token'); if(tk) headers['X-ZZ-Token'] = tk;
  const r = await fetch(path, {headers, cache:'no-store', ...opt});
  const txt = await r.text();
  let data; try{ data = JSON.parse(txt); }catch(_){ data = {error:txt.slice(0,300)}; }
  if(!r.ok) throw new Error((data && data.error) || ('HTTP '+r.status));
  return data;
}

/* ---------- 主题：自动/明亮/暗色 ---------- */
let PAL = {};
function refreshPalette() {
  const cs = getComputedStyle(document.documentElement);
  const v = (n, f) => (cs.getPropertyValue(n) || '').trim() || f;
  PAL = { blue: v('--blue', '#4aa8ff'), green: v('--green', '#3ddc84'),
          orange: v('--orange', '#ffb454'), red: v('--red', '#ff6b6b'),
          purple: v('--purple', '#b48cff'), yellow: v('--yellow', '#ffd166'),
          grid: v('--grid', '#232a34'), axis: v('--axis', '#7b8698'),
          dim: v('--dim', '#8b97a8'), line: v('--line', '#2a323d') };
  return PAL;
}
refreshPalette();

const THEME_MODES = ['auto', 'light', 'dark'];
const THEME_LABEL = { auto: '🌗 自动', light: '☀️ 明亮', dark: '🌙 暗色' };
let LAST_CHARTS = null;

function themeMode() {
  try { return localStorage.getItem('zz-theme') || 'auto'; } catch (e) { return 'auto'; }
}
function applyTheme(mode, persist) {
  if (persist !== false) { try { localStorage.setItem('zz-theme', mode); } catch (e) {} }
  const sysDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  const dark = mode === 'dark' || (mode === 'auto' && sysDark);
  document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  const btn = $('btn-theme');
  if (btn) { btn.textContent = THEME_LABEL[mode]; btn.title = `外观：自动（跟随系统） → 明亮 → 暗色　（当前 ${THEME_LABEL[mode]}）`; }
  refreshPalette();
  if (LAST_CHARTS) renderCharts(LAST_CHARTS);      // 立刻按新配色重画
}
function cycleTheme() {
  const i = THEME_MODES.indexOf(themeMode());
  applyTheme(THEME_MODES[(i + 1) % THEME_MODES.length]);
}
if (window.matchMedia) {                            // 系统主题变化时，自动模式下实时跟随
  try {
    window.matchMedia('(prefers-color-scheme: dark)')
      .addEventListener('change', () => { if (themeMode() === 'auto') applyTheme('auto', false); });
  } catch (e) {}
}

/* ---------- 极简 Canvas 折线（带缩放/平移） ---------- */
const VIEW = {};        // id → {x0,x1}  手动视图；空=自动适应
const LASTCFG = {};     // id → cfg      重绘用

function dataRange(cfg){
  let xs = [];
  (cfg.series||[]).forEach(s => (s.pts||[]).forEach(p => xs.push(+p[0])));
  return xs.length ? {x0: Math.min(...xs), x1: Math.max(...xs)} : {x0: 0, x1: 1};
}

function attachChart(id){
  const cv = $(id);
  if(!cv || cv._zoomed) return;
  cv._zoomed = true; cv.style.cursor = 'crosshair';
  cv.addEventListener('wheel', e => {
    const cfg = LASTCFG[id]; if(!cfg) return;
    e.preventDefault();
    const R = dataRange(cfg), cur = VIEW[id] || {x0: R.x0, x1: R.x1};
    const span = Math.max(cur.x1 - cur.x0, 1e-9);
    const rect = cv.getBoundingClientRect();
    const frac = Math.min(1, Math.max(0, (e.clientX - rect.left - 46) / Math.max(1, rect.width - 56)));
    const anchor = cur.x0 + frac * span;
    const f = e.deltaY < 0 ? 0.8 : 1.25;                  // 上滚放大
    const nspan = Math.min(R.x1 - R.x0, Math.max(span * f, (R.x1 - R.x0) * 1e-3));
    let nx0 = anchor - frac * nspan, nx1 = nx0 + nspan;
    if(nx0 < R.x0){ nx0 = R.x0; nx1 = Math.min(R.x1, nx0 + nspan); }
    if(nx1 > R.x1){ nx1 = R.x1; nx0 = Math.max(R.x0, nx1 - nspan); }
    VIEW[id] = {x0: nx0, x1: nx1}; draw(id, cfg);
  }, {passive: false});
  let drag = null;
  cv.addEventListener('mousedown', e => { drag = {x: e.clientX, cfg: LASTCFG[id], view: VIEW[id] || null}; cv.style.cursor = 'grabbing'; });
  window.addEventListener('mouseup', () => { if(drag){ drag = null; cv.style.cursor = 'crosshair'; } });
  cv.addEventListener('mousemove', e => {
    if(!drag || !drag.cfg) return;
    const R = dataRange(drag.cfg), cur = drag.view || {x0: R.x0, x1: R.x1};
    const span = Math.max(cur.x1 - cur.x0, 1e-9);
    const rect = cv.getBoundingClientRect();
    const dx = (e.clientX - drag.x) / Math.max(1, rect.width - 56) * span;
    let nx0 = cur.x0 - dx, nx1 = cur.x1 - dx;
    if(nx0 < R.x0){ nx0 = R.x0; nx1 = R.x0 + span; }
    if(nx1 > R.x1){ nx1 = R.x1; nx0 = R.x1 - span; }
    VIEW[id] = {x0: nx0, x1: nx1}; draw(id, drag.cfg);
  });
  cv.addEventListener('dblclick', () => { delete VIEW[id]; draw(id, LASTCFG[id]); });
}

function renderLegend(id, cfg){
  const cv = $(id); if(!cv) return;
  const series = (cfg.series || []).filter((s) => (s.pts || []).length);
  let lg = document.getElementById(id + '-legend');
  if (series.length < 2) { if (lg) lg.style.display = 'none'; return; }
  if (!lg) {
    lg = document.createElement('div');
    lg.id = id + '-legend'; lg.className = 'legend';
    cv.parentNode.insertBefore(lg, cv);          // 放在标题和图之间
  }
  lg.style.display = 'flex';
  lg.innerHTML = series.map((s) => {
    const last = s.pts[s.pts.length - 1];
    const val = s.fmtLast ? s.fmtLast(last[1]) : fmt(last[1], 2);
    return `<span><i style="background:${s.color}"></i>${s.name || ''}<b>${val}</b></span>`;
  }).join('');
}

function draw(id, cfg){
  LASTCFG[id] = cfg;
  const cv = $(id); if(!cv || !cv.clientWidth) return;
  renderLegend(id, cfg);
  const dpr = window.devicePixelRatio||1, w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w*dpr; cv.height = h*dpr;
  const g = cv.getContext('2d'); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,w,h);
  const P = {l:46,r:10,t:10,b:26};
  const ss = (cfg.series||[]).filter(s=>s.pts && s.pts.length);
  if(!ss.length){ g.fillStyle=PAL.axis; g.font='12px sans-serif'; g.fillText('暂无数据', P.l, h/2); return; }
  const R = dataRange(cfg);
  const view = VIEW[id];
  let x0 = view ? view.x0 : R.x0, x1 = view ? view.x1 : R.x1;
  let ys=[]; ss.forEach(s=>s.pts.forEach(p=>{ if(+p[0]>=x0 && +p[0]<=x1) ys.push(+p[1]); }));
  if(!ys.length) ys = [0, 1];
  let y0=cfg.ymin!==undefined?cfg.ymin:Math.min(...ys), y1=cfg.ymax!==undefined?cfg.ymax:Math.max(...ys);
  if(x1===x0){x0-=1;x1+=1;} if(y1===y0){y1=y0+1;}
  const pad=(y1-y0)*0.08; y0-=pad; y1+=pad;
  const X=v=>P.l+(v-x0)/(x1-x0)*(w-P.l-P.r), Y=v=>h-P.b-(v-y0)/(y1-y0)*(h-P.t-P.b);
  g.strokeStyle=PAL.grid; g.fillStyle=PAL.axis; g.font='10px sans-serif';
  for(let i=0;i<=4;i++){ const v=y0+(y1-y0)*i/4, y=Y(v);
    g.beginPath(); g.moveTo(P.l,y); g.lineTo(w-P.r,y); g.stroke();
    g.fillText(cfg.fmtY?cfg.fmtY(v):v.toFixed(2), 3, y+3); }
  // 横轴：两端标刻度值，底部居中写明这条轴是什么
  const xf = cfg.xFmt || ((v) => String(Math.round(v)));
  g.font = '10px sans-serif'; g.fillStyle = PAL.axis;
  g.fillText(xf(x0), P.l, h - 15);
  const xe = xf(x1); g.fillText(xe, w - P.r - g.measureText(xe).width, h - 15);
  if (cfg.xLabel) {
    g.fillStyle = PAL.dim; g.font = '10px sans-serif';
    const t = cfg.xLabel;
    g.fillText(t, P.l + (w - P.l - P.r - g.measureText(t).width) / 2, h - 3);
  }
  ss.forEach(s=>{ g.strokeStyle=s.color; g.lineWidth=1.6; g.beginPath();
    s.pts.forEach((p,i)=>{ const x=X(+p[0]), y=Y(+p[1]); i?g.lineTo(x,y):g.moveTo(x,y); }); g.stroke();
    const last=s.pts[s.pts.length-1]; g.fillStyle=s.color; g.beginPath(); g.arc(X(+last[0]),Y(+last[1]),2.6,0,7); g.fill(); });
  if(cfg.hline!==undefined){ g.strokeStyle=PAL.axis; g.setLineDash([4,4]); g.beginPath();
    g.moveTo(P.l,Y(cfg.hline)); g.lineTo(w-P.r,Y(cfg.hline)); g.stroke(); g.setLineDash([]); }
}

/* ---------- 监控 ---------- */
function card(k,v,s){ return `<div class="card"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s||''}</div></div>`; }
function renderSnapshot(s){
  const L=s.live||{}, inf=L.infer||{}, fit=L.fit||{}, ev=L.eval||{}, hw=s.hardware||{}, tr=s.trainer||{};
  const last=(s.rounds||[]).slice(-1)[0]||{}, sp=last.selfplay||{}, gpu=hw.gpu;
  // Buffer：优先用心跳里的实时值（整轮会一直在涨），没有才退回上一轮 perf 记录
  const sims = (s.heartbeat && s.heartbeat.sims_per_move) || sp.sims_per_move;
  const buf = (L.buffer && L.buffer.size != null)
      ? {size: L.buffer.size, capacity: L.buffer.capacity, live: true}
      : {size: last.buffer_size, capacity: last.buffer_capacity, live: false};
  const inferStale = (inf.age_s||0) > 15;
  const gpmLive = (L.gpm !== undefined && L.gpm !== null) ? L.gpm : null;
  $('cards').innerHTML = [
    card('引擎吞吐', fmt(inf.req_s,0)+' <span style="font-size:11px">req/s</span>',
         (inferStale? `日志 ${fmt(inf.age_s,0)}s 未更新 · ` : '') +
         'batch 均值 '+fmt(inf.batch,0)+' · 排队 '+fmt(inf.pending,0),
         '推理引擎每秒处理多少个局面（瞬时值，用相邻两条日志的 total 之差算）。batch 均值 = 每次前向一起算几个局面，越大越省 GPU。排队 = 已进队列、还没被凑进 batch 的请求数。'),
    card('前向耗时', fmt(inf.fwd,2)+' ms', '凑批等待 '+fmt(inf.window,2)+' ms',
         '一次 GPU 前向要多久；凑批等待 = 为了凑出更大的 batch 而多等的毫秒数。GPU 每次前向有固定开销，所以 batch 越大，摊到每个局面上越便宜。'),
    card('每手模拟', sims? fmt(sims,0) : '—', '并行 leaf K=' + fmt(s.meta.batch_leaves,0),
         'MCTS 的每一步搜索多少次（随代数从 400 爬到 800）。K = 一手棋同时挂起的并行 leaf 数（virtual loss），决定 batch 能凑多大。'),
    card('局/分钟', gpmLive!==null? fmt(gpmLive,1) : fmt(sp.games_per_min,1),
         (gpmLive!==null? '本轮实时' : '上轮 ') + ' · 模拟/秒 '+fmt(sp.sims_per_s,0),
         '自博弈吞吐。轮内按"已完成局数 ÷ 已用时间"实时算；没有心跳时退回上一轮的记录。看速度要同时看 sims/秒，因为每手模拟次数在爬坡。'),
    card('单次模拟耗时', fmt((sp.mcts||{}).select_ms_per_sim,2)+' ms',
         '选叶子 '+fmt((sp.mcts||{}).select_ms_per_sim,2)+' · 等推理 '+fmt((sp.mcts||{}).wait_ms_per_sim,2)+'（上轮）',
         '每模拟一次的时长，拆成"CPU 选叶子"和"等推理返回"两段。两者之比说明瓶颈在 CPU 搜索还是 GPU 推理。'),
    card('训练步数',
         fit.step? fmt(fit.step,0) + (fit.fit_total_steps? `<span style="font-size:12px;color:var(--dim)"> / ${fmt(fit.fit_total_steps,0)}</span>`:'') : '—',
         fit.loss!==undefined? `loss ${fmt(fit.loss,3)} · policy ${fmt(fit.policy_loss,3)} · value ${fmt(fit.value_loss,3)}`
             : (fit.fit_seconds? `本轮 fit 用时 ${fmt(fit.fit_seconds,1)}s` : '—'),
         '本轮 fit 已走的优化步数 / 本轮总步数（每轮约 8 遍 × Buffer ÷ batch = 1171 步）。loss 是最近一步的总损失，后面拆成 policy / value。'),
    card('验证集', fit.val?fmt(fit.val.policy,3):'—',
         fit.val? `value ${fmt(fit.val.value,3)}` + (fit.val.kl!==undefined? ` · KL ${fmt(fit.val.kl,3)}`:'')
                  + (fit.val.n_samples? ` · ${fmt(fit.val.n_samples,0)} 样本`:'')
                  + (fit.val.stale? ' · 上一代' : ((fit.val.age_s||0) > 600? ` · ${fmt((fit.val.age_s||0)/60,0)} 分钟前` : '')) : '—',
         '验证集（holdout，永不参与训练）上的 loss。policy 是交叉熵，其中约 85% 是目标自身熵的下限，真正能学的是 KL。'
         + ' value 是价值头 MSE。标“上一代”表示本轮 fit 还没跑完，显示的是上一个模型的测量值。'),
    card('评估胜率', ev.win_rate!==undefined?(ev.win_rate*100).toFixed(1)+'%':'—',
         ev.games? `胜 ${ev.win} · 负 ${ev.lose} · 和 ${ev.draw}（共 ${ev.games} 局）· 得分 ${fmt(ev.win + ev.draw/2,1)}/${ev.games}`
             : '—',
         `新模型（第 ${fmt(ev.new||0,0)} 代）对守擂模型（第 ${fmt(ev.best||0,0)} 代）的战绩。`
         + `和棋算半分，得分率超过门槛 ${(100*((s.meta||{}).win_threshold||0.52)).toFixed(0)}% 就晋升为新的守擂模型。`
         + ' 例：胜 28 · 负 10 · 和 49 = 得分 52.5/87 = 60.3%。'),
    card('CPU', fmt(hw.cpu,1)+'%', (hw.cores||'—')+' 核 · load '+((hw.load||[])[0]||'—'),
         '整机 CPU 占用。selfplay 阶段 MCTS 选叶子是主要消耗。'),
    card('GPU', gpu?fmt(gpu.util,0)+'%':'—', gpu?`${fmt(gpu.mem_used_mb/1024,1)}/${fmt(gpu.mem_total_mb/1024,0)} GB · ${fmt(gpu.temp_c,0)}°C · ${fmt(gpu.power_w,0)}W`:'不可用',
         'GPU 利用率/显存/温度/功耗。自博弈时推理是间歇的，利用率天然有波动；fit 阶段会拉满。'),
    card('内存', fmt((hw.mem||{}).pct,1)+'%', `${fmt((hw.mem||{}).used_gb,1)}/${fmt((hw.mem||{}).total_gb,1)} GB`,
         '物理内存占用（Buffer 常驻内存占大头）。'),
    card('Buffer', buf.size==null? '—' : (buf.capacity? `${(buf.size/1000).toFixed(1)}k<span style="font-size:12px;color:var(--dim)"> / ${(buf.capacity/1000).toFixed(0)}k</span>` : fmt(buf.size,0)),
         (buf.capacity? `样本 ${(100*buf.size/buf.capacity).toFixed(0)}% 满 · ` : '样本 · ') + (buf.live? '实时' : '上轮记录'),
         '训练池样本数 / 容量。满了会淘汰最旧的样本。数值来自训练进程心跳，轮内实时增长。'),
    card('已完成轮数', fmt((s.rounds||[]).length,0)+' 轮', '第 '+fmt(L.iteration||last.iteration,0)+' 代 · 每轮 '+fmt(s.meta.games_per_round,0)+' 局',
         '完成的自博弈 + 训练轮数（跨重启累计）。“代”是模型版本号，每完成一轮加 1。'),
  ].join('');
  $('appver').textContent = 'v' + (s.meta.version || '?');
  $('appver').title = '应用版本（构建 ' + (s.meta.revision || '未知') + '）· 表示/网络/模型代数见下行';
  $('meta').textContent = `${s.meta.game} · ${s.meta.profile} · ${s.meta.device} · ${s.meta.net.n_filters}x${s.meta.net.n_res_blocks}`
      + ` · K=${s.meta.batch_leaves}` + (sims? ` · ${fmt(sims,0)} sims/手` : '')
      + (s.meta.profile_source==='default'? '（神经网络取面板默认值，训练未在跑）' : '');
  window.__lastProgress = L.progress || {};
  const ph = L.phase || 'idle', p = L.progress, running = tr.running;
  let chip = 'pill ' + (ph==='stalled'?'stalled':''), txt;
  if(!running){
    txt = '未运行' + (tr.exit_code!=null? ` · 上次退出码 ${tr.exit_code}` : '');
  } else {
    txt = ph + (p && p.done!=null? ` ${p.done}/${p.total||'?'}` : '');
    if(L.source === 'heartbeat') txt += ` · 心跳 ${fmt(L.age_s,1)}s 前`;
    else if((L.age_s||0) > 30){ chip += ' stalled'; txt += ` · 日志 ${fmt(L.age_s,0)}s 未更新`; }
  }
  $('phase').className = chip; $('phase').textContent = txt;
  document.body.classList.toggle('stalled', !running);
  // 训练进程
  $('trainer').textContent = running? `训练中 pid ${tr.pid} · ${secs(tr.uptime_s)}` : '训练未运行';
  $('trainer').className = 'pill ' + (running?'run':'stop');
  $('btn-stop').disabled = !running;
  $('btn-start2').disabled = running;
  const pending = !!tr.stop_after_round;
  $('btn-stop-round').disabled = !running || pending;
  $('btn-stop').disabled = !running;
  $('pending-stop').style.display = pending? 'inline-block' : 'none';
  if(pending) $('phase').textContent = '待退出 · ' + $('phase').textContent;
  TRAIN_RUNNING = running;
  const tn = $('trainer_now');
  if(tn) tn.textContent = running ? `● 训练进程运行中 pid ${tr.pid} · ${Math.round((tr.uptime_s||0)/60)} 分钟`
    : (tr.exit_code!==null && tr.exit_code!==undefined ? `○ 训练进程上次退出码 ${tr.exit_code}` : '○ 训练进程未运行');
  const cr=s.crashes||[]; $('crashes').style.display = cr.length?'block':'none'; $('crash_txt').textContent = cr.join('\n');
}
function hwClock(v){
  const d = new Date((typeof v === 'number' ? v : 0) * 1000);
  if (isNaN(d.getTime())) return '';
  return `${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`;
}
function renderCharts(d){
  LAST_CHARTS = d;
  refreshPalette();
  const R=d.rounds||{}, tb=d.tb||{}, hw=d.hw||[];
  draw('c_gpm',{series:[{name:'局/分钟',pts:R.games_per_min,color:PAL.blue}],fmtY:v=>v.toFixed(0),fmtLast:v=>v.toFixed(1),xLabel:'累计轮数'});
  draw('c_batch',{series:[{name:'batch 均值',pts:R.batch_mean,color:PAL.green}],ymin:0,fmtY:v=>v.toFixed(0),xLabel:'累计轮数'});
  draw('c_mcts',{series:[{name:'选叶子',pts:R.select_ms,color:PAL.orange},{name:'等推理返回',pts:R.wait_ms,color:PAL.purple}],
       ymin:0,fmtY:v=>v.toFixed(1),fmtLast:v=>v.toFixed(2),xLabel:'累计轮数'});
  const lp=t=>(tb[t]||[]).map(([s,v])=>[s,v]);
  draw('c_loss_p',{series:[{name:'train',pts:lp('Loss/policy'),color:PAL.blue},{name:'val',pts:lp('Val/policy'),color:PAL.red}],fmtY:v=>v.toFixed(2),xLabel:'训练步数（step）'});
  draw('c_loss_v',{series:[{name:'train',pts:lp('Loss/value'),color:PAL.purple},{name:'val',pts:lp('Val/value'),color:PAL.yellow}],fmtY:v=>v.toFixed(3),xLabel:'训练步数（step）'});
  draw('c_kl',{series:[{name:'KL（可学部分）',pts:lp('Val/policy_kl'),color:PAL.green},{name:'目标熵（下限）',pts:lp('Val/policy_entropy'),color:PAL.dim}],
       ymin:0,fmtY:v=>v.toFixed(2),xLabel:'训练步数（step）'});
  draw('c_win',{series:[{name:'评估胜率',pts:R.eval_win_rate,color:PAL.green}],ymin:0,ymax:1,hline:0.52,fmtY:v=>(v*100).toFixed(0)+'%',fmtLast:v=>(v*100).toFixed(1)+'%',xLabel:'累计轮数'});
  const hp=k=>hw.filter(s=>s[k]!==null&&s[k]!==undefined).map(s=>[s.t,s[k]]);
  draw('c_hw',{series:[{name:'CPU',pts:hp('cpu'),color:PAL.blue},{name:'GPU',pts:hp('gpu'),color:PAL.green},{name:'内存',pts:hp('mem'),color:PAL.orange}],
       ymin:0,ymax:100,fmtY:v=>v.toFixed(0),xLabel:'时间',xFmt:hwClock});
  ['c_gpm','c_batch','c_mcts','c_loss_p','c_loss_v','c_kl','c_win','c_hw'].forEach(attachChart);
}

/* ---------- 资产 ---------- */
async function loadArtifacts(){
  const game=$('a_game').value, profile=$('a_profile').value;
  try{
    const d = await api(`/api/admin/artifacts?game=${game}&profile=${profile}`);
    $('art_summary').innerHTML = `best_index <b>${d.best_index}</b> · 最新 <b>${d.latest_index}</b> · `
      + `checkpoint ${d.checkpoints.length} 个 · data ${gb(d.totals.data)} · runs ${gb(d.totals.runs)} · 磁盘余 ${d.disk_free_gb} GB`;
    $('t_ckpt').innerHTML = `<tr><th><input type=checkbox class=ckall data-t=t_ckpt></th><th>代</th><th>大小</th><th>时间</th><th></th></tr>` + d.checkpoints.slice().reverse().map(c=>`
      <tr><td><input type=checkbox class=ck data-kind=checkpoint ${isSel('t_ckpt',c.path)?'checked':''} value="${esc(c.path)}"></td><td class="num">${c.iteration}${c.iteration===d.best_index?' <span class="pill run">best</span>':''}</td>
          <td class="num">${gb(c.size)}</td><td class="hint">${new Date(c.mtime*1000).toLocaleString()}</td>
          <td><button class="mini" onclick="promote(${c.iteration})">设为best</button>
              <button class="mini danger" onclick="delArtifact('checkpoint','${c.path}','delete')" title="删除此代">${ICO_TRASH}</button></td></tr>`).join('');
    $('t_runs').innerHTML = `<tr><th><input type=checkbox class=ckall data-t=t_runs></th><th>run</th><th>大小</th><th></th></tr>` + d.runs.slice(0,20).map(r=>`
      <tr><td><input type=checkbox class=ck data-kind=run ${isSel('t_runs',r.path)?'checked':''} value="${esc(r.path)}"></td><td class="hint">${esc(r.name)}</td><td class="num">${gb(r.size)}</td>
          <td><button class="mini danger" onclick="delArtifact('run','${r.path}','delete')" title="删除此 run">${ICO_TRASH}</button></td></tr>`).join('');
    tidySel('t_ckpt', d.checkpoints.map(c=>c.path));
    tidySel('t_runs', d.runs.slice(0,20).map(r=>r.path));
    loadPools(game);
  }catch(e){ toast('资产读取失败: '+e.message); }
}
async function loadPools(game){
  const t = $('t_pools');
  try{
    const d = await api('/api/admin/buffers?game='+encodeURIComponent(game));
    const pools = (d.buffers||[]).filter(b=>b.kind==='named');
    const num = (v)=> v>=1000 ? (v/1000).toFixed(1)+'k' : String(v??0);
    t.innerHTML = '<tr><th><input type=checkbox class=ckall data-t=t_pools></th><th>池</th><th>已用/容量</th><th>来源</th><th></th></tr>' + (pools.length ? pools.map(b=>{
      const src = b.source ? `${b.source.model} 网络 · 第 ${b.source.iteration} 代` : '尚无数据';
      return `
      <tr title="${esc(b.note||'')}\n输入特征编码：v${b.feature_version??'?'} · 棋种 ${b.game}\n更新：${esc(b.updated_at||b.created_at||'—')}" style="cursor:help">
          <td><input type=checkbox class=ck data-kind=pool ${isSel('t_pools',b.ref)?'checked':''} value="${esc(b.ref)}"></td>
          <td><b>${esc(b.ref)}</b></td>
          <td class="num">${num(b.size)} / ${num(b.capacity)}</td>
          <td class="hint">${esc(src)}<br>${esc((b.updated_at||'').slice(5,16)).replace('T',' ')}</td>
          <td><button class="mini" title="克隆为新池" onclick="clonePool('${esc(b.ref)}')">${ICO_COPY}</button> <button class="mini danger" title="删除此池（数据不可恢复）" onclick="delPool('${esc(b.ref)}')">${ICO_TRASH}</button></td></tr>`;
    }).join('') : (selMap.t_pools.clear(), '<tr><td colspan="5" class="hint">本棋种还没有经验池——训练表单里可直接新建</td></tr>'));
  syncBtn('t_pools');
  }catch(e){ t.innerHTML = `<tr><td class="hint">经验池读取失败：${esc(e.message)}</td></tr>`; }
}
async function delPool(name){
  if(window.prompt(`删除经验池 ${name}？池内全部对局数据不可恢复。输入池名确认：`, '') !== name) return;
  try{
    const r = await api('/api/admin/buffers/delete', {method:'POST',
      body: JSON.stringify({game: $('a_game').value, name, confirm:'delete-pool'})});
    if(r.ok){ toast(`经验池 ${name} 已删除`, true); loadPools($('a_game').value); loadBuffers(); }
    else toast('删除失败: '+r.error);
  }catch(e){ toast('删除失败: '+e.message); }
}
async function clonePool(src){
  try{
    const r = await api('/api/admin/buffers/clone', {method:'POST', body: JSON.stringify({game: $('a_game').value, from: src})});
    if(r.ok){ toast(`${src} 已克隆为 ${r.buffer.name}，原池不动`, true); loadPools($('a_game').value); }
    else toast('克隆失败: '+r.error);
  }catch(e){ toast('克隆失败: '+e.message); }
}
/* 勾选状态活在 DOM 之外：资产页每 15s 自动刷新、切棋种会整表重绘，
   innerHTML 重建会把 checkbox 抹掉——用 Map(路径→kind) 记账，重绘时回填 */
const selMap = {t_ckpt:new Map(), t_pools:new Map(), t_runs:new Map()};
const isSel  = (tid,p)=> selMap[tid].has(p);
function bindBatchUI(){
  document.addEventListener('change', ev=>{
    const t = ev.target;
    if(!t.classList) return;
    if(t.classList.contains('ck')){
      const tid = t.closest('table').id;
      t.checked ? selMap[tid].set(t.value, t.dataset.kind) : selMap[tid].delete(t.value);
      syncBtn(tid);
    } else if(t.classList.contains('ckall')){
      const box = document.getElementById(t.dataset.t);
      box.querySelectorAll('.ck').forEach(c=>{ c.checked = t.checked;
        t.checked ? selMap[t.dataset.t].set(c.value, c.dataset.kind) : selMap[t.dataset.t].delete(c.value); });
      syncBtn(t.dataset.t);
    }
  });
  document.addEventListener('click', ev=>{
    const b = ev.target.closest ? ev.target.closest('.bb') : null;
    if(b) batchDel(b.dataset.t);
  });
}
function syncBtn(tid){
  const btn = document.querySelector(`.bb[data-t="${tid}"]`); if(!btn) return;
  const n = selMap[tid].size;
  btn.hidden = !n; btn.textContent = `删除 ${n}`;
}
const ICO_COPY = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:-2px"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>';
const ICO_TRASH = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" style="vertical-align:-2px"><path d="M3 6h18"/><path d="M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6M14 11v6"/></svg>';
/* 重绘后调用：清掉本次列表里已不存在的勾选（文件被删/切了棋种），并校正表头全选态 */
function tidySel(tid, rows){
  const alive = new Set(rows);
  [...selMap[tid].keys()].forEach(p=>{ if(!alive.has(p)) selMap[tid].delete(p); });
  const box = document.getElementById(tid);
  if(box){ const all=[...box.querySelectorAll('.ck')];
    const head=box.querySelector('.ckall');
    if(head) head.checked = all.length>0 && all.every(c=>c.checked); }
  ['t_ckpt','t_pools','t_runs'].forEach(syncBtn);
}
async function batchDel(tid){
  const items = [...selMap[tid]].map(([path,kind])=>({kind, path}));
  if(!items.length) return;
  if(!window.confirm(`删除 ${items.length} 项？不可恢复：\n` + items.map(i=>i.path.split('/').pop()).join('\n').slice(0,600))) return;
  try{
    const r = await api('/api/admin/artifacts/delete_batch', {method:'POST',
      body: JSON.stringify({game:$('a_game').value, profile:$('a_profile').value, items, confirm:'delete'})});
    if(r.ok){
      const skip = (r.skipped||[]).length;
      selMap[tid].clear();
      toast(`已删 ${(r.deleted||[]).length} 项` + (skip?`，跳过 ${skip} 项：`+r.skipped.map(x=>x.path.split('/').pop()+'('+x.error+')').join('；'):''), !skip);
      loadArtifacts(); loadPools($('a_game').value); loadBuffers();
    } else toast('批量删除失败: '+r.error);
  }catch(e){ toast('批量删除失败: '+e.message); }
}
bindBatchUI();
async function delArtifact(kind, path, confirm){
  if(!window.confirm(`删除？\n${path}`)) return;
  try{
    const r = await api('/api/admin/artifacts/delete', {method:'POST', body: JSON.stringify({kind, path, confirm, game:$('a_game').value, profile:$('a_profile').value})});
    r.ok ? toast('已删除: '+r.deleted, true) : toast('删除失败: '+r.error);
    loadArtifacts();
  }catch(e){ toast('删除失败: '+e.message); }
}
async function promote(it){
  const v = it;
  if(!v){ toast('未指定代数'); return; }
  try{
    const r = await api('/api/admin/best', {method:'POST', body: JSON.stringify({iteration:v, game:$('a_game').value, profile:$('a_profile').value})});
    r.ok ? toast('best_index 已设为 '+r.best_index, true) : toast('失败: '+r.error);
    loadArtifacts();
  }catch(e){ toast('失败: '+e.message); }
}

/* ---------- 神经网络配置 ---------- */
let PROFILES = [];
async function loadProfiles(){
  const game = $('f_game').value;
  let d;
  try{ d = await api('/api/admin/profiles?game='+encodeURIComponent(game)); }
  catch(e){ $('pf_rows').innerHTML = `<tr><td colspan="5" class="hint">读取失败: ${esc(e.message)}</td></tr>`; return; }
  PROFILES = d.profiles || [];
  $('pf_rows').innerHTML = PROFILES.map(p=>`<tr>
      <td class="num">${esc(p.name)}${p.builtin?'':' <span class="pill" style="font-size:10px">自定义</span>'}</td>
      <td class="hint">${esc(String(p.net.n_filters||'?'))}×${esc(String(p.net.n_res_blocks||'?'))} · vh ${esc(String(p.net.n_value_hidden_channels||'?'))}${p.net.use_se?' · SE':''}</td>
      <td class="hint">${p.builtin?'内置':'profiles.json'}</td>
      <td class="num">${p.models}${p.best>=0?` <span class="pill" style="font-size:10px">best ${p.best}</span>`:''}</td>
      <td>${p.builtin?'':`<button class="mini danger" onclick="pfDel('${esc(p.name)}',${p.models})">删名</button>`}</td></tr>`).join('')
    || '<tr><td colspan="5" class="hint">无</td></tr>';
  // 两个神经网络下拉：内置在前、自定义带 ✦；保留当前选中值
  ['f_profile','a_profile'].forEach(id=>{
    const sel = $(id), old = sel.value;
    sel.innerHTML = PROFILES.map(p=>`<option value="${esc(p.name)}">${esc(p.name)}${p.builtin?'':' ✦'} ${p.net.n_filters||'?'}×${p.net.n_res_blocks||'?'}</option>`).join('');
    if (PROFILES.some(p=>p.name===old)) sel.value = old;
  });
}
$('pf_add').onclick = async ()=>{
  const body = {name: $('pf_name').value.trim(), net: {
    n_filters: +$('pf_filters').value, n_res_blocks: +$('pf_blocks').value,
    n_value_hidden_channels: +$('pf_vh').value, use_se: $('pf_se').value==='1'}};
  try{
    const r = await api('/api/admin/profiles/save', {method:'POST', body: JSON.stringify(body)});
    r.ok ? (toast(`已创建配置 ${body.name}`, true), $('pf_name').value='', loadProfiles()) : toast('创建失败: '+r.error);
  }catch(e){ toast('创建失败: '+e.message); }
};
async function pfDel(name, models){
  const extra = models ? `\n名下还有 ${models} 个模型文件（保留在磁盘，仅移除配置名）。` : '';
  if(!window.confirm(`删除配置「${name}」？${extra}`)) return;
  try{
    const body = {name};
    if(models) body.confirm = 'delete-profile-with-models';
    const r = await api('/api/admin/profiles/delete', {method:'POST', body: JSON.stringify(body)});
    r.ok ? (toast('已删除配置 '+name, true), loadProfiles()) : toast('删除被拒: '+r.error);
  }catch(e){ toast('删除失败: '+e.message); }
}

/* ---------- 训练控制 ---------- */
function launchOpts(){
  const o = {game:$('f_game').value, profile:$('f_profile').value};
  if($('f_workers').value) o.workers = +$('f_workers').value;
  if($('f_games').value) o.games_per_round = +$('f_games').value;
  if($('f_leaves').value) o.batch_leaves = +$('f_leaves').value;
  if($('f_maxbatch').value) o.max_batch = +$('f_maxbatch').value;
  o.device = $('f_device').value;
  const st = $('f_start').value;
  if(st === 'fork'){
    const n = +$('f_startn').value;
    if(!n){ toast('「从指定代数分叉」需要填起始代数'); return null; }
    o.start_from = String(n);
  } else if(st){ o.start_from = st; }
  const bufv = $('f_buffer').value;
  if(!bufv || bufv==='__new__'){ toast('经验池必须选择：选已有池或选「新建经验池」'); return null; }
  o.buffer = bufv;
  return o;
}
async function startTraining(){
  const o = launchOpts();
  if(!o) return;
  $('launch_hint').textContent = '正在启动…';
  try{
    const r = await api('/api/admin/train/start', {method:'POST', body: JSON.stringify(o)});
    if(r.ok){ toast('训练已启动 pid '+r.status.pid, true); }
    else { toast('启动失败: '+r.error+'\n\n'+(r.status.tail||[]).slice(-12).join('\n')); }
  }catch(e){ toast('启动失败: '+e.message); }
  $('launch_hint').textContent = '';
  tick();
}
async function stopTraining(mode){
  const done = ((window.__lastProgress||{}).done) || 0;
  const ask = mode === 'interrupt'
      ? '等本轮结束？（发 SIGINT：取消还没开跑的局，把手上正在下的局收完入池、存盘后退出；fit/评估若还没开始会跳过）'
      : '立即结束？会直接 kill -9：本轮已下完的 ' + done + ' 局不会存盘，全白跑。确定要硬杀吗？';
  if(!window.confirm(ask)) return;
  try{
    const r = await api('/api/admin/train/stop', {method:'POST', body: JSON.stringify({mode})});
    toast(r.ok? (r.message || '已停止') : ('停止失败: '+r.error), !!r.ok);
  }catch(e){ toast('停止失败: '+e.message); }
  tick();
}

/* ---------- 日志 ---------- */
const LIVE = '● 实时';
let LOGS_CACHE = [];
function logOpt(x){ const label = x.path==='current' ? x.name : String(x.path||x.name).split('/').pop();
  return `<option value="${esc(x.path)}" title="${esc(x.path)}">${esc(label)} (${gb(x.size||0)})</option>`; }
function filesOf(cat){
  if(cat === LIVE) return [{path:'current', name:'当前训练进程（自动跟随最新 console 日志）', size:0}];
  return LOGS_CACHE.filter(x=>x.cat===cat);
}
function catOrder(){
  // 分类 = 日志目录名；排序 = 该目录内最新一份的时间（最近在写的排最前）
  const last = {};
  LOGS_CACHE.forEach(x=>{ if(!(x.cat in last) || x.mtime > last[x.cat]) last[x.cat] = x.mtime; });
  const cats = Object.keys(last).sort((a,b)=>last[b]-last[a]);
  if(TRAIN_RUNNING) cats.unshift(LIVE);
  return cats;
}
function renderFileSel(keep){
  const fs = filesOf($('l_cat').value), sel = $('l_name');
  sel.innerHTML = fs.map(logOpt).join('') || '<option value="">（暂无）</option>';
  const vals = [...sel.options].map(o=>o.value).filter(Boolean);
  sel.value = (keep && vals.includes(keep)) ? keep : (vals[0] || '');   // 组内 mtime 倒序，默认最新
}
function renderCatSel(keep){
  const cats = catOrder(), sel = $('l_cat');
  sel.innerHTML = cats.map(c=>`<option value="${esc(c)}" title="${c===LIVE?'当前训练进程的实时输出':'来自 logs/'+c+'/'}">${esc(c)}</option>`).join('');
  if(keep && cats.includes(keep)) sel.value = keep;
  else if(TRAIN_RUNNING) sel.value = LIVE;
  renderFileSel();
}
async function loadLogList(){
  try{ LOGS_CACHE = (await api('/api/admin/logs/list')).logs || []; }
  catch(e){ return; }
  renderCatSel($('l_cat').value);
  loadLog();
}
$('l_cat').onchange = ()=>{ renderFileSel(); loadLog(); };
/* ANSI SGR → 内联着色：棋盘等终端渲染的日志不再显示 \x1b[91m 乱码 */
/* ANSI → HTML：完整 SGR（前景/背景 16+256+RGB、粗体/暗淡/斜体/下划线/反显/删除线、
   各类复位），非 SGR 控制序列（tqdm 的 ESC[2K、光标移动、隐藏光标、OSC 标题…）静默吞掉不显示 */
function ansiHtml(s){
  const BASE = ['#000','#c0342b','#3a8f4f','#b7950b','#2471a3','#8e44ad','#148f77','#c8c8c8',
                '#666','#e74c3c','#2ecc71','#f1c40f','#5dade2','#c39bd3','#76d7c4','#fff'];
  const x256 = (n) => {
    if (n < 16) return BASE[n];
    if (n >= 232) { const v = 8 + (n - 232) * 10; return `rgb(${v},${v},${v})`; }
    n -= 16; const c = [0,95,135,175,215,255];
    return `rgb(${c[(n/36|0)%6]},${c[(n/6|0)%6]},${c[n%6]})`;
  };
  const re = /\x1b\[([0-9;?]*)([A-Za-z])|\x1b\][^\x07]*?(?:\x07|\x1b\\)|\x1b./g;
  let out = '', last = 0, st = {}, m;
  const flush = (t) => {
    if (!t) return;
    let css = (st.b ? 'font-weight:700;' : '') + (st.f ? 'opacity:.6;' : '')
            + (st.i ? 'font-style:italic;' : '') + (st.u ? 'text-decoration:underline;' : '')
            + (st.s ? 'text-decoration:line-through;' : '') + (st.v ? 'visibility:hidden;' : '')
            + (st.fg ? `color:${st.fg};` : '') + (st.bg ? `background:${st.bg};` : '')
            + (st.inv ? 'filter:invert(1);' : '');
    out += css ? `<span style="${css}">${esc(t)}</span>` : esc(t);
  };
  while ((m = re.exec(s))) {
    flush(s.slice(last, m.index)); last = re.lastIndex;
    if (m[2] !== 'm' || (m[1] && m[1].includes('?'))) continue;      // 非 SGR：吞掉
    const p = (m[1] || '0').split(';').map(Number);
    for (let i = 0; i < p.length; i++) {
      const c = p[i];
      if (c === 0) st = {};
      else if (c === 1) st.b = 1; else if (c === 2) st.f = 1; else if (c === 3) st.i = 1;
      else if (c === 4) st.u = 1; else if (c === 7) st.inv = 1; else if (c === 8) st.v = 1;
      else if (c === 9) st.s = 1;
      else if (c === 22) { delete st.b; delete st.f; } else if (c === 23) delete st.i;
      else if (c === 24) delete st.u; else if (c === 27) delete st.inv; else if (c === 28) delete st.v;
      else if (c === 29) delete st.s;
      else if (c >= 30 && c <= 37) st.fg = BASE[c - 30];
      else if (c === 38) { if (p[i+1] === 5) { st.fg = x256(p[i+2]); i += 2; } else if (p[i+1] === 2) { st.fg = `rgb(${p[i+2]},${p[i+3]},${p[i+4]})`; i += 4; } }
      else if (c === 39) delete st.fg;
      else if (c >= 40 && c <= 47) st.bg = BASE[c - 40];
      else if (c === 48) { if (p[i+1] === 5) { st.bg = x256(p[i+2]); i += 2; } else if (p[i+1] === 2) { st.bg = `rgb(${p[i+2]},${p[i+3]},${p[i+4]})`; i += 4; } }
      else if (c === 49) delete st.bg;
      else if (c >= 90 && c <= 97) st.fg = BASE[c - 90 + 8];
      else if (c >= 100 && c <= 107) st.bg = BASE[c - 100 + 8];
    }
  }
  flush(s.slice(last));
  return out;
}
async function loadLog(){
  const name=$('l_name').value, lines=$('l_lines').value||200, grep=$('l_grep').value||'';
  try{
    const key = name === 'current' ? 'name=current' : `path=${encodeURIComponent(name)}`;
    const d = await api(`/api/admin/logs?${key}&lines=${lines}&grep=${encodeURIComponent(grep)}`);
    if(!d.ok){ $('log_view').textContent = d.error||'读取失败'; return; }
    const pre = $('log_view'); const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 40;
    pre.innerHTML = d.lines.length ? ansiHtml(d.lines.join('\n')) : '（空）';
    if(atBottom) pre.scrollTop = pre.scrollHeight;
  }catch(e){ $('log_view').textContent = '读取失败: '+e.message; }
}

/* ---------- 任务 ---------- */
let taskFirstLoad = true;
async function loadTasks(rebuild){
  let list;
  try{ list = (await api('/api/admin/tasks')).tasks; }
  catch(e){ if(rebuild) $('task-list').innerHTML = '<div class="hint">任务列表读取失败: '+esc(e.message)+'</div>'; return; }
  if(rebuild){
    const keep = {};
    document.querySelectorAll('#task-list [data-t]').forEach(el=>{
      keep[el.dataset.t+'.'+el.dataset.k] = (el.type==='checkbox') ? el.checked : el.value; });
    $('task-list').innerHTML = list.map(t=>{
      const opts = Object.entries(t.opts||{}).map(([k,spec])=>{
        const key = t.name+'.'+k;
        if(spec.type==='flag'){
          const on = keep[key]===true ? ' checked':'';
          return `<label class="hint" style="display:inline-flex;align-items:center;gap:5px"><input type="checkbox" data-t="${t.name}" data-k="${k}"${on}>${esc(k)}</label>`;
        }
        const ph = spec.default!==undefined ? String(spec.default) : '';
        const type = spec.type==='int' ? 'number' : 'text';
        const val = (typeof keep[key]==='string') ? keep[key] : '';
        return `<span class="field" style="min-width:120px"><label>${esc(k)}</label>`
             + `<input type="${type}" placeholder="默认 ${esc(ph)}" data-t="${t.name}" data-k="${k}" value="${esc(val)}"></span>`;
      }).join('');
      return `<div class="panel" style="margin-bottom:10px"><h2>${esc(t.name)} <span class="lg">${t.kind==='long'?'常驻':'一次性'}</span> <span class="pill task-st" data-st="${t.name}"></span></h2>
        <div class="hint" style="margin:2px 0 8px">${esc(t.desc)}</div>
        <div class="row">${opts}
          <button onclick="taskAct('start','${t.name}')">启动</button>
          <button class="danger" onclick="taskAct('stop','${t.name}')">停止</button>
          <button onclick="taskShowLog('${t.name}')">查看日志</button></div></div>`;
    }).join('');
  }
  list.forEach(t=>{
    const el = document.querySelector(`.task-st[data-st="${t.name}"]`); if(!el) return;
    el.className = 'pill task-st ' + (t.running?'run':'');
    el.textContent = t.running ? `运行中 pid ${t.pid}`
      : (t.exit_code!==null && t.exit_code!==undefined ? `上次退出 ${t.exit_code}` : '未运行');
  });
}
async function taskAct(act, name){
  const params = {};
  document.querySelectorAll(`#task-list [data-t="${name}"]`).forEach(el=>{
    if(el.type==='checkbox'){ if(el.checked) params[el.dataset.k] = '1'; }
    else if(el.value!=='') params[el.dataset.k] = el.value;
  });
  try{
    const r = await api(`/api/admin/tasks/${act}`, {method:'POST', body: JSON.stringify({name, params})});
    if(r.ok) toast(act==='start' ? (r.message||'已启动') : (r.message||'已停止'), true);
    else toast(`${act} 失败: ${r.error}` + ((r.log_tail||[]).length ? '\n' + r.log_tail.slice(-3).join('\n') : ''));
    if(act==='start' && r.ok) taskShowLog(name);
  }catch(e){ toast(`${act} 失败: ${e.message}`); }
  loadTasks(false);
}
async function taskShowLog(name){
  try{
    const d = await api(`/api/admin/tasks/log?name=${name}&tail=300`);
    $('task-log-name').textContent = `　${name}${d.running?'（运行中）':''}　${d.path||''}`;
    $('task-log').innerHTML = (d.lines||[]).length ? ansiHtml(d.lines.join('\n')) : (d.note||'（空）');
  }catch(e){ $('task-log').textContent = '日志读取失败: '+e.message; }
}

/* ---------- 主循环 ---------- */
let tab = 'monitor';
let TRAIN_RUNNING = false;
function switchTab(name){
  tab = name;
  document.querySelectorAll('nav a').forEach(a=>a.classList.toggle('on', a.dataset.tab===name));
  document.querySelectorAll('.tab').forEach(s=>s.classList.toggle('on', s.id==='tab-'+name));
  if(name==='monitor'){ tick(); tickSeries(); }
  if(name==='artifacts') loadArtifacts();
  if(name==='logs'){ loadLogList(); }
  if(name==='tasks') loadTasks(true);
}
document.querySelectorAll('nav a').forEach(a=>a.addEventListener('click',e=>{e.preventDefault(); switchTab(a.dataset.tab);}));
async function tick(){
  if(tab!=='monitor' && tab!=='train' && tab!=='logs') return;
  try{
    const s = await api('/api/admin/snapshot'); renderSnapshot(s);
  }catch(e){ $('phase').textContent = '拉取失败: '+e.message; }
}
async function tickSeries(){
  if(tab!=='monitor') return;
  try{ renderCharts(await api('/api/admin/series')); }catch(e){ /* 静默 */ }
}
$('btn-theme').onclick = () => cycleTheme();
applyTheme(themeMode(), false);      // 同步按钮文案 + 图表配色（不改 localStorage）
$('btn-start2').onclick = startTraining;
$('btn-stop').onclick = () => stopTraining('kill');
$('btn-stop-round').onclick = () => stopTraining('interrupt');
$('btn-refresh-art').onclick = loadArtifacts;
$('btn-load-log').onclick = loadLog;
$('a_game').onchange = $('a_profile').onchange = loadArtifacts;
$('f_game').onchange = ()=>{ loadProfiles(); loadBuffers(); };
$('f_start').onchange = ()=> $('f_startn_w').classList.toggle('hidden', $('f_start').value!=='fork');
$('f_buffer').onchange = async (e)=>{
  const v = e.target.value;
  if(v !== '__new__') return;
  try{
    const r = await api('/api/admin/buffers/create', {method:'POST',
      body: JSON.stringify({game: $('f_game').value})});          // 不传名字：后端按 poolN 自动编号
    if(r.ok){ toast('已新建经验池 '+(r.buffer&&r.buffer.name), true); await loadBuffers();
              if(r.buffer) $('f_buffer').value = r.buffer.name; }
    else toast('新建失败: '+r.error);
  }catch(err){ toast('新建失败: '+err.message); }
  sel0();
};
function sel0(){ const x=$('f_buffer'); if(!x.value || x.value==='__new__') x.selectedIndex = 0; }
async function loadBuffers(){
  let list = [];
  try{ list = (await api('/api/admin/buffers?game='+encodeURIComponent($('f_game').value))).buffers || []; }
  catch(e){ return; }
  const sel = $('f_buffer'), keep = sel.value;
  const pools = list.filter(b=>b.kind==='named');
  let html = '';
  html += pools.map(b=>{
    const used = b.size ?? b.main_size ?? 0, cap = b.capacity ?? b.main_capacity;
    const src = b.source ? `最新写入：${b.source.model} 网络 · 第 ${b.source.iteration} 代` : '尚无数据';
    const o = document.createElement('option');
    o.value = b.ref;
    o.textContent = `${b.ref}　${used}${cap ? '/'+cap : ''}`;
    o.title = `${b.ref} · 棋种 ${b.game}\n输入特征编码：v${b.feature_version??'?'}\n${src}\n更新：${b.updated_at||b.created_at||'—'}${b.note?('\n备注：'+b.note):''}`;
    return o.outerHTML;
  }).join('');
  const nxt = 'pool' + (pools.reduce((m,b)=>{ const t=/^(\d+)$/.exec((b.ref||'').replace(/^pool/,'')) ? +b.ref.slice(4) : m; return Math.max(m, t||0); }, 0) + 1);
  html += `<option value="__new__" title="自动按编号创建：${nxt}">＋ 新建经验池（${nxt}）</option>`;
  sel.innerHTML = html;
  if(pools.some(b=>b.ref===keep)) sel.value = keep;               // 默认停在第一个池（下拉无"不选"）
}
$('l_name').onchange = loadLog;
window.addEventListener('resize', ()=>{ if(tab==='monitor') tickSeries(); });

(async function boot(){
  try{
    const env = await api('/api/admin/env');
    const sug = env.suggested || {game: env.game, profile: env.profile};
    $('f_game').value = sug.game; $('a_game').value = sug.game;
    if([...$('f_profile').options].some(o=>o.value===sug.profile)){ $('f_profile').value = sug.profile; $('a_profile').value = sug.profile; }
    if(env.workers) $('f_workers').value = env.workers;
    if(env.games_per_round) $('f_games').value = env.games_per_round;
    if(env.batch_leaves) $('f_leaves').value = env.batch_leaves;
    if(env.max_batch) $('f_maxbatch').value = env.max_batch;
    const devs = env.devices || [{value:'cuda:0',label:'cuda:0'},{value:'cpu',label:'cpu'}];
    $('f_device').innerHTML = devs.map(d=>`<option value="${esc(d.value)}">${esc(d.label)}</option>`).join('');
    if(devs.some(d=>d.value===env.device)) $('f_device').value = env.device;
    $('launch_hint').textContent = `当前生效：${env.net.n_filters}x${env.net.n_res_blocks} · 线程 ${env.workers} · 每轮 ${env.games_per_round} 局 · K=${env.batch_leaves}`;
  }catch(e){ /* 忽略 */ }
  loadProfiles(); loadBuffers();
  tick(); tickSeries();
  setInterval(tick, 2000);
  setInterval(()=>{ if(tab==='tasks') loadTasks(false); }, 5000);
  setInterval(tickSeries, 8000);
  setInterval(()=>{ if(tab==='logs' && $('l_follow').checked) loadLog(); }, 5000);
  setInterval(()=>{ if(tab==='artifacts') loadArtifacts(); }, 15000);
})();

globalThis.__P=()=>selMap;