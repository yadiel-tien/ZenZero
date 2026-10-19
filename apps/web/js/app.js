/* ZenZero Web 前端主控：启动器 / 对局状态机 / AI 会话 / 渲染 / 音效
 * 布局坐标与 pygame 版 apps/ui 保持一致（舞台 600x800，棋盘中心 300,400） */
(function () {
  'use strict';

  /* ================= 工具 ================= */
  const $ = (id) => document.getElementById(id);
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  let toastTimer = null;
  function toast(msg, isErr = false, ms = 2600) {
    const t = $('toast');
    t.textContent = msg;
    t.className = isErr ? 'err' : '';
    t.classList.remove('hidden');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.add('hidden'), ms);
  }

  // 全局异常兜底：不再静默白屏，5s 节流防刷屏
  let lastErrAt = 0;
  window.addEventListener('error', (e) => {
    const now = Date.now();
    if (now - lastErrAt > 5000) { lastErrAt = now; toast('页面异常：' + (e.message || '资源加载失败'), true, 4000); }
  });
  window.addEventListener('unhandledrejection', (e) => {
    const now = Date.now();
    if (now - lastErrAt > 5000) {
      lastErrAt = now;
      toast('异步异常：' + ((e.reason && e.reason.message) || e.reason || '未知'), true, 4000);
    }
  });

  /* ================= 常量 ================= */
  const CENTER = { x: 300, y: 400 };
  const GAMES = {
    ChineseChess: {
      label: '中国象棋', rows: 10, cols: 9, grid: 54,
      sides: ['红方', '黑方'], sideCls: ['si-red', 'si-black'],
      winImgs: ['red_win.png', 'black_win.png', 'draw_win.png'],
      pieceSize: 65
    },
    Gomoku: {
      label: '五子棋', rows: 15, cols: 15, grid: 35.2857,
      sides: ['黑方', '白方'], sideCls: ['si-black', 'si-white'],
      winImgs: ['black_win.png', 'white_win.png', 'draw_win.png'],
      pieceSize: 33
    }
  };
  const TURN_LIMIT = 100; // 每步 100 秒（对应 Timer(limit=100000ms)）

  /* ================= 音效 ================= */
  const Sound = {
    sfxOn: localStorage.getItem('zz_sfx') !== '0',
    bgmOn: localStorage.getItem('zz_bgm') === '1',
    unlocked: false,
    clips: {},
    bgmEl: null, bgmName: '',
    init() {
      const files = {
        click: 'ui_click.wav', place: 'place_stone.mp3', piece: 'piece_down.mp3',
        capture: 'capture.mp3', check: 'check.mp3', checkmate: 'checkmate.mp3', win: 'win.mp3'
      };
      for (const [k, f] of Object.entries(files)) {
        const a = new Audio('assets/sound/' + f);
        a.preload = 'none';  // 懒加载：首次播放时才拉数据，首屏省 ~1MB
        this.clips[k] = a;
      }
    },
    play(name) {
      if (!this.sfxOn || !this.unlocked) return;
      const a = this.clips[name];
      if (!a) return;
      try { a.currentTime = 0; a.play().catch(() => {}); } catch (e) { /* noop */ }
    },
    setBgm(name) { // null 停止
      if (!this.bgmEl || this.bgmName !== name) {
        this.stopBgm();
        if (name) {
          this.bgmEl = new Audio('assets/sound/' + name);
          this.bgmEl.preload = 'none';
          this.bgmEl.loop = true;
          this.bgmEl.volume = 0.35;
          this.bgmName = name;
        }
      }
      if (this.bgmEl && this.bgmOn && this.unlocked) this.bgmEl.play().catch(() => {});
    },
    stopBgm() {
      if (this.bgmEl) { this.bgmEl.pause(); this.bgmEl.src = ''; }
      this.bgmEl = null; this.bgmName = '';
    },
    unlock() {
      if (this.unlocked) return;
      this.unlocked = true;
      if (this.bgmName && this.bgmOn) this.bgmEl.play().catch(() => {});
    }
  };

  /* ================= 粒子背景（对应 launcher.py 的墨点粒子） ================= */
  (function particles() {
    const cv = $('particles'), ctx = cv.getContext('2d');
    const COLORS = ['rgba(40,40,40,', 'rgba(139,0,0,', 'rgba(80,80,80,', 'rgba(110,100,90,'];
    let W = 0, H = 0, ps = [];
    function fit() {
      W = cv.width = window.innerWidth;
      H = cv.height = window.innerHeight;
    }
    fit(); window.addEventListener('resize', fit);
    for (let i = 0; i < 35; i++) {
      ps.push({
        x: Math.random() * W, y: Math.random() * H, sp: .7 + Math.random() * .7,
        sz: 2 + Math.random() * 4, ph: Math.random() * Math.PI * 2,
        amp: 15 + Math.random() * 25, ws: .02 + Math.random() * .02,
        col: COLORS[(Math.random() * COLORS.length) | 0], a: .08 + Math.random() * .12
      });
    }
    (function loop() {
      ctx.clearRect(0, 0, W, H);
      for (const p of ps) {
        p.ph += p.ws; p.y += p.sp;
        if (p.y > H + 20) { p.y = -20; p.x = Math.random() * W; p.ph = Math.random() * 6.28; }
        const x = p.x + Math.sin(p.ph) * p.amp;
        ctx.beginPath();
        ctx.fillStyle = p.col + p.a + ')';
        ctx.arc(x, p.y, p.sz, 0, 6.29);
        ctx.fill();
      }
      requestAnimationFrame(loop);
    })();
  })();

  /* ================= 舞台缩放 ================= */
  const mqNarrow = window.matchMedia('(max-width: 1040px)');

  function fitStage() {
    const stage = $('stage');
    const inGame = !$('panel').classList.contains('hidden');
    const wide = !mqNarrow.matches && inGame;
    const drawerH = mqNarrow.matches && inGame ? 46 : 0;  // 窄屏底部抽屉把手高度
    const availW = wide ? window.innerWidth - 330 - 78 : window.innerWidth - 12;
    const s = Math.min(availW / 600, (window.innerHeight - 12 - drawerH) / 800, 1.4);
    stage.style.transform = `scale(${s})`;
    const wrap = $('stage-wrap');
    wrap.style.width = 600 * s + 'px';
    wrap.style.height = 800 * s + 'px';
    G && (G.scale = s);
  }
  window.addEventListener('resize', fitStage);

  /* ================= 坐标换算 ================= */
  function grid2px(r, c) { // 交叉点左上角（棋子图锚点）
    if (G.name === 'ChineseChess') {
      let x = (c - 4) * 54 + CENTER.x - 27, y = (r - 4.5) * 54 + CENTER.y - 27;
      x = Math.trunc(x); y = Math.trunc(y);
      return { x: x - 4 - Math.floor(x / 200), y: y - Math.floor(y / 100), size: 65 };
    }
    const g = 35.2857;
    return { x: Math.trunc((c - 7) * g + CENTER.x - 17), y: Math.trunc((r - 7) * g + CENTER.y - 17), size: 33 };
  }
  function px2grid(px, py) {
    const cfg = GAMES[G.name];
    const col = Math.round((px - CENTER.x) / cfg.grid + (cfg.cols - 1) / 2);
    const row = Math.round((py - CENTER.y) / cfg.grid + (cfg.rows - 1) / 2);
    return (col >= 0 && col < cfg.cols && row >= 0 && row < cfg.rows) ? [row, col] : null;
  }
  function ev2stage(e) {
    const rect = $('board-wrap').getBoundingClientRect();
    const s = G.scale || 1;
    return { x: (e.clientX - rect.left) / s + 0 * rect.left, y: (e.clientY - rect.top) / s };
  }

  /* ================= 全局游戏状态 ================= */
  let G = null;
  let launcher = null;

  function newGameEnv() {
    G.lastPlaced = null;
    if (G.name === 'ChineseChess') {
      G.state = ChessEnv.initialState();
    } else {
      G.board = GomokuEnv.initialState();
    }
  }

  /* ================= 启动器 ================= */
  function initDropdown(el, onChange) {
    const btn = el.querySelector('.dd-btn'), list = el.querySelector('.dd-list');
    let value = null;
    btn.addEventListener('click', (ev) => {
      ev.stopPropagation();
      Sound.unlock(); Sound.play('click');
      document.querySelectorAll('.dd-list').forEach((l) => { if (l !== list) l.classList.add('hidden'); });
      list.classList.toggle('hidden');
    });
    document.addEventListener('click', () => list.classList.add('hidden'));
    return {
      setOptions(opts, defIdx) {
        list.innerHTML = '';
        value = null; btn.textContent = '请选择 AI 模型';
        opts.forEach((o, i) => {
          const li = document.createElement('li');
          li.textContent = o.label;
          li.addEventListener('click', (ev) => {
            ev.stopPropagation();
            value = o.value;
            btn.textContent = o.label;
            list.classList.add('hidden');
            onChange && onChange(value);
          });
          if (i === defIdx) { li.click(); }
          list.appendChild(li);
        });
        list.classList.add('hidden');
      },
      get value() { return value; }
    };
  }

  launcher = {
    game: null, mode: null,
    dd1: null, dd2: null,
    modelsCache: {},
    init() {
      this.dd1 = initDropdown($('dd1'));
      this.dd2 = initDropdown($('dd2'));
      $$('#l-step-game .card-btn').forEach((b) => b.addEventListener('click', () => {
        Sound.unlock(); Sound.play('click');
        this.game = b.dataset.game; this.showStep('mode');
      }));
      $$('#l-step-mode .row-btn').forEach((b) => b.addEventListener('click', () => {
        Sound.play('click');
        if (b.dataset.back) { this.showStep('game'); return; }
        this.mode = b.dataset.mode;
        if (this.mode === 'HvH') { this.start(); return; }
        this.showStep('model');
      }));
      $$('#l-step-model .row-btn').forEach((b) => b.addEventListener('click', () => {
        if (b.dataset.back) { Sound.play('click'); this.showStep('mode'); return; }
        Sound.play('click');
        this.start();
      }));
      Sound.setBgm('launcher_bgm.wav');
      this.showStep('game');
    },
    showStep(step) {
      $('l-step-game').classList.toggle('hidden', step !== 'game');
      $('l-step-mode').classList.toggle('hidden', step !== 'mode');
      $('l-step-model').classList.toggle('hidden', step !== 'model');
      const subs = { game: '选择对弈游戏', mode: '选择对战模式', model: '选择 AI 模型' };
      let sub = subs[step];
      if (this.game && step !== 'game') sub = GAMES[this.game].label + ' · ' + sub;
      $('l-sub').textContent = sub;
      if (step === 'model') this.loadModels();
    },
    async loadModels() {
      const envName = this.game;
      $('btn-launch').disabled = true;
      $('model-hint').classList.add('hidden');
      $('dd2-box').classList.toggle('hidden', this.mode !== 'AIvAI');
      $('dd1-label').textContent = this.mode === 'AIvAI' ? 'AI 1（执先手）' : '你的对手（AI 执后手）';
      let data = this.modelsCache[envName];
      try {
        data = await API.getModels(envName);
        this.modelsCache[envName] = data;
        setServerDot('ok');
      } catch (e) {
        setServerDot('err');
        $('model-hint').classList.remove('hidden');
        $('model-hint').innerHTML = `无法获取模型列表：<br>${e.message}`;
        return;
      }
      const indices = (data.indices || []).slice().sort((a, b) => b - a);
      const best = data.best_index;
      if (!indices.length) {
        $('model-hint').classList.remove('hidden');
        $('model-hint').innerHTML = `未检测到模型文件。<br>请从 GitHub Release 下载 .pt 放入 data/${envName}/ 目录。`;
        return;
      }
      const opts = indices.map((i) => ({
        value: i,
        label: i === best ? `最强 AI (v${i})` : (i === -1 ? '原生模型' : `AI v${i}`)
      }));
      const bestIdx = Math.max(0, opts.findIndex((o) => o.value === best));
      this.dd1.setOptions(opts, bestIdx);
      this.dd2.setOptions(opts, bestIdx);
      $('btn-launch').disabled = false;
    },
    start() {
      const mode = this.mode;
      let players;
      if (mode === 'HvH') players = [mkHuman('玩家一'), mkHuman('玩家二')];
      else if (mode === 'HvAI') players = [mkHuman('玩家'), mkAI(this.dd1.value)];
      else players = [mkAI(this.dd1.value), mkAI(this.dd2.value)];
      if (mode !== 'HvH') {
        if (mode === 'HvAI' && this.dd1.value == null) { toast('请先选择 AI 模型', true); return; }
        if (mode === 'AIvAI' && (this.dd1.value == null || this.dd2.value == null)) { toast('请为双方选择 AI 模型', true); return; }
      }
      deviceNoted = false;
      if ($('p-device')) { $('p-device').classList.add('hidden'); $('p-device').innerHTML = ''; }
      G = {
        name: this.game, mode, players, serverDevice: null, expose: location.search.includes('expose=1'),
        ptm: 0, lastAction: -1, steps: 0, winner: 2,
        status: 'new', history: [], selected: null, safeCache: new Map(),
        timers: { 0: TURN_LIMIT, 1: TURN_LIMIT }, undoStack: [],
        aiBusy: false, token: 0, scale: 1,
        hover: null, winStones: []
      };
      newGameEnv();
      if (G.expose) window.__zz = G;
      $('launcher').classList.add('hidden');
      $('stage').classList.remove('hidden');
      $('panel').classList.remove('hidden');
      fitStage();
      setupGameScreen();
      renderDeviceInfo();
      Sound.setBgm('bgm.mp3');
      render();
      renderControls();
      renderMoves();
      updateInfos();
    }
  };

  function mkHuman(desc) { return { type: 'human', desc }; }
  function mkAI(modelId) { return { type: 'ai', desc: `AI·v${modelId}`, modelId, pid: null, winRate: -1 }; }

  /* ================= 服务器状态 ================= */
  function setServerDot(state) {
    const d = $('srv-dot');
    d.className = 'srv-ind ' + (state === 'ok' ? 'ok' : state === 'err' ? 'err' : 'mid');
    $('srv-url').textContent = (location.protocol === 'file:' || state === 'err')
      ? (API.baseUrl()) : (location.origin + API.baseUrl());
  }

  /* ================= 对局界面 ================= */
  function setupGameScreen() {
    const cfg = GAMES[G.name];
    $('info-top').classList.remove('hidden');
    $('info-bottom').classList.remove('hidden');
    $('p-mode').textContent = cfg.label + ' · ' +
      ({ HvH: '双人同屏 · 无需后端', HvAI: '人机对战', AIvAI: 'AI 观战' })[G.mode];
    $('board-wrap').classList.toggle('gomoku', G.name === 'Gomoku');
    $('board-img').src = G.name === 'ChineseChess' ? 'assets/graphics/chess/board.svg' : 'assets/graphics/gomoku/board.jpeg';
    $('p-subtitle').textContent = cfg.label + ' · ' +
      ({ HvH: '双人同屏', HvAI: '人机对战', AIvAI: 'AI 观战' })[G.mode];
    $('wr-label-0').textContent = cfg.sides[0];
    $('wr-label-1').textContent = cfg.sides[1];
  }

  function renderControls() {
    const playing = G.status === 'playing';
    $('btn-back').classList.remove('hidden');
    const humanTurn = playing && currentIsHuman() && !G.aiBusy && G.mode !== 'AIvAI';
    $('btn-resign').classList.toggle('hidden', !humanTurn);
    $('btn-undo').classList.toggle('hidden', !(humanTurn && G.undoStack.length > 0));
    $('logo').classList.toggle('hidden', playing);
    const showNew = !playing;
    $('btn-start').classList.toggle('hidden', !showNew);
    $('btn-reverse').classList.toggle('hidden', !showNew || G.mode === 'AIvAI');
    $('btn-start').textContent = G.status === 'finished' ? '再来一局' : '开始对弈';
    const first = G.players[0];
    $('btn-reverse').textContent = '先手：' + (first.type === 'human' ? '玩家' : first.desc);
    $('btn-reverse').classList.toggle('ai-first', first.type === 'ai');
    $('stage').classList.toggle('playing', playing);
  }

  function infoHtml(sideIdx) {
    const p = G.players[sideIdx];
    const cfg = GAMES[G.name];
    const isTurn = G.status === 'playing' && G.ptm === sideIdx;
    const other = 1 - sideIdx;
    let timeStr = '';
    if (G.status === 'playing' || G.status === 'finished') {
      timeStr = String(G.timers[sideIdx]);
    }
    const wr = p.type === 'ai' && p.winRate !== -1 ? (p.winRate * 100).toFixed(1) + '%' : '--';
    return `
      <div class="si-line ${isTurn ? 'si-turn' : ''}" style="padding:2px 14px">
        <span class="${cfg.sideCls[sideIdx]}">${p.desc} · ${cfg.sides[sideIdx]}</span>
        ${isTurn && p.type === 'ai' ? '<span class="si-thinking"><i></i>AI 思考中<span id="think-sec"></span></span>' : ''}
        ${timeStr !== '' ? `<span class="si-timer ${isTurn && G.timers[sideIdx] <= 10 && G.status === 'playing' ? 'danger' : ''}" id="timer-${sideIdx}">${timeStr}s</span>` : ''}
      </div>
      <div class="si-wr">胜率 ${wr}</div>`;
  }

  function updateInfos() {
    if (!G) return;
    const top = $('info-top'), bottom = $('info-bottom');
    const show = G.status === 'playing';
    top.classList.toggle('hidden', !show);
    bottom.classList.toggle('hidden', !show);
    if (show) { top.innerHTML = infoHtml(1); bottom.innerHTML = infoHtml(0); }
    renderControls();
    updateWr();
  }

  function updateWr() {
    for (const s of [0, 1]) {
      const p = G.players[s];
      const v = p.type === 'ai' ? p.winRate : -1;
      $('wr-val-' + s).textContent = v === -1 ? '--' : (v * 100).toFixed(1) + '%';
      $('wr-' + s).style.width = v === -1 ? '50%' : clamp(v * 100, 2, 100) + '%';
    }
  }

  /* ---- 渲染棋盘 ---- */
  function mkImg(src, cls, x, y) {
    const i = document.createElement('img');
    i.src = src; i.className = cls; i.alt = '';
    i.style.left = x + 'px'; i.style.top = y + 'px';
    i.draggable = false;
    return i;
  }
  function render() {
    const under = $('l-under'), pieces = $('l-pieces'), over = $('l-over');
    under.innerHTML = ''; pieces.innerHTML = ''; over.innerHTML = '';
    if (!G) return;
    const dir = G.name === 'ChineseChess' ? 'assets/graphics/chess/' : 'assets/graphics/gomoku/';

    // 最后一步标记（象棋在棋子下方，对应 pygame draw_last_mark 顺序）
    if (G.name === 'ChineseChess') {
      if (G.history.length) {
        const last = G.history[G.history.length - 1];
        const [r, c, tr, tc] = ChessEnv.action2move(last.action);
        let p = grid2px(r, c); under.appendChild(mkImg(dir + 'blue_dot.png', 'mark', p.x, p.y));
        p = grid2px(tr, tc); under.appendChild(mkImg(dir + 'blue_circle.png', 'mark', p.x, p.y));
      }
      for (let r = 0; r < 10; r++) for (let c = 0; c < 9; c++) {
        const piece = ChessEnv.boardLayer(G.state)[r * 9 + c];
        if (piece === -1 || (G.selected && G.selected[0] === r && G.selected[1] === c)) continue;
        const p = grid2px(r, c);
        pieces.appendChild(mkImg(dir + 'low_shadow.png', 'mark', p.x, p.y));
        const isNew = G.lastPlaced && G.lastPlaced[0] === r && G.lastPlaced[1] === c;
        pieces.appendChild(mkImg(dir + `piece${piece}.png`, 'piece-chess' + (isNew ? ' new' : ''), p.x, p.y));
      }
      // 选中棋子放大 + 光圈
      if (G.selected) {
        const [r, c] = G.selected;
        const piece = ChessEnv.boardLayer(G.state)[r * 9 + c];
        const p = grid2px(r, c);
        over.appendChild(mkImg(dir + 'high_shadow.png', 'mark shadow-big', p.x + 65 / 2 - 52, p.y + 65 / 2 - 52));
        over.appendChild(mkImg(dir + `piece${piece}.png`, 'piece-chess selected', p.x, p.y));
        for (const [a, safe] of G.safeCache) {
          const [, , tr, tc] = ChessEnv.action2move(a);
          const q = grid2px(tr, tc);
          const el = document.createElement('div');
          el.className = 'move-dot ' + (safe ? 'safe' : 'danger');
          el.style.left = (q.x + 32.5) + 'px'; el.style.top = (q.y + 32.5) + 'px'; // 交叉点中心（margin 已偏移半径）
          over.appendChild(el);
        }
      }
    } else {
      // 五子棋
      for (let i = 0; i < 225; i++) {
        const v = G.board[i];
        if (!v) continue;
        const r = Math.floor(i / 15), cc = i % 15;
        const p = grid2px(r, cc);
        const isNew = G.lastPlaced && G.lastPlaced[0] === r && G.lastPlaced[1] === cc;
        pieces.appendChild(mkImg(dir + (v === 1 ? 'black.png' : 'white.png'), 'stone' + (isNew ? ' new' : ''), p.x, p.y));
      }
      if (G.lastAction !== -1 && G.history.length) {
        const a = G.history[G.history.length - 1].action;
        const p = grid2px(Math.floor(a / 15), a % 15);
        over.appendChild(mkImg(dir + 'circle.png', 'mark sm', p.x, p.y));
      }
      if (G.status === 'finished') {
        for (const [r, c] of G.winStones) {
          const p = grid2px(r, c);
          over.appendChild(mkImg(dir + 'circle.png', 'mark sm', p.x, p.y));
        }
        G.history.forEach((h, idx) => {
          const p = grid2px(Math.floor(h.action / 15), h.action % 15);
          const el = document.createElement('div');
          el.className = 'step-no';
          el.style.left = p.x + 'px'; el.style.top = p.y + 'px';
          el.style.color = h.player === 0 ? '#fff' : '#000';
          el.textContent = idx + 1;
          over.appendChild(el);
        });
      }
      syncGhost();
    }
  }

  function syncGhost() {
    const g = $('gomoku-ghost');
    if (!g) return;
    const show = G && G.name === 'Gomoku' && G.status === 'playing' && currentIsHuman() &&
      !G.aiBusy && G.hover && !G.board[G.hover[0] * 15 + G.hover[1]];
    if (!show) { g.classList.add('hidden'); return; }
    const p = grid2px(G.hover[0], G.hover[1]);
    g.style.left = p.x + 'px'; g.style.top = p.y + 'px';
    g.classList.remove('hidden');
  }

  function currentIsHuman() { return G.players[G.ptm].type === 'human'; }

  /* ---- 棋谱 ---- */
  function renderMoves() {
    const ol = $('moves');
    ol.innerHTML = '';
    G.history.forEach((h, i) => {
      const li = document.createElement('li');
      li.className = h.player === 0 ? 'red-move' : 'black-move';
      li.innerHTML = h.text + (h.note ? `<em>${h.note}</em>` : '');
      ol.appendChild(li);
    });
    ol.scrollTop = ol.scrollHeight;
  }

  /* ================= 落子与规则 ================= */
  function applyAction(action) {
    const mover = G.ptm;
    if (G.players[mover].type === 'human') pushUndo();
    let text = '', note = '', ended = false, winner = 2, reason = '';

    if (G.name === 'ChineseChess') {
      const [r, c, tr, tc] = ChessEnv.action2move(action);
      const preBoard = ChessEnv.boardLayer(G.state).slice();
      const captured = preBoard[tr * 9 + tc];
      text = ChessEnv.describeMove(preBoard, action);
      note = `(${r},${c})→(${tr},${tc})`;
      G.state = ChessEnv.virtualStep(G.state, action);
      G.lastPlaced = [tr, tc];
      G.history.push({ action, player: mover, text, note });
      Sound.play('piece');
      // 胜负：checkWinner（吃王/长将/和）+ 绝杀判定，与 core/env + pygame UI 对齐
      let mate = false, check = false;
      const result = ChessEnv.checkWinner(G.state, mover);
      if (result !== 2) {
        ended = true;
        winner = result === 1 ? mover : (result === 0 ? -1 : 1 - mover);
        reason = result === 0 ? '和棋' : '绝杀';
        mate = reason === '绝杀';
      } else {
        mate = ChessEnv.isCheckmate(G.state, 1 - mover);
        if (mate) { ended = true; winner = mover; reason = '绝杀'; }
        else check = ChessEnv.isCheck(G.state, 1 - mover);
      }
      // 音效优先级（对应 play_place_sound）：绝杀 > 将军 > 吃子
      if (mate) Sound.play('checkmate');
      else if (check) Sound.play('check');
      else if (captured !== -1 && captured !== 4 && captured !== 11) Sound.play('capture');
    } else {
      const row = Math.floor(action / 15), col = action % 15;
      G.board[action] = mover === 0 ? 1 : 2;
      G.lastPlaced = [row, col];
      text = `${row + 1},${col + 1}`;
      G.history.push({ action, player: mover, text });
      Sound.play('place');
      const stones = GomokuEnv.getWinStones(G.board, action, G.board[action]);
      if (stones.length) { ended = true; winner = mover; reason = '五连'; G.winStones = stones; }
      else if (GomokuEnv.isDraw(G.board)) { ended = true; winner = -1; reason = '棋盘已满'; }
    }
    if (ended) { note = reason; }
    if (!ended) {
      G.ptm = 1 - mover;
      G.steps++;
      G.lastAction = action;
      G.selected = null; G.safeCache.clear();
      resetTimer();
    } else {
      G.lastAction = action;
    }
    render(); renderMoves(); updateInfos();
    if (ended) { finishGame(winner, reason); return; }
    advance();
  }

  async function finishGame(winner, reason) {
    G.status = 'finished';
    Sound.play('win');
    Sound.stopBgm();
    const cfg = GAMES[G.name];
    $('result-badge').src = `assets/graphics/${G.name === 'ChineseChess' ? 'chess' : 'gomoku'}/` +
      cfg.winImgs[winner === -1 ? 2 : winner];
    $('result-text').textContent = winner === -1 ? '和棋 · ' + reason
      : `${cfg.sides[winner]} ${G.players[winner].desc} 获胜 · ${reason}`;
    $('dlg-result').classList.remove('hidden');
    render(); renderControls(); updateInfos();
  }

  // 悔棋快照：HvAI 仅人类行棋压栈（一次回退 = 撤销「人着+AI应着」成对）；HvH 逐着回退
  function pushUndo() {
    G.undoStack.push({
      state: G.state ? G.state.slice() : null,
      board: G.board ? G.board.slice() : null,
      ptm: G.ptm, lastAction: G.lastAction, steps: G.steps,
      histLen: G.history.length, timers: { 0: G.timers[0], 1: G.timers[1] },
      winRates: G.players.map((p) => (p.type === 'ai' ? p.winRate : -1))
    });
    if (G.undoStack.length > 80) G.undoStack.shift();
  }

  function undo() {
    if (!G || G.status !== 'playing' || G.aiBusy || !G.undoStack.length) return;
    const s = G.undoStack.pop();
    if (s.state) G.state = s.state; else if (s.board) G.board = s.board;
    G.ptm = s.ptm; G.lastAction = s.lastAction; G.steps = s.steps;
    G.history.length = s.histLen;
    G.timers = s.timers;
    G.lastPlaced = null; G.selected = null; G.safeCache.clear();
    G.players.forEach((p, i) => { if (p.type === 'ai') p.winRate = s.winRates[i]; });
    // AI 会话无需通知：下次轮到我方 update 时服务端检测到局面不匹配会自动 set_root
    render(); renderMoves(); updateInfos();
    advance();
  }

  function resign() {
    if (G && G.status === 'playing') finishGame(1 - G.ptm, '认输');
  }

  /* ================= AI 会话 ================= */
  let deviceNoted = false;
  let lastDeviceInfo = null;   // {current, device_note, devices:[...]}
  let deviceBusy = false;

  function applyDeviceInfo(device, note) {
    if (!device) return;
    if (lastDeviceInfo && lastDeviceInfo.current === device && lastDeviceInfo.device_note === (note || '')) return;
    lastDeviceInfo = { current: device, device_note: note || '', devices: lastDeviceInfo?.devices };
    if (G) G.serverDevice = device;
    renderDeviceInfo();
    if (note && !deviceNoted && G) {
      deviceNoted = true;
      toast('后端已降级 CPU 推理：' + note, true, 6000);
    }
  }

  function renderDeviceInfo() {
    const el = $('p-device');
    if (!el || !G || !lastDeviceInfo) { if (el && !G) return; if (el) el.classList.add('hidden'); return; }
    el.classList.remove('hidden');
    const { current, device_note, devices } = lastDeviceInfo;
    let html;
    if (!devices || !devices.length) {
      html = '推理设备：<b>' + current + '</b>';
    } else {
      html = '<div class="p-label">推理设备' + (deviceSwitchEnabled ? '（点击切换）' : '') + '</div>';
      for (const d of devices) {
        const isActive = d.id === current || (d.id === 'cuda' && String(current).startsWith('cuda'));
        const usable = d.status === 'available' || d.status === 'active';
        const vramTxt = d.vram ? `<i>${d.vram.free}/${d.vram.total}G</i>` : '';
        let mark, cls = 'dev-row';
        if (isActive) { mark = '使用中'; cls += ' active'; }
        else if (d.status === 'available') { mark = '切换'; cls += ' jump'; }
        else if (d.status === 'unavailable') { mark = '不可用'; cls += ' off'; }
        else { mark = '不可用'; cls += ' off'; }
        const why = d.reason || (d.status === 'available' && device_note && current === 'cpu' ? '降级：' + device_note : '');
        html += `<button class="${cls}" data-dev="${d.id}" ${isActive || !usable || !deviceSwitchEnabled ? 'disabled' : ''}>` +
          `<b>${d.id}</b><span class="dr-name">${d.name}</span>${vramTxt}<em>${mark}</em>` +
          (why && !isActive ? `<span class="dr-why" title="${why.replace(/"/g, '&quot;')}">${why}</span>` : '') +
          `</button>`;
      }
    }
    if (device_note && current === 'cpu' && devices) {
      const brief = device_note.length > 42 ? device_note.slice(0, 42) + '…' : device_note;
      html = `<div class="warn dr-note" title="${device_note.replace(/"/g, '&quot;')}">⚠ ${brief}</div>` + html;
    }
    el.innerHTML = html;
  }
  const deviceSwitchEnabled = true;

  // 点击切换设备（委托，列表可反复重绘）
  document.addEventListener('click', async (ev) => {
    const btn = ev.target.closest('#p-device .jump');
    if (!btn || deviceBusy) return;
    if (G && G.aiBusy) { toast('AI 正在思考，稍后再切换', true); return; }
    const id = btn.dataset.dev;
    deviceBusy = true;
    btn.querySelector('em').textContent = '迁移中…';
    try {
      const r = await API.setDevice(id);
      toast('推理设备已切换：' + r.device);
      await probeServer(true);
    } catch (e) {
      toast('切换失败：' + e.message, true, 5000);
      renderDeviceInfo();
    } finally {
      deviceBusy = false;
    }
  });

  // 服务信息与推理设备：打开页面即探测
  async function probeServer(force) {
    try {
      const r = await API.devices();
      setServerDot('ok');
      lastDeviceInfo = { current: r.current, device_note: r.device_note, devices: r.devices };
      if (G) G.serverDevice = r.current;
      renderDeviceInfo();
      if (r.device_note && !deviceNoted && G) {
        deviceNoted = true;
        toast('后端已降级 CPU 推理：' + r.device_note, true, 6000);
      }
    } catch (e) {
      if (e.status === 404) {   // 旧版后端无 /devices：退回 /info
        try {
          const r = await API.info();
          setServerDot('ok');
          applyDeviceInfo(r.device, r.device_note);
        } catch (e2) { setServerDot('err'); }
      } else {
        setServerDot('err');
      }
    }
  }

  function stateArrayForAPI() {
    return G.name === 'ChineseChess'
      ? ChessEnv.stateToNested(G.state)
      : GomokuEnv.stateToNested(G.board, G.ptm);
  }

  async function aiTurn() {
    if (!G || G.status !== 'playing') return;
    const p = G.players[G.ptm];
    if (p.type !== 'ai') return;
    const token = G.token;
    G.aiBusy = true;
    G.aiThinkStart = Date.now();
    if (lastDeviceInfo && lastDeviceInfo.current === 'cpu' && !G.cpuThinkWarned) {
      G.cpuThinkWarned = true;
      toast('CPU 推理较慢，每步约需十几秒到一分钟，请稍候', false, 4000);
    }
    updateInfos();
    // 后端重启/超时回收会让旧 pid 变 404：清掉重建会话重试一次（update 携带完整局面，可无损续战）
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        if (p.pid == null) {
          const r = await API.setup(p.modelId, G.name);
          p.pid = r.pid;
          if (r.model_id != null) p.modelId = r.model_id;
          if (r.device) applyDeviceInfo(r.device, r.device_note);
        }
        const ru = await API.update(p.pid, stateArrayForAPI(), G.lastAction, G.ptm);
        if (token !== G.token) return;
        if (ru.win_rate != null) p.winRate = ru.win_rate;
        updateInfos();
        const ra = await API.getAction(p.pid);
        if (token !== G.token) return;
        p.winRate = ra.win_rate != null ? ra.win_rate : p.winRate;
        if (ra.model_id != null) p.modelId = ra.model_id;
        G.aiBusy = false;
        if (ra.action == null || ra.action < 0) throw new Error('AI 未返回有效着法');
        applyAction(ra.action);
        return;
      } catch (e) {
        if (token !== G.token) return;
        if (e.status === 404 && attempt === 0) {
          p.pid = null;
          toast('AI 会话已失效（后端重启过？），正在自动重建…', false, 3000);
          continue;
        }
        G.aiBusy = false;
        setServerDot('err');
        toast('AI 服务异常：' + e.message, true, 4000);
        updateInfos();
        return;
      }
    }
  }

  function advance() {
    if (G.status !== 'playing') return;
    if (G.players[G.ptm].type === 'ai') {
      setTimeout(aiTurn, G.mode === 'AIvAI' ? 420 : 120);
    } else {
      G.aiBusy = false;
      updateInfos();
    }
  }

  /* ================= 心跳 ================= */
  setInterval(async () => {
    if (!G || $('stage').classList.contains('hidden')) return;
    for (const p of G.players) {
      if (p.type === 'ai' && p.pid) {
        try {
          const r = await API.heartbeat(p.pid);
          if (r.win_rate != null && G.status === 'playing') p.winRate = r.win_rate;
          setServerDot('ok');
          probeServer();
          updateInfos();
        } catch (e) {
          if (e.status === 404) {
            // 会话已被服务端回收/重启清空：放弃旧 pid，轮到 AI 时自动重建
            p.pid = null;
            setServerDot('ok');
            toast('检测到后端会话已重置，AI 将在下一手自动重建会话', false, 3500);
          }
          /* 其余(断网等)静默，下个周期重试 */
        }
      }
    }
  }, 20000);

  /* ================= 计时器（人类每步限时，AI 免计） ================= */
  function resetTimer() { G.timers[G.ptm] = TURN_LIMIT; }
  setInterval(() => {
    if (!G || G.status !== 'playing') return;
    const mover = G.ptm;
    G.timers[mover]--;  // AI 回合同样倒数（展示用），但超时不判负
    const el = document.getElementById('timer-' + mover); // 只改文本，不重建信息条（防闪烁/spinner 重启）
    if (el) {
      el.textContent = G.timers[mover] + 's';
      el.classList.toggle('danger', G.players[mover].type === 'human' && G.timers[mover] <= 10);
    }
    const th = document.getElementById('think-sec');
    if (th && G.aiThinkStart) th.textContent = ' ' + Math.round((Date.now() - G.aiThinkStart) / 1000) + 's';
    if (G.timers[mover] <= 0 && G.players[mover].type === 'human') finishGame(1 - mover, '超时');
  }, 1000);

  /* ================= 输入 ================= */
  function boardClick(e) {
    if (!G || G.status !== 'playing' || !currentIsHuman() || G.aiBusy) return;
    Sound.unlock();
    const { x, y } = ev2stage(e);
    const grid = px2grid(x, y);
    if (!grid) return;
    const [r, c] = grid;
    if (G.name === 'Gomoku') {
      const action = GomokuEnv.move2action(r, c);
      if (G.board[action]) { return; }
      applyAction(action);
    } else {
      const board = ChessEnv.boardLayer(G.state);
      const piece = board[r * 9 + c];
      if (G.selected) {
        const [sr, sc] = G.selected;
        const a = ChessEnv.move2action(sr, sc, r, c);
        if (a >= 0 && G.safeCache.has(a)) {
          applyAction(a);
          return;
        }
        // 点到己方可走子 → 改选；否则取消选择（与 pygame 略优）
        if (piece !== -1 && ChessEnv.ownerOf(piece) === G.ptm) {
          selectChess(r, c);
        } else {
          Sound.play('place'); // 无效点击（对应 piece_sound）
          G.selected = null; G.safeCache.clear();
          render();
        }
      } else if (piece !== -1 && ChessEnv.ownerOf(piece) === G.ptm) {
        selectChess(r, c);
      }
    }
  }

  function selectChess(r, c) {
    G.selected = [r, c];
    G.safeCache.clear();
    for (const a of ChessEnv.getValidActionsFromPos(G.state, r, c)) {
      const next = ChessEnv.virtualStep(G.state, a);
      const [, , tr, tc] = ChessEnv.action2move(a);
      const captured = ChessEnv.boardLayer(G.state)[tr * 9 + tc];
      let safe = !ChessEnv.isCheck(next, G.ptm);
      if (captured === 4 || captured === 11) safe = true; // 吃王即胜，不算送将
      G.safeCache.set(a, safe);
    }
    Sound.play('piece');
    render();
  }

  function boardMove(e) {
    if (!G || G.name !== 'Gomoku') return;
    const { x, y } = ev2stage(e);
    const g = px2grid(x, y);
    const prev = G.hover;
    const changed = (!g) !== (!prev) || (g && prev && (g[0] !== prev[0] || g[1] !== prev[1]));
    G.hover = g;
    if (changed) syncGhost();  // 只移动幽灵子，不重建棋盘层（防闪烁）
  }

  /* ================= 控件事件 ================= */
  $('board-wrap').addEventListener('click', boardClick);
  $('board-wrap').addEventListener('mousemove', boardMove);
  $('btn-back').addEventListener('click', () => {
    if (!G) return;
    Sound.play('click');
    if (G.status === 'playing') $('dlg-quit').classList.remove('hidden');
    else backToLauncher();
  });
  $('btn-resign').addEventListener('click', () => { Sound.play('click'); resign(); });
  $('btn-undo').addEventListener('click', () => { Sound.play('click'); undo(); });
  $('btn-start').addEventListener('click', () => {
    Sound.unlock(); Sound.play('click');
    if (G.status === 'finished') { // 重开一局：本地重置 + 通知服务端 reset
      G.token++;
      for (const p of G.players) {
        if (p.type === 'ai' && p.pid) { API.reset(p.pid).catch(() => {}); p.pid = null; p.winRate = -1; }
      }
      G.ptm = 0; G.lastAction = -1; G.steps = 0; G.winner = 2;
      G.history = []; G.selected = null; G.safeCache.clear(); G.winStones = []; G.lastPlaced = null; G.undoStack = [];
      newGameEnv();
      render(); renderMoves();
    }
    G.status = 'playing';
    resetTimer();
    $('dlg-result').classList.add('hidden');
    Sound.setBgm('bgm.mp3');
    renderControls(); updateInfos();
    advance();
  });
  $('btn-reverse').addEventListener('click', () => {
    Sound.play('click');
    G.players.reverse();
    renderControls(); updateInfos();
  });
  $('btn-quit-yes').addEventListener('click', () => { $('dlg-quit').classList.add('hidden'); backToLauncher(); });
  $('btn-quit-no').addEventListener('click', () => { $('dlg-quit').classList.add('hidden'); });

  function backToLauncher() {
    if (G) {
      G.token++;
      for (const p of G.players) if (p.type === 'ai' && p.pid) API.reset(p.pid).catch(() => {});
    }
    if (G) G.hover = null;
    syncGhost();
    G = null;
    $('info-top').classList.add('hidden');
    $('info-bottom').classList.add('hidden');
    $('btn-back').classList.add('hidden');
    $('stage').classList.add('hidden');
    $('panel').classList.add('hidden');
    $('dlg-result').classList.add('hidden');
    $('dlg-quit').classList.add('hidden');
    $('launcher').classList.remove('hidden');
    Sound.setBgm('launcher_bgm.wav');
    launcher.showStep('game');
    render();
  }

  /* ---- 设置 ---- */
  $('btn-settings').addEventListener('click', () => {
    $('in-url').value = localStorage.getItem('zz_server_url') || '';
    $('dlg-settings').classList.remove('hidden');
  });
  $('btn-close-settings').addEventListener('click', () => $('dlg-settings').classList.add('hidden'));
  $('btn-save-url').addEventListener('click', () => {
    API.setBaseUrl($('in-url').value.trim());
    launcher.modelsCache = {};
    deviceNoted = false;
    lastDeviceInfo = null;
    setServerDot('mid');
    probeServer();
    $('dlg-settings').classList.add('hidden');
    toast('已保存服务地址');
  });

  $('btn-sfx').addEventListener('click', (e) => {
    Sound.unlock();
    Sound.sfxOn = !Sound.sfxOn;
    localStorage.setItem('zz_sfx', Sound.sfxOn ? '1' : '0');
    e.currentTarget.classList.toggle('on', Sound.sfxOn);
  });
  $('btn-bgm').addEventListener('click', (e) => {
    Sound.unlock();
    Sound.bgmOn = !Sound.bgmOn;
    localStorage.setItem('zz_bgm', Sound.bgmOn ? '1' : '0');
    e.currentTarget.classList.toggle('on', Sound.bgmOn);
    if (Sound.bgmOn && Sound.bgmName) Sound.bgmEl && Sound.bgmEl.play().catch(() => {});
    else Sound.stopBgm() , Sound.setBgm(G ? 'bgm.mp3' : 'launcher_bgm.wav');
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && G) {
      if (!$('dlg-result').classList.contains('hidden')) return;
      if (G.status === 'playing') $('dlg-quit').classList.remove('hidden');
      else backToLauncher();
    }
  });
  document.addEventListener('pointerdown', () => Sound.unlock(), { once: true });

  // 窄屏：点面板头部展开/收起棋谱抽屉
  document.querySelector('#panel .p-head').addEventListener('click', () => {
    if (mqNarrow.matches) $('panel').classList.toggle('open');
  });
  mqNarrow.addEventListener('change', () => $('panel').classList.remove('open'));

  /* ================= 启动 ================= */
  Sound.init();
  $('btn-sfx').classList.toggle('on', Sound.sfxOn);
  $('btn-bgm').classList.toggle('on', Sound.bgmOn);
  setServerDot('mid');
  probeServer();
  API.getModels('Gomoku').then((d) => { launcher.modelsCache['Gomoku'] = d; }).catch(() => {});
  launcher.init();
  fitStage();
})();
