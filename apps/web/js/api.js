/* 与 play_server (Flask) 的 HTTP 客户端，协议对齐 core/player/ai_client.py */
(function (global) {
  'use strict';

  function baseUrl() {
    // 同源部署时为空；跨机器在设置中填入，如 http://192.168.0.126:5000/
    const saved = localStorage.getItem('zz_server_url');
    if (saved && saved.trim()) return ensureSlash(saved.trim());
    if (location.protocol === 'file:') return 'http://127.0.0.1:5000/';
    return '/';
  }
  function ensureSlash(u) { return /\/$/.test(u) ? u : u + '/'; }

  async function post(path, payload, timeoutMs = 120000) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
      const resp = await fetch(ensureSlash(baseUrl()) + path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: ctrl.signal
      });
      let data = null;
      try { data = await resp.json(); } catch (e) { /* 非 JSON */ }
      if (!resp.ok) {
        const err = new Error((data && data.error) ? String(data.error).split('\n')[0] : `HTTP ${resp.status}`);
        err.status = resp.status;
        throw err;
      }
      return data;
    } catch (err) {
      if (err.status !== undefined) throw err;           // 带状态码的 HTTP 错误原样上抛
      if (err.name === 'AbortError') throw new Error('推理服务响应超时');
      const e2 = new Error(`无法连接推理服务（${err.message}）。请确认已运行 python -m scripts.play_server`);
      e2.status = 0;
      throw e2;
    } finally {
      clearTimeout(timer);
    }
  }

  async function get(path, timeoutMs = 8000) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
      const resp = await fetch(ensureSlash(baseUrl()) + path, { signal: ctrl.signal });
      if (!resp.ok) { const e = new Error(`HTTP ${resp.status}`); e.status = resp.status; throw e; }
      return await resp.json();
    } finally { clearTimeout(timer); }
  }

  global.API = {
    baseUrl,
    info: () => get('info'),
    devices: () => get('devices'),
    setDevice: (device) => post('device', { device }, 15000),
    setBaseUrl(u) { localStorage.setItem('zz_server_url', u || ''); },
    getModels: (envName) => post('models', { env_name: envName }, 8000),
    setup: (modelId, envClass) => post('setup', { model_id: modelId, env_class: envClass }, 180000),
    // 单步 MCTS 在 CPU 上可能需要几分钟，超时放宽避免误报"服务异常"
    update: (pid, array, action, playerToMove) => post('update', { pid, array, action, player_to_move: playerToMove }, 900000),
    getAction: (pid) => post('get_action', { pid }, 60000),
    reset: (pid) => post('reset', { pid }),
    heartbeat: (pid) => post('heartbeat', { pid }, 10000)
  };
})(typeof window !== 'undefined' ? window : globalThis);
