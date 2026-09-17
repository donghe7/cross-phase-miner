'use strict';
const $ = id => document.getElementById(id);
const state = { metas: [], byId: new Map(), cards: new Map(), tiles: new Map(), rows: new Map(), config: null,
  snapshot: null, selected: null, followed: null, detail: null, detailBusy: false, query: '',
  receivedAt: 0, connected: false, started: false, detailError: false, speedPending: false, speedError: false,
  mapScope: 'zone', mapPage: 0, fleetPage: 0, robotButtons: new Map() };
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const percent = value => !Number.isFinite(value) ? '—' : `${(value * 100).toFixed(1).replace(/\.0$/, '')}%`;
const shortId = id => (id || '').replace('robot_', 'R');
const nameOf = id => state.byId.get(id)?.name || id || '起点未上报';
const label = ['暂无数据', '实时红灯', '实时绿灯', '预测红灯', '预测绿灯', '共识不足'];
const colors = ['unknown', 'red', 'green', 'predicted', 'predicted', 'amber'];
const colorName = color => ({RED:'红灯', GREEN:'绿灯', UNKNOWN:'未知'})[color] || '未知';
const setHTML = (node, html) => { if (node.innerHTML !== html) node.innerHTML = html; };
const now = () => state.snapshot ? state.snapshot.sim_now + (state.connected ? (performance.now() - state.receivedAt) / 1000 * state.snapshot.speed : 0) : 0;
// Freshness belongs to the server snapshot that contains the observation.
// Extrapolating wall time at high simulation speeds would invent a dropout between UI refreshes.
const age = timestamp => Math.max(0, (state.snapshot?.sim_now || 0) - timestamp);
const robotStale = robot => robot.action === 'TRAVEL'
  ? robot.travel_arrives_at == null || state.snapshot.sim_now > robot.travel_arrives_at + state.config.live_stale_seconds
  : age(robot.updated_at) > state.config.live_stale_seconds;
const robotAction = robot => robotStale(robot) ? '位置待更新' : robot.action === 'TRAVEL' ? '行驶中'
  : robot.action === 'CROSS' ? '通过路口' : robot.action === 'TIMEOUT' ? '超时离开' : robot.mode === 'scout' ? '侦察采样' : '等待通行';
const observationStale = observation => !state.connected || age(observation.timestamp) > state.config.live_stale_seconds;
const activeRobots = () => state.snapshot?.robots || [];
const observationOf = robot => robot.observation;
const simTime = timestamp => timestamp == null ? '—' : new Date((timestamp + 9 * 3600) * 1000).toISOString().slice(11, 19);
const modelLabel = status => ({unlearned:'未建模', learning:'学习中', reliable:'可预测', unavailable:'数据待更新'})[status] || '未建模';
const summaryOf = (id, model = null) => state.snapshot?.learning?.[id] || {
  status:model ? (model.reliable ? 'reliable' : 'learning') : state.snapshot?.learning ? 'unlearned' : 'unavailable',
  period:state.snapshot?.period, confidence:model?.confidence ?? null, samples:model?.samples ?? 0,
  history:{}, visits:{robots:0, visits:0, scout:0, normal:0, cross:0, timeout:0}};
const progressOf = summary => summary.progress || (!Number.isFinite(state.config.model_min_samples) || !Number.isFinite(state.config.model_min_confidence) ? {
  required_samples:null, remaining_samples:null, sample_progress:0, required_confidence:null,
  confidence_met:false, estimate_basis:'unavailable', estimated_scout_visits:null, pending_updates:0
} : {
  samples:summary.samples, required_samples:state.config.model_min_samples,
  remaining_samples:Math.max(0, state.config.model_min_samples - summary.samples),
  sample_progress:Math.min(1, summary.samples / state.config.model_min_samples),
  confidence_met:(summary.confidence || 0) >= state.config.model_min_confidence,
  required_confidence:state.config.model_min_confidence,
  estimated_scout_visits:[Math.ceil(state.config.model_min_samples / 4), Math.ceil(state.config.model_min_samples / 3)],
  estimate_basis:summary.status === 'reliable' ? 'ready' : 'planning', pending_updates:0
});
const estimateLabel = progress => {
  if (progress.estimate_basis === 'ready') return '已满足预测条件';
  if (progress.estimate_basis === 'confidence') return '样本已够，需继续提高置信度';
  if (progress.estimate_basis === 'no_transitions') return '近期未采到跳变，暂无法估算';
  if (!progress.estimated_scout_visits) return '学习进度待更新';
  const [low, high] = progress.estimated_scout_visits;
  return `预计再侦察 ${low === high ? low : low + '–' + high} 次补齐样本`;
};
const firstModelLabel = summary => summary.history?.first_model ? simTime(summary.history.first_model.sim_time)
  : summary.confidence == null ? '尚未生成' : '旧模型 · 时间未记录';

async function getJSON(url) {
  const response = await fetch(url, {signal: AbortSignal.timeout(6000)});
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  return response.json();
}
async function boot() {
  try {
    const [metas, config] = await Promise.all([getJSON('/v1/intersections'), getJSON('/v1/config')]);
    state.config = config;
    state.metas = metas.intersections;
    state.metas.forEach((meta, index) => state.byId.set(meta.id, {...meta, index}));
    $('obs-threshold').textContent = percent(config.obs_confidence_threshold);
    $('consensus-threshold').textContent = percent(config.consensus_threshold);
    $('all-summary').textContent = `全部 ${state.metas.length.toLocaleString()} 个路口 · 展开全城概览`;
    buildMap();
    $('error').hidden = true;
    connect();
    if (!state.started) {
      state.started = true;
      setInterval(paint, 500);
      setInterval(refreshDetail, 200);
    }
  } catch (error) {
    $('error').hidden = false;
    $('error').textContent = '无法读取服务器配置，3 秒后重试。请确认 signal server 已启动。';
    setConnection(false);
    setTimeout(boot, 3000);
  }
}
function buildMap() {
  $('zone-map').replaceChildren(); $('field').replaceChildren();
  state.cards.clear(); state.tiles.clear();
  state.metas.forEach(meta => {
    const tile = document.createElement('button');
    tile.className = 'tile'; tile.title = `${meta.name} · ${meta.id}`;
    tile.setAttribute('aria-label', tile.title);
    tile.addEventListener('click', () => selectCrossing(meta.id, true));
    state.tiles.set(meta.id, tile); $('field').append(tile);
    {
      const card = document.createElement('button');
      card.className = 'crossing';
      card.hidden = true;
      card.addEventListener('click', () => selectCrossing(meta.id, true));
      state.cards.set(meta.id, card); $('zone-map').append(card);
    }
  });
}
let reconnectTimer;
function connect() {
  let socket;
  try { socket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`); }
  catch (error) { fallback(); return; }
  socket.onmessage = event => {
    try { applySnapshot(JSON.parse(event.data)); setConnection(true); }
    catch (error) { setConnection(false); }
  };
  socket.onerror = () => setConnection(false);
  socket.onclose = () => { setConnection(false); fallback(); };
}
async function fallback() {
  try { applySnapshot(await getJSON('/v1/snapshot')); setConnection(true, '轮询快照'); }
  catch (error) { setConnection(false); }
  clearTimeout(reconnectTimer);
  reconnectTimer = setTimeout(connect, 2000);
}
function setConnection(up, text = '实时连接') {
  state.connected = up;
  paintSpeed();
  $('connection').textContent = up ? text : '连接中断 · 正在重试';
  $('connection').className = `connection ${up ? 'on' : 'off'}`;
  if (state.snapshot) {
    $('error').hidden = up;
    $('error').textContent = '连接中断：以下为最后收到的快照，当前位置和观测可能已过期。';
  }
}
function applySnapshot(snapshot) {
  state.snapshot = snapshot; state.receivedAt = performance.now();
  if (!state.speedPending && !state.speedError) $('speed-status').textContent = `当前 ${snapshot.speed}× · 全局生效`;
  paintSpeed();
  if (!state.selected) {
    const robot = snapshot.robots[0];
    if (robot) followRobot(robot.robot_id);
    else selectCrossing(state.metas.find(m => m.in_zone)?.id || state.metas[0]?.id);
  }
  if (state.followed) {
    const robot = snapshot.robots.find(r => r.robot_id === state.followed);
    const target = robot?.intersection_id || robot?.destination_id;
    if (target && target !== state.selected) selectCrossing(target);
  }
  paint();
}

function paintSpeed() {
  document.querySelectorAll('#speed-options button').forEach(button => {
    button.disabled = !state.connected || !state.snapshot || state.speedPending;
    button.setAttribute('aria-pressed', String(Number(button.dataset.speed) === state.snapshot?.speed));
  });
}

async function changeSpeed(speed) {
  if (state.speedPending || !state.connected || !state.snapshot || state.snapshot.speed === speed) return;
  state.speedPending = true;
  state.speedError = false;
  $('speed-status').textContent = `正在切换至 ${speed}×…`;
  paintSpeed();
  try {
    const response = await fetch('/v1/clock/speed', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({speed}), signal: AbortSignal.timeout(6000)
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    // Refresh the authoritative snapshot, including all time-based readouts.
    applySnapshot(await getJSON('/v1/snapshot'));
    $('speed-status').textContent = `已切换至 ${state.snapshot.speed}× · 全局生效`;
    refreshDetail();
  } catch (error) {
    state.speedError = true;
    $('speed-status').textContent = '切换未确认，请检查连接后重试';
  } finally {
    state.speedPending = false;
    paintSpeed();
  }
}

document.querySelectorAll('#speed-options button').forEach(button => {
  button.addEventListener('click', () => changeSpeed(Number(button.dataset.speed)));
});
function selectCrossing(id, manual = false) {
  if (!id) return;
  if (manual) state.followed = null;
  if (state.selected !== id) {
    state.selected = id; state.detail = null; state.detailError = false;
    $('detail').innerHTML = '<p class="empty">正在读取路口观测…</p>';
  }
  refreshDetail(); paint();
  if (manual && window.innerWidth <= 850) document.querySelector('.detail-panel').scrollIntoView({behavior:'smooth', block:'start'});
}
function followRobot(id) {
  state.followed = id;
  const index = activeRobots().slice().sort((a, b) => a.robot_id.localeCompare(b.robot_id)).findIndex(r => r.robot_id === id);
  if (index >= 0) state.fleetPage = Math.floor(index / 5);
  const robot = activeRobots().find(r => r.robot_id === id);
  if (robot) selectCrossing(robot.intersection_id || robot.destination_id);
  paint();
}
async function refreshDetail() {
  if (!state.selected || state.detailBusy) return;
  const id = state.selected;
  state.detailBusy = true;
  try {
    const detail = await getJSON(`/v1/intersections/${encodeURIComponent(id)}`);
    if (state.selected === id) { state.detail = detail; state.detailError = false; paintDetail(); }
  } catch (error) {
    if (state.selected === id) {
      state.detailError = true;
      $('detail-status').textContent = '详情更新失败';
      if (!state.detail) $('detail').innerHTML = '<p class="empty">路口详情暂时无法读取，正在重试。</p>';
    }
  } finally {
    state.detailBusy = false;
    if (state.selected !== id) refreshDetail();
  }
}
function paint() {
  const snapshot = state.snapshot;
  if (!snapshot) return;
  const seconds = Math.floor((now() + 9 * 3600) % 86400);
  $('clock').textContent = [Math.floor(seconds / 3600), Math.floor(seconds % 3600 / 60), seconds % 60].map(n => String(n).padStart(2, '0')).join(':');
  $('clock-meta').textContent = `${snapshot.period === 'day' ? '日间' : '夜间'} / KST · ${snapshot.speed}× 仿真`;
  const robots = activeRobots();
  const travel = robots.filter(r => r.action === 'TRAVEL' && !robotStale(r)).length;
  const stale = robots.filter(robotStale).length;
  $('robot-count').textContent = robots.length;
  $('robot-summary').textContent = `${robots.length - travel - stale} 台在路口 · ${travel} 台行驶中 · ${stale} 台待更新`;
  const counts = snapshot.counts;
  $('observed-count').textContent = counts.observed_red + counts.observed_green;
  $('coverage-summary').textContent = `${counts.predicted_red + counts.predicted_green} 处模型预测 · ${counts.disputed} 处共识不足`;
  paintTracking(robots); paintFleet(robots); paintMap(robots); paintDetail();
  const stats = snapshot.stats;
  const metrics = [['数据库', state.config.database_backend === 'postgresql' ? 'PostgreSQL' : 'SQLite'],
    ['观测上传', state.config.observation_transport === 'mqtt' ? 'MQTT' : 'HTTP'],
    ['上传帧数', stats.observations.toLocaleString()], ['低于门槛丢弃', stats.rejected.toLocaleString()],
    ['可靠 / 已学模型', `${stats.models_reliable} / ${stats.models}`], ['摄取速度', `${stats.obs_per_sec}/s`], ['学习队列', stats.queue_depth]];
  setHTML($('ledger'), metrics.map(([k,v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join(''));
}
function paintTracking(robots) {
  const robot = robots.find(r => r.robot_id === state.followed);
  if (!robot) {
    setHTML($('tracking'), `<p class="empty">${robots.length ? '点击下方机器人编号，跟随它的当前位置与观测。当前可自由查看路口。' : '暂无机器人上报。启动 fleet 后，会显示当前位置和行驶目的地。'}</p>`);
    return;
  }
  const travelling = robot.action === 'TRAVEL';
  const eta = travelling && robot.travel_arrives_at != null ? Math.max(0, robot.travel_arrives_at - now()) : null;
  const duration = robot.travel_arrives_at - robot.travel_started_at;
  const progress = duration > 0 ? Math.max(0, Math.min(100, (now() - robot.travel_started_at) / duration * 100)) : 0;
  const location = travelling ? `${nameOf(robot.origin_id)} → ${nameOf(robot.destination_id)}` : nameOf(robot.intersection_id);
  const note = robotStale(robot) ? '上报已过期，实际位置待确认'
    : travelling ? `${eta > 0 ? `预计 ${Math.ceil(eta)} 仿真秒后到达` : '等待到达上报'} · 行程进度估算，无 GPS 定位`
    : `${robotAction(robot)} · ${age(robot.updated_at).toFixed(1)} 仿真秒前上报`;
  if (!$('track-content')) {
    $('tracking').innerHTML = '<div id="track-content" class="track-content"></div><button id="stop-follow" type="button">取消跟随</button>';
    $('stop-follow').onclick = () => { state.followed = null; paint(); };
  }
  setHTML($('track-content'), `<span class="robot-icon">${esc(shortId(robot.robot_id))}</span><div class="track-info"><span class="eyebrow">正在跟随${travelling ? ' · 行驶目的地' : ' · 当前路口'}</span><strong title="${esc(location)}">${esc(location)}</strong><p title="${esc(note)}">${esc(note)}</p><div class="track-progress" style="visibility:${travelling ? 'visible' : 'hidden'}"><i style="width:${progress}%"></i></div></div>`);
}
function paintFleet(robots) {
  robots = robots.slice().sort((a, b) => a.robot_id.localeCompare(b.robot_id));
  const pages = Math.max(1, Math.ceil(robots.length / 5));
  state.fleetPage = Math.max(0, Math.min(state.fleetPage, pages - 1));
  const visible = new Set(robots.slice(state.fleetPage * 5, state.fleetPage * 5 + 5).map(r => r.robot_id));
  $('fleet-count').textContent = `全部 ${robots.length} 台机器人`;
  $('fleet-page').textContent = robots.length ? `第 ${state.fleetPage + 1} / ${pages} 页 · 显示 ${state.fleetPage * 5 + 1}–${Math.min(robots.length, state.fleetPage * 5 + 5)} / ${robots.length} 台` : '每页 5 台 · 暂无机器人';
  $('fleet-prev').disabled = state.fleetPage === 0;
  $('fleet-next').disabled = state.fleetPage >= pages - 1;
  $('fleet-empty').hidden = robots.length > 0;
  const ids = new Set(robots.map(r => r.robot_id));
  for (const [id, row] of state.rows) if (!ids.has(id)) { row.remove(); state.rows.delete(id); }
  for (const [id, button] of state.robotButtons) if (!ids.has(id)) { button.remove(); state.robotButtons.delete(id); }
  robots.forEach(robot => {
    let picker = state.robotButtons.get(robot.robot_id);
    if (!picker) {
      picker = document.createElement('button'); picker.type = 'button';
      picker.textContent = shortId(robot.robot_id);
      picker.setAttribute('aria-label', `定位 ${robot.robot_id}`);
      picker.onclick = () => followRobot(robot.robot_id);
      state.robotButtons.set(robot.robot_id, picker); $('robot-picker').append(picker);
    }
    picker.setAttribute('aria-pressed', String(state.followed === robot.robot_id));
    picker.title = `${shortId(robot.robot_id)} · ${robotAction(robot)} · ${nameOf(robot.intersection_id || robot.destination_id)}`;
    let row = state.rows.get(robot.robot_id);
    if (!row) {
      row = document.createElement('tr');
      for (let i = 0; i < 5; i++) row.append(document.createElement('td'));
      const button = document.createElement('button'); button.className = 'robot-button';
      button.textContent = shortId(robot.robot_id); button.setAttribute('aria-label', `跟随 ${robot.robot_id}`);
      button.onclick = () => followRobot(robot.robot_id); row.children[0].append(button);
      state.rows.set(robot.robot_id, row); $('fleet').append(row);
    }
    row.classList.toggle('selected', state.followed === robot.robot_id);
    row.hidden = !visible.has(robot.robot_id);
    row.children[0].firstChild.setAttribute('aria-pressed', String(state.followed === robot.robot_id));
    const obs = observationOf(robot);
    const stale = robotStale(robot);
    const travelling = robot.action === 'TRAVEL';
    setHTML(row.children[1], travelling
      ? `<span>${esc(nameOf(robot.origin_id))} → ${esc(nameOf(robot.destination_id))}</span><small>行驶途中 · 未报告 GPS</small>`
      : `<span>${esc(nameOf(robot.intersection_id))}</span><small>${esc(robot.intersection_id)}${stale ? ' · 最后上报位置' : ''}</small>`);
    row.children[1].title = row.children[1].textContent;
    setHTML(row.children[2], `<span class="badge ${stale ? 'amber' : travelling ? 'blue' : robot.action === 'CROSS' ? 'green' : ''}">${robotAction(robot)}</span>`);
    setHTML(row.children[3], obs ? `<span class="${obs.color === 'RED' ? 'red-text' : obs.color === 'GREEN' ? 'green-text' : 'muted'}">${colorName(obs.color)}</span><small>${observationStale(obs) ? '已过期 · ' : ''}${age(obs.timestamp).toFixed(1)} 仿真秒前</small>` : '<span class="muted">—</span><small>暂无观测</small>');
    setHTML(row.children[4], obs ? `<strong>${percent(obs.confidence)}</strong> / ${percent(state.config.obs_confidence_threshold)}<small class="${obs.accepted ? '' : 'red-text'}">${obs.accepted ? '该帧已接收' : '低于门槛 · 已丢弃'}</small>` : `<span class="muted">— / ${percent(state.config.obs_confidence_threshold)}</span>`);
  });
  const order = robots.map(r => r.robot_id).join('|');
  if (state.fleetOrder !== order) {
    robots.forEach(robot => {
      $('fleet').append(state.rows.get(robot.robot_id));
      $('robot-picker').append(state.robotButtons.get(robot.robot_id));
    });
    state.fleetOrder = order;
  }
}
function paintMap(robots) {
  const at = new Map(), inbound = new Map();
  robots.forEach(robot => {
    const id = robot.intersection_id || robot.destination_id;
    const map = robot.intersection_id ? at : inbound;
    if (!map.has(id)) map.set(id, []);
    map.get(id).push(robot);
  });
  const matches = state.metas.filter(meta => (state.mapScope === 'all' || meta.in_zone) &&
    (!state.query || `${meta.name} ${meta.id} ${meta.district}`.toLowerCase().includes(state.query)));
  const pages = Math.max(1, Math.ceil(matches.length / 48));
  state.mapPage = Math.max(0, Math.min(state.mapPage, pages - 1));
  const visible = new Set(matches.slice(state.mapPage * 48, (state.mapPage + 1) * 48).map(m => m.id));
  $('map-page').textContent = `${matches.length} 个路口 · 第 ${state.mapPage + 1} / ${pages} 页`;
  $('map-prev').disabled = state.mapPage === 0;
  $('map-next').disabled = state.mapPage >= pages - 1;
  state.metas.forEach((meta, index) => {
    const code = state.snapshot.tiles[index]?.[0] || 0;
    const hit = !state.query || `${meta.name} ${meta.id} ${meta.district}`.toLowerCase().includes(state.query);
    const tile = state.tiles.get(meta.id);
    tile.dataset.s = code; tile.hidden = !hit; tile.classList.toggle('selected', meta.id === state.selected);
    const summary = summaryOf(meta.id);
    const totals = summary.visits;
    tile.title = `${meta.name} · ${label[code]} · 模型 ${percent(summary.confidence)} ${modelLabel(summary.status)} · 已记录 ${totals.robots} 台 / ${totals.visits} 次到访 · 首次模型 ${firstModelLabel(summary)}`;
    tile.setAttribute('aria-label', tile.title);
    tile.setAttribute('aria-pressed', String(meta.id === state.selected));
    const card = state.cards.get(meta.id);
    if (!card) return;
    card.hidden = !visible.has(meta.id);
    if (card.hidden) return;
    const present = at.get(meta.id) || [], approaching = inbound.get(meta.id) || [];
    card.classList.toggle('selected', meta.id === state.selected);
    card.classList.toggle('occupied', present.some(r => !robotStale(r)));
    card.setAttribute('aria-pressed', String(meta.id === state.selected));
    const tags = present.map(r => `<span class="badge ${robotStale(r) ? 'amber' : 'blue'}">${esc(shortId(r.robot_id))}${robotStale(r) ? ' 待更新' : ' 在此'}</span>`)
      .concat(approaching.map(r => `<span class="badge ${robotStale(r) ? 'amber' : ''}">→ ${esc(shortId(r.robot_id))} ${robotStale(r) ? '待确认' : '前往'}</span>`));
    const current = present.filter(r => !robotStale(r));
    const crossing = current.filter(r => r.action === 'CROSS').length;
    const scouting = current.filter(r => r.action === 'WAIT' && r.mode === 'scout').length;
    const waiting = current.filter(r => r.action === 'WAIT' && r.mode !== 'scout').length;
    const progress = progressOf(summary);
    let modelText = `<span class="card-model"><span>模型置信度 <strong>${percent(summary.confidence)}</strong></span><span class="badge ${summary.status === 'reliable' ? 'green' : ''}">${modelLabel(summary.status)}</span></span>` +
      `<span class="card-confidence" aria-hidden="true"><i style="width:${(summary.confidence || 0) * 100}%"></i></span>` +
      `<span class="card-evidence">样本 ${summary.samples} / ${progress.required_samples ?? '—'} 次跳变 · ${summary.period === 'night' ? '夜间' : '日间'}</span>` +
      `<span class="card-progress">${estimateLabel(progress)}</span>` +
      `<span class="card-evidence">${progress.estimate_basis === 'unavailable' ? '重启后端后刷新页面' : progress.pending_updates ? `${progress.pending_updates} 批已上传，等待学习完成` : progress.remaining_samples ? `还缺 ${progress.remaining_samples} 次跳变 · 到访次数为估算` : progress.confidence_met ? '样本与置信度均达标' : '继续采样，次数取决于观测质量'}</span>` +
      `<span class="card-evidence">首次模型 ${firstModelLabel(summary)}</span>` +
      `<span class="card-evidence">首次可预测 ${summary.history?.first_reliable ? simTime(summary.history.first_reliable.sim_time) : summary.status === 'reliable' ? '时间未记录' : '尚未达到'}</span>` +
      `<span class="card-visits">已记录 <b>${totals.robots}</b> 台机器人 / <b>${totals.visits}</b> 次到访</span>` +
      `<span class="card-evidence">侦察 ${totals.scout} 次 · 普通 ${totals.normal} 次 · 超时 ${totals.timeout} 次</span>` +
      `<span class="card-current">当前：侦察 ${scouting} · 等待 ${waiting} · 通行 ${crossing}</span>`;
    if (!state.snapshot.learning) modelText = '<span class="card-progress">学习详情待更新</span><span class="card-evidence">重启后端后可查看进度与到访记录</span>';
    setHTML(card, `<span class="crossing-top"><span><i class="dot ${colors[code]}"></i>${label[code]}</span>${meta.is_hub ? '<span>枢纽</span>' : ''}</span><span class="name" title="${esc(meta.name)}">${esc(meta.name)}</span>${modelText}<span class="occupants">${tags.join('') || '<span class="vacant">暂无机器人驻留</span>'}</span>`);
    card.setAttribute('aria-label', `${tile.title}，${present.map(r => shortId(r.robot_id)).join('、') || '无机器人驻留'}`);
  });
  $('no-results').hidden = matches.length > 0;
}
function meter(value, threshold) {
  return `<div class="meter" aria-hidden="true"><i style="width:${Math.max(0, Math.min(100, (value || 0) * 100))}%"></i>${Number.isFinite(threshold) ? `<b style="left:${Math.max(0, Math.min(100, threshold * 100))}%"></b>` : ''}</div>`;
}
function signalComparison(detail, stale, drift) {
  const live = detail.live;
  const observed = !stale && live.state === 'observed' && live.age != null &&
    live.age <= state.config.live_stale_seconds;
  const observedColor = observed ? live.color : null;
  const observationTitle = stale ? '观测待更新' : live.state === 'disputed' ? '共识不足'
    : observedColor ? colorName(observedColor) : '暂无观测';
  const observationNote = stale ? '快照已过期' : live.state === 'disputed' ? '机器人观测存在分歧'
    : observed ? `${live.voters} 台有效投票 · 快照内 ${Math.max(0, live.age).toFixed(1)} 秒前`
    : '等待机器人有效观测';
  // Older backends only expose a fused decision; do not label it a model-only prediction.
  const prediction = detail.model_prediction;
  const predictedColor = !stale ? prediction?.color : null;
  const remaining = predictedColor === 'GREEN' ? prediction.green_remaining : prediction?.seconds_to_green;
  const expired = predictedColor && remaining != null && remaining - drift <= 0;
  const predictionTitle = stale ? '预测待更新' : expired ? '相位待更新'
    : predictedColor ? colorName(predictedColor) : detail.model?.reliable ? '预测待更新' : '暂无可靠模型';
  const predictionNote = stale ? '快照已过期' : expired ? '等待最新模型相位'
    : predictedColor && remaining != null ? `${predictedColor === 'GREEN' ? '绿灯剩余' : '距绿灯'}约 ${Math.max(0, remaining - drift).toFixed(0)} 秒`
    : detail.model?.reliable ? '重启后端后刷新页面' : '等待学习达标';
  const panel = (kind, title, color, status, note, extra) => `<section class="signal-column ${kind}" aria-label="${title}"><h4>${title}</h4><div class="signal-readout"><div class="signal-lamp ${color === 'RED' ? 'red' : color === 'GREEN' ? 'green' : ''}" aria-hidden="true"><i></i><i></i></div><strong>${status}</strong></div><p>${note}</p><small>${extra}</small></section>`;
  const mismatch = observedColor && predictedColor && !expired && observedColor !== predictedColor;
  return `<div class="signal-comparison">${panel('observed-signal', '实时观测', observedColor, observationTitle, observationNote, '来自机器人视觉')}${panel('predicted-signal', '模型预测', expired ? null : predictedColor, predictionTitle, predictionNote, `模型置信度 ${percent(detail.model?.confidence)}`)}</div>${mismatch ? '<p class="signal-mismatch">观测与预测不一致，请以实时观测确认灯色。</p>' : ''}`;
}
function paintDetail() {
  const detail = state.detail;
  if (!detail || detail.intersection_id !== state.selected) return;
  const config = state.config;
  const stale = state.detailError || !state.connected || now() - detail.sim_now > Math.max(3 * state.snapshot.speed, config.live_stale_seconds);
  $('detail-status').textContent = state.detailError ? '详情更新失败' : stale ? '快照可能过期' : state.followed ? '跟随中' : '已选路口';
  const drift = stale ? 0 : Math.max(0, now() - detail.sim_now);
  const observers = activeRobots().filter(r => r.intersection_id === detail.intersection_id && r.observation?.intersection_id === detail.intersection_id);
  const live = detail.live;
  const consensus = live.voters ? `${percent(live.agreement)} ${live.agreement + 1e-9 >= config.consensus_threshold ? '≥' : '<'} ${percent(config.consensus_threshold)}` : `— / ${percent(config.consensus_threshold)}`;
  const consensusNote = !live.voters ? '没有有效投票，暂不判断共识。' : `${live.voters} 台机器人有效投票 · ${live.agreement + 1e-9 >= config.consensus_threshold ? '达到共识门槛' : '低于门槛，实时灯色不采纳'}`;
  const model = detail.model;
  const learning = detail.learning || summaryOf(detail.intersection_id, model);
  setHTML($('detail'), `<h3 class="detail-title">${esc(detail.meta.name)}</h3><p class="detail-sub">${esc(detail.intersection_id)} · ${esc(detail.meta.district)}${detail.meta.in_zone ? ' · 配送区' : ''}</p>
    ${signalComparison(detail, stale, drift)}
    ${learningPanel(learning)}
    ${detail.learning ? visitsPanel(learning.visits, detail.recent_visits || []) : ''}
    <section class="detail-section"><h3>这盏灯的观测 threshold</h3><div class="gate-line"><span>仿真检测频率</span><strong>${Number.isFinite(config.obs_rate_hz) ? config.obs_rate_hz + ' FPS' : '—'}</strong></div><p class="hint">按仿真秒采样，倍速不改变采样间隔。</p><div class="gate-line"><span>单帧置信度门槛</span><strong>${percent(config.obs_confidence_threshold)}</strong></div><p class="hint">下方显示各机器人最近一帧；黑色刻度为接收门槛。</p>
    ${observers.length ? observers.map(robot => {
      const obs = robot.observation;
      return `<div class="observation-card"><div class="gate-line"><strong>${esc(shortId(robot.robot_id))} · ${colorName(obs.color)}</strong><span class="badge ${obs.accepted ? 'green' : 'red'}">${obs.accepted ? '该帧已接收' : '该帧已丢弃'}</span></div><div class="gate-line"><span>confidence / threshold</span><strong>${percent(obs.confidence)} / ${percent(config.obs_confidence_threshold)}</strong></div>${meter(obs.confidence, config.obs_confidence_threshold)}<p class="hint">${observationStale(obs) ? '观测已过期 · ' : ''}${age(obs.timestamp).toFixed(1)} 仿真秒前${obs.accepted ? '' : ' · 低于单帧门槛'}</p></div>`;
    }).join('') : '<p class="empty">暂无驻留机器人上传的观测，当前置信度为 —。</p>'}
    </section><section class="detail-section"><h3>车队共识</h3><div class="gate-line"><span>加权一致率 / 门槛</span><strong>${consensus}</strong></div>${meter(live.voters ? live.agreement : 0, config.consensus_threshold)}<p class="detail-note">${consensusNote}</p><p class="detail-note">单台机器人投票时一致率为 100%，不等于视觉置信度。共识仅使用最近 ${config.vote_window_seconds} 仿真秒内通过门槛的投票。</p></section>
    <section class="detail-section"><h3>周期学习 <span class="badge ${model?.reliable ? 'green' : ''}">${model?.reliable ? '可预测' : '学习中'}</span></h3>${model ? `<table class="kv"><thead><tr><th>时长</th><th>学习值</th><th>仿真真值</th></tr></thead><tbody>${[['总周期','T_cycle'],['红灯','T_red'],['绿灯','T_green']].map(([name,key]) => `<tr><td>${name}</td><td>${model[key].toFixed(1)}s</td><td>${detail.truth[key] == null ? '—' : detail.truth[key].toFixed(1) + 's'}</td></tr>`).join('')}</tbody></table><p class="detail-note">${model.samples} 次跳变样本 · 真值仅用于仿真对照。</p>` : '<p class="empty">尚无周期模型。机器人将通过侦察采样学习信号周期。</p>'}<p class="detail-note">此路口累计接收 ${live.accepted.toLocaleString()} 帧，丢弃 ${live.rejected.toLocaleString()} 帧。</p></section>`);
}
$('search').addEventListener('input', event => { state.query = event.target.value.trim().toLowerCase(); state.mapPage = 0; if (state.snapshot) paintMap(activeRobots()); });
$('map-scope').addEventListener('change', event => { state.mapScope = event.target.value; state.mapPage = 0; if (state.snapshot) paintMap(activeRobots()); });
$('map-prev').onclick = () => { state.mapPage--; paintMap(activeRobots()); };
$('map-next').onclick = () => { state.mapPage++; paintMap(activeRobots()); };
$('fleet-prev').onclick = () => { state.fleetPage--; paintFleet(activeRobots()); };
$('fleet-next').onclick = () => { state.fleetPage++; paintFleet(activeRobots()); };
boot();

function learningPanel(summary) {
  const config = state.config;
  if (progressOf(summary).estimate_basis === 'unavailable') return `<section class="detail-section learning-panel"><h3>模型学习进度</h3><div class="model-score"><strong>${percent(summary.confidence)}</strong><span>${modelLabel(summary.status)}</span></div><p class="detail-note">后端尚未提供学习进度数据，请重启后端后刷新页面。</p></section>`;
  const history = summary.history || {};
  const milestone = (title, event, fallback) => `<li><span>${title}</span><strong>${event ? simTime(event.sim_time) + ' 仿真' : fallback}</strong>${event ? `<small>${esc(new Date(event.wall_time * 1000).toLocaleString('zh-CN', {timeZone:'Asia/Seoul', hour12:false}))} KST 记录 · ${event.samples} 次跳变 · ${percent(event.confidence)}<br>触发更新：${event.robots.map(shortId).map(esc).join('、') || '—'}</small>` : ''}</li>`;
  const progress = progressOf(summary);
  const basis = {ready:'当前时段模型已可用于预测，后续观测仍会更新模型。',
    confidence:'样本数量已达标，但置信度仍不足。需要更稳定的周期观测，无法承诺固定到访次数。',
    no_transitions:`最近 ${progress.recent_scout_visits} 次相关侦察没有采到本时段跳变，需要继续驻留观察灯色切换。`,
    history:`按最近 ${progress.recent_scout_visits} 次相关侦察平均每次 ${progress.mean_transitions_per_scout} 个本时段跳变估算。`,
    planning:'暂无侦察产出历史，按仿真中一次完整侦察约采到 3–4 次跳变估算。'}[progress.estimate_basis];
  return `<section class="detail-section learning-panel"><div class="gate-line"><h3>模型学习进度</h3><span class="badge ${summary.status === 'reliable' ? 'green' : ''}">${modelLabel(summary.status)}</span></div><div class="model-score"><strong>${percent(summary.confidence)}</strong><span>模型置信度 · ${summary.period === 'night' ? '夜间' : '日间'}</span></div>${meter(summary.confidence, config.model_min_confidence)}
    <div class="learning-check"><span>① 跳变样本</span><strong>${summary.samples} / ${progress.required_samples}</strong><span class="badge ${progress.remaining_samples ? 'amber' : 'green'}">${progress.remaining_samples ? `还缺 ${progress.remaining_samples} 次` : '已达标'}</span></div>
    <div class="sample-progress" role="progressbar" aria-label="跳变样本收集进度" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Math.round(progress.sample_progress * 100)}"><i style="width:${progress.sample_progress * 100}%"></i></div>
    <div class="learning-check"><span>② 模型置信度</span><strong>≥ ${percent(progress.required_confidence)}</strong><span class="badge ${progress.confidence_met ? 'green' : 'amber'}">${progress.confidence_met ? '已达标' : '未达标'}</span></div>
    <div class="learning-estimate"><strong>${estimateLabel(progress)}</strong><p>${basis}</p>${summary.status !== 'reliable' ? '<p>估算用于补齐样本，不保证模型届时可靠；普通路过可能没有跳变，日间与夜间分别学习。</p>' : ''}</div>
    <p class="learning-pending">${progress.pending_updates ? `${progress.pending_updates} 批已上传，等待学习完成` : '机器人结束到访后才上传跳变；正在采集的数据尚未计入。'}</p>
    <p class="detail-note">两项同时达标才能预测。进度条仅表示样本数量；模型置信度是学习器评分，不是视觉识别准确率。</p><ol class="milestones">${milestone('首次生成模型', history.first_model, summary.confidence == null ? '尚未生成' : '旧模型，首次时间未记录')}${milestone('首次达到可预测', history.first_reliable, summary.status === 'reliable' ? '旧模型，首次时间未记录' : '尚未达到')}${milestone('最近一次更新', history.last_update, '暂无更新记录')}</ol></section>`;
}
function visitsPanel(totals, records) {
  return `<section class="detail-section"><h3>机器人到访记录</h3><p class="visit-summary">已记录 <strong>${totals.robots}</strong> 台不同机器人，<strong>${totals.visits}</strong> 次已结束到访</p><p class="detail-note">侦察 ${totals.scout} 次 · 普通 ${totals.normal} 次<br>已通行 ${totals.cross} 次 · 超时离开 ${totals.timeout} 次</p><div class="visit-history">${records.map(record => `<div class="visit-entry"><div class="gate-line"><strong>${esc(shortId(record.robot_id))} · ${record.mode === 'scout' ? '侦察' : record.mode === 'normal' ? '普通' : esc(record.mode)}</strong><span class="badge ${record.action === 'CROSS' ? 'green' : 'amber'}">${record.action === 'CROSS' ? '已通行' : '超时离开'}</span></div><p>${simTime(record.arrival_time)} → ${simTime(record.depart_time)} 仿真</p><p>等待 ${Number(record.waited).toFixed(1)} 秒 · 上报 ${record.transitions} 次跳变</p></div>`).join('') || '<p class="empty">暂无已结束到访；在场机器人见路口卡片。</p>'}</div><p class="detail-note">最近 ${records.length} 条，按实际接收时间排序。统计从启用本功能后开始，历史缺失记录不会补造；跳变上报不代表已完成学习。</p></section>`;
}
