'use strict';
const $ = (id) => document.getElementById(id);
const state = {
  metas: [],
  byId: new Map(),
  cards: new Map(),
  tiles: new Map(),
  rows: new Map(),
  config: null,
  snapshot: null,
  selected: null,
  followed: null,
  detail: null,
  detailBusy: false,
  query: '',
  receivedAt: 0,
  connected: false,
  started: false,
  detailError: false,
  speedPending: false,
  speedError: false,
  mapScope: 'zone',
  mapPage: 0,
  fleetPage: 0,
  robotButtons: new Map(),
};
const esc = (value) =>
  String(value ?? '').replace(
    /[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c],
  );
const percent = (value) =>
  !Number.isFinite(value) ? '—' : `${(value * 100).toFixed(1).replace(/\.0$/, '')}%`;
const shortId = (id) => (id || '').replace('robot_', 'R');
const nameOf = (id) => state.byId.get(id)?.name || id || '출발지 미보고';
const label = [
  '데이터 없음',
  '실시간 빨간불',
  '실시간 초록불',
  '예측 빨간불',
  '예측 초록불',
  '합의 부족',
];
const colors = ['unknown', 'red', 'green', 'predicted', 'predicted', 'amber'];
const colorName = (color) =>
  ({ RED: '빨간불', GREEN: '초록불', UNKNOWN: '알 수 없음' })[color] || '알 수 없음';
const setHTML = (node, html) => {
  if (node.innerHTML !== html) node.innerHTML = html;
};
const now = () =>
  state.snapshot
    ? state.snapshot.sim_now +
      (state.connected ? ((performance.now() - state.receivedAt) / 1000) * state.snapshot.speed : 0)
    : 0;
// Freshness belongs to the server snapshot that contains the observation.
// Extrapolating wall time at high simulation speeds would invent a dropout between UI refreshes.
const age = (timestamp) => Math.max(0, (state.snapshot?.sim_now || 0) - timestamp);
const robotStale = (robot) =>
  robot.action === 'TRAVEL'
    ? robot.travel_arrives_at == null ||
      state.snapshot.sim_now > robot.travel_arrives_at + state.config.live_stale_seconds
    : age(robot.updated_at) > state.config.live_stale_seconds;
const robotAction = (robot) =>
  robotStale(robot)
    ? '위치 업데이트 대기'
    : robot.action === 'TRAVEL'
      ? '이동 중'
      : robot.action === 'CROSS'
        ? '교차로 통과'
        : robot.action === 'TIMEOUT'
          ? '시간 초과로 떠남'
          : robot.mode === 'scout'
            ? '정찰 샘플링'
            : '통행 대기';
const observationStale = (observation) =>
  !state.connected || age(observation.timestamp) > state.config.live_stale_seconds;
const activeRobots = () => state.snapshot?.robots || [];
const observationOf = (robot) => robot.observation;
const simTime = (timestamp) =>
  timestamp == null ? '—' : new Date((timestamp + 9 * 3600) * 1000).toISOString().slice(11, 19);
const modelLabel = (status) =>
  ({
    unlearned: '모델 없음',
    learning: '학습 중',
    reliable: '예측 가능',
    unavailable: '데이터 업데이트 대기',
  })[status] || '모델 없음';
const summaryOf = (id, model = null) =>
  state.snapshot?.learning?.[id] || {
    status: model
      ? model.reliable
        ? 'reliable'
        : 'learning'
      : state.snapshot?.learning
        ? 'unlearned'
        : 'unavailable',
    period: state.snapshot?.period,
    confidence: model?.confidence ?? null,
    samples: model?.samples ?? 0,
    history: {},
    visits: { robots: 0, visits: 0, scout: 0, normal: 0, cross: 0, timeout: 0 },
  };
const progressOf = (summary) =>
  summary.progress ||
  (!Number.isFinite(state.config.model_min_samples) ||
  !Number.isFinite(state.config.model_min_confidence)
    ? {
        required_samples: null,
        remaining_samples: null,
        sample_progress: 0,
        required_confidence: null,
        confidence_met: false,
        estimate_basis: 'unavailable',
        estimated_scout_visits: null,
        pending_updates: 0,
      }
    : {
        samples: summary.samples,
        required_samples: state.config.model_min_samples,
        remaining_samples: Math.max(0, state.config.model_min_samples - summary.samples),
        sample_progress: Math.min(1, summary.samples / state.config.model_min_samples),
        confidence_met: (summary.confidence || 0) >= state.config.model_min_confidence,
        required_confidence: state.config.model_min_confidence,
        estimated_scout_visits: [
          Math.ceil(state.config.model_min_samples / 4),
          Math.ceil(state.config.model_min_samples / 3),
        ],
        estimate_basis: summary.status === 'reliable' ? 'ready' : 'planning',
        pending_updates: 0,
      });
const estimateLabel = (progress) => {
  if (progress.estimate_basis === 'ready') return '예측 조건 충족';
  if (progress.estimate_basis === 'confidence') return '샘플은 충분, confidence를 더 높여야 함';
  if (progress.estimate_basis === 'no_transitions') return '최근 전환을 수집하지 못해 추정 불가';
  if (!progress.estimated_scout_visits) return '학습 진행 상황 업데이트 대기';
  const [low, high] = progress.estimated_scout_visits;
  return `정찰 약 ${low === high ? low : low + '–' + high} 회 더 진행하면 샘플 보충 예상`;
};
const firstModelLabel = (summary) =>
  summary.history?.first_model
    ? simTime(summary.history.first_model.sim_time)
    : summary.confidence == null
      ? '아직 생성되지 않음'
      : '이전 모델 · 시각 미기록';

async function getJSON(url) {
  const response = await fetch(url, { signal: AbortSignal.timeout(6000) });
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  return response.json();
}
async function boot() {
  try {
    const [metas, config] = await Promise.all([
      getJSON('/v1/intersections'),
      getJSON('/v1/config'),
    ]);
    state.config = config;
    state.metas = metas.intersections;
    state.metas.forEach((meta, index) => state.byId.set(meta.id, { ...meta, index }));
    $('obs-threshold').textContent = percent(config.obs_confidence_threshold);
    $('consensus-threshold').textContent = percent(config.consensus_threshold);
    $('all-summary').textContent =
      `전체 교차로 ${state.metas.length.toLocaleString()}개 · 도시 전체 개요 펼치기`;
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
    $('error').textContent =
      '서버 설정을 읽을 수 없습니다. 3초 후 다시 시도합니다. signal server가 실행 중인지 확인하세요.';
    setConnection(false);
    setTimeout(boot, 3000);
  }
}
function buildMap() {
  $('zone-map').replaceChildren();
  $('field').replaceChildren();
  state.cards.clear();
  state.tiles.clear();
  state.metas.forEach((meta) => {
    const tile = document.createElement('button');
    tile.className = 'tile';
    tile.title = `${meta.name} · ${meta.id}`;
    tile.setAttribute('aria-label', tile.title);
    tile.addEventListener('click', () => selectCrossing(meta.id, true));
    state.tiles.set(meta.id, tile);
    $('field').append(tile);
    {
      const card = document.createElement('button');
      card.className = 'crossing';
      card.hidden = true;
      card.addEventListener('click', () => selectCrossing(meta.id, true));
      state.cards.set(meta.id, card);
      $('zone-map').append(card);
    }
  });
}
let reconnectTimer;
function connect() {
  let socket;
  try {
    socket = new WebSocket(
      `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`,
    );
  } catch (error) {
    fallback();
    return;
  }
  socket.onmessage = (event) => {
    try {
      applySnapshot(JSON.parse(event.data));
      setConnection(true);
    } catch (error) {
      setConnection(false);
    }
  };
  socket.onerror = () => setConnection(false);
  socket.onclose = () => {
    setConnection(false);
    fallback();
  };
}
async function fallback() {
  try {
    applySnapshot(await getJSON('/v1/snapshot'));
    setConnection(true, '스냅샷 폴링');
  } catch (error) {
    setConnection(false);
  }
  clearTimeout(reconnectTimer);
  reconnectTimer = setTimeout(connect, 2000);
}
function setConnection(up, text = '실시간 연결') {
  state.connected = up;
  paintSpeed();
  $('connection').textContent = up ? text : '연결 끊김 · 재시도 중';
  $('connection').className = `connection ${up ? 'on' : 'off'}`;
  if (state.snapshot) {
    $('error').hidden = up;
    $('error').textContent =
      '연결 끊김: 아래는 마지막으로 받은 스냅샷으로, 현재 위치와 관측이 오래되었을 수 있습니다.';
  }
}
function applySnapshot(snapshot) {
  state.snapshot = snapshot;
  state.receivedAt = performance.now();
  if (!state.speedPending && !state.speedError)
    $('speed-status').textContent = `현재 ${snapshot.speed}× · 전체 적용`;
  paintSpeed();
  if (!state.selected) {
    const robot = snapshot.robots[0];
    if (robot) followRobot(robot.robot_id);
    else selectCrossing(state.metas.find((m) => m.in_zone)?.id || state.metas[0]?.id);
  }
  if (state.followed) {
    const robot = snapshot.robots.find((r) => r.robot_id === state.followed);
    const target = robot?.intersection_id || robot?.destination_id;
    if (target && target !== state.selected) selectCrossing(target);
  }
  paint();
}

function paintSpeed() {
  document.querySelectorAll('#speed-options button').forEach((button) => {
    button.disabled = !state.connected || !state.snapshot || state.speedPending;
    button.setAttribute(
      'aria-pressed',
      String(Number(button.dataset.speed) === state.snapshot?.speed),
    );
  });
}

async function changeSpeed(speed) {
  if (state.speedPending || !state.connected || !state.snapshot || state.snapshot.speed === speed)
    return;
  state.speedPending = true;
  state.speedError = false;
  $('speed-status').textContent = `${speed}×로 전환 중…`;
  paintSpeed();
  try {
    const response = await fetch('/v1/clock/speed', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ speed }),
      signal: AbortSignal.timeout(6000),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    // Refresh the authoritative snapshot, including all time-based readouts.
    applySnapshot(await getJSON('/v1/snapshot'));
    $('speed-status').textContent = `${state.snapshot.speed}×로 전환됨 · 전체 적용`;
    refreshDetail();
  } catch (error) {
    state.speedError = true;
    $('speed-status').textContent = '전환 미확인, 연결 확인 후 다시 시도하세요';
  } finally {
    state.speedPending = false;
    paintSpeed();
  }
}

document.querySelectorAll('#speed-options button').forEach((button) => {
  button.addEventListener('click', () => changeSpeed(Number(button.dataset.speed)));
});
function selectCrossing(id, manual = false) {
  if (!id) return;
  if (manual) state.followed = null;
  if (state.selected !== id) {
    state.selected = id;
    state.detail = null;
    state.detailError = false;
    $('detail').innerHTML = '<p class="empty">교차로 관측을 불러오는 중…</p>';
  }
  refreshDetail();
  paint();
  if (manual && window.innerWidth <= 850)
    document.querySelector('.detail-panel').scrollIntoView({ behavior: 'smooth', block: 'start' });
}
function followRobot(id) {
  state.followed = id;
  const index = activeRobots()
    .slice()
    .sort((a, b) => a.robot_id.localeCompare(b.robot_id))
    .findIndex((r) => r.robot_id === id);
  if (index >= 0) state.fleetPage = Math.floor(index / 5);
  const robot = activeRobots().find((r) => r.robot_id === id);
  if (robot) selectCrossing(robot.intersection_id || robot.destination_id);
  paint();
}
async function refreshDetail() {
  if (!state.selected || state.detailBusy) return;
  const id = state.selected;
  state.detailBusy = true;
  try {
    const detail = await getJSON(`/v1/intersections/${encodeURIComponent(id)}`);
    if (state.selected === id) {
      state.detail = detail;
      state.detailError = false;
      paintDetail();
    }
  } catch (error) {
    if (state.selected === id) {
      state.detailError = true;
      $('detail-status').textContent = '상세 업데이트 실패';
      if (!state.detail)
        $('detail').innerHTML =
          '<p class="empty">교차로 상세를 일시적으로 읽을 수 없습니다. 재시도 중입니다.</p>';
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
  $('clock').textContent = [
    Math.floor(seconds / 3600),
    Math.floor((seconds % 3600) / 60),
    seconds % 60,
  ]
    .map((n) => String(n).padStart(2, '0'))
    .join(':');
  $('clock-meta').textContent =
    `${snapshot.period === 'day' ? '주간' : '야간'} / KST · ${snapshot.speed}× 시뮬레이션`;
  const robots = activeRobots();
  const travel = robots.filter((r) => r.action === 'TRAVEL' && !robotStale(r)).length;
  const stale = robots.filter(robotStale).length;
  $('robot-count').textContent = robots.length;
  $('robot-summary').textContent =
    `${robots.length - travel - stale} 대 교차로에 있음 · ${travel} 대 이동 중 · ${stale} 대 업데이트 대기`;
  const counts = snapshot.counts;
  $('observed-count').textContent = counts.observed_red + counts.observed_green;
  $('coverage-summary').textContent =
    `${counts.predicted_red + counts.predicted_green} 곳 모델 예측 · ${counts.disputed} 곳 합의 부족`;
  paintTracking(robots);
  paintFleet(robots);
  paintMap(robots);
  paintDetail();
  const stats = snapshot.stats;
  const metrics = [
    ['데이터베이스', state.config.database_backend === 'postgresql' ? 'PostgreSQL' : 'SQLite'],
    ['관측 업로드', state.config.observation_transport === 'mqtt' ? 'MQTT' : 'HTTP'],
    ['업로드 프레임 수', stats.observations.toLocaleString()],
    ['임계값 미달로 폐기', stats.rejected.toLocaleString()],
    ['신뢰 가능 / 학습된 모델', `${stats.models_reliable} / ${stats.models}`],
    ['수집 속도', `${stats.obs_per_sec}/s`],
    ['학습 큐', stats.queue_depth],
  ];
  setHTML(
    $('ledger'),
    metrics.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join(''),
  );
}
function paintTracking(robots) {
  const robot = robots.find((r) => r.robot_id === state.followed);
  if (!robot) {
    setHTML(
      $('tracking'),
      `<p class="empty">${robots.length ? '아래 로봇 번호를 클릭해 현재 위치와 관측을 따라가세요. 지금은 교차로를 자유롭게 볼 수 있습니다.' : '아직 로봇 보고가 없습니다. fleet을 시작하면 현재 위치와 이동 목적지가 표시됩니다.'}</p>`,
    );
    return;
  }
  const travelling = robot.action === 'TRAVEL';
  const eta =
    travelling && robot.travel_arrives_at != null
      ? Math.max(0, robot.travel_arrives_at - now())
      : null;
  const duration = robot.travel_arrives_at - robot.travel_started_at;
  const progress =
    duration > 0
      ? Math.max(0, Math.min(100, ((now() - robot.travel_started_at) / duration) * 100))
      : 0;
  const location = travelling
    ? `${nameOf(robot.origin_id)} → ${nameOf(robot.destination_id)}`
    : nameOf(robot.intersection_id);
  const note = robotStale(robot)
    ? '보고가 오래되어 실제 위치 확인 필요'
    : travelling
      ? `${eta > 0 ? `약 ${Math.ceil(eta)} 시뮬레이션 초 후 도착 예정` : '도착 보고 대기 중'} · 경로 진행률은 추정치이며 GPS 위치 아님`
      : `${robotAction(robot)} · ${age(robot.updated_at).toFixed(1)} 시뮬레이션 초 전 보고`;
  if (!$('track-content')) {
    $('tracking').innerHTML =
      '<div id="track-content" class="track-content"></div><button id="stop-follow" type="button">따라가기 취소</button>';
    $('stop-follow').onclick = () => {
      state.followed = null;
      paint();
    };
  }
  setHTML(
    $('track-content'),
    `<span class="robot-icon">${esc(shortId(robot.robot_id))}</span><div class="track-info"><span class="eyebrow">따라가는 중${travelling ? ' · 이동 목적지' : ' · 현재 교차로'}</span><strong title="${esc(location)}">${esc(location)}</strong><p title="${esc(note)}">${esc(note)}</p><div class="track-progress" style="visibility:${travelling ? 'visible' : 'hidden'}"><i style="width:${progress}%"></i></div></div>`,
  );
}
function paintFleet(robots) {
  robots = robots.slice().sort((a, b) => a.robot_id.localeCompare(b.robot_id));
  const pages = Math.max(1, Math.ceil(robots.length / 5));
  state.fleetPage = Math.max(0, Math.min(state.fleetPage, pages - 1));
  const visible = new Set(
    robots.slice(state.fleetPage * 5, state.fleetPage * 5 + 5).map((r) => r.robot_id),
  );
  $('fleet-count').textContent = `전체 로봇 ${robots.length} 대`;
  $('fleet-page').textContent = robots.length
    ? `페이지 ${state.fleetPage + 1} / ${pages} · ${state.fleetPage * 5 + 1}–${Math.min(robots.length, state.fleetPage * 5 + 5)} / ${robots.length} 대 표시`
    : '페이지당 5 대 · 로봇 없음';
  $('fleet-prev').disabled = state.fleetPage === 0;
  $('fleet-next').disabled = state.fleetPage >= pages - 1;
  $('fleet-empty').hidden = robots.length > 0;
  const ids = new Set(robots.map((r) => r.robot_id));
  for (const [id, row] of state.rows)
    if (!ids.has(id)) {
      row.remove();
      state.rows.delete(id);
    }
  for (const [id, button] of state.robotButtons)
    if (!ids.has(id)) {
      button.remove();
      state.robotButtons.delete(id);
    }
  robots.forEach((robot) => {
    let picker = state.robotButtons.get(robot.robot_id);
    if (!picker) {
      picker = document.createElement('button');
      picker.type = 'button';
      picker.textContent = shortId(robot.robot_id);
      picker.setAttribute('aria-label', `${robot.robot_id} 위치 보기`);
      picker.onclick = () => followRobot(robot.robot_id);
      state.robotButtons.set(robot.robot_id, picker);
      $('robot-picker').append(picker);
    }
    picker.setAttribute('aria-pressed', String(state.followed === robot.robot_id));
    picker.title = `${shortId(robot.robot_id)} · ${robotAction(robot)} · ${nameOf(robot.intersection_id || robot.destination_id)}`;
    let row = state.rows.get(robot.robot_id);
    if (!row) {
      row = document.createElement('tr');
      for (let i = 0; i < 5; i++) row.append(document.createElement('td'));
      const button = document.createElement('button');
      button.className = 'robot-button';
      button.textContent = shortId(robot.robot_id);
      button.setAttribute('aria-label', `${robot.robot_id} 따라가기`);
      button.onclick = () => followRobot(robot.robot_id);
      row.children[0].append(button);
      state.rows.set(robot.robot_id, row);
      $('fleet').append(row);
    }
    row.classList.toggle('selected', state.followed === robot.robot_id);
    row.hidden = !visible.has(robot.robot_id);
    row.children[0].firstChild.setAttribute(
      'aria-pressed',
      String(state.followed === robot.robot_id),
    );
    const obs = observationOf(robot);
    const stale = robotStale(robot);
    const travelling = robot.action === 'TRAVEL';
    setHTML(
      row.children[1],
      travelling
        ? `<span>${esc(nameOf(robot.origin_id))} → ${esc(nameOf(robot.destination_id))}</span><small>이동 중 · GPS 미보고</small>`
        : `<span>${esc(nameOf(robot.intersection_id))}</span><small>${esc(robot.intersection_id)}${stale ? ' · 마지막 보고 위치' : ''}</small>`,
    );
    row.children[1].title = row.children[1].textContent;
    setHTML(
      row.children[2],
      `<span class="badge ${stale ? 'amber' : travelling ? 'blue' : robot.action === 'CROSS' ? 'green' : ''}">${robotAction(robot)}</span>`,
    );
    setHTML(
      row.children[3],
      obs
        ? `<span class="${obs.color === 'RED' ? 'red-text' : obs.color === 'GREEN' ? 'green-text' : 'muted'}">${colorName(obs.color)}</span><small>${observationStale(obs) ? '만료됨 · ' : ''}${age(obs.timestamp).toFixed(1)} 시뮬레이션 초 전</small>`
        : '<span class="muted">—</span><small>관측 없음</small>',
    );
    setHTML(
      row.children[4],
      obs
        ? `<strong>${percent(obs.confidence)}</strong> / ${percent(state.config.obs_confidence_threshold)}<small class="${obs.accepted ? '' : 'red-text'}">${obs.accepted ? '해당 프레임 수신됨' : '임계값 미달 · 폐기됨'}</small>`
        : `<span class="muted">— / ${percent(state.config.obs_confidence_threshold)}</span>`,
    );
  });
  const order = robots.map((r) => r.robot_id).join('|');
  if (state.fleetOrder !== order) {
    robots.forEach((robot) => {
      $('fleet').append(state.rows.get(robot.robot_id));
      $('robot-picker').append(state.robotButtons.get(robot.robot_id));
    });
    state.fleetOrder = order;
  }
}
function paintMap(robots) {
  const at = new Map(),
    inbound = new Map();
  robots.forEach((robot) => {
    const id = robot.intersection_id || robot.destination_id;
    const map = robot.intersection_id ? at : inbound;
    if (!map.has(id)) map.set(id, []);
    map.get(id).push(robot);
  });
  const matches = state.metas.filter(
    (meta) =>
      (state.mapScope === 'all' || meta.in_zone) &&
      (!state.query ||
        `${meta.name} ${meta.id} ${meta.district}`.toLowerCase().includes(state.query)),
  );
  const pages = Math.max(1, Math.ceil(matches.length / 48));
  state.mapPage = Math.max(0, Math.min(state.mapPage, pages - 1));
  const visible = new Set(
    matches.slice(state.mapPage * 48, (state.mapPage + 1) * 48).map((m) => m.id),
  );
  $('map-page').textContent = `교차로 ${matches.length}개 · 페이지 ${state.mapPage + 1} / ${pages}`;
  $('map-prev').disabled = state.mapPage === 0;
  $('map-next').disabled = state.mapPage >= pages - 1;
  state.metas.forEach((meta, index) => {
    const code = state.snapshot.tiles[index]?.[0] || 0;
    const hit =
      !state.query ||
      `${meta.name} ${meta.id} ${meta.district}`.toLowerCase().includes(state.query);
    const tile = state.tiles.get(meta.id);
    tile.dataset.s = code;
    tile.hidden = !hit;
    tile.classList.toggle('selected', meta.id === state.selected);
    const summary = summaryOf(meta.id);
    const totals = summary.visits;
    tile.title = `${meta.name} · ${label[code]} · 모델 ${percent(summary.confidence)} ${modelLabel(summary.status)} · 기록된 로봇 ${totals.robots} 대 / 방문 ${totals.visits} 회 · 첫 모델 ${firstModelLabel(summary)}`;
    tile.setAttribute('aria-label', tile.title);
    tile.setAttribute('aria-pressed', String(meta.id === state.selected));
    const card = state.cards.get(meta.id);
    if (!card) return;
    card.hidden = !visible.has(meta.id);
    if (card.hidden) return;
    const present = at.get(meta.id) || [],
      approaching = inbound.get(meta.id) || [];
    card.classList.toggle('selected', meta.id === state.selected);
    card.classList.toggle(
      'occupied',
      present.some((r) => !robotStale(r)),
    );
    card.setAttribute('aria-pressed', String(meta.id === state.selected));
    const tags = present
      .map(
        (r) =>
          `<span class="badge ${robotStale(r) ? 'amber' : 'blue'}">${esc(shortId(r.robot_id))}${robotStale(r) ? ' 업데이트 대기' : ' 여기 있음'}</span>`,
      )
      .concat(
        approaching.map(
          (r) =>
            `<span class="badge ${robotStale(r) ? 'amber' : ''}">→ ${esc(shortId(r.robot_id))} ${robotStale(r) ? '확인 대기' : '이동 중'}</span>`,
        ),
      );
    const current = present.filter((r) => !robotStale(r));
    const crossing = current.filter((r) => r.action === 'CROSS').length;
    const scouting = current.filter((r) => r.action === 'WAIT' && r.mode === 'scout').length;
    const waiting = current.filter((r) => r.action === 'WAIT' && r.mode !== 'scout').length;
    const progress = progressOf(summary);
    let modelText =
      `<span class="card-model"><span>모델 confidence <strong>${percent(summary.confidence)}</strong></span><span class="badge ${summary.status === 'reliable' ? 'green' : ''}">${modelLabel(summary.status)}</span></span>` +
      `<span class="card-confidence" aria-hidden="true"><i style="width:${(summary.confidence || 0) * 100}%"></i></span>` +
      `<span class="card-evidence">샘플 ${summary.samples} / ${progress.required_samples ?? '—'} 회 전환 · ${summary.period === 'night' ? '야간' : '주간'}</span>` +
      `<span class="card-progress">${estimateLabel(progress)}</span>` +
      `<span class="card-evidence">${progress.estimate_basis === 'unavailable' ? '백엔드 재시작 후 페이지 새로고침' : progress.pending_updates ? `${progress.pending_updates} 개 배치 업로드됨, 학습 완료 대기 중` : progress.remaining_samples ? `${progress.remaining_samples} 회 전환 부족 · 방문 횟수는 추정치` : progress.confidence_met ? '샘플과 confidence 모두 충족' : '계속 샘플링 중, 횟수는 관측 품질에 따라 달라짐'}</span>` +
      `<span class="card-evidence">첫 모델 ${firstModelLabel(summary)}</span>` +
      `<span class="card-evidence">처음 예측 가능 ${summary.history?.first_reliable ? simTime(summary.history.first_reliable.sim_time) : summary.status === 'reliable' ? '시각 미기록' : '아직 미달성'}</span>` +
      `<span class="card-visits">기록됨 <b>${totals.robots}</b> 대 로봇 / <b>${totals.visits}</b> 회 방문</span>` +
      `<span class="card-evidence">정찰 ${totals.scout} 회 · 일반 ${totals.normal} 회 · 시간 초과 ${totals.timeout} 회</span>` +
      `<span class="card-current">현재: 정찰 ${scouting} · 대기 ${waiting} · 통행 ${crossing}</span>`;
    if (!state.snapshot.learning)
      modelText =
        '<span class="card-progress">학습 상세 업데이트 대기</span><span class="card-evidence">백엔드 재시작 후 진행 상황과 방문 기록을 볼 수 있습니다</span>';
    setHTML(
      card,
      `<span class="crossing-top"><span><i class="dot ${colors[code]}"></i>${label[code]}</span>${meta.is_hub ? '<span>허브</span>' : ''}</span><span class="name" title="${esc(meta.name)}">${esc(meta.name)}</span>${modelText}<span class="occupants">${tags.join('') || '<span class="vacant">머무는 로봇 없음</span>'}</span>`,
    );
    card.setAttribute(
      'aria-label',
      `${tile.title}, ${present.map((r) => shortId(r.robot_id)).join(', ') || '머무는 로봇 없음'}`,
    );
  });
  $('no-results').hidden = matches.length > 0;
}
function meter(value, threshold) {
  return `<div class="meter" aria-hidden="true"><i style="width:${Math.max(0, Math.min(100, (value || 0) * 100))}%"></i>${Number.isFinite(threshold) ? `<b style="left:${Math.max(0, Math.min(100, threshold * 100))}%"></b>` : ''}</div>`;
}
function signalComparison(detail, stale, drift) {
  const live = detail.live;
  const observed =
    !stale &&
    live.state === 'observed' &&
    live.age != null &&
    live.age <= state.config.live_stale_seconds;
  const observedColor = observed ? live.color : null;
  const observationTitle = stale
    ? '관측 업데이트 대기'
    : live.state === 'disputed'
      ? '합의 부족'
      : observedColor
        ? colorName(observedColor)
        : '관측 없음';
  const observationNote = stale
    ? '스냅샷 만료됨'
    : live.state === 'disputed'
      ? '로봇 관측이 서로 엇갈림'
      : observed
        ? `유효 투표 ${live.voters} 대 · 스냅샷 기준 ${Math.max(0, live.age).toFixed(1)} 초 전`
        : '로봇의 유효 관측 대기 중';
  // Older backends only expose a fused decision; do not label it a model-only prediction.
  const prediction = detail.model_prediction;
  const predictedColor = !stale ? prediction?.color : null;
  const remaining =
    predictedColor === 'GREEN' ? prediction.green_remaining : prediction?.seconds_to_green;
  const expired = predictedColor && remaining != null && remaining - drift <= 0;
  const predictionTitle = stale
    ? '예측 업데이트 대기'
    : expired
      ? '위상 업데이트 대기'
      : predictedColor
        ? colorName(predictedColor)
        : detail.model?.reliable
          ? '예측 업데이트 대기'
          : '신뢰할 수 있는 모델 없음';
  const predictionNote = stale
    ? '스냅샷 만료됨'
    : expired
      ? '최신 모델 위상 대기 중'
      : predictedColor && remaining != null
        ? `${predictedColor === 'GREEN' ? '초록불 남음' : '초록불까지'} 약 ${Math.max(0, remaining - drift).toFixed(0)} 초`
        : detail.model?.reliable
          ? '백엔드 재시작 후 페이지 새로고침'
          : '학습 목표 달성 대기 중';
  const panel = (kind, title, color, status, note, extra) =>
    `<section class="signal-column ${kind}" aria-label="${title}"><h4>${title}</h4><div class="signal-readout"><div class="signal-lamp ${color === 'RED' ? 'red' : color === 'GREEN' ? 'green' : ''}" aria-hidden="true"><i></i><i></i></div><strong>${status}</strong></div><p>${note}</p><small>${extra}</small></section>`;
  const mismatch = observedColor && predictedColor && !expired && observedColor !== predictedColor;
  return `<div class="signal-comparison">${panel('observed-signal', '실시간 관측', observedColor, observationTitle, observationNote, '로봇 시각에서 수집')}${panel('predicted-signal', '모델 예측', expired ? null : predictedColor, predictionTitle, predictionNote, `모델 confidence ${percent(detail.model?.confidence)}`)}</div>${mismatch ? '<p class="signal-mismatch">관측과 예측이 일치하지 않습니다. 실시간 관측으로 신호색을 확인하세요.</p>' : ''}`;
}
function paintDetail() {
  const detail = state.detail;
  if (!detail || detail.intersection_id !== state.selected) return;
  const config = state.config;
  const stale =
    state.detailError ||
    !state.connected ||
    now() - detail.sim_now > Math.max(3 * state.snapshot.speed, config.live_stale_seconds);
  $('detail-status').textContent = state.detailError
    ? '상세 업데이트 실패'
    : stale
      ? '스냅샷이 오래되었을 수 있음'
      : state.followed
        ? '따라가는 중'
        : '선택된 교차로';
  const drift = stale ? 0 : Math.max(0, now() - detail.sim_now);
  const observers = activeRobots().filter(
    (r) =>
      r.intersection_id === detail.intersection_id &&
      r.observation?.intersection_id === detail.intersection_id,
  );
  const live = detail.live;
  const consensus = live.voters
    ? `${percent(live.agreement)} ${live.agreement + 1e-9 >= config.consensus_threshold ? '≥' : '<'} ${percent(config.consensus_threshold)}`
    : `— / ${percent(config.consensus_threshold)}`;
  const consensusNote = !live.voters
    ? '유효 투표가 없어 합의를 판단하지 않습니다.'
    : `${live.voters} 대 로봇 유효 투표 · ${live.agreement + 1e-9 >= config.consensus_threshold ? '합의 임계값 도달' : '임계값 미달, 실시간 신호색 미채택'}`;
  const model = detail.model;
  const learning = detail.learning || summaryOf(detail.intersection_id, model);
  setHTML(
    $('detail'),
    `<h3 class="detail-title">${esc(detail.meta.name)}</h3><p class="detail-sub">${esc(detail.intersection_id)} · ${esc(detail.meta.district)}${detail.meta.in_zone ? ' · 배송 구역' : ''}</p>
    ${signalComparison(detail, stale, drift)}
    ${learningPanel(learning)}
    ${detail.learning ? visitsPanel(learning.visits, detail.recent_visits || []) : ''}
    <section class="detail-section"><h3>이 신호의 관측 threshold</h3><div class="gate-line"><span>시뮬레이션 감지 빈도</span><strong>${Number.isFinite(config.obs_rate_hz) ? config.obs_rate_hz + ' FPS' : '—'}</strong></div><p class="hint">시뮬레이션 초 단위로 샘플링되며, 배속은 샘플링 간격을 바꾸지 않습니다.</p><div class="gate-line"><span>단일 프레임 confidence 임계값</span><strong>${percent(config.obs_confidence_threshold)}</strong></div><p class="hint">아래에는 각 로봇의 최근 프레임이 표시됩니다. 검은색 눈금이 수신 임계값입니다.</p>
    ${
      observers.length
        ? observers
            .map((robot) => {
              const obs = robot.observation;
              return `<div class="observation-card"><div class="gate-line"><strong>${esc(shortId(robot.robot_id))} · ${colorName(obs.color)}</strong><span class="badge ${obs.accepted ? 'green' : 'red'}">${obs.accepted ? '해당 프레임 수신됨' : '해당 프레임 폐기됨'}</span></div><div class="gate-line"><span>confidence / threshold</span><strong>${percent(obs.confidence)} / ${percent(config.obs_confidence_threshold)}</strong></div>${meter(obs.confidence, config.obs_confidence_threshold)}<p class="hint">${observationStale(obs) ? '관측 만료됨 · ' : ''}${age(obs.timestamp).toFixed(1)} 시뮬레이션 초 전${obs.accepted ? '' : ' · 단일 프레임 임계값 미달'}</p></div>`;
            })
            .join('')
        : '<p class="empty">머무는 로봇이 업로드한 관측이 없습니다. 현재 confidence는 —입니다.</p>'
    }
    </section><section class="detail-section"><h3>플릿 합의</h3><div class="gate-line"><span>가중 일치율 / 임계값</span><strong>${consensus}</strong></div>${meter(live.voters ? live.agreement : 0, config.consensus_threshold)}<p class="detail-note">${consensusNote}</p><p class="detail-note">로봇 한 대가 투표하면 일치율은 100%이며 시각 confidence와는 다릅니다. 합의에는 최근 ${config.vote_window_seconds} 시뮬레이션 초 이내에 임계값을 통과한 투표만 사용됩니다.</p></section>
    <section class="detail-section"><h3>주기 학습 <span class="badge ${model?.reliable ? 'green' : ''}">${model?.reliable ? '예측 가능' : '학습 중'}</span></h3>${
      model
        ? `<table class="kv"><thead><tr><th>길이</th><th>학습값</th><th>시뮬레이션 실제값</th></tr></thead><tbody>${[
            ['총 주기', 'T_cycle'],
            ['빨간불', 'T_red'],
            ['초록불', 'T_green'],
          ]
            .map(
              ([name, key]) =>
                `<tr><td>${name}</td><td>${model[key].toFixed(1)}s</td><td>${detail.truth[key] == null ? '—' : detail.truth[key].toFixed(1) + 's'}</td></tr>`,
            )
            .join(
              '',
            )}</tbody></table><p class="detail-note">${model.samples} 회 독립 전환 샘플 · 실제값은 시뮬레이션 대조용입니다.</p>`
        : '<p class="empty">아직 주기 모델이 없습니다. 로봇이 정찰 샘플링으로 신호 주기를 학습합니다.</p>'
    }<p class="detail-note">이 교차로에서 누적 ${live.accepted.toLocaleString()} 프레임 수신, ${live.rejected.toLocaleString()} 프레임 폐기.</p></section>`,
  );
}
$('search').addEventListener('input', (event) => {
  state.query = event.target.value.trim().toLowerCase();
  state.mapPage = 0;
  if (state.snapshot) paintMap(activeRobots());
});
$('map-scope').addEventListener('change', (event) => {
  state.mapScope = event.target.value;
  state.mapPage = 0;
  if (state.snapshot) paintMap(activeRobots());
});
$('map-prev').onclick = () => {
  state.mapPage--;
  paintMap(activeRobots());
};
$('map-next').onclick = () => {
  state.mapPage++;
  paintMap(activeRobots());
};
$('fleet-prev').onclick = () => {
  state.fleetPage--;
  paintFleet(activeRobots());
};
$('fleet-next').onclick = () => {
  state.fleetPage++;
  paintFleet(activeRobots());
};
boot();

function learningPanel(summary) {
  const config = state.config;
  if (progressOf(summary).estimate_basis === 'unavailable')
    return `<section class="detail-section learning-panel"><h3>모델 학습 진행 상황</h3><div class="model-score"><strong>${percent(summary.confidence)}</strong><span>${modelLabel(summary.status)}</span></div><p class="detail-note">백엔드가 아직 학습 진행 데이터를 제공하지 않습니다. 백엔드 재시작 후 페이지를 새로고침하세요.</p></section>`;
  const history = summary.history || {};
  const milestone = (title, event, fallback) =>
    `<li><span>${title}</span><strong>${event ? simTime(event.sim_time) + ' 시뮬레이션' : fallback}</strong>${event ? `<small>${esc(new Date(event.wall_time * 1000).toLocaleString('ko-KR', { timeZone: 'Asia/Seoul', hour12: false }))} KST 기록 · ${event.samples} 회 전환 · ${percent(event.confidence)}<br>업데이트 트리거: ${event.robots.map(shortId).map(esc).join(', ') || '—'}</small>` : ''}</li>`;
  const progress = progressOf(summary);
  const basis = {
    ready: '현재 시간대 모델을 예측에 사용할 수 있으며, 이후 관측도 모델을 계속 업데이트합니다.',
    confidence:
      '샘플 수는 충족했지만 confidence가 아직 부족합니다. 더 안정적인 주기 관측이 필요하며 고정된 방문 횟수를 약속할 수 없습니다.',
    no_transitions: `최근 ${progress.recent_scout_visits} 회 관련 정찰에서 이 시간대 전환을 수집하지 못했습니다. 머물며 신호색 전환을 계속 관찰해야 합니다.`,
    history: `최근 ${progress.recent_scout_visits} 회 관련 정찰에서 평균 ${progress.mean_transitions_per_scout} 회의 이 시간대 전환을 기준으로 추정했습니다.`,
    planning:
      '정찰 이력이 아직 없어 시뮬레이션의 한 번의 완전한 정찰이 약 3–4 회 전환을 수집한다고 가정해 추정합니다.',
  }[progress.estimate_basis];
  return `<section class="detail-section learning-panel"><div class="gate-line"><h3>모델 학습 진행 상황</h3><span class="badge ${summary.status === 'reliable' ? 'green' : ''}">${modelLabel(summary.status)}</span></div><div class="model-score"><strong>${percent(summary.confidence)}</strong><span>모델 confidence · ${summary.period === 'night' ? '야간' : '주간'}</span></div>${meter(summary.confidence, config.model_min_confidence)}
    <div class="learning-check"><span>① 독립 전환 샘플</span><strong>${summary.samples} / ${progress.required_samples}</strong><span class="badge ${progress.remaining_samples ? 'amber' : 'green'}">${progress.remaining_samples ? `${progress.remaining_samples} 회 부족` : '충족됨'}</span></div>
    <div class="sample-progress" role="progressbar" aria-label="전환 샘플 수집 진행률" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Math.round(progress.sample_progress * 100)}"><i style="width:${progress.sample_progress * 100}%"></i></div>
    <div class="learning-check"><span>② 모델 confidence</span><strong>≥ ${percent(progress.required_confidence)}</strong><span class="badge ${progress.confidence_met ? 'green' : 'amber'}">${progress.confidence_met ? '충족됨' : '미충족'}</span></div>
    <p class="detail-note">같은 전환을 여러 로봇이 관측해도 샘플은 1회입니다. 연속 관측으로 확인한 완전한 주기: ${summary.complete_cycles ?? 0}회.</p>
    <p class="detail-note">업데이트 전 전환 시각 오차: ${Number.isFinite(summary.timing_mae) ? summary.timing_mae.toFixed(2) + '초 MAE · 최근 ' + summary.timing_evaluations + '회' : '평가할 새 전환 대기 중'}. 주기 내 가장 가까운 경계 기준이며 전체 예측 정확도는 아닙니다.</p>
    ${summary.sample_basis === 'legacy' ? '<p class="detail-note">이전 버전 모델은 파라미터만 보존되며 독립 샘플을 다시 확인합니다.</p>' : ''}
    <div class="learning-estimate"><strong>${estimateLabel(progress)}</strong><p>${basis}</p>${summary.status !== 'reliable' ? '<p>추정은 샘플 보충용이며 그 시점의 모델 신뢰성을 보장하지 않습니다. 일반 통행에는 전환이 없을 수 있으며 주간과 야간은 별도로 학습됩니다.</p>' : ''}</div>
    <p class="learning-pending">${progress.pending_updates ? `${progress.pending_updates} 개 배치 업로드됨, 학습 완료 대기 중` : '로봇이 방문을 끝낸 후에야 전환을 업로드합니다. 수집 중인 데이터는 아직 반영되지 않았습니다.'}</p>
    <p class="detail-note">두 조건을 모두 충족해야 예측할 수 있습니다. 진행률 막대는 샘플 수만 나타내며, 모델 confidence는 학습기 점수이지 시각 인식 정확도가 아닙니다.</p><ol class="milestones">${milestone('첫 모델 생성', history.first_model, summary.confidence == null ? '아직 생성되지 않음' : '이전 모델, 첫 시각 미기록')}${milestone('처음 예측 가능 달성', history.first_reliable, summary.status === 'reliable' ? '이전 모델, 첫 시각 미기록' : '아직 미달성')}${milestone('마지막 업데이트', history.last_update, '업데이트 기록 없음')}</ol></section>`;
}
function visitsPanel(totals, records) {
  return `<section class="detail-section"><h3>로봇 방문 기록</h3><p class="visit-summary">서로 다른 로봇 <strong>${totals.robots}</strong> 대, 종료된 방문 <strong>${totals.visits}</strong> 회 기록됨</p><p class="detail-note">정찰 ${totals.scout} 회 · 일반 ${totals.normal} 회<br>통행 완료 ${totals.cross} 회 · 시간 초과로 떠남 ${totals.timeout} 회</p><div class="visit-history">${records.map((record) => `<div class="visit-entry"><div class="gate-line"><strong>${esc(shortId(record.robot_id))} · ${record.mode === 'scout' ? '정찰' : record.mode === 'normal' ? '일반' : esc(record.mode)}</strong><span class="badge ${record.action === 'CROSS' ? 'green' : 'amber'}">${record.action === 'CROSS' ? '통행 완료' : '시간 초과로 떠남'}</span></div><p>${simTime(record.arrival_time)} → ${simTime(record.depart_time)} 시뮬레이션</p><p>대기 ${Number(record.waited).toFixed(1)} 초 · ${record.transitions} 회 전환 보고</p></div>`).join('') || '<p class="empty">종료된 방문이 없습니다. 현재 있는 로봇은 교차로 카드를 참조하세요.</p>'}</div><p class="detail-note">최근 ${records.length} 건, 실제 수신 시간순 정렬. 통계는 이 기능 활성화 이후부터 집계되며, 누락된 과거 기록은 복원하지 않습니다. 전환 보고가 학습 완료를 의미하지는 않습니다.</p></section>`;
}
