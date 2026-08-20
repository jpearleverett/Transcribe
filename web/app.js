/* Transcribe — front end.
 *
 * Guiding constraint: on Android, Chrome freezes timers and often drops open
 * connections for a backgrounded tab, and may discard the tab entirely under
 * memory pressure. So the browser never *drives* a transcription — the Termux
 * server does. This page only renders server state, and every path back into
 * it (SSE, polling, a cold page load) reconstructs the same view.
 */
'use strict';

const $ = (id) => document.getElementById(id);
const api = (path, opts) => fetch(path, opts).then(async (r) => {
  const text = await r.text();
  let data;
  try { data = text ? JSON.parse(text) : {}; } catch { data = { error: text }; }
  if (!r.ok) throw new Error(data.error || `Request failed (${r.status})`);
  return data;
});

const state = {
  jobs: new Map(),
  current: null,       // job id shown in the detail view
  result: null,        // { segments, speakers, meta }
  engines: [],
  config: {},
  showTimestamps: true,
  merged: true,
  search: '',
  activeSegment: -1,
  activeEl: null,      // cached DOM node for the active segment
  activeWords: [],     // its word spans, cached alongside
  upload: null,        // in-flight XMLHttpRequest
};

/* ------------------------------------------------------------------ *
 * Utilities
 * ------------------------------------------------------------------ */

function hms(sec) {
  sec = Math.max(0, Math.floor(sec || 0));
  const h = Math.floor(sec / 3600), m = Math.floor(sec / 60) % 60, s = sec % 60;
  const mm = h ? String(m).padStart(2, '0') : String(m);
  return (h ? h + ':' : '') + mm + ':' + String(s).padStart(2, '0');
}

function bytes(n) {
  if (!n) return '';
  const u = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n < 10 && i > 0 ? n.toFixed(1) : Math.round(n)} ${u[i]}`;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

let toastTimer;
function toast(msg, ms = 2600) {
  const el = $('toast');
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, ms);
}

function speakerColor(label) {
  // Stable per-transcript colour: hash the label so a speaker keeps its colour
  // across re-renders and after a rename.
  const s = String(label ?? '');
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
  return `var(--spk-${h % 10})`;
}

function speakerName(label) {
  const names = (state.result && state.result.speakers) || {};
  if (names[label]) return names[label];
  if (label == null) return 'Speaker';
  const s = String(label);
  if (/^SPEAKER_/i.test(s)) return 'Speaker ' + (s.split('_')[1] || '').replace(/^0+(?=\d)/, '');
  return 'Speaker ' + s;
}

function ask(title, label, value) {
  return new Promise((resolve) => {
    const dlg = $('promptDialog');
    $('promptTitle').textContent = title;
    $('promptLabel').textContent = label;
    const input = $('promptInput');
    input.value = value || '';
    const onClose = () => {
      dlg.removeEventListener('close', onClose);
      resolve(dlg.returnValue === 'ok' ? input.value.trim() : null);
    };
    dlg.addEventListener('close', onClose);
    dlg.showModal();
    setTimeout(() => { input.focus(); input.select(); }, 50);
  });
}

/* ------------------------------------------------------------------ *
 * Live updates: SSE with a polling fallback
 * ------------------------------------------------------------------ */

let es = null;
let pollTimer = null;
let sseFailures = 0;
let reconnectTimer = null;

function connectLive() {
  if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
  if (es) { es.close(); es = null; }
  try {
    es = new EventSource('/api/events');
  } catch {
    startPolling();
    return;
  }

  es.onopen = () => { sseFailures = 0; stopPolling(); };

  es.onmessage = (ev) => {
    let data;
    try { data = JSON.parse(ev.data); } catch { return; }
    if (data.type === 'job') {
      applyJob(data.job);
    } else if (data.type === 'deleted') {
      state.jobs.delete(data.id);
      renderJobList();
      if (state.current === data.id) showList();
    }
  };

  es.onerror = () => {
    // Backgrounding the tab kills the stream; that is normal, not an error to
    // show the user. Reconnect with backoff and poll meanwhile so a phone that
    // blocks SSE entirely still updates.
    sseFailures++;
    if (es) { es.close(); es = null; }
    startPolling();
    // One pending reconnect at a time: without this, every error queues another
    // timer and a flapping connection turns into a burst of reconnects.
    if (reconnectTimer) return;
    const delay = Math.min(1000 * Math.pow(2, Math.min(sseFailures, 5)), 30000);
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      if (!document.hidden) connectLive();
    }, delay);
  };
}

function startPolling() {
  if (pollTimer) return;
  pollTimer = setInterval(refreshJobs, 3000);
}

function stopPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

document.addEventListener('visibilitychange', () => {
  if (!document.hidden) {
    // Coming back from the lock screen: resync immediately, then reconnect.
    refreshJobs();
    if (!es) connectLive();
    if (state.current) refreshDetail(state.current);
  }
});

function applyJob(job) {
  const prev = state.jobs.get(job.id);
  state.jobs.set(job.id, job);
  renderJobList();
  if (state.current === job.id) {
    renderDetailHeader(job);
    const finished = prev && prev.status !== job.status && job.status === 'done';
    if (finished || (job.status === 'done' && !state.result)) refreshDetail(job.id);
  }
  if (prev && prev.status === 'running' && job.status === 'done' && state.current !== job.id) {
    toast(`"${job.name}" is ready`);
  }
}

async function refreshJobs() {
  try {
    const data = await api('/api/jobs');
    state.jobs = new Map(data.jobs.map((j) => [j.id, j]));
    renderJobList();
    if (state.current) {
      const j = state.jobs.get(state.current);
      if (j) renderDetailHeader(j);
    }
  } catch { /* offline or server restarting; the next tick retries */ }
}

/* ------------------------------------------------------------------ *
 * Upload
 * ------------------------------------------------------------------ */

function pickFile() { $('fileInput').click(); }

$('pickBtn').addEventListener('click', pickFile);
$('fileInput').addEventListener('change', (e) => {
  const f = e.target.files && e.target.files[0];
  if (f) uploadFile(f);
  e.target.value = '';   // let the same file be picked twice in a row
});

// Desktop convenience; harmless on a phone.
const dz = $('dropZone');
['dragenter', 'dragover'].forEach((t) => dz.addEventListener(t, (e) => {
  e.preventDefault(); dz.classList.add('drag');
}));
['dragleave', 'drop'].forEach((t) => dz.addEventListener(t, (e) => {
  e.preventDefault(); dz.classList.remove('drag');
}));
dz.addEventListener('drop', (e) => {
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (f) uploadFile(f);
});

function uploadFile(file) {
  const maxMb = state.config.max_upload_mb || 2048;
  if (file.size > maxMb * 1024 * 1024) {
    toast(`That file is ${bytes(file.size)} — the limit is ${maxMb} MB.`);
    return;
  }
  if (file.size === 0) { toast('That file is empty.'); return; }

  const engine = $('engineSelect').value;
  const eng = state.engines.find((e) => e.name === engine);
  if (eng && eng.needs_key && !eng.has_key) {
    toast(`${eng.label} needs an API key first.`);
    openSettings();
    return;
  }

  const params = new URLSearchParams({
    name: file.name,
    engine,
    language: $('languageSelect').value,
    num_speakers: $('speakersSelect').value,
  });

  // XMLHttpRequest, not fetch: upload progress events are the whole point, and
  // fetch's request streaming still isn't a portable way to get them.
  const xhr = new XMLHttpRequest();
  state.upload = xhr;
  xhr.open('POST', '/api/upload?' + params.toString());
  xhr.setRequestHeader('Content-Type', file.type || 'application/octet-stream');

  $('uploadProgress').hidden = false;
  $('upName').textContent = file.name;
  $('upPct').textContent = '0%';
  $('upBar').style.width = '0%';

  xhr.upload.onprogress = (e) => {
    if (!e.lengthComputable) return;
    const pct = Math.round((e.loaded / e.total) * 100);
    $('upPct').textContent = pct + '%';
    $('upBar').style.width = pct + '%';
  };

  xhr.onload = () => {
    state.upload = null;
    $('uploadProgress').hidden = true;
    let data = {};
    try { data = JSON.parse(xhr.responseText); } catch { /* handled below */ }
    if (xhr.status >= 200 && xhr.status < 300 && data.job) {
      applyJob(data.job);
      openJob(data.job.id);
    } else {
      toast(data.error || `Upload failed (${xhr.status})`);
    }
  };

  xhr.onerror = () => {
    state.upload = null;
    $('uploadProgress').hidden = true;
    toast('Upload failed — is the Termux server still running?');
  };

  xhr.onabort = () => {
    state.upload = null;
    $('uploadProgress').hidden = true;
  };

  xhr.send(file);
}

$('upCancel').addEventListener('click', () => { if (state.upload) state.upload.abort(); });

/* ------------------------------------------------------------------ *
 * Job list
 * ------------------------------------------------------------------ */

const STAGE_TEXT = {
  queued: 'Queued', starting: 'Starting', converting: 'Converting audio',
  uploading: 'Uploading to engine', transcribing: 'Transcribing',
  diarizing: 'Identifying speakers', aligning: 'Aligning words',
  polling: 'Waiting for the engine', downloading: 'Fetching result',
  assembling: 'Building transcript', done: 'Done', failed: 'Failed',
  cancelled: 'Cancelled', interrupted: 'Interrupted',
};

function renderJobList() {
  const list = $('jobList');
  const jobs = [...state.jobs.values()].sort((a, b) => b.created - a.created);
  $('emptyState').hidden = jobs.length > 0;
  list.innerHTML = '';

  for (const j of jobs) {
    const el = document.createElement('button');
    el.className = 'job';
    el.type = 'button';
    const sub = [];
    if (j.duration) sub.push(hms(j.duration));
    if (j.size) sub.push(bytes(j.size));
    if (j.engine) sub.push(j.engine);
    const running = j.status === 'running' || j.status === 'pending';

    el.innerHTML = `
      <span class="dot ${esc(j.status)}"></span>
      <span class="job-body">
        <span class="job-name">${esc(j.name)}</span>
        <span class="job-sub">
          <span>${running ? esc(STAGE_TEXT[j.stage] || j.stage) : esc(statusText(j))}</span>
          ${sub.map((s) => `<span>${esc(s)}</span>`).join('')}
        </span>
        ${running ? `<span class="job-bar bar"><span class="bar-fill" style="width:${Math.round((j.progress || 0) * 100)}%"></span></span>` : ''}
      </span>`;
    el.addEventListener('click', () => openJob(j.id));
    list.appendChild(el);
  }
}

function statusText(j) {
  if (j.status === 'done') return new Date(j.created * 1000).toLocaleString();
  if (j.status === 'failed') return 'Failed';
  if (j.status === 'cancelled') return 'Cancelled';
  return STAGE_TEXT[j.stage] || j.stage;
}

/* ------------------------------------------------------------------ *
 * Detail view
 * ------------------------------------------------------------------ */

function showList() {
  state.current = null;
  state.result = null;
  stopAudio();
  $('listView').hidden = false;
  $('detailView').hidden = true;
  $('backBtn').hidden = true;
  $('player').hidden = true;
  $('title').textContent = 'Transcribe';
  if (location.hash) history.pushState({}, '', location.pathname);
}

function openJob(id) {
  state.current = id;
  state.result = null;
  state.search = '';
  $('searchInput').value = '';
  $('listView').hidden = true;
  $('detailView').hidden = false;
  $('backBtn').hidden = false;
  $('transcript').innerHTML = '';
  if (location.hash !== '#' + id) history.pushState({ id }, '', '#' + id);
  const job = state.jobs.get(id);
  if (job) renderDetailHeader(job);
  refreshDetail(id);
}

$('backBtn').addEventListener('click', showList);
window.addEventListener('popstate', () => {
  const id = location.hash.slice(1);
  if (id && state.jobs.has(id)) openJob(id); else showList();
});

function renderDetailHeader(job) {
  $('title').textContent = job.name;
  const bits = [];
  if (job.duration) bits.push(hms(job.duration));
  if (job.engine) bits.push(job.engine + (job.model ? ` · ${job.model}` : ''));
  if (job.language) bits.push(job.language);
  if (job.size) bits.push(bytes(job.size));
  $('detailMeta').innerHTML = bits.map((b) => `<span>${esc(b)}</span>`).join('');

  const running = job.status === 'running' || job.status === 'pending';
  $('progressPanel').hidden = !running;
  $('errorPanel').hidden = job.status !== 'failed' && job.status !== 'cancelled';
  $('toolbar').hidden = job.status !== 'done';

  if (running) {
    $('stageLabel').textContent = STAGE_TEXT[job.stage] || job.stage;
    const pct = Math.round((job.progress || 0) * 100);
    $('stagePct').textContent = job.progress > 0 ? pct + '%' : '';
    const fill = $('stageBar');
    fill.classList.toggle('indeterminate', !job.progress);
    fill.style.width = job.progress ? pct + '%' : '';
    $('jobLog').textContent = (job.log || []).map((l) => `${l.t}s  ${l.m}`).join('\n');
  }
  if (!$('errorPanel').hidden) {
    $('errorText').textContent = job.error || 'Cancelled.';
  }

  const audio = $('audio');
  if (job.has_audio) {
    const src = `/api/jobs/${job.id}/audio`;
    if (!audio.src.endsWith(src)) audio.src = src;
    $('player').hidden = false;
  } else {
    $('player').hidden = true;
  }
}

async function refreshDetail(id) {
  const job = state.jobs.get(id);
  if (job && job.status !== 'done') return;
  try {
    const data = await api(`/api/jobs/${id}/result?merged=${state.merged ? 1 : 0}`);
    if (state.current !== id) return;
    state.result = data;
    renderSpeakers();
    renderTranscript();
  } catch (e) {
    if (state.current === id) toast(e.message);
  }
}

function renderSpeakers() {
  const wrap = $('speakerChips');
  wrap.innerHTML = '';
  const stats = (state.result && state.result.stats) || [];
  if (stats.length < 1) return;
  for (const s of stats) {
    const btn = document.createElement('button');
    btn.className = 'chip';
    btn.type = 'button';
    btn.style.color = speakerColor(s.speaker);
    btn.innerHTML = `${esc(speakerName(s.speaker))} <span class="share">${Math.round((s.share || 0) * 100)}%</span>`;
    btn.title = 'Tap to rename';
    btn.addEventListener('click', () => renameSpeaker(s.speaker));
    wrap.appendChild(btn);
  }
}

async function renameSpeaker(label) {
  const current = (state.result.speakers || {})[label] || '';
  const name = await ask('Rename speaker', `Who is ${speakerName(label)}?`, current);
  if (name === null) return;
  const speakers = { ...(state.result.speakers || {}) };
  if (name) speakers[label] = name; else delete speakers[label];
  state.result.speakers = speakers;
  renderSpeakers();
  renderTranscript();
  try {
    await api(`/api/jobs/${state.current}/speakers`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ speakers }),
    });
  } catch (e) { toast(e.message); }
}

function renderTranscript() {
  const box = $('transcript');
  // The cached nodes are about to be replaced.
  state.activeSegment = -1;
  state.activeEl = null;
  state.activeWords = [];
  box.innerHTML = '';
  box.classList.toggle('hide-times', !state.showTimestamps);
  const segs = (state.result && state.result.segments) || [];
  if (!segs.length) {
    $('noMatches').hidden = true;
    box.innerHTML = '<p class="empty">This recording came back empty — no speech was detected.</p>';
    return;
  }

  const needle = state.search.trim().toLowerCase();
  let shown = 0;
  const frag = document.createDocumentFragment();
  let lastSpeaker = Symbol('none');

  segs.forEach((seg, i) => {
    const hit = !needle || seg.text.toLowerCase().includes(needle);
    if (needle && !hit) return;
    shown++;

    const turn = document.createElement('div');
    turn.className = 'turn';
    const color = speakerColor(seg.speaker);

    if (seg.speaker !== lastSpeaker || needle) {
      const head = document.createElement('div');
      head.className = 'turn-head';
      head.innerHTML =
        `<span class="turn-speaker" style="color:${color}">${esc(speakerName(seg.speaker))}</span>` +
        `<button class="turn-time" data-t="${seg.start}">${hms(seg.start)}</button>`;
      turn.appendChild(head);
      lastSpeaker = seg.speaker;
    }

    const body = document.createElement('div');
    body.className = 'turn-body seg';
    body.style.color = color;
    body.dataset.index = String(i);
    body.dataset.start = String(seg.start);
    body.dataset.end = String(seg.end);
    body.innerHTML = renderSegText(seg, needle);
    turn.appendChild(body);
    frag.appendChild(turn);
  });

  box.appendChild(frag);
  $('noMatches').hidden = shown > 0;
}

function renderSegText(seg, needle) {
  // With word timings we emit per-word spans so playback can highlight the
  // exact word. Without them we fall back to plain (still searchable) text.
  if (seg.words && seg.words.length && !needle) {
    return seg.words.map((w) =>
      `<span class="w" data-s="${w.start}" data-e="${w.end}">${esc(w.text)}</span>`
    ).join(' ');
  }
  const text = esc(seg.text);
  if (!needle) return text;
  const re = new RegExp('(' + needle.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'ig');
  return text.replace(re, '<mark>$1</mark>');
}

$('transcript').addEventListener('click', (e) => {
  const t = e.target.closest('[data-t], .w, .seg');
  if (!t) return;
  const time = t.dataset.t ?? t.dataset.s ?? t.dataset.start;
  if (time === undefined) return;
  seekTo(parseFloat(time));
});

let searchTimer;
$('searchInput').addEventListener('input', (e) => {
  clearTimeout(searchTimer);
  const v = e.target.value;
  searchTimer = setTimeout(() => { state.search = v; renderTranscript(); }, 160);
});

/* ------------------------------------------------------------------ *
 * Playback
 * ------------------------------------------------------------------ */

const audio = $('audio');
let rafId = null;
const RATES = [1, 1.25, 1.5, 1.75, 2, 0.75];
let rateIndex = 0;

$('playBtn').addEventListener('click', () => {
  if (audio.paused) audio.play().catch((e) => toast('Playback failed: ' + e.message));
  else audio.pause();
});

audio.addEventListener('play', () => { setPlayIcon(true); tick(); });
audio.addEventListener('pause', () => { setPlayIcon(false); cancelAnimationFrame(rafId); });
audio.addEventListener('ended', () => { setPlayIcon(false); cancelAnimationFrame(rafId); });
audio.addEventListener('loadedmetadata', () => { $('timeLabel').textContent = hms(audio.duration); });

function setPlayIcon(playing) {
  $('playIcon').innerHTML = playing
    ? '<path fill="currentColor" d="M6 5h4v14H6zm8 0h4v14h-4z"/>'
    : '<path fill="currentColor" d="M8 5v14l11-7z"/>';
}

$('rateBtn').addEventListener('click', () => {
  rateIndex = (rateIndex + 1) % RATES.length;
  audio.playbackRate = RATES[rateIndex];
  $('rateBtn').textContent = RATES[rateIndex] + '×';
});

let seeking = false;
$('seek').addEventListener('input', () => { seeking = true; });
$('seek').addEventListener('change', (e) => {
  if (audio.duration) audio.currentTime = (e.target.value / 1000) * audio.duration;
  seeking = false;
});

function seekTo(t) {
  if (!isFinite(t)) return;
  if (!audio.src) { toast('The audio for this transcript is no longer on disk.'); return; }
  audio.currentTime = Math.max(0, t);
  audio.play().catch(() => { /* a tap is required first on some devices */ });
}

function tick() {
  // rAF rather than timeupdate: timeupdate fires only ~4x/second, which makes
  // word highlighting visibly lag the audio.
  const t = audio.currentTime;
  if (!seeking && audio.duration) {
    $('seek').value = String(Math.round((t / audio.duration) * 1000));
  }
  $('timeLabel').textContent = hms(t);
  highlight(t);
  rafId = requestAnimationFrame(tick);
}

function highlight(t) {
  const segs = (state.result && state.result.segments) || [];
  if (!segs.length) return;
  let idx = state.activeSegment;
  if (idx < 0 || idx >= segs.length || t < segs[idx].start || t > segs[idx].end) {
    idx = binarySearchSegment(segs, t);
  }

  if (idx !== state.activeSegment) {
    if (state.activeEl) state.activeEl.classList.remove('active');
    state.activeSegment = idx;
    // Cache the element and its word spans. This runs on every animation
    // frame, and re-querying the document 60 times a second over a transcript
    // with thousands of nodes is what makes long recordings feel sluggish.
    state.activeEl = idx >= 0 ? document.querySelector(`.seg[data-index="${idx}"]`) : null;
    state.activeWords = state.activeEl ? [...state.activeEl.querySelectorAll('.w')] : [];
    if (state.activeEl) {
      state.activeEl.classList.add('active');
      const r = state.activeEl.getBoundingClientRect();
      if (r.top < 70 || r.bottom > window.innerHeight - 90) {
        state.activeEl.scrollIntoView({ block: 'center', behavior: 'smooth' });
      }
    }
  }

  for (const w of state.activeWords) {
    const on = t >= parseFloat(w.dataset.s) && t <= parseFloat(w.dataset.e);
    if (on !== w.classList.contains('active')) w.classList.toggle('active', on);
  }
}

function binarySearchSegment(segs, t) {
  let lo = 0, hi = segs.length - 1, found = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (t < segs[mid].start) hi = mid - 1;
    else if (t > segs[mid].end) { found = mid; lo = mid + 1; }
    else return mid;
  }
  return found >= 0 && t - segs[found].end < 2 ? found : -1;
}

function stopAudio() {
  audio.pause();
  audio.removeAttribute('src');
  audio.load();
  cancelAnimationFrame(rafId);
  state.activeSegment = -1;
  state.activeEl = null;
  state.activeWords = [];
}

/* ------------------------------------------------------------------ *
 * Menus / actions
 * ------------------------------------------------------------------ */

function toggleMenu(menu) {
  const wasHidden = menu.hidden;
  document.querySelectorAll('.menu').forEach((m) => { m.hidden = true; });
  menu.hidden = !wasHidden;
}
$('exportBtn').addEventListener('click', (e) => { e.stopPropagation(); toggleMenu($('exportMenu')); });
$('moreBtn').addEventListener('click', (e) => { e.stopPropagation(); toggleMenu($('moreMenu')); });
document.addEventListener('click', () => { document.querySelectorAll('.menu').forEach((m) => { m.hidden = true; }); });

$('exportMenu').addEventListener('click', (e) => {
  const btn = e.target.closest('button[data-fmt]');
  if (!btn) return;
  // A plain navigation, not fetch+blob: Chrome on Android handles a normal
  // download with a Content-Disposition header far more reliably than a
  // synthesised <a download> click.
  window.location.href = `/api/jobs/${state.current}/export.${btn.dataset.fmt}`;
});

$('copyBtn').addEventListener('click', async () => {
  try {
    const text = await fetch(`/api/jobs/${state.current}/export.txt?inline=1`).then((r) => r.text());
    await navigator.clipboard.writeText(text);
    toast('Transcript copied');
  } catch {
    toast('Copy failed — use Export instead');
  }
});

$('toggleTimestamps').addEventListener('click', () => {
  state.showTimestamps = !state.showTimestamps;
  $('toggleTimestamps').textContent = state.showTimestamps ? 'Hide timestamps' : 'Show timestamps';
  renderTranscript();
});

$('toggleMerge').addEventListener('click', () => {
  state.merged = !state.merged;
  $('toggleMerge').textContent = state.merged ? 'Split into short segments' : 'Merge into paragraphs';
  refreshDetail(state.current);
});

$('rerunBtn').addEventListener('click', async () => {
  const job = state.jobs.get(state.current);
  // Only engines that are actually ready, and not the one already used.
  const options = state.engines.filter((e) =>
    e.available && (e.has_key || !e.needs_key) && e.name !== 'mock' && e.name !== (job && job.engine));
  if (!options.length) {
    toast('No other engine is set up yet — add a key in Settings.');
    return;
  }
  const labels = options.map((e, i) => `${i + 1}. ${e.label}`).join('\n');
  const pick = await ask('Re-run with another engine',
                         `The same audio, a second opinion:\n${labels}\n\nEnter a number`, '1');
  const idx = parseInt(pick, 10) - 1;
  if (!(idx >= 0 && idx < options.length)) return;
  try {
    const data = await api(`/api/jobs/${state.current}/rerun`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ engine: options[idx].name }),
    });
    applyJob(data.job);
    openJob(data.job.id);
    toast(`Re-running with ${options[idx].label}`);
  } catch (e) { toast(e.message); }
});

$('renameJobBtn').addEventListener('click', async () => {
  const job = state.jobs.get(state.current);
  const name = await ask('Rename', 'Transcript name', job ? job.name : '');
  if (!name) return;
  try {
    const data = await api(`/api/jobs/${state.current}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
    applyJob(data.job);
  } catch (e) { toast(e.message); }
});

$('deleteJobBtn').addEventListener('click', async () => {
  const job = state.jobs.get(state.current);
  if (!confirm(`Delete "${job ? job.name : 'this transcript'}"? This also removes the audio.`)) return;
  try {
    await api(`/api/jobs/${state.current}`, { method: 'DELETE' });
    state.jobs.delete(state.current);
    showList();
    renderJobList();
  } catch (e) { toast(e.message); }
});

$('cancelJobBtn').addEventListener('click', async () => {
  try { await api(`/api/jobs/${state.current}/cancel`, { method: 'POST' }); }
  catch (e) { toast(e.message); }
});

$('retryBtn').addEventListener('click', async () => {
  try {
    const data = await api(`/api/jobs/${state.current}/retry`, { method: 'POST' });
    applyJob(data.job);
  } catch (e) { toast(e.message); }
});

/* ------------------------------------------------------------------ *
 * Settings
 * ------------------------------------------------------------------ */

const LANGUAGES = [
  ['auto', 'Auto-detect'], ['en', 'English'], ['es', 'Spanish'], ['fr', 'French'],
  ['de', 'German'], ['it', 'Italian'], ['pt', 'Portuguese'], ['nl', 'Dutch'],
  ['ru', 'Russian'], ['pl', 'Polish'], ['tr', 'Turkish'], ['uk', 'Ukrainian'],
  ['ar', 'Arabic'], ['hi', 'Hindi'], ['zh', 'Chinese'], ['ja', 'Japanese'],
  ['ko', 'Korean'], ['vi', 'Vietnamese'], ['id', 'Indonesian'], ['sv', 'Swedish'],
  ['da', 'Danish'], ['no', 'Norwegian'], ['fi', 'Finnish'], ['cs', 'Czech'],
  ['el', 'Greek'], ['he', 'Hebrew'], ['th', 'Thai'], ['ro', 'Romanian'], ['hu', 'Hungarian'],
];

function fillLanguages(sel, value) {
  sel.innerHTML = LANGUAGES.map(([v, l]) => `<option value="${v}">${l}</option>`).join('');
  sel.value = value || 'auto';
}

function fillEngines(sel, value) {
  sel.innerHTML = state.engines.map((e) => {
    const warn = e.needs_key && !e.has_key ? ' — needs key' : (e.available ? '' : ' — unavailable');
    return `<option value="${esc(e.name)}">${esc(e.label)}${esc(warn)}</option>`;
  }).join('');
  sel.value = value && state.engines.some((e) => e.name === value) ? value : (state.engines[0] || {}).name;
}

function updateSpeakerHintNote(eng) {
  // Do not let the UI promise a setting the chosen engine cannot use.
  const note = $('speakersNote');
  const select = $('speakersSelect');
  if (!note) return;
  if (eng && eng.supports_speaker_count === false) {
    // Still worth setting: the engine cannot take the hint, but the app
    // applies it afterwards by merging any extra speakers it invented.
    note.textContent = `${eng.label} has no speaker-count setting of its own, so this `
      + 'is applied afterwards — extra speakers it invents get merged into '
      + 'whoever was talking around them. AssemblyAI and ElevenLabs can also '
      + 'use it during transcription, which works better.';
    note.classList.remove('warn');
    select.disabled = false;
    select.removeAttribute('title');
  } else {
    note.textContent = 'Telling it the exact number of speakers, when you know it, '
      + 'is the single biggest accuracy win for diarization.';
    note.classList.remove('warn');
    select.disabled = false;
    select.removeAttribute('title');
  }
}

function updateEngineNote() {
  const eng = state.engines.find((e) => e.name === $('engineSelect').value);
  const note = $('engineNote');
  updateSpeakerHintNote(eng);
  if (!eng) { note.textContent = ''; return; }
  let text = eng.description || '';
  let warn = false;
  if (eng.needs_key && !eng.has_key) { text = `${eng.label} needs an API key. Open Settings to add one.`; warn = true; }
  else if (!eng.available) { text = eng.unavailable_reason || `${eng.label} isn't available on this device.`; warn = true; }
  note.textContent = text;
  note.classList.toggle('warn', warn);
  $('engineConfigBtn').hidden = !warn;
}

$('engineSelect').addEventListener('change', () => {
  updateEngineNote();
  api('/api/config', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ engine: $('engineSelect').value }),
  }).catch(() => {});
});
$('engineConfigBtn').addEventListener('click', openSettings);
$('settingsBtn').addEventListener('click', openSettings);

function openSettings() {
  fillEngines($('setEngine'), state.config.engine);
  fillLanguages($('setLanguage'), state.config.language);
  $('setMaxDur').value = state.config.max_dur ?? 30;
  $('setMaxGap').value = state.config.max_gap ?? 1.0;

  const wrap = $('keyFields');
  wrap.innerHTML = '';
  for (const eng of state.engines) {
    if (!eng.needs_key && !(eng.config_fields || []).length) continue;
    const hint = (state.config.key_hint || {})[eng.name];
    const fromEnv = (state.config.from_env || []).includes(eng.name);
    const row = document.createElement('div');
    row.className = 'key-row';
    let html = `<div class="key-head">
        <span>${esc(eng.label)}${hint ? ` <span class="key-set">key set${fromEnv ? ' from environment' : ''} (${esc(hint)})</span>` : ''}</span>
        ${eng.signup_url ? `<a href="${esc(eng.signup_url)}" target="_blank" rel="noopener">Get a key</a>` : ''}
      </div>`;
    if (eng.needs_key) {
      html += `<input type="password" data-engine="${esc(eng.name)}"
             placeholder="${hint ? (fromEnv ? 'Paste a key to override the environment' : 'API key — leave blank to keep') : 'Paste API key'}"
             autocomplete="off" autocapitalize="off" autocorrect="off" spellcheck="false">`;
    }
    for (const f of eng.config_fields || []) {
      html += `<input type="${esc(f.type || 'text')}" data-config="${esc(f.key)}"
             value="${esc(f.value ?? '')}" placeholder="${esc(f.placeholder || f.label)}"
             aria-label="${esc(f.label)}"
             autocomplete="off" autocapitalize="off" autocorrect="off" spellcheck="false">
             ${f.help ? `<p class="hint small">${esc(f.help)}</p>` : ''}`;
    }
    if (eng.key_help) html += `<p class="hint small">${esc(eng.key_help)}</p>`;
    row.innerHTML = html;
    wrap.appendChild(row);
  }
  $('serverInfo').textContent = state.config.server_info || '';
  $('settingsDialog').showModal();
}

$('settingsDialog').addEventListener('close', async () => {
  if ($('settingsDialog').returnValue !== 'save') return;
  const keys = {};
  $('keyFields').querySelectorAll('input[data-engine]').forEach((i) => {
    if (i.value.trim()) keys[i.dataset.engine] = i.value.trim();
  });
  const extra = {};
  $('keyFields').querySelectorAll('input[data-config]').forEach((i) => {
    extra[i.dataset.config] = i.value.trim();
  });
  try {
    const data = await api('/api/config', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ...extra,
        engine: $('setEngine').value,
        language: $('setLanguage').value,
        max_dur: parseFloat($('setMaxDur').value),
        max_gap: parseFloat($('setMaxGap').value),
        keys,
      }),
    });
    state.config = data.config;
    state.engines = data.engines;
    fillEngines($('engineSelect'), state.config.engine);
    fillLanguages($('languageSelect'), state.config.language);
    updateEngineNote();
    toast('Settings saved');
  } catch (e) { toast(e.message); }
});

/* ------------------------------------------------------------------ *
 * Boot
 * ------------------------------------------------------------------ */

async function boot() {
  try {
    const data = await api('/api/config');
    state.config = data.config;
    state.engines = data.engines;
  } catch {
    toast('Cannot reach the server. Is it still running in Termux?');
  }
  fillEngines($('engineSelect'), state.config.engine);
  fillLanguages($('languageSelect'), state.config.language);
  $('speakersSelect').value = String(state.config.num_speakers || 0);
  updateEngineNote();

  await refreshJobs();
  connectLive();

  const id = location.hash.slice(1);
  if (id && state.jobs.has(id)) openJob(id);
}

boot();
