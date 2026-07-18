"use strict";

const $ = (sel) => document.querySelector(sel);

const el = {
  connDot: $("#conn-dot"),
  statusText: $("#status-text"),
  startBtn: $("#start-btn"),
  stopBtn: $("#stop-btn"),
  endBtn: $("#end-btn"),
  counter: $("#counter"),
  warnings: $("#warnings"),
  transcript: $("#transcript"),
  partial: $("#partial"),
  words: $("#words"),
  wordsCount: $("#words-count"),
  analysis: $("#analysis"),
  knownWords: $("#known-words"),
  analyzeBtn: $("#analyze-btn"),
  analyzeStatus: $("#analyze-status"),
  results: $("#results"),
  learnCount: $("#learn-count"),
  learnList: $("#learn-list"),
  rejectedList: $("#rejected-list"),
  summary: $("#summary"),
  downloadBtn: $("#download-btn"),
  savedPath: $("#saved-path"),
};

let recording = false;
let uniqueCount = 0;
let lastResult = null;
const seenWords = new Set();

// ---------------------------------------------------------------------------
// WebSocket live feed
// ---------------------------------------------------------------------------
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    el.connDot.classList.add("on");
    el.statusText.textContent = "connected";
  };
  ws.onclose = () => {
    el.connDot.classList.remove("on");
    el.statusText.textContent = "disconnected — retrying…";
    setTimeout(connect, 1500);
  };
  ws.onmessage = (ev) => handleEvent(JSON.parse(ev.data));
}

function handleEvent(msg) {
  switch (msg.type) {
    case "snapshot":
      renderSnapshot(msg.session);
      break;
    case "partial":
      el.partial.textContent = msg.text;
      break;
    case "final":
      el.partial.textContent = "";
      if (msg.text) appendTranscript(msg.text);
      (msg.new_words || []).forEach(addWord);
      if (msg.stats) setCount(msg.stats.unique_words);
      break;
    case "info":
      flash(msg.message);
      break;
    case "error":
      warn(msg.message);
      break;
  }
}

function renderSnapshot(session) {
  if (!session) return;
  setCount(session.unique_words || 0);
  el.words.innerHTML = "";
  seenWords.clear();
  (session.words || []).forEach((w) =>
    addWord({ word: w.word, surface: w.surface, context: w.context, count: w.count })
  );
}

function appendTranscript(text) {
  const p = document.createElement("p");
  p.textContent = text;
  el.transcript.appendChild(p);
  el.transcript.parentElement.scrollTop = el.transcript.parentElement.scrollHeight;
}

function addWord(w) {
  if (seenWords.has(w.word)) return;
  seenWords.add(w.word);
  const item = document.createElement("div");
  item.className = "word-item";
  item.innerHTML = `
    <div class="w">${escapeHtml(w.surface || w.word)}${
      w.count ? `<span class="cnt">×${w.count}</span>` : ""
    }</div>
    <div class="ctx">${escapeHtml(w.context || "")}</div>`;
  el.words.prepend(item);
}

function setCount(n) {
  uniqueCount = n;
  el.counter.textContent = `${n} unique word${n === 1 ? "" : "s"}`;
  el.wordsCount.textContent = n;
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------
async function api(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  return res.json();
}

el.startBtn.onclick = async () => {
  const r = await api("/api/start");
  if (!r.ok) return warn(r.message);
  recording = true;
  el.connDot.classList.add("rec");
  el.statusText.textContent = "recording";
  el.startBtn.disabled = true;
  el.stopBtn.disabled = false;
  el.endBtn.disabled = false;
  flash(r.message);
};

el.stopBtn.onclick = async () => {
  const r = await api("/api/stop");
  recording = false;
  el.connDot.classList.remove("rec");
  el.connDot.classList.add("on");
  el.statusText.textContent = "stopped";
  el.startBtn.disabled = false;
  el.stopBtn.disabled = true;
  flash(r.message);
};

el.endBtn.onclick = async () => {
  if (recording) await el.stopBtn.onclick();
  el.analysis.classList.remove("hidden");
  el.analysis.scrollIntoView({ behavior: "smooth" });
};

el.analyzeBtn.onclick = async () => {
  el.analyzeBtn.disabled = true;
  el.analyzeStatus.textContent = "Analyzing (matching → lemmatizing → AI)…";
  const r = await api("/api/analyze", { known_words: el.knownWords.value });
  el.analyzeBtn.disabled = false;
  if (!r.ok) {
    el.analyzeStatus.textContent = "";
    return warn(r.message || "Analysis failed.");
  }
  el.analyzeStatus.textContent = "Done.";
  lastResult = r;
  renderResults(r);
};

el.downloadBtn.onclick = () => {
  if (!lastResult) return;
  const blob = new Blob([JSON.stringify(lastResult, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "words-to-learn.json";
  a.click();
  URL.revokeObjectURL(a.href);
};

// ---------------------------------------------------------------------------
// Results rendering
// ---------------------------------------------------------------------------
function renderResults(r) {
  el.results.classList.remove("hidden");
  const s = r.summary || {};
  el.learnCount.textContent = s.to_learn ?? (r.words_to_learn || []).length;
  el.summary.innerHTML = `
    <span>Total unique: <b>${s.total_unique ?? 0}</b></span>
    <span>Known (exact): <b>${s.already_known_exact ?? 0}</b></span>
    <span>Known (lemma): <b>${s.already_known_lemma ?? 0}</b></span>
    <span>AI-rejected: <b>${s.ai_rejected ?? 0}</b></span>
    <span>To learn: <b>${s.to_learn ?? 0}</b></span>
    <span>Model: <b>${escapeHtml(s.model || "")}</b></span>
    <span>Lemmatizer: <b>${escapeHtml(s.lemmatizer || "")}</b></span>`;

  el.savedPath.textContent = r.saved_to ? `Saved to ${r.saved_to}` : "";

  el.learnList.innerHTML = "";
  (r.words_to_learn || []).forEach((w) => {
    const card = document.createElement("div");
    card.className = "learn-card";
    card.innerHTML = `
      <div class="head">
        <span class="lemma">${escapeHtml(w.lemma || w.word)}</span>
        <span class="surface">heard as “${escapeHtml(w.surface || w.word)}” ×${w.count || 1}</span>
      </div>
      <div class="ctx">${escapeHtml(w.context || "")}</div>
      ${w.reason ? `<div class="reason">${escapeHtml(w.reason)}</div>` : ""}`;
    el.learnList.appendChild(card);
  });

  el.rejectedList.innerHTML = "";
  const rejected = [
    ...(r.already_known_exact || []).map((w) => [w, "known (exact)"]),
    ...(r.already_known_lemma || []).map((w) => [w, "known (lemma)"]),
    ...(r.ai_rejected || []).map((w) => [w.word, `AI: ${w.reason || "not useful"}`]),
  ];
  rejected.forEach(([w, why]) => {
    const div = document.createElement("div");
    div.className = "rej-item";
    div.innerHTML = `<b>${escapeHtml(w)}</b> — ${escapeHtml(why)}`;
    el.rejectedList.appendChild(div);
  });

  el.results.scrollIntoView({ behavior: "smooth" });
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function warn(text) {
  const div = document.createElement("div");
  div.className = "warn";
  div.textContent = text;
  el.warnings.appendChild(div);
  setTimeout(() => div.remove(), 8000);
}
function flash(text) {
  el.statusText.textContent = text;
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

// ---------------------------------------------------------------------------
async function init() {
  connect();
  try {
    const st = await (await fetch("/api/status")).json();
    if (!st.has_speechmatics_key)
      warn("SPEECHMATICS_API_KEY not set — recording will not start.");
    if (!st.has_openai_key)
      warn("OPENAI_API_KEY not set — AI filtering step will be skipped (words kept unfiltered).");
    if (st.session) renderSnapshot(st.session);
  } catch (e) {
    /* status is best-effort */
  }
}
init();
