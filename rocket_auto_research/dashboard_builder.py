from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rocket_auto_research.auto_research.problem_definition import default_problem_stages
from rocket_auto_research.replay_enrichment import augment_trajectory_with_balloon_snapshots


INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-Hant" data-theme="day">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>火箭自動研究儀表板 | Rocket Auto Research Dashboard</title>
  <link rel="stylesheet" href="./styles.css">
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
</head>
<body>
  <div class="page-shell">
    <header class="hero">
      <div>
        <p class="eyebrow bilingual"><span class="zh">火箭 GNC 自動研究</span><span class="en">Rocket GNC Auto Research</span></p>
        <h1 class="bilingual"><span class="zh">研究儀表板</span><span class="en">Research Dashboard</span></h1>
        <p class="subtitle bilingual"><span class="zh">瀏覽世代演化、檢視最佳策略，並用 3D 回放歷史軌跡。</span><span class="en">Browse generations, inspect best strategies, and replay historical trajectories in 3D.</span></p>
      </div>
      <div class="hero-meta">
        <label class="theme-switch">
          <span class="bilingual"><span class="zh">頁面主題</span><span class="en">Theme</span></span>
          <select id="theme-select">
            <option value="day">白天模式 / Day</option>
            <option value="night">黑夜模式 / Night</option>
          </select>
        </label>
        <div class="pill" id="generated-at">載入建置資訊中...<br>Loading build metadata...</div>
        <div class="pill" id="best-strategy-pill">最佳策略載入中...<br>Best strategy pending...</div>
      </div>
    </header>

    <section class="summary-grid" id="summary-grid"></section>

    <section class="panel control-panel">
      <div class="panel-heading">
        <div>
          <p class="section-kicker bilingual"><span class="zh">執行控制</span><span class="en">Runtime</span></p>
          <h2 class="bilingual"><span class="zh">自動研究控制台</span><span class="en">Auto Research Control</span></h2>
        </div>
        <div class="pill" id="research-status-pill">檢查本機控制伺服器中...<br>Checking local control server...</div>
      </div>
      <div class="control-row stacked">
        <label class="wide">
          <span class="bilingual"><span class="zh">研究設定檔</span><span class="en">Research config</span></span>
          <select id="research-config"></select>
        </label>
        <label>
          <span class="bilingual"><span class="zh">工作執行數</span><span class="en">Workers</span></span>
          <select id="worker-count"></select>
        </label>
        <label>
          <span class="bilingual"><span class="zh">每代實驗數</span><span class="en">Population size</span></span>
          <select id="population-size"></select>
        </label>
        <button id="research-start" type="button"><span class="zh">開始</span><span class="en">Start</span></button>
        <button id="research-pause" type="button" class="secondary"><span class="zh">暫停</span><span class="en">Pause</span></button>
        <button id="research-stop" type="button"><span class="zh">停止</span><span class="en">Stop</span></button>
      </div>
      <div id="research-config-help" class="config-help">Select a config to see its purpose and recommended use.</div>
      <div class="notes-block">
        <h3 class="bilingual"><span class="zh">狀態</span><span class="en">Status</span></h3>
        <div id="research-status-text" class="research-status-block">儀表板正在等待本機控制伺服器。<br>The dashboard is waiting for the local control server.</div>
      </div>
      <pre id="research-log" class="code-block compact">尚無研究日誌。
No research log yet.</pre>
    </section>

    <section class="panel">
      <div class="panel-heading">
        <div>
          <p class="section-kicker bilingual"><span class="zh">演化趨勢</span><span class="en">Evolution</span></p>
          <h2 class="bilingual"><span class="zh">世代趨勢</span><span class="en">Generation Trend</span></h2>
        </div>
      </div>
      <div class="control-row">
        <label>
          <span class="bilingual"><span class="zh">顯示範圍</span><span class="en">Trend window</span></span>
          <select id="generation-window">
            <option value="all">全部 / All</option>
            <option value="20">最近 20 代 / Last 20</option>
            <option value="50">最近 50 代 / Last 50</option>
            <option value="100">最近 100 代 / Last 100</option>
            <option value="250">最近 250 代 / Last 250</option>
            <option value="500">最近 500 代 / Last 500</option>
            <option value="1000">最近 1000 代 / Last 1000</option>
          </select>
        </label>
        <label>
          <span class="bilingual"><span class="zh">適應值尺度</span><span class="en">Fitness scale</span></span>
          <select id="generation-scale">
            <option value="linear">線性 / Linear</option>
            <option value="log">對數 / Log</option>
          </select>
        </label>
        <div id="generation-stats" class="pill trend-pill">顯示全部世代 / Showing all generations</div>
      </div>
      <div id="generation-chart" class="chart"></div>
    </section>

    <section class="panel">
      <div class="panel-heading">
        <div>
          <p class="section-kicker bilingual"><span class="zh">實驗演進</span><span class="en">Experiment Evolution</span></p>
          <h2 class="bilingual"><span class="zh">自動研究進展</span><span class="en">Auto Research Progress</span></h2>
        </div>
      </div>
      <div id="progress-summary" class="pill trend-pill">載入實驗演進中 / Loading experiment evolution</div>
      <div class="control-row">
        <label>
          <span class="bilingual"><span class="zh">實驗範圍</span><span class="en">Experiment window</span></span>
          <select id="progress-window">
            <option value="all">全部 / All</option>
            <option value="20">最近 20 次 / Last 20</option>
            <option value="50">最近 50 次 / Last 50</option>
            <option value="100">最近 100 次 / Last 100</option>
            <option value="250">最近 250 次 / Last 250</option>
            <option value="500">最近 500 次 / Last 500</option>
          </select>
        </label>
        <label>
          <span class="bilingual"><span class="zh">Y 軸指標</span><span class="en">Y-axis metric</span></span>
          <select id="progress-metric">
            <option value="final_fitness">Final fitness</option>
            <option value="mean_popped">Mean popped</option>
          </select>
        </label>
      </div>
      <div id="progress-chart" class="chart"></div>
    </section>

    <section class="layout-grid">
      <div class="panel left-column">
        <div class="panel-heading">
          <div>
            <p class="section-kicker bilingual"><span class="zh">執行瀏覽</span><span class="en">Run Browser</span></p>
            <h2 class="bilingual"><span class="zh">歷史實驗</span><span class="en">Historical Experiments</span></h2>
          </div>
        </div>
        <div class="control-row">
          <label>
            <span class="bilingual"><span class="zh">策略</span><span class="en">Strategy</span></span>
            <select id="strategy-filter"></select>
          </label>
          <label>
            <span class="bilingual"><span class="zh">模擬介面</span><span class="en">Adapter</span></span>
            <select id="adapter-filter"></select>
          </label>
          <label class="wide">
            <span class="bilingual"><span class="zh">搜尋</span><span class="en">Search</span></span>
            <input id="search-filter" type="search" placeholder="搜尋實驗編號、註記或策略 / Search experiment id, note, or strategy">
          </label>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>實驗<br>Experiment</th>
                <th>時間<br>Time</th>
                <th>耗時<br>Duration</th>
                <th>策略<br>Strategy</th>
                <th>介面<br>Adapter</th>
                <th>適應度<br>Fitness</th>
                <th>命中數<br>Pops</th>
              </tr>
            </thead>
            <tbody id="run-table-body"></tbody>
          </table>
        </div>
      </div>

      <div class="panel right-column">
        <div class="panel-heading">
          <div>
            <p class="section-kicker bilingual"><span class="zh">回放控制</span><span class="en">Replay Control</span></p>
            <h2 class="bilingual"><span class="zh">軌跡回放</span><span class="en">Trajectory Replay</span></h2>
          </div>
        </div>
        <div class="control-row stacked">
          <label class="wide">
            <span class="bilingual"><span class="zh">實驗</span><span class="en">Experiment</span></span>
            <select id="run-select"></select>
          </label>
          <label>
            <span class="bilingual"><span class="zh">隨機種子</span><span class="en">Seed</span></span>
            <select id="seed-select"></select>
          </label>
          <label>
            <span class="bilingual"><span class="zh">回放速度</span><span class="en">Replay speed</span></span>
            <select id="speed-select">
              <option value="1">1x</option>
              <option value="2">2x</option>
              <option value="4">4x</option>
              <option value="8">8x</option>
            </select>
          </label>
          <label class="wide">
            <span class="bilingual"><span class="zh">回放尺度</span><span class="en">Replay scale</span></span>
            <div class="inline-range">
              <input id="scale-slider" type="range" min="0.6" max="2.4" step="0.1" value="1">
              <span id="scale-readout">1.0x</span>
            </div>
          </label>
        </div>
        <div class="replay-actions">
          <button id="play-toggle" type="button"><span class="zh">播放</span><span class="en">Play</span></button>
          <button id="reset-replay" type="button"><span class="zh">重設</span><span class="en">Reset</span></button>
          <span id="frame-readout">影格 0 / 0<br>Frame 0 / 0</span>
        </div>
        <input id="frame-slider" type="range" min="0" max="0" value="0">
        <div id="replay-plot" class="chart replay-chart"></div>
        <div id="timeline-chart" class="chart timeline-chart"></div>
      </div>
    </section>

    <section class="layout-grid">
      <div class="panel left-column">
        <div class="panel-heading">
          <div>
            <p class="section-kicker bilingual"><span class="zh">目前選擇</span><span class="en">Selection</span></p>
            <h2 class="bilingual"><span class="zh">執行細節</span><span class="en">Run Details</span></h2>
          </div>
        </div>
        <div id="run-details" class="detail-grid"></div>
        <div class="notes-block">
          <h3 class="bilingual"><span class="zh">失敗註記</span><span class="en">Failure Notes</span></h3>
          <ul id="failure-notes"></ul>
        </div>
      </div>
      <div class="panel right-column">
        <div class="panel-heading">
          <div>
            <p class="section-kicker bilingual"><span class="zh">設定內容</span><span class="en">Configuration</span></p>
            <h2 class="bilingual"><span class="zh">策略參數</span><span class="en">Strategy Parameters</span></h2>
          </div>
        </div>
        <pre id="params-viewer" class="code-block"></pre>
        <div class="notes-block">
          <h3 class="bilingual"><span class="zh">交叉驗證</span><span class="en">Cross Validation</span></h3>
          <div id="cross-validation"></div>
        </div>
      </div>
    </section>

    <section class="panel">
      <div class="panel-heading">
        <div>
          <p class="section-kicker bilingual"><span class="zh">報告預覽</span><span class="en">Reports</span></p>
          <h2 class="bilingual"><span class="zh">世代報告預覽</span><span class="en">Generation Report Preview</span></h2>
        </div>
      </div>
      <div id="report-list" class="report-list"></div>
    </section>
  </div>

  <script src="./data/index.js"></script>
  <script src="./app.js"></script>
</body>
</html>
"""


STYLES_CSS = """* {
  box-sizing: border-box;
}

:root {
  color-scheme: light;
  --bg: #f3efe3;
  --paper: rgba(255, 252, 244, 0.85);
  --panel: rgba(255, 250, 240, 0.94);
  --ink: #20211d;
  --muted: #676b62;
  --accent: #1d6b69;
  --accent-strong: #b25424;
  --border: rgba(32, 33, 29, 0.12);
  --shadow: 0 18px 40px rgba(66, 53, 31, 0.12);
  --input-bg: #ffffff;
  --thead-bg: #fff8ee;
  --code-bg: #141716;
  --code-ink: #ecf3ee;
  --plot-bg: rgba(255,255,255,0.55);
}

:root[data-theme="night"] {
  color-scheme: dark;
  --bg: #090c10;
  --paper: rgba(23, 30, 36, 0.9);
  --panel: rgba(18, 24, 30, 0.94);
  --ink: #edf3f7;
  --muted: #aebac4;
  --accent: #6ec2bd;
  --accent-strong: #f09b62;
  --border: rgba(255, 255, 255, 0.12);
  --shadow: 0 18px 40px rgba(0, 0, 0, 0.35);
  --input-bg: #1a232a;
  --thead-bg: #1a232a;
  --code-bg: #0b1014;
  --code-ink: #eef6fb;
  --plot-bg: rgba(18,24,30,0.82);
}

body {
  margin: 0;
  font-family: "Segoe UI", "Trebuchet MS", sans-serif;
  color: var(--ink);
  background: var(--bg);
}

.page-shell {
  width: min(1440px, calc(100vw - 32px));
  margin: 0 auto;
  padding: 28px 0 44px;
}

.hero,
.panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 24px;
  box-shadow: var(--shadow);
}

.hero {
  display: flex;
  justify-content: space-between;
  gap: 24px;
  padding: 28px;
  align-items: flex-start;
  margin-bottom: 20px;
}

.eyebrow,
.section-kicker {
  margin: 0 0 8px;
  font-size: 0.76rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--accent);
}

.hero h1,
.panel h2 {
  margin: 0;
  font-family: Georgia, "Times New Roman", serif;
  font-weight: 700;
}

.hero h1 {
  font-size: clamp(2.1rem, 5vw, 3.6rem);
}

.subtitle {
  max-width: 720px;
  margin: 12px 0 0;
  color: var(--muted);
  line-height: 1.55;
}

.hero-meta {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  justify-content: flex-end;
  align-items: flex-start;
}

.theme-switch {
  min-width: 180px;
}

.theme-switch select {
  min-width: 180px;
}

.bilingual {
  display: inline-flex;
  flex-direction: column;
  gap: 2px;
}

.bilingual .zh {
  font-weight: 600;
}

.bilingual .en {
  font-size: 0.82em;
  opacity: 0.88;
  font-weight: 500;
}

.pill {
  padding: 10px 14px;
  border-radius: 999px;
  background: rgba(29, 107, 105, 0.08);
  border: 1px solid rgba(29, 107, 105, 0.16);
  font-size: 0.92rem;
}

.summary-grid,
.layout-grid {
  display: grid;
  gap: 18px;
}

.summary-grid {
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  margin-bottom: 20px;
}

.summary-card {
  background: var(--paper);
  border-radius: 20px;
  padding: 18px;
  border: 1px solid var(--border);
}

.summary-card .label {
  color: var(--muted);
  font-size: 0.88rem;
}

.summary-card .value {
  font-size: 1.9rem;
  font-weight: 700;
  margin-top: 8px;
}

.summary-card .caption {
  margin-top: 6px;
  color: var(--muted);
  font-size: 0.85rem;
}

.panel {
  padding: 22px;
  margin-bottom: 18px;
}

.control-panel .code-block.compact {
  max-height: 220px;
}

.config-help {
  margin-top: -2px;
  margin-bottom: 12px;
  padding: 12px 14px;
  border-radius: 14px;
  border: 1px solid var(--border);
  background: var(--paper);
  color: var(--muted);
  line-height: 1.5;
  font-size: 0.9rem;
}

.research-status-block {
  display: grid;
  gap: 12px;
}

.status-visuals {
  display: grid;
  gap: 12px;
}

.status-summary {
  padding: 14px 16px;
  border-radius: 16px;
  border: 1px solid var(--border);
  background: var(--paper);
  line-height: 1.5;
}

.status-summary strong {
  color: var(--ink);
}

.status-visual-panel {
  padding: 14px 16px;
  border-radius: 16px;
  border: 1px solid var(--border);
  background: var(--paper);
}

.status-visual-title {
  margin-bottom: 10px;
  color: var(--muted);
  font-size: 0.8rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.status-visual-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: center;
  margin-bottom: 10px;
}

.status-visual-head .status-visual-title {
  margin-bottom: 0;
}

.status-inline-select {
  min-height: 34px;
  padding: 6px 10px;
  font-size: 0.84rem;
}

.metric-stack,
.failure-stack {
  display: grid;
  gap: 10px;
}

.metric-row,
.failure-row {
  display: grid;
  gap: 6px;
}

.metric-head,
.failure-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: baseline;
}

.metric-name,
.failure-name {
  font-weight: 600;
  color: var(--ink);
}

.metric-value,
.failure-value {
  color: var(--muted);
  font-size: 0.88rem;
}

.metric-track,
.failure-track {
  position: relative;
  height: 12px;
  border-radius: 999px;
  background: rgba(29, 107, 105, 0.10);
  overflow: hidden;
}

.metric-fill,
.failure-fill {
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  border-radius: 999px;
}

.metric-fill.primary {
  background: linear-gradient(90deg, #1d6b69, #2d8c88);
}

.metric-fill.secondary {
  background: linear-gradient(90deg, #b25424, #cf7a2d);
}

.metric-fill.success {
  background: linear-gradient(90deg, #2f7d32, #4caf50);
}

.metric-fill.warning {
  background: linear-gradient(90deg, #c28b1f, #e0b13d);
}

.metric-fill.danger {
  background: linear-gradient(90deg, #a93d22, #d95a2d);
}

.failure-fill {
  background: linear-gradient(90deg, #b25424, #cf7a2d);
}

.metric-marker {
  position: absolute;
  top: -2px;
  bottom: -2px;
  width: 2px;
  background: var(--ink);
  opacity: 0.65;
}

.pipeline {
  display: grid;
  gap: 10px;
}

.pipeline-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 10px;
}

.pipeline-node {
  padding: 12px;
  border-radius: 14px;
  border: 1px solid var(--border);
  background: rgba(29, 107, 105, 0.05);
}

.pipeline-node.current {
  border-color: rgba(29, 107, 105, 0.45);
  background: rgba(29, 107, 105, 0.12);
}

.pipeline-node.passed {
  border-color: rgba(45, 140, 136, 0.35);
  background: rgba(45, 140, 136, 0.10);
}

.pipeline-node.upcoming {
  opacity: 0.8;
}

.pipeline-state {
  margin-top: 6px;
  color: var(--muted);
  font-size: 0.82rem;
}

.status-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 10px;
}

.status-item {
  padding: 12px 14px;
  border-radius: 14px;
  border: 1px solid var(--border);
  background: var(--paper);
  min-width: 0;
}

.status-label {
  margin-bottom: 6px;
  color: var(--muted);
  font-size: 0.8rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.status-value {
  font-size: 0.96rem;
  font-weight: 600;
  line-height: 1.45;
  color: var(--ink);
  word-break: break-word;
}

.status-value.compact {
  font-size: 0.88rem;
  font-weight: 500;
}

.trend-pill {
  align-self: flex-end;
  white-space: nowrap;
}

.panel-heading {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
  margin-bottom: 18px;
}

.layout-grid {
  grid-template-columns: minmax(340px, 0.9fr) minmax(420px, 1.1fr);
}

.left-column,
.right-column {
  min-width: 0;
}

.control-row {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin-bottom: 14px;
}

.control-row.stacked {
  align-items: flex-end;
}

label {
  display: flex;
  flex-direction: column;
  gap: 8px;
  font-size: 0.88rem;
  color: var(--muted);
}

label.wide {
  flex: 1 1 260px;
}

.inline-range {
  display: flex;
  align-items: center;
  gap: 10px;
}

.inline-range input[type="range"] {
  flex: 1 1 auto;
  margin: 0;
}

.inline-range span {
  min-width: 44px;
  text-align: right;
  color: var(--ink);
  font-weight: 600;
}

input,
select,
button {
  font: inherit;
}

input,
select {
  border-radius: 12px;
  border: 1px solid var(--border);
  background: var(--input-bg);
  color: var(--ink);
  padding: 10px 12px;
  min-height: 42px;
}

button {
  border: none;
  border-radius: 12px;
  background: linear-gradient(135deg, var(--accent), #2d8c88);
  color: white;
  padding: 10px 16px;
  cursor: pointer;
  font-weight: 600;
}

button .zh,
button .en {
  display: block;
  line-height: 1.2;
}

button.secondary {
  background: linear-gradient(135deg, var(--accent-strong), #cf7a2d);
}

button:disabled {
  cursor: not-allowed;
  opacity: 0.5;
}

.replay-actions {
  display: flex;
  gap: 12px;
  align-items: center;
  margin-bottom: 12px;
}

#frame-slider {
  width: 100%;
  margin-bottom: 16px;
}

.chart {
  width: 100%;
  min-height: 320px;
}

.replay-chart {
  min-height: 460px;
}

.timeline-chart {
  min-height: 240px;
}

.table-wrap {
  max-height: 560px;
  overflow: auto;
  border: 1px solid var(--border);
  border-radius: 18px;
}

table {
  width: 100%;
  border-collapse: collapse;
  background: var(--input-bg);
}

thead {
  position: sticky;
  top: 0;
  background: var(--thead-bg);
  z-index: 1;
}

th,
td {
  text-align: left;
  padding: 12px 14px;
  border-bottom: 1px solid rgba(32, 33, 29, 0.08);
  font-size: 0.92rem;
}

tbody tr {
  cursor: pointer;
}

tbody tr:hover,
tbody tr.selected {
  background: rgba(29, 107, 105, 0.08);
}

.detail-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px;
}

.detail-card {
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 12px;
  background: rgba(255, 255, 255, 0.72);
}

.detail-card .label {
  display: block;
  color: var(--muted);
  font-size: 0.82rem;
}

.detail-card .value {
  display: block;
  font-size: 1.1rem;
  font-weight: 700;
  margin-top: 6px;
}

.notes-block {
  margin-top: 18px;
}

.notes-block h3 {
  margin: 0 0 10px;
  font-size: 1rem;
}

.notes-block ul {
  margin: 0;
  padding-left: 18px;
  color: var(--muted);
  line-height: 1.55;
}

.code-block {
  margin: 0;
  background: var(--code-bg);
  color: var(--code-ink);
  border-radius: 18px;
  padding: 16px;
  max-height: 420px;
  overflow: auto;
  font-size: 0.82rem;
}

.report-list {
  display: grid;
  gap: 12px;
}

.report-card {
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 14px;
  background: rgba(255, 255, 255, 0.74);
}

.report-card h3 {
  margin: 0 0 8px;
  font-size: 1rem;
}

.report-card p {
  margin: 0;
  color: var(--muted);
  line-height: 1.55;
  white-space: pre-wrap;
}

.cross-validation-card {
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 12px;
  background: rgba(255, 255, 255, 0.7);
  margin-bottom: 10px;
}

:root[data-theme="night"] .cross-validation-card,
:root[data-theme="night"] .detail-card,
:root[data-theme="night"] .report-card {
  background: rgba(255, 255, 255, 0.04);
}

@media (max-width: 1080px) {
  .layout-grid {
    grid-template-columns: 1fr;
  }

  .hero {
    flex-direction: column;
  }

  .hero-meta {
    justify-content: flex-start;
  }
}
"""


APP_JS = """(() => {
  const data = window.__DASHBOARD_DATA__;
  const trajectoryCache = window.__TRAJECTORY_CACHE__ = window.__TRAJECTORY_CACHE__ || {};
  const state = {
    runs: [],
    selectedRunId: null,
    selectedSeed: null,
    frameIndex: 0,
    timer: null,
    speed: 1,
    scale: 1,
    replayBounds: null,
    controlStatus: null,
    controlPollHandle: null,
    availableConfigs: [],
    generationWindow: "all",
    generationScale: "linear",
    progressWindow: "all",
    progressMetric: "final_fitness",
    failureSource: "current_experiment",
  };

  const els = {
    generatedAt: document.getElementById("generated-at"),
    bestStrategyPill: document.getElementById("best-strategy-pill"),
    themeSelect: document.getElementById("theme-select"),
    summaryGrid: document.getElementById("summary-grid"),
    generationWindow: document.getElementById("generation-window"),
    generationScale: document.getElementById("generation-scale"),
    generationStats: document.getElementById("generation-stats"),
    progressSummary: document.getElementById("progress-summary"),
    progressWindow: document.getElementById("progress-window"),
    progressMetric: document.getElementById("progress-metric"),
    researchStatusPill: document.getElementById("research-status-pill"),
    researchStatusText: document.getElementById("research-status-text"),
    researchConfig: document.getElementById("research-config"),
    workerCount: document.getElementById("worker-count"),
    populationSize: document.getElementById("population-size"),
    researchConfigHelp: document.getElementById("research-config-help"),
    researchStart: document.getElementById("research-start"),
    researchPause: document.getElementById("research-pause"),
    researchStop: document.getElementById("research-stop"),
    researchLog: document.getElementById("research-log"),
    strategyFilter: document.getElementById("strategy-filter"),
    adapterFilter: document.getElementById("adapter-filter"),
    searchFilter: document.getElementById("search-filter"),
    runTableBody: document.getElementById("run-table-body"),
    runSelect: document.getElementById("run-select"),
    seedSelect: document.getElementById("seed-select"),
    speedSelect: document.getElementById("speed-select"),
    scaleSlider: document.getElementById("scale-slider"),
    scaleReadout: document.getElementById("scale-readout"),
    playToggle: document.getElementById("play-toggle"),
    resetReplay: document.getElementById("reset-replay"),
    frameReadout: document.getElementById("frame-readout"),
    frameSlider: document.getElementById("frame-slider"),
    runDetails: document.getElementById("run-details"),
    failureNotes: document.getElementById("failure-notes"),
    paramsViewer: document.getElementById("params-viewer"),
    crossValidation: document.getElementById("cross-validation"),
    reportList: document.getElementById("report-list"),
  };

  function fmt(value, digits = 2) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return "n/a";
    }
    return Number(value).toFixed(digits);
  }

  function formatTimestamp(value) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric) || numeric <= 0) {
      return "n/a";
    }
    const date = new Date(numeric * 1000);
    if (Number.isNaN(date.getTime())) {
      return "n/a";
    }
    return date.toLocaleString("zh-TW", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
  }

  function formatDurationSeconds(value) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric) || numeric < 0) {
      return "n/a";
    }
    return `${numeric.toFixed(2)}s`;
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function dualLine(zh, en) {
    return `<span class="zh">${escapeHtml(zh)}</span><span class="en">${escapeHtml(en)}</span>`;
  }

  function setDualText(element, zh, en) {
    element.innerHTML = dualLine(zh, en);
  }

  function setButtonLabel(element, zh, en) {
    element.innerHTML = `<span class="zh">${escapeHtml(zh)}</span><span class="en">${escapeHtml(en)}</span>`;
  }

  function shortPath(value) {
    if (!value) {
      return "n/a";
    }
    const normalized = String(value).replace(/\\\\/g, "/");
    const parts = normalized.split("/").filter(Boolean);
    return parts.length ? parts[parts.length - 1] : normalized;
  }

  function humanizeToken(value) {
    if (!value) {
      return "n/a";
    }
    return String(value)
      .replace(/[_-]+/g, " ")
      .replace(/\\s+/g, " ")
      .trim();
  }

  function titleCase(value) {
    return humanizeToken(value)
      .split(" ")
      .filter(Boolean)
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join(" ");
  }

  function populateWorkerOptions(selectedValue, preserveCurrent = false) {
    if (!els.workerCount) {
      return;
    }
    const current = preserveCurrent ? Number(els.workerCount.value) : Number(selectedValue);
    const fallback = Number(selectedValue);
    const desiredSource = Number.isFinite(current) && current >= 1 ? current : fallback;
    const desired = Number.isFinite(desiredSource) && desiredSource >= 1 ? Math.min(32, Math.max(1, desiredSource)) : 1;
    const options = [];
    for (let index = 1; index <= 32; index += 1) {
      options.push(`<option value="${index}">${index}</option>`);
    }
    els.workerCount.innerHTML = options.join("");
    els.workerCount.value = String(desired);
  }

  function populatePopulationOptions(selectedValue, preserveCurrent = false) {
    if (!els.populationSize) {
      return;
    }
    const allowed = [4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64];
    const current = preserveCurrent ? Number(els.populationSize.value) : Number(selectedValue);
    const fallback = Number(selectedValue);
    const desiredSource = Number.isFinite(current) && current >= 1 ? current : fallback;
    const desired = Number.isFinite(desiredSource) && desiredSource >= 1 ? desiredSource : 6;
    const values = allowed.includes(desired) ? allowed : [...allowed, desired].sort((a, b) => a - b);
    els.populationSize.innerHTML = values.map((value) => `<option value="${value}">${value}</option>`).join("");
    els.populationSize.value = String(desired);
  }

  function statusCard(labelZh, labelEn, value, secondary = "") {
    const secondaryBlock = secondary
      ? `<div class="status-value compact">${escapeHtml(secondary)}</div>`
      : "";
    return `
      <div class="status-item">
        <div class="status-label bilingual">${dualLine(labelZh, labelEn)}</div>
        <div class="status-value">${escapeHtml(value || "n/a")}</div>
        ${secondaryBlock}
      </div>
    `;
  }

  function percentLabel(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) {
      return "n/a";
    }
    return `${(number * 100).toFixed(0)}%`;
  }

  function promotionDecisionLabel(value) {
    const normalized = String(value || "").trim().toLowerCase();
    if (normalized === "advance") {
      return "Advance";
    }
    if (normalized === "stay") {
      return "Stay";
    }
    if (normalized === "final_stage") {
      return "Final stage";
    }
    return "Unknown";
  }

  function promotionTone(currentValue, targetValue) {
    const current = Number(currentValue);
    const target = Number(targetValue);
    if (!Number.isFinite(current) || !Number.isFinite(target) || target <= 0) {
      return "secondary";
    }
    if (current >= target) {
      return "success";
    }
    if (current >= target * 0.75) {
      return "warning";
    }
    return "danger";
  }

  function clamp01(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) {
      return 0;
    }
    return Math.max(0, Math.min(1, number));
  }

  function metricBar(labelZh, labelEn, currentValue, targetValue, currentText, targetText, variant = "primary") {
    const currentWidth = clamp01(currentValue) * 100;
    const targetLeft = clamp01(targetValue) * 100;
    return `
      <div class="metric-row">
        <div class="metric-head">
          <div class="metric-name bilingual">${dualLine(labelZh, labelEn)}</div>
          <div class="metric-value">${escapeHtml(`${currentText} / ${targetText}`)}</div>
        </div>
        <div class="metric-track">
          <div class="metric-fill ${escapeHtml(variant)}" style="width:${currentWidth.toFixed(1)}%"></div>
          <div class="metric-marker" style="left:${targetLeft.toFixed(1)}%"></div>
        </div>
      </div>
    `;
  }

  function stageCatalogMap() {
    const map = new Map();
    (data.stage_catalog || []).forEach((stage, index) => {
      map.set(stage.stage_id, { ...stage, index });
    });
    return map;
  }

  function resolveStageContext(status) {
    const apiContext = status?.stage_context || {};
    const currentStageId = String(status?.current_stage || apiContext?.stage_id || data.latest_plan?.stage_id || "").trim();
    const catalog = stageCatalogMap();
    const fallback = catalog.get(currentStageId) || null;
    const pipeline = Array.isArray(apiContext?.pipeline) && apiContext.pipeline.length
      ? apiContext.pipeline
      : Array.from(catalog.values()).map((stage) => ({
          stage_id: stage.stage_id,
          title: stage.title,
          threshold: stage.success_threshold,
          best_success_rate: stage.stage_id === currentStageId ? Number(data.latest_plan?.recent_stage_success_rate || 0) : 0,
          state: stage.stage_id === currentStageId ? "current" : stage.index < (fallback?.index ?? 0) ? "passed" : "upcoming",
        }));
    const nextStage = fallback && typeof fallback.index === "number"
      ? (data.stage_catalog || [])[fallback.index + 1] || null
      : null;
    return {
      stage_id: apiContext?.stage_id || currentStageId || "n/a",
      stage_title: apiContext?.stage_title || fallback?.title || titleCase(currentStageId || "n/a"),
      stage_description: apiContext?.stage_description || fallback?.description || "n/a",
      success_threshold: apiContext?.success_threshold ?? fallback?.success_threshold ?? null,
      recent_stage_success_rate: apiContext?.recent_stage_success_rate ?? data.latest_plan?.recent_stage_success_rate ?? null,
      planner_rationale: apiContext?.planner_rationale || data.latest_plan?.rationale || "n/a",
      planner_notes: apiContext?.planner_notes || data.latest_plan?.planner_notes || [],
      next_stage_id: apiContext?.next_stage_id || nextStage?.stage_id || null,
      next_stage_title: apiContext?.next_stage_title || nextStage?.title || "n/a",
      promotion_decision: apiContext?.promotion_decision || "unknown",
      promotion_reason: apiContext?.promotion_reason || "n/a",
      pipeline,
      failure_breakdowns: apiContext?.failure_breakdowns || {},
    };
  }

  function stagePipeline(pipeline) {
    if (!Array.isArray(pipeline) || pipeline.length === 0) {
      return "";
    }
    const nodes = pipeline.map((stage) => {
      const state = String(stage?.state || "upcoming");
      const best = percentLabel(stage?.best_success_rate);
      const threshold = percentLabel(stage?.threshold);
      const stateLabel = state === "passed" ? "Passed" : state === "current" ? "Current" : "Upcoming";
      return `
        <div class="pipeline-node ${escapeHtml(state)}">
          <div class="metric-name">${escapeHtml(stage?.title || stage?.stage_id || "n/a")}</div>
          <div class="pipeline-state">${escapeHtml(`${stateLabel} · ${best} / ${threshold}`)}</div>
        </div>
      `;
    }).join("");
    return `<div class="pipeline-row">${nodes}</div>`;
  }

  function failureBars(entries) {
    if (!Array.isArray(entries) || entries.length === 0) {
      return `<div class="status-summary bilingual">${dualLine("目前沒有可用的失敗統計。", "No failure breakdown available yet.")}</div>`;
    }
    return `
      <div class="failure-stack">
        ${entries.map((entry) => `
          <div class="failure-row">
            <div class="failure-head">
              <div class="failure-name">${escapeHtml(titleCase(entry?.name || "unknown"))}</div>
              <div class="failure-value">${escapeHtml(`${percentLabel(entry?.rate)} · count ${entry?.count ?? 0}`)}</div>
            </div>
            <div class="failure-track">
              <div class="failure-fill" style="width:${(clamp01(entry?.rate) * 100).toFixed(1)}%"></div>
            </div>
          </div>
        `).join("")}
      </div>
    `;
  }

  function selectedFailureEntries(stageContext) {
    const breakdowns = stageContext?.failure_breakdowns || {};
    return breakdowns[state.failureSource] || [];
  }

  function applyTheme(theme) {
    const nextTheme = theme === "night" ? "night" : "day";
    document.documentElement.dataset.theme = nextTheme;
    els.themeSelect.value = nextTheme;
    window.localStorage.setItem("rocket-dashboard-theme", nextTheme);
  }

  function themeColor(variableName, fallback = "rgba(0,0,0,0)") {
    const value = getComputedStyle(document.documentElement).getPropertyValue(variableName).trim();
    return value || fallback;
  }

  async function apiGet(path) {
    const response = await fetch(path, { cache: "no-store" });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `GET ${path} failed`);
    }
    return response.json();
  }

  async function apiPost(path, payload = {}) {
    const response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.error || `POST ${path} failed`);
    }
    return data;
  }

  function makeOption(select, label, value) {
    const option = document.createElement("option");
    option.textContent = label;
    option.value = value;
    select.appendChild(option);
  }

  function renderSummary() {
    const summaryCards = [
      { labelZh: "Runs indexed", labelEn: "Runs indexed", value: data.summary.total_runs, captionZh: `${data.summary.total_trajectories} replayable seed traces`, captionEn: `${data.summary.total_trajectories} replayable seed traces` },
      { labelZh: "Best fitness", labelEn: "Best fitness", value: fmt(data.summary.best_final_fitness, 1), captionZh: data.summary.best_experiment_id || "n/a", captionEn: data.summary.best_experiment_id || "n/a" },
      { labelZh: "Best mean pops", labelEn: "Best mean pops", value: fmt(data.summary.best_mean_popped, 1), captionZh: data.summary.best_strategy_name || "n/a", captionEn: data.summary.best_strategy_name || "n/a" },
      { labelZh: "Strategies", labelEn: "Strategies", value: data.summary.strategy_count, captionZh: data.summary.strategy_names.join(", "), captionEn: data.summary.strategy_names.join(", ") },
    ];
    els.summaryGrid.innerHTML = "";
    summaryCards.forEach((card) => {
      const node = document.createElement("div");
      node.className = "summary-card";
      node.innerHTML = `<div class="label bilingual">${dualLine(card.labelZh, card.labelEn)}</div><div class="value">${card.value}</div><div class="caption bilingual">${dualLine(card.captionZh, card.captionEn)}</div>`;
      els.summaryGrid.appendChild(node);
    });
    setDualText(els.generatedAt, `Generated ${data.generated_at}`, `Generated ${data.generated_at}`);
    const snapshotId = data.best_config?.experiment_id || data.summary.best_experiment_id || "n/a";
    const snapshotStrategy = data.best_config?.strategy_name || data.summary.best_strategy_name || "n/a";
    setDualText(els.bestStrategyPill, `Snapshot ${snapshotStrategy} (${snapshotId})`, `Snapshot ${snapshotStrategy} (${snapshotId})`);
  }

  function renderConfigOptions(configs) {
    const currentValue = els.researchConfig.value;
    els.researchConfig.innerHTML = "";
    const groupedConfigs = new Map();
    configs.forEach((entry) => {
      const groupLabel = entry.group || "Configs";
      if (!groupedConfigs.has(groupLabel)) {
        groupedConfigs.set(groupLabel, []);
      }
      groupedConfigs.get(groupLabel).push(entry);
    });
    groupedConfigs.forEach((entries, groupLabel) => {
      const optgroup = document.createElement("optgroup");
      optgroup.label = groupLabel;
      entries.forEach((entry) => {
        const option = document.createElement("option");
        option.textContent = entry.label;
        option.value = entry.value;
        option.dataset.mode = entry.mode || "";
        optgroup.appendChild(option);
      });
      els.researchConfig.appendChild(optgroup);
    });
    if (configs.length === 0) {
      makeOption(els.researchConfig, "No configs found", "");
      els.researchConfig.disabled = true;
      return;
    }
    els.researchConfig.disabled = false;
    const preferred = configs.some((entry) => entry.value === currentValue) ? currentValue : configs[0].value;
    els.researchConfig.value = preferred;
    renderSelectedConfigHelp();
  }

  function selectedConfigEntry() {
    return state.availableConfigs.find((entry) => entry.value === els.researchConfig.value) || null;
  }

  function renderSelectedConfigHelp() {
    const entry = selectedConfigEntry();
    if (!entry) {
      els.researchConfigHelp.textContent = "Select a config to see its purpose and recommended use.";
      return;
    }
    els.researchConfigHelp.innerHTML = `
      <strong>${escapeHtml(entry.label)}</strong><br>
      <span>${escapeHtml(entry.summary || "")}</span><br>
      <span>${escapeHtml(entry.recommended_use || "")}</span>
    `;
  }

  function syncResearchButtons(status) {
    const phase = status?.status || "idle";
    const running = !!status?.running;
    const paused = phase === "paused" || phase === "pausing";
    els.researchStart.disabled = running;
    els.researchPause.disabled = !running;
    els.researchStop.disabled = !running;
    setButtonLabel(els.researchStart, "Start", "Start");
    setButtonLabel(els.researchPause, paused ? "Resume" : "Pause", paused ? "Resume" : "Pause");
    setButtonLabel(els.researchStop, "Stop", "Stop");
  }

  function renderResearchStatus(status) {
    state.controlStatus = status;
    const phase = status?.status || "idle";
    const generationLabel = status?.generation_label || "n/a";
    const experimentLabel = status?.current_experiment_id || "n/a";
    const configLabel = status?.config_path || els.researchConfig.value || "n/a";
    const stageContext = resolveStageContext(status);
    const stageLabel = stageContext?.stage_title || titleCase(status?.current_stage || "n/a");
    const strategyLabel = status?.current_strategy_name || "n/a";
    const currentExperimentIndex = Number(status?.current_experiment_index);
    const populationSize = Number(status?.population_size);
    const experimentProgress = Number.isFinite(currentExperimentIndex) && Number.isFinite(populationSize) && populationSize > 0
      ? `${currentExperimentIndex}/${populationSize}`
      : "n/a";
    const completedExperiments = Number(status?.completed_experiments);
    const progressDetail = Number.isFinite(completedExperiments) && Number.isFinite(populationSize) && populationSize > 0
      ? `${completedExperiments} completed / ${populationSize} per generation`
      : Number.isFinite(completedExperiments)
        ? `${completedExperiments} completed`
        : "n/a";
    const message = status?.message || "Idle.";
    const updatedAt = status?.updated_at || status?.started_at || "n/a";
    const bufferedPending = Number(status?.buffered_results_pending);
    const bufferedPersisted = Number(status?.persisted_results_count);
    const flushTotal = Number(status?.buffered_results_flush_total);
    const flushCompleted = Number(status?.buffered_results_flush_completed);
    const flushInProgress = !!status?.buffered_results_flushing;
    const bufferLabel = Number.isFinite(bufferedPending) ? `${bufferedPending}` : "n/a";
    const bufferDetail = Number.isFinite(bufferedPersisted) ? `${bufferedPersisted} persisted` : "n/a";
    const flushLabel = Number.isFinite(flushCompleted) && Number.isFinite(flushTotal) && flushTotal > 0
      ? `${flushCompleted}/${flushTotal}`
      : flushInProgress
        ? "starting"
        : "idle";
    const flushDetail = flushInProgress ? "writing buffered results" : "no active flush";
    const stageObjective = stageContext?.stage_description || "n/a";
    const stageThreshold = percentLabel(stageContext?.success_threshold);
    const recentSuccess = percentLabel(stageContext?.recent_stage_success_rate);
    const nextStage = stageContext?.next_stage_title || "n/a";
    const plannerRationale = stageContext?.planner_rationale || "n/a";
    const promotionDecision = promotionDecisionLabel(stageContext?.promotion_decision);
    const promotionReason = stageContext?.promotion_reason || "n/a";
    const stagePipelineMarkup = stagePipeline(stageContext?.pipeline || []);
    const failureBreakdownMarkup = failureBars(selectedFailureEntries(stageContext));
    const currentExperimentProgressValue = Number.isFinite(currentExperimentIndex) && Number.isFinite(populationSize) && populationSize > 0
      ? currentExperimentIndex / populationSize
      : 0;
    const generationTotal = status?.continuous ? null : Number(status?.total_generations);
    const generationProgressValue = Number.isFinite(generationTotal) && generationTotal > 0
      ? (Number(status?.current_generation || 0) + 1) / generationTotal
      : 0;
    const configuredWorkers = Number(status?.configured_workers);
    const activeWorkers = Number(status?.active_workers);
    const workerMode = String(status?.worker_mode || "serial");
    const syncWorkersFromStatus = !!status?.running || ["starting", "running", "paused", "pausing", "stopping"].includes(phase);
    populateWorkerOptions(configuredWorkers, !syncWorkersFromStatus);
    const configuredPopulationSize = Number(status?.population_size);
    populatePopulationOptions(configuredPopulationSize, !syncWorkersFromStatus);
    const workerLabel = Number.isFinite(activeWorkers) ? `${activeWorkers}` : "n/a";
    const availableWorkerMax = Number(status?.worker_limits?.max);
    const workerDetail = Number.isFinite(configuredWorkers)
      ? `${workerMode} / configured ${configuredWorkers}${Number.isFinite(availableWorkerMax) ? ` / max ${availableWorkerMax}` : ""}`
      : workerMode;
    const phaseMap = {
      idle: ["Idle", "Idle"],
      starting: ["Starting", "Starting"],
      running: ["Running", "Running"],
      pausing: ["Pausing", "Pausing"],
      paused: ["Paused", "Paused"],
      stopping: ["Stopping", "Stopping"],
      stopped: ["Stopped", "Stopped"],
      completed: ["Completed", "Completed"],
      error: ["Error", "Error"],
    };
    const [phaseZh, phaseEn] = phaseMap[phase] || [phase, phase];
    setDualText(els.researchStatusPill, `Research ${phaseZh}`, `Research ${phaseEn}`);
    const summaryZh = phase === "running"
      ? `目前位於 ${stageLabel} stage，第 ${generationLabel} 代，正在評估第 ${experimentProgress} 個實驗。`
      : `${phaseZh}。${message}`;
    const summaryEn = phase === "running"
      ? `Currently in stage ${stageLabel}, generation ${generationLabel}, evaluating experiment ${experimentProgress}.`
      : `${phaseEn}. ${message}`;
    const summaryTextZh = phase === "running"
      ? `目前在 ${stageLabel} 階段，第 ${generationLabel} 代，正在評估第 ${experimentProgress} 個實驗。`
      : phase === "stopping" && flushInProgress
        ? `正在停止研究，並將 RAM 緩衝結果寫入磁碟：${flushLabel}。`
        : `${phaseZh}。${message}`;
    const summaryTextEn = phase === "running"
      ? `Currently in stage ${stageLabel}, generation ${generationLabel}, evaluating experiment ${experimentProgress}.`
      : phase === "stopping" && flushInProgress
        ? `Stopping research and flushing buffered results: ${flushLabel}.`
        : `${phaseEn}. ${message}`;
    els.researchStatusText.innerHTML = `
      <div class="status-summary bilingual">
        ${dualLine(summaryTextZh, summaryTextEn)}
      </div>
      <div class="status-visuals">
        <div class="status-visual-panel">
          <div class="status-visual-title bilingual">${dualLine("升階進度", "Promotion Progress")}</div>
          <div class="metric-stack">
            ${metricBar("本階段成功率", "Stage success rate", stageContext?.recent_stage_success_rate, stageContext?.success_threshold, recentSuccess, stageThreshold, promotionTone(stageContext?.recent_stage_success_rate, stageContext?.success_threshold))}
            ${metricBar("本代實驗進度", "Generation experiment progress", currentExperimentProgressValue, 1.0, experimentProgress, populationSize > 0 ? String(populationSize) : "n/a", "secondary")}
            ${generationTotal && generationTotal > 0 ? metricBar("整體代數進度", "Generation progress", generationProgressValue, 1.0, generationLabel, String(generationTotal), "primary") : ""}
          </div>
        </div>
        <div class="status-visual-panel">
          <div class="status-visual-title bilingual">${dualLine("研究階段流程", "Stage Pipeline")}</div>
          ${stagePipelineMarkup}
        </div>
        <div class="status-visual-panel">
          <div class="status-visual-head">
            <div class="status-visual-title bilingual">${dualLine("主要失敗結構", "Failure Breakdown")}</div>
            <select class="status-inline-select" id="failure-source-select">
              <option value="current_experiment">Current experiment</option>
              <option value="latest_best">Latest best</option>
              <option value="recent_stage_aggregate">Recent stage aggregate</option>
            </select>
          </div>
          ${failureBreakdownMarkup}
        </div>
      </div>
      <div class="status-grid">
        ${statusCard("階段", "Stage", stageLabel)}
        ${statusCard("階段目標", "Stage objective", stageObjective)}
        ${statusCard("過關門檻", "Promotion threshold", stageThreshold)}
        ${statusCard("近期成功率", "Recent success rate", recentSuccess)}
        ${statusCard("升階判定", "Promotion decision", promotionDecision, promotionReason)}
        ${statusCard("下一階段", "Next stage", nextStage)}
        ${statusCard("世代", "Generation", generationLabel)}
        ${statusCard("實驗進度", "Experiment progress", experimentProgress, progressDetail)}
        ${statusCard("策略", "Strategy", strategyLabel)}
        ${statusCard("實驗 ID", "Experiment ID", experimentLabel)}
        ${statusCard("Workers", "Workers", workerLabel, workerDetail)}
        ${statusCard("RAM 緩衝", "RAM buffer", bufferLabel, bufferDetail)}
        ${statusCard("寫檔進度", "Flush progress", flushLabel, flushDetail)}
        ${statusCard("設定檔", "Config", shortPath(configLabel), configLabel)}
        ${statusCard("訊息", "Message", message)}
        ${statusCard("規劃理由", "Planner rationale", plannerRationale)}
        ${statusCard("更新時間", "Updated", updatedAt)}
      </div>
    `;
    const failureSourceSelect = document.getElementById("failure-source-select");
    if (failureSourceSelect) {
      failureSourceSelect.value = state.failureSource;
      failureSourceSelect.onchange = (event) => {
        state.failureSource = event.target.value || "current_experiment";
        renderResearchStatus(state.controlStatus);
      };
    }
    els.researchLog.textContent = status?.log_tail || "No research log yet.\\nNo research log yet.";
    syncResearchButtons(status);
  }

  async function refreshResearchStatus() {
    try {
      const [statusPayload, configPayload] = await Promise.all([
        apiGet("/api/status"),
        apiGet("/api/configs"),
      ]);
      const configs = configPayload.configs || [];
      const configsChanged = JSON.stringify(configs) !== JSON.stringify(state.availableConfigs);
      state.availableConfigs = configs;
      if (configsChanged) {
        renderConfigOptions(configs);
      }
      renderResearchStatus(statusPayload);
    } catch (error) {
      setDualText(els.researchStatusPill, "Control server offline", "Control server offline");
      els.researchStatusText.innerHTML = `
        <div class="status-summary bilingual">
          ${dualLine(
            `請先啟動本機控制伺服器，才能使用研究控制功能。${error.message}`,
            `Start the local dashboard server to enable runtime controls. ${error.message}`
          )}
        </div>
      `;
      els.researchLog.textContent = "No research log available.\\nNo research log available.";
      els.researchStart.disabled = true;
      els.researchPause.disabled = true;
      els.researchStop.disabled = true;
    }
  }

  function selectedGenerationTrend() {
    const trend = Array.isArray(data.generation_trend) ? data.generation_trend : [];
    const rawWindow = state.generationWindow || "all";
    if (rawWindow === "all") {
      return trend;
    }
    const limit = Number(rawWindow);
    if (!Number.isFinite(limit) || limit <= 0) {
      return trend;
    }
    return trend.slice(Math.max(0, trend.length - limit));
  }

  function renderGenerationStats(visibleTrend, fullTrend) {
    const visibleCount = visibleTrend.length;
    const totalCount = fullTrend.length;
    const firstGeneration = visibleCount ? visibleTrend[0].generation : "n/a";
    const lastGeneration = visibleCount ? visibleTrend[visibleCount - 1].generation : "n/a";
    const scaleLabel = state.generationScale === "log" ? "log" : "linear";
    els.generationStats.innerHTML = dualLine(
      `顯示 ${visibleCount} / ${totalCount} 代 · 範圍 ${firstGeneration}-${lastGeneration} · ${scaleLabel} 尺度`,
      `Showing ${visibleCount} / ${totalCount} generations · range ${firstGeneration}-${lastGeneration} · ${scaleLabel} scale`
    );
  }

  function renderGenerationChart() {
    const fullTrend = Array.isArray(data.generation_trend) ? data.generation_trend : [];
    const trend = selectedGenerationTrend();
    const useLogScale = state.generationScale === "log" && trend.every((entry) => Number(entry.best_final_fitness) > 0);
    renderGenerationStats(trend, fullTrend);
    const trace = {
      x: trend.map((entry) => entry.generation),
      y: trend.map((entry) => entry.best_final_fitness),
      text: trend.map((entry) => `${entry.strategy_name} (${entry.experiment_id})`),
      type: "scatter",
      mode: "lines+markers",
      line: { color: "#1d6b69", width: 3 },
      marker: { color: "#b25424", size: 8 },
    };
    Plotly.newPlot("generation-chart", [trace], {
      margin: { l: 48, r: 12, t: 12, b: 48 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: themeColor("--plot-bg", "rgba(255,255,255,0.55)"),
      xaxis: { title: "Generation" },
      yaxis: { title: "Best final fitness", type: useLogScale ? "log" : "linear" },
    }, { displayModeBar: false, responsive: true });
  }

  function renderResearchProgressChartLegacy() {
    const progress = Array.isArray(data.research_progress) ? data.research_progress : [];
    if (!progress.length) {
      els.progressSummary.innerHTML = dualLine("尚無研究進展資料", "No research progress data yet");
      els.progressSummary.innerHTML = dualLine("尚無實驗演進資料", "No experiment evolution data yet");
      Plotly.newPlot("progress-chart", [], {
        margin: { l: 48, r: 12, t: 12, b: 48 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: themeColor("--plot-bg", "rgba(255,255,255,0.55)"),
      }, { displayModeBar: false, responsive: true });
      return;
    }
    const kept = progress.filter((entry) => entry.kept);
    const discarded = progress.filter((entry) => !entry.kept);
    const best = progress.map((entry) => entry.running_best);
    els.progressSummary.innerHTML = dualLine(
      `共 ${progress.length} 次實驗，保留 ${kept.length} 次有效提升`,
      `${progress.length} experiments, ${kept.length} kept improvements`
    );
    const traces = [
      {
        x: discarded.map((entry) => entry.index),
        y: discarded.map((entry) => entry.final_fitness),
        text: discarded.map((entry) => `${entry.strategy_name} (${entry.experiment_id})`),
        type: "scatter",
        mode: "markers",
        name: "Discarded",
        marker: { color: "rgba(140,140,140,0.45)", size: 8 },
      },
      {
        x: kept.map((entry) => entry.index),
        y: kept.map((entry) => entry.final_fitness),
        text: kept.map((entry) => entry.annotation || `${entry.strategy_name} (${entry.experiment_id})`),
        type: "scatter",
        mode: "markers+text",
        name: "Kept",
        textposition: "top right",
        textfont: { color: themeColor("--accent", "#1d6b69"), size: 12 },
        marker: { color: "#2fbf71", size: 11, line: { color: "#0d5d35", width: 1.5 } },
      },
      {
        x: progress.map((entry) => entry.index),
        y: best,
        text: progress.map((entry) => entry.annotation || ""),
        type: "scatter",
        mode: "lines",
        name: "Running best",
        line: { color: "#5dc98f", width: 4, shape: "hv" },
      },
    ];
    Plotly.newPlot("progress-chart", traces, {
      margin: { l: 56, r: 12, t: 12, b: 48 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: themeColor("--plot-bg", "rgba(255,255,255,0.55)"),
      xaxis: { title: "Experiment #" },
      yaxis: { title: "Final fitness (higher is better)" },
      legend: { orientation: "h", x: 1, xanchor: "right", y: 1.12 },
    }, { displayModeBar: false, responsive: true });
  }

  function refreshThemeSensitiveViews() {
    renderGenerationChart();
    renderResearchProgressChart();
    if (state.selectedRunId) {
      renderSelectedReplay();
    }
  }

  function renderResearchProgressChart() {
    const progress = Array.isArray(data.research_progress) ? data.research_progress : [];
    if (!progress.length) {
      els.progressSummary.innerHTML = dualLine("撠?弦?脣?鞈?", "No research progress data yet");
      Plotly.newPlot("progress-chart", [], {
        margin: { l: 48, r: 12, t: 12, b: 48 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: themeColor("--plot-bg", "rgba(255,255,255,0.55)"),
      }, { displayModeBar: false, responsive: true });
      return;
    }
    const maxExperiments = state.progressWindow === "all" ? null : Number(state.progressWindow);
    const visibleProgress = Number.isFinite(maxExperiments) && maxExperiments > 0 ? progress.slice(-maxExperiments) : progress;
    const metricConfig = state.progressMetric === "mean_popped"
      ? {
          key: "mean_popped",
          keptKey: "kept_popped",
          runningBestKey: "running_best_popped",
          annotationKey: "annotation_popped",
          summaryLabelZh: "平均打到數",
          summaryLabelEn: "mean popped",
          axisTitle: "Mean popped (higher is better)",
        }
      : {
          key: "final_fitness",
          keptKey: "kept_fitness",
          runningBestKey: "running_best_fitness",
          annotationKey: "annotation_fitness",
          summaryLabelZh: "最終 fitness",
          summaryLabelEn: "final fitness",
          axisTitle: "Final fitness (higher is better)",
        };
    const kept = visibleProgress.filter((entry) => entry[metricConfig.keptKey]);
    const discarded = visibleProgress.filter((entry) => !entry[metricConfig.keptKey]);
    const best = visibleProgress.map((entry) => entry[metricConfig.runningBestKey]);
    const windowLabelZh = state.progressWindow === "all" ? "全部" : `最近 ${state.progressWindow} 次`;
    const windowLabelEn = state.progressWindow === "all" ? "all experiments" : `last ${state.progressWindow}`;
    els.progressSummary.innerHTML = dualLine(
      `${windowLabelZh} / 顯示 ${visibleProgress.length} 次，保留 ${kept.length} 次以 ${metricConfig.summaryLabelZh} 計算的提升`,
      `Showing ${windowLabelEn}: ${visibleProgress.length} experiments, ${kept.length} kept improvements by ${metricConfig.summaryLabelEn}`
    );
    const traces = [
      {
        x: discarded.map((entry) => entry.index),
        y: discarded.map((entry) => entry[metricConfig.key]),
        text: discarded.map((entry) => `${entry.strategy_name} (${entry.experiment_id})`),
        type: "scatter",
        mode: "markers",
        name: "Discarded",
        marker: { color: "rgba(140,140,140,0.45)", size: 8 },
      },
      {
        x: kept.map((entry) => entry.index),
        y: kept.map((entry) => entry[metricConfig.key]),
        text: kept.map((entry) => entry[metricConfig.annotationKey] || `${entry.strategy_name} (${entry.experiment_id})`),
        type: "scatter",
        mode: "markers+text",
        name: "Kept",
        textposition: "top right",
        textfont: { color: themeColor("--accent", "#1d6b69"), size: 12 },
        marker: { color: "#2fbf71", size: 11, line: { color: "#0d5d35", width: 1.5 } },
      },
      {
        x: visibleProgress.map((entry) => entry.index),
        y: best,
        text: visibleProgress.map((entry) => entry[metricConfig.annotationKey] || ""),
        type: "scatter",
        mode: "lines",
        name: "Running best",
        line: { color: "#5dc98f", width: 4, shape: "hv" },
      },
    ];
    Plotly.newPlot("progress-chart", traces, {
      margin: { l: 56, r: 12, t: 12, b: 48 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: themeColor("--plot-bg", "rgba(255,255,255,0.55)"),
      xaxis: { title: "Experiment #" },
      yaxis: { title: metricConfig.axisTitle },
      legend: { orientation: "h", x: 1, xanchor: "right", y: 1.12 },
    }, { displayModeBar: false, responsive: true });
  }

  function renderResearchProgressChart() {
    const progress = Array.isArray(data.research_progress) ? data.research_progress : [];
    if (!progress.length) {
      els.progressSummary.innerHTML = dualLine("尚無實驗演進資料", "No experiment evolution data yet");
      Plotly.newPlot("progress-chart", [], {
        margin: { l: 48, r: 12, t: 12, b: 48 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: themeColor("--plot-bg", "rgba(255,255,255,0.55)"),
      }, { displayModeBar: false, responsive: true });
      return;
    }
    const maxExperiments = state.progressWindow === "all" ? null : Number(state.progressWindow);
    const visibleProgress = Number.isFinite(maxExperiments) && maxExperiments > 0 ? progress.slice(-maxExperiments) : progress;
    const metricConfig = state.progressMetric === "mean_popped"
      ? {
          key: "mean_popped",
          keptKey: "kept_popped",
          runningBestKey: "running_best_popped",
          annotationKey: "annotation_popped",
          summaryLabelZh: "平均打到數",
          summaryLabelEn: "mean popped",
          axisTitle: "Mean popped (higher is better)",
        }
      : {
          key: "final_fitness",
          keptKey: "kept_fitness",
          runningBestKey: "running_best_fitness",
          annotationKey: "annotation_fitness",
          summaryLabelZh: "最終 fitness",
          summaryLabelEn: "final fitness",
          axisTitle: "Final fitness (higher is better)",
        };
    const kept = visibleProgress.filter((entry) => entry[metricConfig.keptKey]);
    const discarded = visibleProgress.filter((entry) => !entry[metricConfig.keptKey]);
    const best = visibleProgress.map((entry) => entry[metricConfig.runningBestKey]);
    const windowLabelZh = state.progressWindow === "all" ? "全部" : `最近 ${state.progressWindow} 次`;
    const windowLabelEn = state.progressWindow === "all" ? "all experiments" : `last ${state.progressWindow}`;
    els.progressSummary.innerHTML = dualLine(
      `${windowLabelZh} / 顯示 ${visibleProgress.length} 次，保留 ${kept.length} 次以 ${metricConfig.summaryLabelZh} 計算的提升`,
      `Showing ${windowLabelEn}: ${visibleProgress.length} experiments, ${kept.length} kept improvements by ${metricConfig.summaryLabelEn}`
    );
    const traces = [
      {
        x: discarded.map((entry) => entry.index),
        y: discarded.map((entry) => entry[metricConfig.key]),
        text: discarded.map((entry) => `${entry.strategy_name} (${entry.experiment_id})`),
        type: "scatter",
        mode: "markers",
        name: "Discarded",
        marker: { color: "rgba(140,140,140,0.45)", size: 8 },
      },
      {
        x: kept.map((entry) => entry.index),
        y: kept.map((entry) => entry[metricConfig.key]),
        text: kept.map((entry) => entry[metricConfig.annotationKey] || `${entry.strategy_name} (${entry.experiment_id})`),
        type: "scatter",
        mode: "markers+text",
        name: "Kept",
        textposition: "top right",
        textfont: { color: themeColor("--accent", "#1d6b69"), size: 12 },
        marker: { color: "#2fbf71", size: 11, line: { color: "#0d5d35", width: 1.5 } },
      },
      {
        x: visibleProgress.map((entry) => entry.index),
        y: best,
        text: visibleProgress.map((entry) => entry[metricConfig.annotationKey] || ""),
        type: "scatter",
        mode: "lines",
        name: "Running best",
        line: { color: "#5dc98f", width: 4, shape: "hv" },
      },
    ];
    Plotly.newPlot("progress-chart", traces, {
      margin: { l: 56, r: 12, t: 12, b: 48 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: themeColor("--plot-bg", "rgba(255,255,255,0.55)"),
      xaxis: { title: "Experiment #" },
      yaxis: { title: metricConfig.axisTitle },
      legend: { orientation: "h", x: 1, xanchor: "right", y: 1.12 },
    }, { displayModeBar: false, responsive: true });
  }

  function populateFilters() {
    const strategies = ["all", ...new Set(data.runs.map((run) => run.strategy_name))];
    const adapters = ["all", ...new Set(data.runs.map((run) => run.adapter || "unknown"))];
    els.strategyFilter.innerHTML = "";
    els.adapterFilter.innerHTML = "";
    strategies.forEach((strategy) => makeOption(els.strategyFilter, strategy === "all" ? "All" : strategy, strategy));
    adapters.forEach((adapter) => makeOption(els.adapterFilter, adapter === "all" ? "All" : adapter, adapter));
  }

  function filteredRuns() {
    const strategy = els.strategyFilter.value || "all";
    const adapter = els.adapterFilter.value || "all";
    const query = (els.searchFilter.value || "").trim().toLowerCase();
    return data.runs.filter((run) => {
      const strategyOk = strategy === "all" || run.strategy_name === strategy;
      const adapterOk = adapter === "all" || (run.adapter || "unknown") === adapter;
      const haystack = `${run.experiment_id} ${run.strategy_name} ${run.note || ""}`.toLowerCase();
      const queryOk = !query || haystack.includes(query);
      return strategyOk && adapterOk && queryOk;
    });
  }

  function replayPointCount(trajectory) {
    return Number(trajectory?.points || 0);
  }

  function runHasAnimatedReplay(run) {
    return (run?.trajectory_index || []).some((entry) => replayPointCount(entry) > 1);
  }

  function preferredRun(runs) {
    return runs.find((run) => runHasAnimatedReplay(run)) || runs[0] || null;
  }

  function preferredSeedEntry(run) {
    if (!run || !run.trajectory_index || run.trajectory_index.length === 0) {
      return null;
    }
    const ranked = [...run.trajectory_index].sort((left, right) => {
      const leftAnimated = replayPointCount(left) > 1 ? 1 : 0;
      const rightAnimated = replayPointCount(right) > 1 ? 1 : 0;
      if (rightAnimated !== leftAnimated) {
        return rightAnimated - leftAnimated;
      }
      const leftPopped = Number(left?.popped || 0);
      const rightPopped = Number(right?.popped || 0);
      if (rightPopped !== leftPopped) {
        return rightPopped - leftPopped;
      }
      const leftScore = Number(left?.score || 0);
      const rightScore = Number(right?.score || 0);
      if (rightScore !== leftScore) {
        return rightScore - leftScore;
      }
      const leftFrames = replayPointCount(left);
      const rightFrames = replayPointCount(right);
      if (rightFrames !== leftFrames) {
        return rightFrames - leftFrames;
      }
      return Number(left?.seed || 0) - Number(right?.seed || 0);
    });
    return ranked[0] || null;
  }

  function updateRunCollections() {
    state.runs = filteredRuns();
    els.runTableBody.innerHTML = "";
    els.runSelect.innerHTML = "";
    state.runs.forEach((run) => {
      const row = document.createElement("tr");
      row.dataset.experimentId = run.experiment_id;
      row.innerHTML = `
        <td>${run.experiment_id}</td>
        <td>${formatTimestamp(run.completed_at)}</td>
        <td>${formatDurationSeconds(run.summary.mean_duration)}</td>
        <td>${run.strategy_name}</td>
        <td>${run.adapter || "unknown"}</td>
        <td>${fmt(run.summary.final_fitness, 1)}</td>
        <td>${fmt(run.summary.mean_popped, 1)}</td>
      `;
      row.addEventListener("click", () => selectRun(run.experiment_id));
      els.runTableBody.appendChild(row);
      makeOption(
        els.runSelect,
        `${run.experiment_id} | ${formatTimestamp(run.completed_at)} | ${formatDurationSeconds(run.summary.mean_duration)} | ${run.strategy_name} | fitness ${fmt(run.summary.final_fitness, 1)}`,
        run.experiment_id,
      );
    });
    const currentExists = state.runs.some((run) => run.experiment_id === state.selectedRunId);
    if (!currentExists && state.runs.length > 0) {
      state.selectedRunId = preferredRun(state.runs)?.experiment_id || state.runs[0].experiment_id;
    }
    els.runSelect.value = state.selectedRunId || "";
    syncSelectedRow();
    populateSeedOptions();
  }

  function syncSelectedRow() {
    Array.from(els.runTableBody.querySelectorAll("tr")).forEach((row) => {
      row.classList.toggle("selected", row.dataset.experimentId === state.selectedRunId);
    });
  }

  function currentRun() {
    return data.runs.find((run) => run.experiment_id === state.selectedRunId) || null;
  }

  function populateSeedOptions(forcePreferred = false) {
    const run = currentRun();
    els.seedSelect.innerHTML = "";
    if (!run) {
      return;
    }
    run.trajectory_index.forEach((trajectory) => {
      const replayLabel = replayPointCount(trajectory) > 1 ? `${replayPointCount(trajectory)} frames` : "static";
      makeOption(
        els.seedSelect,
        `seed ${trajectory.seed} | pops ${trajectory.popped} | score ${fmt(trajectory.score, 1)} | ${replayLabel}`,
        String(trajectory.seed),
      );
    });
    const preferredSeed = preferredSeedEntry(run)?.seed ?? null;
    const currentSeedExists = run.trajectory_index.some((entry) => String(entry.seed) === String(state.selectedSeed));
    if (forcePreferred || !currentSeedExists) {
      state.selectedSeed = preferredSeed;
    }
    els.seedSelect.value = state.selectedSeed !== null ? String(state.selectedSeed) : "";
  }

  function renderRunDetails() {
    const run = currentRun();
    if (!run) {
      return;
    }
    const detailCards = [
      ["Strategy", "Strategy", run.strategy_name],
      ["完成時間", "Completed", formatTimestamp(run.completed_at)],
      ["耗時", "Duration", formatDurationSeconds(run.summary.mean_duration)],
      ["Adapter", "Adapter", run.adapter || "unknown"],
      ["Generation", "Generation", run.generation ?? "n/a"],
      ["Fitness", "Fitness", fmt(run.summary.final_fitness, 1)],
      ["Mean pops", "Mean pops", fmt(run.summary.mean_popped, 2)],
      ["Mean score", "Mean score", fmt(run.summary.mean_score, 1)],
      ["Wind sensitivity", "Wind sensitivity", fmt(run.summary.wind_sensitivity, 2)],
      ["Switches", "Switches", fmt(run.summary.mean_target_switches, 2)],
      ["Dominant failure", "Dominant failure", run.failure_report.dominant_failure || "none"],
    ];
    els.runDetails.innerHTML = detailCards
      .map(([labelZh, labelEn, value]) => `<div class="detail-card"><span class="label bilingual">${dualLine(labelZh, labelEn)}</span><span class="value">${escapeHtml(value)}</span></div>`)
      .join("");
    els.failureNotes.innerHTML = "";
    (run.failure_report.notes || []).forEach((note) => {
      const item = document.createElement("li");
      item.textContent = note;
      els.failureNotes.appendChild(item);
    });
    if (!run.failure_report.notes || run.failure_report.notes.length === 0) {
      els.failureNotes.innerHTML = "<li>No failure notes recorded.<br>No failure notes recorded.</li>";
    }
    els.paramsViewer.textContent = JSON.stringify(run.params, null, 2);
    renderCrossValidation(run.experiment_id);
  }

  function renderCrossValidation(experimentId) {
    const record = data.cross_validations.find((entry) => entry.best_experiment_id === experimentId);
    els.crossValidation.innerHTML = "";
    if (!record) {
      els.crossValidation.innerHTML = dualLine("No cross-validation bundle recorded for this experiment.", "No cross-validation bundle recorded for this experiment.");
      return;
    }
    record.results.forEach((result) => {
      const node = document.createElement("div");
      node.className = "cross-validation-card";
      node.innerHTML = `
        <strong>${escapeHtml(result.adapter)}</strong>
        <div class="bilingual">${dualLine(`fitness ${fmt(result.summary.final_fitness, 1)} | mean pops ${fmt(result.summary.mean_popped, 2)}`, `fitness ${fmt(result.summary.final_fitness, 1)} | mean pops ${fmt(result.summary.mean_popped, 2)}`)}</div>
        <div class="bilingual">${dualLine(`dominant failure: ${result.failure_report.dominant_failure || "none"}`, `dominant failure: ${result.failure_report.dominant_failure || "none"}`)}</div>
      `;
      els.crossValidation.appendChild(node);
    });
  }

  function renderReports() {
    els.reportList.innerHTML = "";
    data.report_previews.forEach((report) => {
      const node = document.createElement("div");
      node.className = "report-card";
      node.innerHTML = `<h3>${report.name}</h3><p>${report.preview}</p>`;
      els.reportList.appendChild(node);
    });
  }

  function trajectoryCacheKey(experimentId, seed) {
    return `${experimentId}:${seed}`;
  }

  function computeReplayBounds(points) {
    const xs = [];
    const ys = [];
    const zs = [];
    points.forEach((point) => {
      xs.push(point.rocket_position.x);
      ys.push(point.rocket_position.y);
      zs.push(point.rocket_position.z);
      (point.balloons || []).forEach((balloon) => {
        xs.push(balloon.position.x);
        ys.push(balloon.position.y);
        zs.push(balloon.position.z);
      });
    });
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const minZ = Math.min(...zs);
    const maxZ = Math.max(...zs);
    const center = {
      x: (minX + maxX) / 2,
      y: (minY + maxY) / 2,
      z: (minZ + maxZ) / 2,
    };
    const halfRange = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 40) / 2 * 1.08;
    return { center, halfRange };
  }

  function sceneLayoutFromBounds(bounds, scale) {
    const span = Math.max(20, bounds.halfRange * scale);
    return {
      xaxis: { title: "X (m)", range: [bounds.center.x - span, bounds.center.x + span] },
      yaxis: { title: "Y (m)", range: [bounds.center.y - span, bounds.center.y + span] },
      zaxis: { title: "Altitude (m)", range: [bounds.center.z - span, bounds.center.z + span] },
      aspectmode: "cube",
      camera: { eye: { x: 1.25, y: -1.65, z: 0.85 } },
    };
  }

  function applyReplayScale() {
    if (!state.replayBounds) {
      return;
    }
    const scene = sceneLayoutFromBounds(state.replayBounds, state.scale);
    Plotly.relayout("replay-plot", {
      "scene.xaxis.range": scene.xaxis.range,
      "scene.yaxis.range": scene.yaxis.range,
      "scene.zaxis.range": scene.zaxis.range,
      "scene.aspectmode": scene.aspectmode,
    });
    els.scaleReadout.textContent = `${fmt(state.scale, 1)}x`;
  }

  function ensureTrajectoryLoaded(experimentId, seed) {
    const run = data.runs.find((entry) => entry.experiment_id === experimentId);
    const trajectoryMeta = run?.trajectory_index.find((entry) => String(entry.seed) === String(seed));
    if (!trajectoryMeta) {
      return Promise.reject(new Error("Trajectory metadata not found."));
    }
    const cacheKey = trajectoryCacheKey(experimentId, seed);
    if (trajectoryCache[cacheKey]) {
      return Promise.resolve(trajectoryCache[cacheKey]);
    }
    return new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = trajectoryMeta.file;
      script.onload = () => resolve(trajectoryCache[cacheKey]);
      script.onerror = () => reject(new Error(`Failed to load ${trajectoryMeta.file}`));
      document.body.appendChild(script);
    });
  }

  function clearReplayPlots(messageZh = "尚無可用回放", messageEn = "No replay available") {
    stopPlayback();
    state.replayBounds = null;
    state.frameIndex = 0;
    Plotly.purge("replay-plot");
    Plotly.purge("timeline-chart");
    Plotly.newPlot("replay-plot", [], {
      margin: { l: 0, r: 0, t: 0, b: 0 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      annotations: [
        {
          text: `${escapeHtml(messageZh)}<br>${escapeHtml(messageEn)}`,
          x: 0.5,
          y: 0.5,
          xref: "paper",
          yref: "paper",
          showarrow: false,
          font: { size: 16, color: themeColor("--muted", "#6b7280") },
        },
      ],
    }, { displayModeBar: false, responsive: true });
    Plotly.newPlot("timeline-chart", [], {
      margin: { l: 40, r: 10, t: 10, b: 40 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: themeColor("--plot-bg", "rgba(255,255,255,0.55)"),
      annotations: [
        {
          text: `${escapeHtml(messageZh)}<br>${escapeHtml(messageEn)}`,
          x: 0.5,
          y: 0.5,
          xref: "paper",
          yref: "paper",
          showarrow: false,
          font: { size: 14, color: themeColor("--muted", "#6b7280") },
        },
      ],
    }, { displayModeBar: false, responsive: true });
    els.frameSlider.min = "0";
    els.frameSlider.max = "0";
    els.frameSlider.value = "0";
    els.frameReadout.innerHTML = dualLine("Frame 0 / 0", "Frame 0 / 0");
    setButtonLabel(els.playToggle, "Static", "Static");
    els.playToggle.disabled = true;
    els.resetReplay.onclick = null;
    els.frameSlider.oninput = null;
  }

  function buildReplayMarkers(points) {
    const popX = [];
    const popY = [];
    const popZ = [];
    const switchX = [];
    const switchY = [];
    const switchZ = [];
    let previousPopped = 0;
    let previousTarget = null;
    points.forEach((point) => {
      if ((point.popped || 0) > previousPopped) {
        popX.push(point.rocket_position.x);
        popY.push(point.rocket_position.y);
        popZ.push(point.rocket_position.z);
      }
      if (previousTarget && point.target_id && point.target_id !== previousTarget) {
        switchX.push(point.rocket_position.x);
        switchY.push(point.rocket_position.y);
        switchZ.push(point.rocket_position.z);
      }
      previousPopped = point.popped || previousPopped;
      previousTarget = point.target_id || previousTarget;
    });
    return { popX, popY, popZ, switchX, switchY, switchZ };
  }

  function updateReplayFrame(points, frameIndex) {
    if (!points.length) {
      return;
    }
    const point = points[frameIndex];
    const balloons = point.balloons || [];
    const freeBalloons = balloons.filter((entry) => !entry.is_target);
    const target = balloons.find((entry) => entry.is_target) || null;
    Plotly.restyle("replay-plot", {
      x: [[point.rocket_position.x]],
      y: [[point.rocket_position.y]],
      z: [[point.rocket_position.z]],
      text: [[`t=${fmt(point.time_s, 2)}s<br>target=${point.target_id || "none"}<br>popped=${point.popped}${point.replay_placeholder ? "<br>placeholder frame" : ""}`]],
    }, [5]);
    Plotly.restyle("replay-plot", {
      x: [freeBalloons.map((entry) => entry.position.x)],
      y: [freeBalloons.map((entry) => entry.position.y)],
      z: [freeBalloons.map((entry) => entry.position.z)],
      text: [freeBalloons.map((entry) => `${entry.balloon_id}<br>${fmt(entry.distance_to_rocket_m, 1)} m`)],
    }, [3]);
    Plotly.restyle("replay-plot", {
      x: [target ? [target.position.x] : []],
      y: [target ? [target.position.y] : []],
      z: [target ? [target.position.z] : []],
      text: [target ? [`${target.balloon_id}<br>target`] : []],
    }, [4]);
    Plotly.restyle("timeline-chart", { x: [[point.time_s, point.time_s]], y: [[0, Math.max(...points.map((entry) => entry.released || 0), 1)]] }, [2]);
    const replayMode = points.length > 1 ? "animated" : "static";
    els.frameReadout.innerHTML = dualLine(
      `Frame ${frameIndex + 1} / ${points.length} · ${replayMode} · visible balloons ${point.visible_balloon_count || balloons.length}${point.replay_placeholder ? " · placeholder" : ""}`,
      `Frame ${frameIndex + 1} / ${points.length} · ${replayMode} · visible balloons ${point.visible_balloon_count || balloons.length}${point.replay_placeholder ? " · placeholder" : ""}`
    );
  }

  function renderReplay(payload) {
    const points = payload.trajectory || [];
    if (points.length === 0) {
      Plotly.purge("replay-plot");
      Plotly.purge("timeline-chart");
      els.frameReadout.innerHTML = dualLine("Frame 0 / 0", "Frame 0 / 0");
      return;
    }
    const x = points.map((point) => point.rocket_position.x);
    const y = points.map((point) => point.rocket_position.y);
    const z = points.map((point) => point.rocket_position.z);
    const time = points.map((point) => point.time_s);
    const popped = points.map((point) => point.popped || 0);
    const released = points.map((point) => point.released || 0);
    const markers = buildReplayMarkers(points);
    const initialBalloons = points[0].balloons || [];
    const initialFreeBalloons = initialBalloons.filter((entry) => !entry.is_target);
    const initialTarget = initialBalloons.find((entry) => entry.is_target) || null;
    state.replayBounds = computeReplayBounds(points);
    state.frameIndex = 0;
    els.frameSlider.min = "0";
    els.frameSlider.max = String(points.length - 1);
    els.frameSlider.value = "0";
    els.scaleSlider.value = String(state.scale);
    els.scaleReadout.textContent = `${fmt(state.scale, 1)}x`;

    const traces = [
      {
        x,
        y,
        z,
        type: "scatter3d",
        mode: "lines",
        line: { color: "#1d6b69", width: 6 },
        name: "Rocket path",
      },
      {
        x: markers.popX,
        y: markers.popY,
        z: markers.popZ,
        type: "scatter3d",
        mode: "markers",
        marker: { color: "#2ca24d", size: 5, symbol: "diamond" },
        name: "Pop events",
      },
      {
        x: markers.switchX,
        y: markers.switchY,
        z: markers.switchZ,
        type: "scatter3d",
        mode: "markers",
        marker: { color: "#b25424", size: 4 },
        name: "Target switch",
      },
      {
        x: initialFreeBalloons.map((entry) => entry.position.x),
        y: initialFreeBalloons.map((entry) => entry.position.y),
        z: initialFreeBalloons.map((entry) => entry.position.z),
        type: "scatter3d",
        mode: "markers",
        marker: { color: "#94b447", size: 3, opacity: 0.72 },
        text: initialFreeBalloons.map((entry) => `${entry.balloon_id}<br>${fmt(entry.distance_to_rocket_m, 1)} m`),
        hoverinfo: "text",
        name: "Balloons",
      },
      {
        x: initialTarget ? [initialTarget.position.x] : [],
        y: initialTarget ? [initialTarget.position.y] : [],
        z: initialTarget ? [initialTarget.position.z] : [],
        type: "scatter3d",
        mode: "markers",
        marker: { color: "#d93a2f", size: 7, symbol: "diamond" },
        text: initialTarget ? [`${initialTarget.balloon_id}<br>target`] : [],
        hoverinfo: "text",
        name: "Current target",
      },
      {
        x: [x[0]],
        y: [y[0]],
        z: [z[0]],
        type: "scatter3d",
        mode: "markers",
        marker: { color: "#111111", size: 7 },
        text: [`t=${fmt(time[0], 2)}s`],
        hoverinfo: "text",
        name: "Current frame",
      },
    ];
    Plotly.newPlot("replay-plot", traces, {
      margin: { l: 0, r: 0, t: 0, b: 0 },
      scene: sceneLayoutFromBounds(state.replayBounds, state.scale),
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      legend: { orientation: "h" },
    }, { responsive: true });

    Plotly.newPlot("timeline-chart", [
      { x: time, y: released, type: "scatter", mode: "lines", line: { color: "#90955a", width: 2 }, name: "Released" },
      { x: time, y: popped, type: "scatter", mode: "lines", line: { color: "#1d6b69", width: 3 }, name: "Popped" },
      { x: [time[0], time[0]], y: [0, Math.max(...released, 1)], type: "scatter", mode: "lines", line: { color: "#111111", dash: "dash" }, name: "Cursor" },
    ], {
      margin: { l: 40, r: 10, t: 10, b: 40 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: themeColor("--plot-bg", "rgba(255,255,255,0.55)"),
      xaxis: { title: "Time (s)" },
      yaxis: { title: "Count" },
      legend: { orientation: "h" },
    }, { displayModeBar: false, responsive: true });

    updateReplayFrame(points, 0);
  }

  function stopPlayback() {
    if (state.timer) {
      window.clearInterval(state.timer);
      state.timer = null;
    }
    setButtonLabel(els.playToggle, "Play", "Play");
  }

  function startPlayback(points) {
    stopPlayback();
    const intervalMs = Math.max(30, Math.round(180 / state.speed));
    state.timer = window.setInterval(() => {
      state.frameIndex += 1;
      if (state.frameIndex >= points.length) {
        state.frameIndex = points.length - 1;
        stopPlayback();
      }
      els.frameSlider.value = String(state.frameIndex);
      updateReplayFrame(points, state.frameIndex);
    }, intervalMs);
    setButtonLabel(els.playToggle, "Pause", "Pause");
  }

  async function selectRun(experimentId) {
    stopPlayback();
    state.selectedRunId = experimentId;
    state.selectedSeed = null;
    syncSelectedRow();
    els.runSelect.value = experimentId;
    populateSeedOptions(true);
    renderRunDetails();
    try {
      await renderSelectedReplay();
    } catch (error) {
      clearReplayPlots("回放載入失敗", `Replay load failed: ${error.message || error}`);
    }
  }

  async function renderSelectedReplay() {
    const run = currentRun();
    if (!run || state.selectedSeed === null) {
      clearReplayPlots("尚無可用回放", "No replay available");
      return;
    }
    const payload = await ensureTrajectoryLoaded(run.experiment_id, state.selectedSeed);
    const points = payload.trajectory || [];
    renderReplay(payload);
    const frameCount = points.length;
    els.playToggle.disabled = frameCount <= 1;
    setButtonLabel(els.playToggle, frameCount <= 1 ? "Static" : "Play", frameCount <= 1 ? "Static" : "Play");
    els.playToggle.onclick = () => {
      if (state.timer) {
        stopPlayback();
      } else {
        startPlayback(points);
      }
    };
    els.resetReplay.onclick = () => {
      stopPlayback();
      state.frameIndex = 0;
      els.frameSlider.value = "0";
      updateReplayFrame(points, 0);
    };
    els.frameSlider.oninput = (event) => {
      stopPlayback();
      state.frameIndex = Number(event.target.value);
      updateReplayFrame(points, state.frameIndex);
    };
  }

  function bindEvents() {
    els.themeSelect.addEventListener("change", (event) => {
      applyTheme(event.target.value);
      refreshThemeSensitiveViews();
    });
    els.generationWindow.addEventListener("change", (event) => {
      state.generationWindow = event.target.value || "all";
      renderGenerationChart();
    });
    els.generationScale.addEventListener("change", (event) => {
      state.generationScale = event.target.value === "log" ? "log" : "linear";
      renderGenerationChart();
    });
    els.progressWindow.addEventListener("change", (event) => {
      state.progressWindow = event.target.value || "all";
      renderResearchProgressChart();
    });
    els.progressMetric.addEventListener("change", (event) => {
      state.progressMetric = event.target.value === "mean_popped" ? "mean_popped" : "final_fitness";
      renderResearchProgressChart();
    });
    els.researchConfig.addEventListener("change", () => {
      renderSelectedConfigHelp();
    });
    els.researchStart.addEventListener("click", async () => {
      try {
        const payload = await apiPost("/api/start", {
          config: els.researchConfig.value,
          parallel_workers: Number(els.workerCount?.value || 1),
          population_size: Number(els.populationSize?.value || 6),
        });
        renderResearchStatus(payload);
      } catch (error) {
        els.researchStatusText.innerHTML = `
          <div class="status-summary bilingual">
            ${dualLine(`開始失敗。${error.message}`, `Start failed. ${error.message}`)}
          </div>
        `;
      }
    });
    els.researchPause.addEventListener("click", async () => {
      try {
        const action = (state.controlStatus?.status === "paused" || state.controlStatus?.status === "pausing") ? "/api/resume" : "/api/pause";
        const payload = await apiPost(action);
        renderResearchStatus(payload);
      } catch (error) {
        els.researchStatusText.innerHTML = `
          <div class="status-summary bilingual">
            ${dualLine(`暫停或恢復失敗。${error.message}`, `Pause or resume failed. ${error.message}`)}
          </div>
        `;
      }
    });
    els.researchStop.addEventListener("click", async () => {
      try {
        const payload = await apiPost("/api/stop");
        renderResearchStatus(payload);
      } catch (error) {
        els.researchStatusText.innerHTML = `
          <div class="status-summary bilingual">
            ${dualLine(`停止失敗。${error.message}`, `Stop failed. ${error.message}`)}
          </div>
        `;
      }
    });
    els.strategyFilter.addEventListener("change", updateRunCollections);
    els.adapterFilter.addEventListener("change", updateRunCollections);
    els.searchFilter.addEventListener("input", updateRunCollections);
    els.runSelect.addEventListener("change", (event) => selectRun(event.target.value));
    els.seedSelect.addEventListener("change", async (event) => {
      stopPlayback();
      state.selectedSeed = Number(event.target.value);
      try {
        await renderSelectedReplay();
      } catch (error) {
        clearReplayPlots("回放載入失敗", `Replay load failed: ${error.message || error}`);
      }
    });
      els.speedSelect.addEventListener("change", (event) => {
        state.speed = Number(event.target.value);
      });
      els.scaleSlider.addEventListener("input", (event) => {
        state.scale = Number(event.target.value);
        applyReplayScale();
      });
    }

  function init() {
    applyTheme(window.localStorage.getItem("rocket-dashboard-theme") || "day");
    populateWorkerOptions(1);
    populatePopulationOptions(6);
    els.generationWindow.value = state.generationWindow;
    els.generationScale.value = state.generationScale;
    els.progressWindow.value = state.progressWindow;
    els.progressMetric.value = state.progressMetric;
    renderSummary();
    renderGenerationChart();
    renderResearchProgressChart();
    populateFilters();
    renderReports();
    bindEvents();
    updateRunCollections();
    refreshResearchStatus();
    state.controlPollHandle = window.setInterval(refreshResearchStatus, 2500);
    if (state.selectedRunId) {
      selectRun(state.selectedRunId);
    }
  }

  init();
})();
"""


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _trajectory_script(cache_key: str, payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False)
    return (
        "window.__TRAJECTORY_CACHE__ = window.__TRAJECTORY_CACHE__ || {};\n"
        f"window.__TRAJECTORY_CACHE__[{json.dumps(cache_key, ensure_ascii=False)}] = {serialized};\n"
    )


def _preview_markdown(path: Path, max_chars: int = 700) -> str:
    text = path.read_text(encoding="utf-8")
    collapsed = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return collapsed[:max_chars] + ("..." if len(collapsed) > max_chars else "")


def _ensure_replayable_trajectory(payload: dict[str, Any]) -> list[dict[str, Any]]:
    trajectory = list(payload.get("trajectory", []))
    if trajectory:
        return trajectory
    summary = payload.get("summary", {})
    metadata = payload.get("metadata", {})
    altitude = float(
        metadata.get("apogee_agl_m")
        or metadata.get("altitude_agl_m")
        or summary.get("min_distance_to_any_balloon")
        or 0.0
    )
    synthetic_frame = {
        "time_s": float(summary.get("duration", 0.0) or 0.0),
        "rocket_position": {"x": 0.0, "y": 0.0, "z": altitude},
        "rocket_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
        "popped": int(summary.get("popped", 0) or 0),
        "released": int(metadata.get("balloons_released", metadata.get("released", 0)) or 0),
        "target_id": None,
        "replay_placeholder": True,
    }
    return [synthetic_frame]


def _leaderboard_generation(path: Path) -> int:
    digits = "".join(character for character in path.stem if character.isdigit())
    return int(digits) if digits else -1


def _scan_runs(results_dir: Path, dashboard_dir: Path) -> tuple[list[dict[str, Any]], int]:
    runs_dir = results_dir / "runs"
    trajectory_dir = dashboard_dir / "data" / "trajectories"
    runs: list[dict[str, Any]] = []
    trajectory_count = 0
    if not runs_dir.exists():
        return runs, trajectory_count
    for run_dir in sorted([path for path in runs_dir.iterdir() if path.is_dir()]):
        spec_path = run_dir / "spec.json"
        summary_path = run_dir / "summary.json"
        failure_path = run_dir / "failure_report.json"
        if not (spec_path.exists() and summary_path.exists() and failure_path.exists()):
            continue
        spec = _read_json(spec_path)
        summary = _read_json(summary_path)
        failure_report = _read_json(failure_path)
        trajectory_index: list[dict[str, Any]] = []
        adapter_name = None
        for trajectory_path in sorted(run_dir.glob("trajectory_seed_*.jsonl")):
            payload = json.loads(trajectory_path.read_text(encoding="utf-8").strip())
            episode_summary = payload.get("summary", {})
            metadata = payload.get("metadata", {})
            adapter_name = adapter_name or metadata.get("adapter") or episode_summary.get("metadata", {}).get("adapter")
            seed = int(episode_summary.get("seed", 0))
            payload["trajectory"] = _ensure_replayable_trajectory(payload)
            payload["trajectory"] = augment_trajectory_with_balloon_snapshots(
                spec,
                adapter_name or "unknown",
                seed,
                list(payload.get("trajectory", [])),
            )
            cache_key = f"{spec['experiment_id']}:{seed}"
            script_name = f"{spec['experiment_id']}_{seed}.js"
            script_relative = f"./data/trajectories/{script_name}"
            _write_text(trajectory_dir / script_name, _trajectory_script(cache_key, payload))
            trajectory_index.append(
                {
                    "seed": seed,
                    "file": script_relative,
                    "points": len(payload.get("trajectory", [])),
                    "score": episode_summary.get("score"),
                    "popped": episode_summary.get("popped"),
                    "duration": episode_summary.get("duration"),
                    "target_switch_count": episode_summary.get("target_switch_count"),
                }
            )
            trajectory_count += 1
        runs.append(
            {
                "experiment_id": spec["experiment_id"],
                "strategy_name": spec["strategy_name"],
                "generation": spec.get("generation"),
                "completed_at": summary_path.stat().st_mtime,
                "note": spec.get("note"),
                "parent_ids": spec.get("parent_ids", []),
                "params": spec.get("params", {}),
                "seeds": spec.get("seeds", []),
                "adapter": adapter_name or "unknown",
                "summary": summary,
                "failure_report": failure_report,
                "trajectory_index": trajectory_index,
            }
        )
    runs.sort(key=lambda run: float(run["summary"].get("final_fitness", 0.0)), reverse=True)
    return runs, trajectory_count


def _format_param_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _best_change_label(previous_run: dict[str, Any] | None, current_run: dict[str, Any]) -> str:
    if previous_run is None:
        return "baseline"
    if previous_run.get("strategy_name") != current_run.get("strategy_name"):
        return f"strategy {previous_run.get('strategy_name')} -> {current_run.get('strategy_name')}"
    previous_params = dict(previous_run.get("params", {}))
    current_params = dict(current_run.get("params", {}))
    changed_entries: list[tuple[float, str]] = []
    for key in sorted(set(previous_params) | set(current_params)):
        old_value = previous_params.get(key)
        new_value = current_params.get(key)
        if old_value == new_value:
            continue
        if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
            scale = max(abs(float(old_value)), abs(float(new_value)), 1.0)
            score = abs(float(new_value) - float(old_value)) / scale
        else:
            score = 1.0
        changed_entries.append(
            (
                score,
                f"{key} {_format_param_value(old_value)}->{_format_param_value(new_value)}",
            )
        )
    if changed_entries:
        changed_entries.sort(key=lambda item: item[0], reverse=True)
        return changed_entries[0][1]
    note = str(current_run.get("note") or "").strip()
    return note.split(";")[0] if note else "kept improvement"


def _build_research_progress(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered_runs = sorted(
        runs,
        key=lambda run: (
            int(run.get("generation")) if run.get("generation") is not None else 10**9,
            float(run.get("completed_at", 0.0)),
            str(run.get("experiment_id", "")),
        ),
    )
    progress: list[dict[str, Any]] = []
    running_best_fitness = float("-inf")
    running_best_popped = float("-inf")
    previous_kept_fitness_run: dict[str, Any] | None = None
    previous_kept_popped_run: dict[str, Any] | None = None
    for index, run in enumerate(ordered_runs, start=1):
        final_fitness = float(run.get("summary", {}).get("final_fitness", 0.0))
        mean_popped = float(run.get("summary", {}).get("mean_popped", 0.0))
        kept_fitness = final_fitness > running_best_fitness + 1e-9
        kept_popped = mean_popped > running_best_popped + 1e-9
        if kept_fitness:
            running_best_fitness = final_fitness
            annotation_fitness = _best_change_label(previous_kept_fitness_run, run)
            previous_kept_fitness_run = run
        else:
            annotation_fitness = ""
        if kept_popped:
            running_best_popped = mean_popped
            annotation_popped = _best_change_label(previous_kept_popped_run, run)
            previous_kept_popped_run = run
        else:
            annotation_popped = ""
        progress.append(
            {
                "index": index,
                "experiment_id": run.get("experiment_id"),
                "strategy_name": run.get("strategy_name"),
                "stage_id": run.get("summary", {}).get("stage_id"),
                "final_fitness": final_fitness,
                "mean_popped": mean_popped,
                "kept_fitness": kept_fitness,
                "kept_popped": kept_popped,
                "running_best_fitness": running_best_fitness,
                "running_best_popped": running_best_popped,
                "annotation_fitness": annotation_fitness,
                "annotation_popped": annotation_popped,
            }
        )
    return progress


def build_dashboard(results_dir: str | Path = "results", output_dir: str | Path | None = None) -> Path:
    resolved_results = Path(results_dir)
    resolved_output = Path(output_dir) if output_dir is not None else resolved_results / "dashboard"
    resolved_output.mkdir(parents=True, exist_ok=True)

    best_config_path = resolved_results / "best_agents" / "best_config.json"
    best_config = _read_json(best_config_path) if best_config_path.exists() else {}
    latest_plan_path = resolved_results / "research_memory" / "latest_plan.json"
    latest_plan = _read_json(latest_plan_path) if latest_plan_path.exists() else {}
    stage_catalog = [
        {
            "stage_id": stage.stage_id,
            "title": stage.title,
            "description": stage.description,
            "success_threshold": stage.success_threshold,
        }
        for stage in default_problem_stages()
    ]

    leaderboard_paths = sorted((resolved_results / "leaderboards").glob("generation_*.json"), key=_leaderboard_generation)
    leaderboards = [
        {
            "name": path.name,
            "generation": _leaderboard_generation(path),
            "entries": _read_json(path),
        }
        for path in leaderboard_paths
    ]
    generation_trend = []
    for bundle in leaderboards:
        if not bundle["entries"]:
            continue
        top_entry = max(bundle["entries"], key=lambda entry: float(entry.get("final_fitness", 0.0)))
        generation_trend.append(
            {
                "generation": bundle["generation"],
                "best_final_fitness": top_entry.get("final_fitness"),
                "experiment_id": top_entry.get("experiment_id"),
                "strategy_name": top_entry.get("strategy_name"),
            }
        )

    cross_validation_dir = resolved_results / "cross_validation"
    cross_validations = []
    if cross_validation_dir.exists():
        for path in sorted(cross_validation_dir.glob("*.json")):
            cross_validations.append(_read_json(path))

    report_dir = resolved_results / "reports"
    report_previews = []
    if report_dir.exists():
        for path in sorted(report_dir.glob("generation_*.md")):
            report_previews.append({"name": path.name, "preview": _preview_markdown(path)})

    runs, trajectory_count = _scan_runs(resolved_results, resolved_output)
    research_progress = _build_research_progress(runs)
    strategy_names = sorted({run["strategy_name"] for run in runs})
    best_run = runs[0] if runs else None
    summary = {
        "total_runs": len(runs),
        "total_trajectories": trajectory_count,
        "best_experiment_id": best_run["experiment_id"] if best_run else best_config.get("experiment_id"),
        "best_strategy_name": best_run["strategy_name"] if best_run else best_config.get("strategy_name"),
        "best_final_fitness": best_run["summary"].get("final_fitness") if best_run else best_config.get("final_fitness"),
        "best_mean_popped": best_run["summary"].get("mean_popped") if best_run else None,
        "strategy_count": len(strategy_names),
        "strategy_names": strategy_names,
    }

    dashboard_payload = {
        "generated_at": best_config.get("generated_at", ""),
        "summary": summary,
        "best_config": best_config,
        "latest_plan": latest_plan,
        "stage_catalog": stage_catalog,
        "leaderboards": leaderboards,
        "generation_trend": generation_trend,
        "cross_validations": cross_validations,
        "report_previews": report_previews,
        "runs": runs,
        "research_progress": research_progress,
    }
    if not dashboard_payload["generated_at"]:
        dashboard_payload["generated_at"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    _write_text(resolved_output / "index.html", INDEX_HTML)
    _write_text(resolved_output / "styles.css", STYLES_CSS)
    _write_text(resolved_output / "app.js", APP_JS)
    _write_text(
        resolved_output / "data" / "index.js",
        f"window.__DASHBOARD_DATA__ = {json.dumps(dashboard_payload, ensure_ascii=False)};\n",
    )
    return resolved_output / "index.html"
