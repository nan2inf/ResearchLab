const state = {
  summary: { servers: [], projects: [], versions: [], runs: [] },
  selectedRun: null,
  events: [],
  eventCursor: 0,
  hardware: null,
  runTarget: null,
  runHardware: null,
  runEnvironments: [],
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[c]));
const statusLabel = { queued: "排队中", running: "运行中", completed: "已完成", failed: "失败", cancelled: "已取消", unknown: "状态未知" };

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail || response.statusText);
  }
  return response.headers.get("content-type")?.includes("json") ? response.json() : response.text();
}

function toast(message, error = false) {
  const element = $("#toast");
  element.textContent = message;
  element.className = `show${error ? " error" : ""}`;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => element.className = "", 3200);
}

async function loadSummary() {
  state.summary = await api("/api/summary");
  renderSummary();
}

function renderSummary() {
  const { servers, projects, versions, runs } = state.summary;
  if (state.selectedRun) {
    state.selectedRun = runs.find(run => run.id === state.selectedRun.id) || state.selectedRun;
  }
  $("#runningCount").textContent = runs.filter(run => run.status === "running").length;
  $("#projectCount").textContent = projects.length;
  $("#versionCount").textContent = versions.length;
  $("#serverCount").textContent = servers.length + 1;
  renderRuns("#recentRuns", runs.slice(0, 6));
  renderRuns("#allRuns", runs);
  renderProjects();
  renderServers();
  populateRunVersions();
}

function projectName(id) {
  return state.summary.projects.find(project => project.id === id)?.name || "未知项目";
}

function serverName(id) {
  return id ? (state.summary.servers.find(server => server.id === id)?.name || "SSH 节点") : "本机";
}

function renderRuns(selector, runs) {
  const root = $(selector);
  if (!runs.length) {
    root.className = "run-list empty-block";
    root.textContent = "还没有实验记录";
    return;
  }
  root.className = "run-list";
  root.innerHTML = runs.map(run => `
    <div class="run-row" data-run="${run.id}" role="button" tabindex="0">
      <strong>${escapeHtml(projectName(run.project_id))} · ${escapeHtml(run.name)}</strong>
      <small class="run-target">${escapeHtml(serverName(run.server_id))} / ${escapeHtml(run.backend.toUpperCase())}</small>
      <span class="status ${escapeHtml(run.status)}">${escapeHtml(statusLabel[run.status] || run.status)}</span>
      <small class="run-time">${formatDate(run.created_at)}</small>
      <span>›</span>
    </div>`).join("");
  $$("[data-run]", root).forEach(row => {
    const open = () => selectRun(row.dataset.run);
    row.addEventListener("click", open);
    row.addEventListener("keydown", event => { if (event.key === "Enter") open(); });
  });
}

function renderProjects() {
  const root = $("#projectCards");
  if (!state.summary.projects.length) {
    root.innerHTML = `<div class="panel empty-block">尚未导入项目。先从一个包含 project.yaml 的目录开始。</div>`;
    return;
  }
  root.innerHTML = state.summary.projects.map(project => {
    const versions = state.summary.versions.filter(version => version.project_id === project.id);
    const tasks = Object.keys(project.manifest.tasks || {});
    return `<article class="project-card">
      <div class="card-top">
        <div><p class="eyebrow">PYTORCH PROJECT</p><h2>${escapeHtml(project.name)}</h2><p>${escapeHtml(project.manifest.description || project.source_path)}</p></div>
        <span class="tag">${tasks.length} TASK${tasks.length === 1 ? "" : "S"}</span>
      </div>
      <div class="version-list">
        ${versions.length ? versions.slice(0, 5).map(version => `
          <div class="version-item">
            <span>${escapeHtml(version.name)}</span>
            <small>${escapeHtml(serverName(version.server_id))}</small>
          </div>`).join("") : `<div class="version-item"><small>还没有代码快照</small></div>`}
      </div>
      <div class="card-actions">
        <button class="button ghost" data-version-project="${project.id}">创建版本</button>
        ${versions.length ? `<button class="button primary" data-run-version="${versions[0].id}">运行最新版本</button>` : ""}
      </div>
    </article>`;
  }).join("");
  $$("[data-version-project]").forEach(button => button.addEventListener("click", () => openVersionDialog(button.dataset.versionProject)));
  $$("[data-run-version]").forEach(button => button.addEventListener("click", () => openRunDialog(button.dataset.runVersion)));
}

function renderServers() {
  const root = $("#serverCards");
  const localCard = `<article class="server-card">
    <div class="card-top"><div><p class="eyebrow">LOCAL</p><h2>本机</h2><p>Windows / Linux 本地训练</p></div><span class="tag">READY</span></div>
    <div class="card-actions"><button class="button ghost" data-probe-server="">检测环境</button></div>
  </article>`;
  root.innerHTML = localCard + state.summary.servers.map(server => `
    <article class="server-card">
      <div class="card-top"><div><p class="eyebrow">SSH · LINUX</p><h2>${escapeHtml(server.name)}</h2><p>${escapeHtml(server.ssh_alias)} · ${escapeHtml(server.remote_root)}</p></div><span class="tag">SSH</span></div>
      <div class="card-actions"><button class="button ghost" data-probe-server="${server.id}">检测硬件与环境</button><button class="button primary" data-conda-server="${server.id}">新建 Conda 环境</button></div>
    </article>`).join("");
  $$("[data-probe-server]").forEach(button => button.addEventListener("click", () => probeServer(button.dataset.probeServer)));
  $$("[data-conda-server]").forEach(button => button.addEventListener("click", () => openCondaDialog(button.dataset.condaServer)));
}

async function probeHardware(serverId = "") {
  const root = $("#deviceList");
  root.className = "device-list loading-block";
  root.textContent = "正在检测硬件…";
  try {
    state.hardware = await api(`/api/hardware${serverId ? `?server_id=${serverId}` : ""}`);
    const devices = state.hardware.devices;
    if (!devices.length) {
      root.className = "device-list empty-block";
      root.textContent = "未检测到 CUDA 或昇腾设备，可使用 CPU 运行。";
      return;
    }
    root.className = "device-list";
    root.innerHTML = devices.map(device => {
      const total = device.memory_total_mb || 0;
      const used = device.memory_used_mb || 0;
      const percent = total ? Math.round(used / total * 100) : Math.round(device.utilization_percent || 0);
      const users = [...new Set((device.processes || []).map(p => p.user).filter(Boolean))];
      return `<div class="device">
        <div class="device-id">${escapeHtml(device.backend === "npu" ? "N" : "G")}${escapeHtml(device.id)}</div>
        <div><h3>${escapeHtml(device.name)}</h3><p>${device.busy ? `使用中${users.length ? ` · ${escapeHtml(users.join(", "))}` : ""}` : "当前空闲"} · ${used || "?"}/${total || "?"} MB</p></div>
        <div class="usage"><strong>${percent}%</strong><small>${device.temperature_c ?? "?"}°C</small></div>
      </div>`;
    }).join("");
  } catch (error) {
    root.className = "device-list empty-block";
    root.textContent = error.message;
  }
}

async function probeServer(serverId) {
  switchView("overview");
  const environments = await api(`/api/environments${serverId ? `?server_id=${serverId}` : ""}`);
  await probeHardware(serverId);
  toast(`检测完成：找到 ${environments.length} 个 Python 环境`);
}

function openVersionDialog(projectId) {
  const form = $("#versionForm");
  form.reset();
  form.elements.project_id.value = projectId;
  form.elements.server_id.innerHTML = `<option value="">本机</option>` + state.summary.servers.map(server => `<option value="${server.id}">${escapeHtml(server.name)}</option>`).join("");
  $("#versionDialog").showModal();
}

function populateRunVersions() {
  const select = $("#runForm").elements.version_id;
  select.innerHTML = state.summary.versions.map(version => `<option value="${version.id}">${escapeHtml(projectName(version.project_id))} · ${escapeHtml(version.name)} · ${escapeHtml(serverName(version.server_id))}</option>`).join("");
  if (select.value) updateRunTaskFields();
}

function openRunDialog(versionId = "") {
  if (!state.summary.versions.length) {
    toast("请先为项目创建一个代码版本", true);
    return;
  }
  const form = $("#runForm");
  form.reset();
  populateRunVersions();
  if (versionId) form.elements.version_id.value = versionId;
  updateRunTaskFields();
  $("#runDialog").showModal();
}

function updateRunTaskFields() {
  const form = $("#runForm");
  const version = state.summary.versions.find(item => item.id === form.elements.version_id.value);
  if (!version) return;
  const tasks = version.manifest.tasks || {};
  const taskSelect = form.elements.task;
  const previous = taskSelect.value;
  taskSelect.innerHTML = Object.keys(tasks).map(name => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join("");
  if (tasks[previous]) taskSelect.value = previous;
  const task = tasks[taskSelect.value];
  $("#parameterFields").innerHTML = Object.entries(task?.parameters || {}).map(([name, spec]) => {
    const label = escapeHtml(spec.label || name);
    const help = spec.help ? `<small>${escapeHtml(spec.help)}</small>` : "";
    if (spec.type === "bool") {
      return `<label>${label}<select data-param="${escapeHtml(name)}"><option value="true" ${spec.default ? "selected" : ""}>是</option><option value="false" ${!spec.default ? "selected" : ""}>否</option></select>${help}</label>`;
    }
    if (spec.choices?.length) {
      return `<label>${label}<select data-param="${escapeHtml(name)}">${spec.choices.map(choice => `<option value="${escapeHtml(choice)}" ${choice === spec.default ? "selected" : ""}>${escapeHtml(choice)}</option>`).join("")}</select>${help}</label>`;
    }
    const inputType = ["int", "float"].includes(spec.type) ? "number" : "text";
    const step = spec.type === "float" ? `step="any"` : "";
    return `<label>${label}<input data-param="${escapeHtml(name)}" type="${inputType}" ${step} value="${escapeHtml(spec.default ?? "")}" ${spec.required ? "required" : ""} />${help}</label>`;
  }).join("");
  loadRunTarget(version);
}

async function loadRunTarget(version, force = false) {
  const form = $("#runForm");
  const backend = form.elements.backend.value;
  const target = `${version.id}:${backend}`;
  if (!force && state.runTarget === target) return;
  state.runTarget = target;
  const chooser = $("#runDeviceChooser");
  chooser.innerHTML = "<span>正在读取目标机器状态与 Python 环境…</span>";
  const query = version.server_id ? `?server_id=${version.server_id}` : "";
  try {
    const [hardware, environments] = await Promise.all([
      api(`/api/hardware${query}`),
      api(`/api/versions/${version.id}/environments?backend=${backend}`),
    ]);
    state.runHardware = hardware;
    state.runEnvironments = environments;
    renderRunDevices();
    renderEnvironmentOptions(environments);
  } catch (error) {
    chooser.innerHTML = `<span>${escapeHtml(error.message)}</span>`;
  }
}

function renderEnvironmentOptions(environments) {
  const form = $("#runForm");
  $("#pythonEnvironments").innerHTML = environments.map(environment =>
    `<option value="${escapeHtml(environment.executable)}">${escapeHtml(environment.name)} · ${environment.compatible ? "兼容" : escapeHtml(environment.issues.join("; "))}</option>`
  ).join("");
  const compatible = environments.find(environment => environment.compatible);
  if (compatible && !form.elements.python.value) form.elements.python.value = compatible.executable;
}

async function refreshRunEnvironments() {
  const form = $("#runForm");
  const version = state.summary.versions.find(item => item.id === form.elements.version_id.value);
  if (!version) return;
  const environments = await api(`/api/versions/${version.id}/environments?backend=${form.elements.backend.value}`);
  state.runEnvironments = environments;
  renderEnvironmentOptions(environments);
}

function renderRunDevices() {
  const chooser = $("#runDeviceChooser");
  const devices = state.runHardware?.devices || [];
  if (!devices.length) {
    chooser.innerHTML = "<span>未检测到加速卡；可直接使用 CPU。</span>";
    return;
  }
  chooser.innerHTML = devices.map(device => {
    const users = [...new Set((device.processes || []).map(process => process.user).filter(Boolean))];
    return `<label class="device-option ${device.busy ? "busy" : ""}">
      <input type="checkbox" data-device-id="${device.id}" data-device-backend="${device.backend}" data-device-busy="${device.busy}" />
      <span><strong>${escapeHtml(device.backend.toUpperCase())}:${escapeHtml(device.id)}</strong><small>${device.busy ? `使用中${users.length ? ` · ${escapeHtml(users.join(","))}` : ""}` : "空闲"} · ${escapeHtml(device.name)}</small></span>
    </label>`;
  }).join("");
  $$("[data-device-id]", chooser).forEach(input => input.addEventListener("change", () => {
    if (input.checked) {
      $$("[data-device-id]", chooser).filter(other => other !== input && other.dataset.deviceBackend !== input.dataset.deviceBackend).forEach(other => other.checked = false);
      $("#runForm").elements.backend.value = input.dataset.deviceBackend;
      refreshRunEnvironments().catch(error => toast(error.message, true));
      if (input.dataset.deviceBusy === "true") toast("所选设备已有进程占用，启动前请确认资源不会冲突", true);
    }
    const selected = $$("[data-device-id]:checked", chooser);
    $("#runForm").elements.devices.value = selected.map(item => item.dataset.deviceId).join(",");
  }));
}

function openCondaDialog(serverId) {
  const form = $("#condaForm");
  form.reset();
  form.elements.server_id.value = serverId;
  form.elements.python_version.value = "3.11";
  $("#condaDialog").showModal();
}

async function selectRun(runId) {
  state.selectedRun = state.summary.runs.find(run => run.id === runId);
  if (!state.selectedRun) return;
  state.events = [];
  state.eventCursor = 0;
  $("#metricTitle").textContent = `${projectName(state.selectedRun.project_id)} · ${state.selectedRun.name}`;
  $("#resultTitle").textContent = `${projectName(state.selectedRun.project_id)} · ${state.selectedRun.name}`;
  $("#cancelRun").classList.toggle("hidden", state.selectedRun.status !== "running");
  switchView("runs");
  await refreshSelectedRun();
}

async function refreshSelectedRun() {
  if (!state.selectedRun) return;
  try {
    const [eventData, log] = await Promise.all([
      api(`/api/runs/${state.selectedRun.id}/events?after=${state.eventCursor}`),
      api(`/api/runs/${state.selectedRun.id}/log`),
    ]);
    state.events.push(...eventData.events);
    state.eventCursor = eventData.next;
    $("#liveLog").textContent = log || "暂无日志输出。";
    renderMetrics();
    renderResults(log);
  } catch (error) {
    toast(error.message, true);
  }
}

function metricSeries() {
  const series = {};
  state.events.filter(event => event.type === "metrics").forEach((event, index) => {
    Object.entries(event.metrics || {}).forEach(([name, value]) => {
      const key = `${event.split || "metrics"}/${name}`;
      (series[key] ||= []).push({ x: event.step ?? event.epoch ?? index + 1, y: Number(value) });
    });
  });
  return series;
}

function renderMetrics() {
  const canvas = $("#metricChart");
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(400, rect.width * ratio);
  canvas.height = 290 * ratio;
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  const width = canvas.width / ratio;
  const height = 290;
  context.clearRect(0, 0, width, height);
  const pad = { left: 42, right: 14, top: 18, bottom: 32 };
  const series = metricSeries();
  const entries = Object.entries(series);
  context.strokeStyle = "#273141";
  context.fillStyle = "#6e7a8e";
  context.font = "10px ui-monospace";
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (height - pad.top - pad.bottom) * i / 4;
    context.beginPath(); context.moveTo(pad.left, y); context.lineTo(width - pad.right, y); context.stroke();
  }
  if (!entries.length) {
    context.fillText("启动实验后，指标将在这里实时出现", pad.left + 12, height / 2);
    $("#metricLegend").innerHTML = "";
    return;
  }
  const points = entries.flatMap(([, values]) => values);
  const minX = Math.min(...points.map(point => point.x));
  const maxX = Math.max(...points.map(point => point.x), minX + 1);
  let minY = Math.min(...points.map(point => point.y));
  let maxY = Math.max(...points.map(point => point.y));
  if (minY === maxY) { minY -= 1; maxY += 1; }
  const colors = ["#a8ff60", "#67d6ff", "#ffc857", "#ff7aaa", "#b89cff", "#67f1ca"];
  entries.forEach(([name, values], index) => {
    context.strokeStyle = colors[index % colors.length];
    context.lineWidth = 2;
    context.beginPath();
    values.forEach((point, pointIndex) => {
      const x = pad.left + (point.x - minX) / (maxX - minX) * (width - pad.left - pad.right);
      const y = pad.top + (maxY - point.y) / (maxY - minY) * (height - pad.top - pad.bottom);
      pointIndex ? context.lineTo(x, y) : context.moveTo(x, y);
    });
    context.stroke();
  });
  context.fillStyle = "#6e7a8e";
  context.fillText(maxY.toPrecision(4), 2, pad.top + 4);
  context.fillText(minY.toPrecision(4), 2, height - pad.bottom);
  context.fillText(String(minX), pad.left, height - 8);
  context.fillText(String(maxX), width - pad.right - 25, height - 8);
  $("#metricLegend").innerHTML = entries.map(([name], index) => `<span><i style="background:${colors[index % colors.length]}"></i>${escapeHtml(name)}</span>`).join("");
}

function renderResults(log) {
  const root = $("#resultContent");
  const run = state.selectedRun;
  const runPath = run.remote_path || run.run_path;
  const matrices = state.events.filter(event => event.type === "matrix");
  const tables = state.events.filter(event => event.type === "table");
  const artifacts = state.events.filter(event => event.type === "artifact");
  root.className = "result-grid";
  root.innerHTML = `
    <section class="result-section"><h3>运行信息</h3><div class="table-wrap"><table class="data-table"><tbody>
      <tr><th>实验目录</th><td><code>${escapeHtml(runPath)}</code></td></tr>
      <tr><th>启动命令</th><td><code>${escapeHtml((run.command || []).join(" "))}</code></td></tr>
      <tr><th>设备</th><td>${escapeHtml(run.backend.toUpperCase())}${run.devices?.length ? ` · ${escapeHtml(run.devices.join(","))}` : ""}</td></tr>
      <tr><th>状态</th><td>${escapeHtml(statusLabel[run.status] || run.status)}</td></tr>
    </tbody></table></div></section>
    <section class="result-section"><h3>指标曲线</h3><canvas id="resultMetricChart" height="250"></canvas><div id="resultMetricLegend" class="legend"></div></section>
    ${matrices.map(renderMatrix).join("")}
    ${tables.map(renderTable).join("")}
    ${artifacts.length ? `<section class="result-section"><h3>实验产物</h3><div class="artifact-grid">${artifacts.map(artifact => {
      const path = `${runPath.replace(/[\\/]+$/, "")}/${artifact.path}`;
      const link = `/api/runs/${state.selectedRun.id}/artifacts/${encodeURI(artifact.path)}`;
      return artifact.artifact_type === "image"
        ? `<figure><img src="${link}" alt="${escapeHtml(artifact.name)}" /><figcaption>${escapeHtml(artifact.name)}<code>${escapeHtml(path)}</code></figcaption></figure>`
        : `<div class="artifact-file"><a class="button ghost" href="${link}">${escapeHtml(artifact.name)}</a><code>${escapeHtml(path)}</code></div>`;
    }).join("")}</div></section>` : ""}
    <section class="result-section"><h3>训练日志</h3><pre>${escapeHtml(log || "暂无日志输出。")}</pre></section>`;
  const original = $("#metricChart");
  const result = $("#resultMetricChart");
  result.width = original.width;
  result.height = original.height;
  result.getContext("2d").drawImage(original, 0, 0);
  $("#resultMetricLegend").innerHTML = $("#metricLegend").innerHTML;
}

function renderMatrix(event) {
  const values = event.values || [];
  const flat = values.flat().map(Number);
  const max = Math.max(...flat, 1);
  const columns = Math.max(...values.map(row => row.length), 1);
  return `<section class="result-section"><h3>${escapeHtml(event.name)}</h3><div class="matrix" style="grid-template-columns:repeat(${columns},1fr)">${values.flatMap(row => row.map(value => `<div class="matrix-cell" style="background:rgba(103,214,255,${0.12 + Number(value) / max * 0.78})">${escapeHtml(value)}</div>`)).join("")}</div></section>`;
}

function renderTable(event) {
  return `<section class="result-section"><h3>${escapeHtml(event.name)}</h3><div class="table-wrap"><table class="data-table"><thead><tr>${(event.columns || []).map(column => `<th>${escapeHtml(column)}</th>`).join("")}</tr></thead><tbody>${(event.rows || []).map(row => `<tr>${row.map(cell => `<td>${escapeHtml(cell)}</td>`).join("")}</tr>`).join("")}</tbody></table></div></section>`;
}

function switchView(view) {
  $$(".view").forEach(element => element.classList.toggle("active", element.id === `${view}View`));
  $$(".nav-item").forEach(button => button.classList.toggle("active", button.dataset.view === view));
  $("#pageTitle").textContent = ({ overview: "训练总览", projects: "项目与版本", runs: "实验运行", servers: "计算节点" })[view];
}

function formatDate(value) {
  if (!value) return "—";
  return new Intl.DateTimeFormat("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" }).format(new Date(value));
}

$$(".nav-item").forEach(button => button.addEventListener("click", () => switchView(button.dataset.view)));
$$("[data-go]").forEach(button => button.addEventListener("click", () => switchView(button.dataset.go)));
$$("[data-dialog]").forEach(button => button.addEventListener("click", () => $(`#${button.dataset.dialog}`).showModal()));
$$('dialog button[value="cancel"]').forEach(button => {
  button.type = "button";
  button.addEventListener("click", () => button.closest("dialog").close());
});
$("#newRunButton").addEventListener("click", () => openRunDialog());
$("#refreshButton").addEventListener("click", async () => { await loadSummary(); await refreshSelectedRun(); toast("状态已刷新"); });
$("#probeLocal").addEventListener("click", () => probeHardware());
$("#runForm").elements.version_id.addEventListener("change", updateRunTaskFields);
$("#runForm").elements.task.addEventListener("change", updateRunTaskFields);
$("#runForm").elements.backend.addEventListener("change", () => {
  const version = state.summary.versions.find(item => item.id === $("#runForm").elements.version_id.value);
  if (version) loadRunTarget(version, true);
});
$("#cancelRun").addEventListener("click", async () => {
  if (!state.selectedRun) return;
  await api(`/api/runs/${state.selectedRun.id}/cancel`, { method: "POST", body: "{}" });
  toast("已发送取消请求");
  await loadSummary();
});

$("#projectForm").addEventListener("submit", async event => {
  event.preventDefault();
  const form = event.currentTarget;
  try {
    await api("/api/projects", { method: "POST", body: JSON.stringify({ source_path: form.elements.source_path.value }) });
    $("#projectDialog").close();
    form.reset();
    await loadSummary();
    switchView("projects");
    toast("项目已导入，原目录未被修改");
  } catch (error) { toast(error.message, true); }
});

$("#serverForm").addEventListener("submit", async event => {
  event.preventDefault();
  const form = event.currentTarget;
  const payload = Object.fromEntries(new FormData(form));
  try {
    await api("/api/servers", { method: "POST", body: JSON.stringify(payload) });
    $("#serverDialog").close();
    form.reset();
    await loadSummary();
    switchView("servers");
    toast("SSH 节点连接成功");
  } catch (error) { toast(error.message, true); }
});

$("#versionForm").addEventListener("submit", async event => {
  event.preventDefault();
  const form = event.currentTarget;
  const projectId = form.elements.project_id.value;
  const payload = { name: form.elements.name.value, server_id: form.elements.server_id.value || null };
  try {
    toast(payload.server_id ? "正在打包并上传代码…" : "正在创建代码快照…");
    await api(`/api/projects/${projectId}/versions`, { method: "POST", body: JSON.stringify(payload) });
    $("#versionDialog").close();
    await loadSummary();
    toast("不可变代码版本已创建");
  } catch (error) { toast(error.message, true); }
});

$("#runForm").addEventListener("submit", async event => {
  event.preventDefault();
  const form = event.currentTarget;
  const params = {};
  $$("[data-param]", form).forEach(input => {
    let value = input.value;
    if (input.type === "number" && value !== "") value = Number(value);
    if (value === "true" || value === "false") value = value === "true";
    params[input.dataset.param] = value;
  });
  const devices = form.elements.devices.value.split(",").map(value => value.trim()).filter(Boolean).map(Number);
  const payload = {
    version_id: form.elements.version_id.value,
    task: form.elements.task.value,
    backend: form.elements.backend.value,
    devices,
    python: form.elements.python.value || null,
    name: form.elements.name.value,
    params,
  };
  try {
    const run = await api("/api/runs", { method: "POST", body: JSON.stringify(payload) });
    $("#runDialog").close();
    await loadSummary();
    toast("实验已启动");
    await selectRun(run.id);
  } catch (error) { toast(error.message, true); }
});

$("#condaForm").addEventListener("submit", async event => {
  event.preventDefault();
  const form = event.currentTarget;
  const serverId = form.elements.server_id.value;
  const payload = {
    name: form.elements.name.value,
    python_version: form.elements.python_version.value,
    install_command: form.elements.install_command.value,
  };
  try {
    toast("正在服务器上创建独立 Conda 环境…");
    const environment = await api(`/api/servers/${serverId}/environments`, { method: "POST", body: JSON.stringify(payload) });
    $("#condaDialog").close();
    toast(`环境已创建：${environment.executable}`);
  } catch (error) { toast(error.message, true); }
});

window.addEventListener("resize", renderMetrics);
setInterval(() => {
  $("#clock").textContent = new Intl.DateTimeFormat("zh-CN", { hour: "2-digit", minute: "2-digit", second: "2-digit" }).format(new Date());
}, 1000);
setInterval(async () => {
  if (state.selectedRun) await refreshSelectedRun();
  if (state.summary.runs.some(run => run.status === "running")) await loadSummary();
}, 2500);

Promise.all([loadSummary(), probeHardware()]).catch(error => toast(error.message, true));
renderMetrics();
