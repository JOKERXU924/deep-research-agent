"use strict";

// This page is a local client. Model text is data, never executable markup.
const $ = (id) => document.getElementById(id);
const TERMINAL = new Set(["completed", "failed", "cancelled", "interrupted"]);
const STATUS = { queued: "等待执行", running: "执行中", cancelling: "正在停止", completed: "执行完成", failed: "执行失败", cancelled: "已取消", interrupted: "已中断" };
const ERRORS = {
  web_live_disabled: "服务未开启真实研究，请使用离线演示。",
  live_disabled: "服务未开启真实研究，请使用离线演示。",
  web_queue_full: "等待队列已满，请稍后再提交。",
  queue_full: "等待队列已满，请稍后再提交。",
  web_capacity: "服务或等待队列已满，请稍后再提交。",
  web_validation: "提交内容不符合要求，请检查问题、场景和选项。",
  web_conflict: "当前任务状态或提交标识不允许此操作，请刷新记录后检查。",
  web_csrf: "安全凭证已失效，请刷新页面后重试。",
  csrf_invalid: "安全凭证已失效，请刷新页面后重试。",
  web_not_found: "没有找到该任务，可能使用了不同的数据库。",
  web_idempotency_conflict: "重复提交标识与请求内容不一致，请重新提交。",
  invalid_request: "提交内容不符合要求，请检查问题和选项。",
  review_validation: "审查结果未通过程序校验，任务已停止。",
  schema_mismatch: "模型返回的 JSON 不符合规定的字段结构，任务已停止；这不代表密钥失效。请查看下方失败调用诊断。",
  invalid_json: "模型返回的内容无法解析为规定的 JSON，任务已停止。请查看下方失败调用诊断。",
  review_audit_io: "审查记录无法保存，任务已停止。",
  config_error: "真实模型配置不可用，请在服务端检查配置。",
  configuration: "所选模型配置或密钥不可用，请在服务端检查；不会自动切换其他服务。",
  web_profile: "所选配置已不可用，请刷新页面后重新选择。",
  model_budget_exceeded: "模型调用预算已耗尽，未发送下一次请求。已有调用与用量仍保留在账本中。",
};
let config = null;
let jobs = [];
let selectedId = null;
let selectedJob = null;
let selectionEpoch = 0;
let jobsEpoch = 0;
let selectedReadSeq = 0;
let auditReadSeq = 0;
let stream = null;
let seenSequence = 0;
let reportLoadedFor = null;
let reportLoadingFor = null;
let submitting = false;
let cancelling = false;
let polling = false;
let disposed = false;
let pendingSubmission = null;
let liveQuestion = "";

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined && text !== null) element.textContent = String(text);
  if (className) element.className = className;
  return element;
}

function showNotice(id, message, tone = "") {
  const element = $(id);
  element.textContent = message;
  element.className = `notice ${tone}${message ? "" : " hidden"}`;
}

function describeError(error) {
  if (error instanceof TypeError || error.name === "AbortError") return "无法确认本地服务的响应，请检查服务是否仍在运行。";
  if (ERRORS[error.code]) return ERRORS[error.code];
  if (error.status === 403) return "请求被本地安全检查拒绝。请刷新页面，并使用服务显示的本机地址。";
  if (error.status === 409) return "任务状态或提交标识发生冲突，请刷新记录后检查。";
  if (error.status === 422 || error.status === 400) return "提交内容不符合要求，请检查问题、场景和确认选项。";
  if (error.status === 429 || error.status === 503) return "服务或队列暂时繁忙，请稍后再试。";
  return "操作未完成，请检查服务状态后重试。";
}

async function api(path, options = {}) {
  const response = await fetch(path, { credentials: "same-origin", cache: "no-store", ...options });
  let body = null;
  try { body = await response.json(); } catch { /* A failed response need not contain JSON. */ }
  if (!response.ok) {
    const error = new Error("Local API request failed");
    error.status = response.status;
    error.code = body?.error?.code || body?.error_code || body?.code || body?.detail?.code;
    throw error;
  }
  if (body === null) throw new TypeError("Missing API response");
  return body;
}

function dateValue(value) {
  if (typeof value === "number") return new Date(value < 100000000000 ? value * 1000 : value);
  return new Date(value);
}

function timeLabel(value, brief = false) {
  if (!value) return "时间未记录";
  const date = dateValue(value);
  if (Number.isNaN(date.getTime())) return "时间未记录";
  return brief ? date.toLocaleTimeString("zh-CN", { hour12: false }) : date.toLocaleString("zh-CN", { hour12: false });
}

function updateForm() {
  const live = $("mode").value === "live";
  const profiles = config?.model_profiles || [];
  $("model-profile-field").classList.toggle("hidden", !live);
  $("model-profile").disabled = !live || submitting || profiles.length === 0;
  const profile = profiles.find((item) => item.id === $("model-profile").value);
  $("model-profile-help").textContent = profile
    ? `${profile.provider} / ${profile.model} · 最多 ${profile.budget.max_calls} 次调用；累计请求 ${profile.budget.max_input_bytes} 字节；预留输出 ${profile.budget.max_output_tokens} tokens。架构上限也会同时生效；不保证费用。`
    : "沿用原 DeepSeek 服务端配置；未启用 M7 配置目录。";
  $("live-options").classList.toggle("hidden", !live);
  $("scenario-field").classList.toggle("hidden", live);
  $("question").readOnly = !live;
  $("question-label-note").textContent = live ? "基于维基百科公开资料" : "固定教学样例";
  $("question-help").textContent = live
    ? "真实研究会检索维基百科公开资料，并发送选定证据给配置的模型；这不等于全网搜索。"
    : "演示使用固定问题、资料与假模型，展示真实执行流程，不会根据任意问题进行检索。";
  $("mode-badge").textContent = live ? "真实研究 · 可能计费" : "离线演示";
  $("submit-job").textContent = submitting ? "正在提交……" : live ? "开始真实研究 ↗" : "开始离线演示 ↗";
  $("submit-job").disabled = !config || submitting || (live && (!config.allow_live || !$("confirm-live").checked));
  $("mode").disabled = !config || submitting;
  $("scenario").disabled = !config || submitting || live;
  const requiresReview = !live && $("scenario").value !== "repair";
  if (requiresReview) $("review").checked = true;
  $("review").disabled = !config || submitting || requiresReview;
  $("compression").disabled = !config || submitting;
  $("robust-contracts").disabled = !config?.supports_robust_contracts || submitting || !live;
  $("rerank-evidence").disabled = !config?.supports_evidence_gate || submitting || !live;
  $("check-support").disabled = !config?.supports_evidence_gate || submitting || !live;
  $("optimize-research").disabled = !config?.supports_research_loop || submitting || !live;
  $("question").disabled = submitting;
  $("memory-scope").disabled = !live || submitting;
  $("confirm-live").disabled = !live || submitting;
  $("compression-help").textContent = live
    ? `服务端编码器：${config?.embedding_backend || "未提供"}；不新增资料`
    : "固定离线编码器，仅验证压缩流程";
}

async function loadConfig() {
  config = await api("/api/config");
  $("model-profile").replaceChildren();
  for (const profile of config.model_profiles || []) {
    const option = node("option", `${profile.label} · ${profile.provider} / ${profile.model}`);
    option.value = profile.id;
    $("model-profile").append(option);
  }
  if (config.default_profile) $("model-profile").value = config.default_profile;
  const liveOption = $("mode").querySelector('option[value="live"]');
  liveOption.disabled = !config.allow_live;
  liveOption.textContent = config.allow_live ? "真实研究 · 可能产生费用" : "真实研究 · 服务尚未开启";
  if (typeof config.demo_question !== "string" || !config.csrf_token) throw new TypeError("Incomplete configuration");
  $("question").value = config.demo_question;
  $("service-info").textContent = `本机服务 · 最多 ${config.max_active} 个任务执行 / ${config.max_queued} 个任务等待`;
  updateForm();
}

function renderJobs() {
  const fragment = document.createDocumentFragment();
  for (const job of jobs) {
    const item = node("li");
    const button = node("button", null, `job-item${job.id === selectedId ? " active" : ""}`);
    button.type = "button";
    if (job.id === selectedId) button.setAttribute("aria-current", "true");
    const title = job.request?.question || "研究任务";
    button.title = title;
    button.append(node("span", title, "job-title"));
    const detail = node("span", null, "job-detail");
    const state = node("span", STATUS[job.status] || "状态未知", "job-state");
    if (Object.hasOwn(STATUS, job.status)) state.classList.add(job.status);
    detail.append(state, node("span", `${job.request?.mode === "live" ? "真实" : "演示"} · ${timeLabel(job.created_at, true)}`));
    button.append(detail);
    button.addEventListener("click", () => { void selectJob(job.id); });
    item.append(button);
    fragment.append(item);
  }
  $("jobs").replaceChildren(fragment);
  $("jobs-status").textContent = jobs.length ? "" : "还没有任务。创建一次离线演示开始体验。";
}

async function refreshJobs() {
  const epoch = ++jobsEpoch;
  try {
    const result = await api("/api/jobs");
    if (disposed || epoch !== jobsEpoch) return;
    jobs = Array.isArray(result.jobs) ? result.jobs : [];
    renderJobs();
  } catch (error) {
    if (epoch === jobsEpoch) $("jobs-status").textContent = describeError(error);
  }
}

function closeStream() {
  if (stream) stream.close();
  stream = null;
}

function renderJob(job) {
  selectedJob = job;
  $("task-empty").classList.add("hidden");
  $("task-content").classList.remove("hidden");
  $("task-question").textContent = job.request?.question || "研究任务";
  $("task-id").textContent = `任务 ${job.id}${job.run_id ? ` / 研究 ${job.run_id}` : ""}`;
  $("task-time").textContent = `创建于 ${timeLabel(job.created_at)}${job.finished_at ? ` · 结束于 ${timeLabel(job.finished_at)}` : ""}`;
  const status = $("task-status");
  status.textContent = job.cancel_requested && !TERMINAL.has(job.status) ? "正在停止" : STATUS[job.status] || "状态未知";
  const tone = job.status === "completed" ? "success" : ["queued", "running"].includes(job.status) ? "active" : job.status === "failed" ? "danger" : "warning";
  status.className = `status-badge ${tone}`;
  $("cancel-job").disabled = cancelling || TERMINAL.has(job.status) || Boolean(job.cancel_requested);
  $("cancel-job").textContent = job.cancel_requested && !TERMINAL.has(job.status) ? "正在停止……" : "停止任务";
  if (job.status === "failed") showNotice("task-notice", `任务未完成。${ERRORS[job.error_code] || "请查看服务端记录中的错误代码；未产生可交付报告。"}${job.error_code ? `（${job.error_code}）` : ""}`, "error");
  else if (job.status === "interrupted") showNotice("task-notice", "任务因服务中断而结束。不会自动恢复或重放真实模型调用，需要重新创建任务。", "warning");
  else if (job.status === "cancelled") showNotice("task-notice", "任务已取消，已执行的步骤和调用记录仍会保留。取消不表示已经发生的模型用量可以撤回。", "warning");
  else if (job.cancel_requested) showNotice("task-notice", "已请求停止，正在等待任务清理并保存最终状态。请以最终状态为准。", "warning");
  else if (job.status === "queued") showNotice("task-notice", "任务已保存，等待执行名额。关闭网页不会自动取消任务。");
  else if (job.status === "completed" && job.result?.memory_write?.status === "failed") showNotice("task-notice", "报告已生成，但写入历史记忆失败。报告仍可查看和下载；本次证据暂不能作为新的共享记忆使用。", "warning");
  else if (job.status === "completed") showNotice("task-notice", "执行流程已结束。报告是否通过审查、是否为降级草稿，请看下方质量说明。");
  else showNotice("task-notice", "");
  if (job.run_id) void loadModelAudit(job.id, selectionEpoch);
  else resetModelAudit();
  if (TERMINAL.has(job.status)) {
    // Drain any saved history before closing: polling may see the final state
    // while the browser is still receiving earlier events from the same job.
    if (!stream) $("stream-status").textContent = "任务已结束";
    if (job.status === "completed") void loadReport(job.id, selectionEpoch);
  }
}

function resetModelAudit() {
  $("model-audit-panel").classList.add("hidden");
  $("model-audit-profile").textContent = "";
  $("model-audit-budget").textContent = "";
  $("model-audit-usage").textContent = "";
  $("model-audit-failures").replaceChildren();
}

function renderModelFailures(calls) {
  const failed = (Array.isArray(calls) ? calls : []).filter((call) => call?.status === "failed").slice(-9);
  const fragment = document.createDocumentFragment();
  if (failed.length) {
    fragment.append(node("h3", "失败调用诊断"));
    fragment.append(node("p", "这里保留每次失败尝试；后续格式修复可能成功。任务最终结果以上方状态为准。失败调用仍可能消耗模型额度。", "help"));
    const list = node("ul");
    for (const call of failed) {
      const item = node("li");
      const stage = typeof call.stage === "string" ? call.stage.slice(0, 100) : "阶段未记录";
      const code = typeof call.details?.error_code === "string" ? call.details.error_code.slice(0, 100) : "错误代码未记录";
      item.append(node("p", `${stage} · ${code}`, "mono"));
      const validation = call.details?.validation;
      if (validation && typeof validation.schema === "string" && Array.isArray(validation.issues)) {
        item.append(node("p", `校验结构：${validation.schema.slice(0, 64)}`, "help"));
        for (const issue of validation.issues.slice(0, 8)) {
          if (typeof issue?.path !== "string" || typeof issue?.kind !== "string") continue;
          item.append(node("p", `位置：${issue.path.slice(0, 160)}；类型：${issue.kind.slice(0, 64)}`, "help"));
        }
        if (validation.truncated || validation.issues.length > 8) item.append(node("p", "仅显示有界诊断摘要，未保存或展示模型原始回答。", "help"));
      } else if (code === "schema_mismatch" || code === "invalid_json") {
        item.append(node("p", "此记录未保存字段诊断，无法确定具体字段；不能据此推断是哪一项填写错误。", "help"));
      }
      list.append(item);
    }
    fragment.append(list);
  }
  $("model-audit-failures").replaceChildren(fragment);
  return failed.length;
}

async function loadModelAudit(id, epoch) {
  const readSeq = ++auditReadSeq;
  try {
    const audit = await api(`/api/jobs/${encodeURIComponent(id)}/model-audit`);
    if (disposed || epoch !== selectionEpoch || id !== selectedId || readSeq !== auditReadSeq) return;
    const budget = audit.model_budget;
    const failures = renderModelFailures(audit.calls);
    $("model-audit-panel").classList.toggle("hidden", !budget && !failures);
    if (!budget) {
      $("model-audit-profile").textContent = failures ? "已保存的模型调用记录" : "";
      $("model-audit-budget").textContent = "";
      $("model-audit-usage").textContent = "";
      return;
    }
    const profile = audit.model_profile;
    $("model-audit-profile").textContent = profile ? `${profile.label} · ${profile.provider} / ${profile.model} · 配置 ${profile.fingerprint.slice(0, 12)}` : "原 DeepSeek 配置 · 本次已启用独立预算";
    $("model-audit-budget").textContent = `已预留 ${budget.reserved.calls} / ${budget.effective_max_calls} 次调用；输入 ${budget.reserved.input_bytes} / ${budget.limits.max_input_bytes} 字节；输出 ${budget.reserved.output_tokens} / ${budget.limits.max_output_tokens} tokens。失败和取消不退回预留。`;
    $("model-audit-usage").textContent = `已知用量：${budget.known_usage.total_tokens} tokens（${budget.usage_known_calls} 次有返回用量）；${budget.usage_unknown_calls} 次用量未知。${budget.usage_complete ? "已预留调用的用量均有记录，不等于账单已结算。" : "当前不能得出总用量或精确费用。"}`;
  } catch {
    if (epoch === selectionEpoch && id === selectedId && readSeq === auditReadSeq) {
      $("model-audit-panel").classList.remove("hidden");
      $("model-audit-profile").textContent = "模型账本暂不可读取";
      $("model-audit-budget").textContent = "保留任务状态，请刷新后再核对；不能把读取失败当作零调用。";
      $("model-audit-usage").textContent = "";
      $("model-audit-failures").replaceChildren();
    }
  }
}

async function refreshSelected(epoch = selectionEpoch) {
  const id = selectedId;
  if (!id) return;
  const readSeq = ++selectedReadSeq;
  const job = await api(`/api/jobs/${encodeURIComponent(id)}`);
  if (disposed || epoch !== selectionEpoch || id !== selectedId || readSeq !== selectedReadSeq) return;
  renderJob(job);
}

function eventText(event) {
  const data = event.data;
  if (typeof data === "string") return data;
  if (typeof data?.message === "string") return data.message;
  if (typeof data?.detail === "string") return data.detail;
  if (typeof data?.status === "string") return STATUS[data.status] || data.status;
  if (typeof data?.stage === "string") return `${data.stage}${data.status ? ` · ${STATUS[data.status] || data.status}` : ""}`;
  return `${event.kind || "记录"}${data ? ` · ${JSON.stringify(data)}` : ""}`;
}

function connectEvents(id, epoch) {
  closeStream();
  const connection = new EventSource(`/api/jobs/${encodeURIComponent(id)}/events?after=${seenSequence}`);
  stream = connection;
  $("stream-status").textContent = "正在连接进度……";
  connection.onopen = () => {
    if (epoch === selectionEpoch && stream === connection) $("stream-status").textContent = "实时进度已连接";
  };
  const receive = (message) => {
    if (disposed || epoch !== selectionEpoch || stream !== connection || selectedId !== id) return;
    let event;
    try { event = JSON.parse(message.data); } catch { return; }
    if (!Number.isSafeInteger(event.seq) || event.seq <= seenSequence) return;
    seenSequence = event.seq;
    const item = node("li", null, "event");
    item.append(node("time", timeLabel(event.created_at, true)), node("p", eventText(event)));
    $("events").append(item);
    while ($("events").childElementCount > 200) $("events").firstElementChild.remove();
    $("events-empty").classList.add("hidden");
    if (message.type === "status") {
      // Always read the authoritative job. A progress message is not a terminal-state commit.
      void refreshSelected(epoch).catch(() => {});
      void refreshJobs();
    }
    if (TERMINAL.has(event.data?.status)) {
      closeStream();
      $("stream-status").textContent = "任务已结束";
    }
  };
  connection.addEventListener("status", receive);
  connection.addEventListener("progress", receive);
  connection.onerror = () => {
    if (epoch !== selectionEpoch || stream !== connection) return;
    if (TERMINAL.has(selectedJob?.status)) {
      closeStream();
      $("stream-status").textContent = "任务已结束";
    } else $("stream-status").textContent = "连接暂断，正在自动续接；状态也会定期核对";
  };
}

async function selectJob(id) {
  if (id === selectedId && selectedJob) {
    try { await refreshSelected(); } catch (error) { showNotice("global-error", describeError(error), "error"); }
    return;
  }
  const epoch = ++selectionEpoch;
  selectedId = id;
  selectedJob = null;
  seenSequence = 0;
  reportLoadedFor = null;
  reportLoadingFor = null;
  cancelling = false;
  closeStream();
  resetModelAudit();
  resetReport();
  $("events").replaceChildren();
  $("events-empty").classList.remove("hidden");
  $("report-panel").classList.add("hidden");
  $("task-status").textContent = "正在读取";
  $("task-content").classList.add("hidden");
  $("task-empty").classList.add("hidden");
  showNotice("global-error", "");
  renderJobs();
  try {
    await refreshSelected(epoch);
    if (epoch !== selectionEpoch || disposed) return;
    // Open even terminal jobs once to read the stored event history.
    connectEvents(id, epoch);
  } catch (error) {
    if (epoch === selectionEpoch) showNotice("global-error", describeError(error), "error");
  }
}

function resetReport() {
  $("report-panel").classList.add("hidden");
  $("report").replaceChildren();
  $("review-details").replaceChildren();
  $("gap-retrieval-details").replaceChildren();
  $("research-map").replaceChildren();
  $("download-markdown").removeAttribute("href");
  $("download-json").removeAttribute("href");
}

function renderReviewDetails(bundle) {
  const container = $("review-details");
  container.replaceChildren();
  const recovery = bundle.contract_recovery;
  if (recovery && recovery.version === 1) {
    container.append(node("h3", "输出修复记录"));
    const repairs = Number.isSafeInteger(recovery.repairs) ? recovery.repairs : "未知";
    container.append(node("p", `研究要点与查询的局部重问：${repairs} / 1 次；审查格式修复另有上限。保留草稿不代表通过审查。`, "help"));
    const ids = (values) => Array.isArray(values)
      ? values.filter(v => typeof v === "string" && /^P[1-8]$/.test(v)).slice(0, 8).join("、") : "";
    for (const attempt of (Array.isArray(recovery.report_attempts) ? recovery.report_attempts.slice(0, 2) : [])) {
      const unresolved = ids(attempt?.unresolved_point_ids);
      if (unresolved) container.append(node("p", `仍未通过输出校验的要点：${unresolved}${attempt.retained_draft ? "；已保留上一份有效草稿" : "；其余有效回答已保留"}。`, "notice warning"));
    }
    const fallback = ids(recovery.query_fallback_point_ids);
    if (fallback) container.append(node("p", `查询修复未成功，沿用原始要点检索词：${fallback}。这不表示已补齐证据。`, "help"));
  }
  const review = bundle.review;
  if (!review) return;
  container.append(node("h3", "当前报告的审查问题"));
  container.append(node("p", "仅展示被选中报告版本对应的 Red 意见，不把最后一轮意见套到旧版本上。模型评分与意见都不是独立事实核查。", "help"));
  const hasIdentity = typeof review.best_hash === "string" && /^[a-f0-9]{64}$/.test(review.best_hash)
    && Number.isSafeInteger(review.best_round) && review.best_round >= 0;
  const matches = hasIdentity && Array.isArray(bundle.review_events)
    ? bundle.review_events.map((entry) => entry?.event).filter((event) => event?.kind === "red"
      && event.report_hash === review.best_hash && event.round === review.best_round
      && event.review?.report_hash === review.best_hash)
    : [];
  // A missing/ambiguous historic snapshot is unavailable, never proof of zero findings.
  const findings = matches.length === 1 ? matches[0].review.findings : null;
  const scope = matches.length === 1 ? matches[0].scope_review : null;
  if ([1, 2, 3].includes(scope?.version) && scope.report_hash === review.best_hash && Array.isArray(scope.advice)) {
    const advice = scope.advice.filter(item => typeof item === "string").slice(0, 6);
    if (advice.length) {
      container.append(node("h4", "补充建议（不阻断）"));
      for (const item of advice) container.append(node("p", item, "help"));
    }
    container.append(node("p", "阻断项按问题要点或原结论及引文绑定；补充建议独立保存。绑定校验不等于事实核实。", "help"));
  }
  const validFindings = Array.isArray(findings) && findings.length <= 12 && findings.every((finding) =>
    finding && typeof finding.id === "string" && typeof finding.description === "string"
    && ["evidence", "logic", "citation", "coverage"].includes(finding.category)
    && ["low", "medium", "high"].includes(finding.severity)
    && Array.isArray(finding.claim_ids) && finding.claim_ids.every((id) => typeof id === "string")
    && Array.isArray(finding.evidence_ids) && finding.evidence_ids.every((id) => typeof id === "string"));
  if (!validFindings || !Number.isSafeInteger(review.finding_count) || review.finding_count !== findings.length) {
    container.append(node("p", "此报告的对应审查明细不可用或记录不一致；不能据此判断没有问题。旧记录可能没有保存完整明细，请核对下载的 JSON 与原文。", "notice warning"));
    return;
  }
  container.append(node("p", `展示第 ${review.best_round} 轮选中版本 · ${findings.length} 条待核对意见`, "review-summary"));
  const labels = { evidence_support: "证据支持", logical_consistency: "逻辑一致性", citation_quality: "引用质量" };
  const scores = matches[0].review.scores;
  if (scores && Object.keys(labels).every((key) => Number.isInteger(scores[key]) && scores[key] >= 0 && scores[key] <= 100)) {
    container.append(node("p", Object.entries(labels).map(([key, label]) => `${label} ${scores[key]}/100`).join(" · "), "help"));
  }
  if (!findings.length) {
    container.append(node("p", "该版本没有列出的审查问题；仍需同时满足评分条件，这不等于事实已核实。", "help"));
    return;
  }
  const categories = { evidence: "证据支持", logic: "逻辑一致性", citation: "引用质量", coverage: "覆盖缺口" };
  const severities = { high: "高", medium: "中", low: "低" };
  const list = node("ol", null, "review-findings");
  for (const finding of findings) {
    const item = node("li");
    item.append(node("p", `${finding.id.slice(0, 12)} · ${categories[finding.category]} · 严重程度：${severities[finding.severity]}`, "review-summary"));
    item.append(node("p", finding.description.slice(0, 500)));
    const refs = (values) => values.slice(0, 4).map((value) => value.slice(0, 32)).join("、") || "未指定";
    item.append(node("p", `关联陈述：${refs(finding.claim_ids)}；关联证据：${refs(finding.evidence_ids)}`, "help"));
    list.append(item);
  }
  container.append(list);
}

function renderGapAudit(bundle) {
  const container = $("gap-retrieval-details");
  container.replaceChildren();
  const count = (value) => Number.isSafeInteger(value) && value >= 0 ? String(value) : "未知";
  const quality = bundle.retrieval_quality;
  if (quality && typeof quality === "object") {
    container.append(node("h3", "检索相关性筛选"));
    container.append(node("p", `接受候选 ${count(quality.accepted_count)} 条；过滤候选 ${count(quality.filtered_count)} 条。`, "help"));
    container.append(node("p", quality.enabled && quality.method === "lexical_heuristic"
      ? "使用词语匹配启发式筛选，可减少明显无关资料，但不保证语义相关或事实正确。"
      : quality.enabled && quality.method === "lexical_with_cross_language_rescue"
        ? "词语筛选后，允许本地 E5 找回跨语言资料；相似度是筛选线索，不代表事实核查通过。"
        : "筛选方法未启用或无法识别，不能据此认定资料相关。", "help"));
    if (quality.selection && typeof quality.selection === "object") {
      const selection = quality.selection;
      container.append(node("h3", "按问题要点选择证据"));
      container.append(node("p", `检索方式：${selection.encoder_kind === "semantic" ? "关键词 + 本地 E5" : "关键词"}；候选片段 ${count(selection.candidates)} 个；选择时上下文 ${count(selection.context_bytes)} / ${count(selection.max_bytes)} 字节。`, "help"));
      if (Array.isArray(selection.targets)) container.append(node("p", `覆盖目标：${selection.targets.slice(0, 8).filter(v => typeof v === "string").map(v => v.slice(0, 300)).join("；")}`, "help"));
      container.append(node("p", selection.version === "multilingual-parent-v2"
        ? "本次使用实验检索，按问题要点分配证据，并从原文术语寻找最多两个关联片段。片段相关不等于支持结论，仍需核对原文。"
        : "覆盖目标由问题和检索词拆分。片段命中目标不等于支持结论，仍需核对原文及审查问题。", "help"));
      if (selection.version === "dependency-window-v3") {
        const windows = Array.isArray(selection.windows) ? selection.windows : [];
        container.append(node("p", `本次为上下文实验选段：排除 ${count(selection.removed_structural)} 个导航或裸标签候选；接受 ${windows.filter(v => v.expanded).length} 次上下文扩展，${windows.filter(v => v.fallback).length} 次因预算保留短片段。结构过滤可能误删短事实，扩展不代表结论已获支持。`, "help"));
      }
    }
  }
  const gap = bundle.gap_retrieval;
  if (!gap || typeof gap !== "object") return;
  container.append(node("h3", "有界缺口补查"));
  const boundedText = (value) => typeof value === "string" ? value.slice(0, 160) : "未记录";
  const statuses = {not_needed: "无需补查", skipped: "已跳过", searching: "补查中", completed: "已完成补查", no_new_evidence: "未新增证据", timed_out: "已超时", cancelled: "已取消", failed: "补查失败"};
  const reasons = {no_queries: "没有可用检索词", retrieval_budget: "检索预算不足", model_budget: "模型预算不足", evidence_capacity: "证据数量达到上限", context_budget: "上下文预算不足", no_new_evidence: "没有可接纳的新证据", timeout: "达到时间限制"};
  const status = Object.hasOwn(statuses, gap.status) ? `${statuses[gap.status]}（${gap.status}）` : boundedText(gap.status);
  const reason = gap.reason === null ? "无额外原因" : Object.hasOwn(reasons, gap.reason)
    ? `${reasons[gap.reason]}（${gap.reason}）` : boundedText(gap.reason);
  container.append(node("p", `补查状态：${status}；原因：${reason}`, "help"));
  container.append(node("p", `补查预算计数：检索 ${count(gap.search_calls)} 次；抓取 ${count(gap.fetch_calls)} 次。`, "help"));
  if (Array.isArray(gap.local_added_evidence_ids)) container.append(node("p", `从已保存原文补出的证据：${gap.local_added_evidence_ids.slice(0, 32).filter(v => typeof v === "string").join("、") || "无"}。未重新下载这些来源。`, "help"));
  container.append(node("p", "先预留并保存记录，再发送请求；计数可能包括未实际发出的尝试，失败不退回预算。补查完成不代表缺口已解决，以上方审查意见和原文为准。", "help"));
  if (Array.isArray(gap.queries) && gap.queries.length) {
    container.append(node("p", `补查检索词：${gap.queries.slice(0, 2).map(boundedText).join("；")}`, "help"));
  }
  const refs = (value) => Array.isArray(value) && value.every((id) => typeof id === "string")
    ? value.slice(0, 12).map((id) => id.slice(0, 80)).join("、") || "无" : "未记录";
  container.append(node("p", `新增来源：${refs(gap.added_source_ids)}；新增证据：${refs(gap.added_evidence_ids)}`, "help"));
}

function renderResearchMap(bundle) {
  const container = $("research-map");
  container.replaceChildren();
  container.append(node("h3", "研究要点 · 结论 · 证据 · 缺口"));
  const map = bundle.research_map;
  if (!map) {
    container.append(node("p", "此历史报告未保存研究要点关联记录。新建任务后可查看。", "help"));
    return;
  }
  if (map.version !== 1 || ![map.points, map.conclusions, map.evidence, map.gaps].every(Array.isArray)) {
    container.append(node("p", "关联记录格式不可识别，请核对下载的 JSON。", "notice warning"));
    return;
  }
  container.append(node("p", "要点来自原问题的初步拆分；结论关联为词语匹配候选。引用可追溯不等于支持结论，缺口仍需核对。", "help"));
  if (map.review_binding === "unavailable") container.append(node("p", "当前报告对应的审查记录不可用，不能据此判断没有审查问题。", "notice warning"));
  const conclusions = new Map(map.conclusions.map(c => [c.id, c]));
  const evidence = new Map(map.evidence.map(e => [e.id, e]));
  const gaps = new Map(map.gaps.map(g => [g.id, g]));
  const sources = new Map((bundle.sources || []).map(s => [s.source_id, s]));
  const status = {no_conclusion: "尚无候选结论", needs_review: "已有候选，待核对", review_flagged: "审查指出问题"};
  const gapLabels = {no_conclusion: "结论缺口", missing_evidence: "引用缺口", support_unverified: "支持关系待核对", review_finding: "审查问题"};
  const priorities = {high: "高", medium: "中", low: "低"};
  const showGap = (g, parent) => {
    parent.append(node("p", `${g.id} · ${gapLabels[g.kind] || "待处理问题"} · 优先级：${priorities[g.priority] || "未知"} · 未解决`, "review-summary"));
    parent.append(node("p", g.description));
    if (g.suggested_queries?.length) parent.append(node("p", `建议补查：${g.suggested_queries.join("；")}（尚未据此发起查询）`, "help"));
  };
  for (const p of map.points.slice(0, 8)) {
    const card = node("section", null, "research-point");
    card.append(node("h4", `${p.id} · ${p.question}`));
    card.append(node("p", status[p.status] || "状态待核对", "tag"));
    for (const id of p.claim_ids || []) {
      const c = conclusions.get(id);
      if (!c) continue;
      card.append(node("p", `${c.id} · 候选结论：${c.text}`));
      for (const eid of c.evidence_ids || []) {
        const e = evidence.get(eid);
        if (!e) continue;
        const detail = node("details", null, "map-evidence");
        detail.append(node("summary", `${e.id} · ${sources.get(e.source_id)?.title || e.source_id} · 查看原文`));
        detail.append(node("blockquote", e.quote));
        detail.append(node("p", `来源 ${e.source_id} · 字符位置 [${e.start}, ${e.end})`, "help"));
        detail.append(node("p", sources.get(e.source_id)?.url || "来源网址未记录", "help"));
        card.append(detail);
      }
    }
    for (const id of p.gap_ids || []) { const g = gaps.get(id); if (g) showGap(g, card); }
    container.append(card);
  }
  const unassigned = map.conclusions.filter(c => !c.point_ids?.length);
  if (unassigned.length) {
    container.append(node("h4", "尚未归入要点的结论"));
    for (const c of unassigned) container.append(node("p", `${c.id} · ${c.text} · 引用：${c.evidence_ids.join("、") || "无有效引用"}`));
  }
  const general = map.gaps.filter(g => !g.point_ids?.length);
  if (general.length) {
    container.append(node("h4", "尚未归入要点的缺口"));
    for (const g of general) showGap(g, container);
  }
}

function renderReport(bundle) {
  const report = bundle.publication?.report || bundle.report;
  if (!report || !Array.isArray(report.sections)) throw new TypeError("Invalid report");
  const fragment = document.createDocumentFragment();
  fragment.append(node("h3", report.title));
  if (bundle.support_assessment) {
    fragment.append(node("p", bundle.publication?.abstained
      ? "本次弃答：没有通过支持判定的结论。"
      : "正文仅展示模型判为充分支持的结论；这不是独立事实核查。", "notice warning"));
    fragment.append(node("p", "JSON 的 report 与下方关联图保留原始候选供审计，publication 才是本次正文采用的内容。", "help"));
    const labels = {supported: "充分支持", partial: "部分支持", contradicted: "与引文矛盾", insufficient: "证据不足"};
    for (const item of bundle.support_assessment.decisions || []) {
      fragment.append(node("p", `${item.claim_id} · ${labels[item.verdict] || "未知"}：${item.reason}`, "help"));
    }
  }
  if (bundle.evidence_rerank) fragment.append(node("p", `证据精排状态：${bundle.evidence_rerank.status}；候选 ${bundle.evidence_rerank.candidates} 个。精排相关性不等于事实支持。`, "help"));
  if (bundle.support_assessment?.version === 2) {
    const statuses = {answered: "已回答", partial: "部分回答", abstained: "未回答 / 弃答"};
    fragment.append(node("h4", "逐要点回答状态（模型判断）"));
    for (const point of bundle.publication?.points || []) fragment.append(node("p", `${point.id} · ${statuses[point.status] || "未知"} · ${point.question}${point.gap_query ? `；剩余检索线索：${point.gap_query}` : ""}`, "help"));
    fragment.append(node("h4", "事实支持与非事实说明"));
    for (const item of bundle.support_assessment.claims || []) fragment.append(node("p", `${item.claim_id} · ${item.kind} · ${item.support}：${item.reason}`, "help"));
    for (const item of bundle.publication?.nonfactual || []) fragment.append(node("p", `${item.kind === "abstention" ? "弃答" : "范围说明"}：${item.text}（不计入事实支持率）`, "help"));
  }
  if (bundle.research_loop) fragment.append(node("p", `查询改写：${bundle.research_loop.query_rewrite?.status || "未知"}；局部修订：${bundle.research_loop.repair?.status || "无需修订"}。补查和修订不等于事实已核实。`, "help"));
  if (bundle.embedding_cache) fragment.append(node("p", `编码缓存：命中 ${bundle.embedding_cache.hits} 条，实际编码 ${bundle.embedding_cache.encoded_texts} 条，批次 ${bundle.embedding_cache.batches}。`, "help"));
  for (const section of report.sections) {
    fragment.append(node("h4", section.heading));
    for (const claim of section.claims || []) {
      const paragraph = node("p", claim.text, "claim");
      for (const citation of claim.citations || []) paragraph.append(node("span", citation, "citation"));
      fragment.append(paragraph);
    }
  }
  const limitations = [...(report.limitations || []), ...(bundle.system_limitations || [])];
  if (limitations.length) {
    const section = node("section", null, "limitations");
    section.append(node("h4", "局限与使用边界"));
    const list = node("ul");
    for (const limitation of limitations) list.append(node("li", limitation));
    section.append(list);
    fragment.append(section);
  }
  $("report").replaceChildren(fragment);
  const review = bundle.review;
  const degraded = Boolean(bundle.execution?.degraded?.length || bundle.execution?.degraded === true || bundle.warnings?.length || bundle.execution?.warnings?.length);
  if (degraded || (review && review.status !== "passed")) showNotice("report-notice", `降级草稿 / 未通过审查。${review?.stop_reason ? `停止原因：${review.stop_reason}。` : ""}请先核对原文，不应直接作为事实结论使用。`, "warning");
  else if (review?.status === "passed") showNotice("report-notice", "通过本轮模型审查。此状态不是事实正确性的证明；Red 与 Blue 使用同一模型接口，仍需核对原文证据。");
  else showNotice("report-notice", "未启用 Red–Blue 审查。本报告仅表示生成流程完成，不代表研究质量已经合格。", "warning");
  renderReviewDetails(bundle);
  renderGapAudit(bundle);
  renderResearchMap(bundle);
  $("report-panel").classList.remove("hidden");
}

async function loadReport(id, epoch) {
  if (reportLoadedFor === id || reportLoadingFor === id) return;
  reportLoadingFor = id;
  try {
    const bundle = await api(`/api/jobs/${encodeURIComponent(id)}/report`);
    if (disposed || epoch !== selectionEpoch || id !== selectedId) return;
    renderReport(bundle);
    $("download-markdown").href = `/api/jobs/${encodeURIComponent(id)}/download/markdown`;
    $("download-json").href = `/api/jobs/${encodeURIComponent(id)}/download/json`;
    reportLoadedFor = id;
  } catch (error) {
    if (!disposed && epoch === selectionEpoch && id === selectedId) {
      resetReport();
      showNotice("task-notice", `任务已结束，但报告读取失败。${describeError(error)}`, "warning");
    }
  } finally {
    if (epoch === selectionEpoch && reportLoadingFor === id) reportLoadingFor = null;
  }
}

async function submitJob(event) {
  event.preventDefault();
  if (submitting || !config) return;
  const live = $("mode").value === "live";
  if (live && (!config.allow_live || !$("confirm-live").checked)) return;
  const request = { mode: live ? "live" : "demo", question: live ? $("question").value.trim() : config.demo_question, scenario: live ? "repair" : $("scenario").value, review: $("review").checked, compression: $("compression").checked, memory_scope: live ? $("memory-scope").value.trim() || null : null, confirm_live: live && $("confirm-live").checked };
  if (live && config.supports_robust_contracts && $("robust-contracts").checked) request.robust_contracts = true;
  if (live && config.supports_evidence_gate && $("rerank-evidence").checked) request.rerank_evidence = true;
  if (live && config.supports_evidence_gate && $("check-support").checked) request.check_support = true;
  if (live && config.supports_research_loop && $("optimize-research").checked) request.optimize_research = true;
  if (live && config.model_profiles?.length) request.profile_id = $("model-profile").value;
  if (!request.question) { $("question").focus(); return; }
  const fingerprint = JSON.stringify(request);
  // Keep this key after an unknown network outcome; retrying the same payload must not create a second job.
  if (!pendingSubmission || pendingSubmission.fingerprint !== fingerprint) pendingSubmission = { fingerprint, key: crypto.randomUUID() };
  submitting = true;
  updateForm();
  showNotice("global-error", "");
  $("submit-status").textContent = "正在保存任务……";
  try {
    const job = await api("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json", "X-CSRF-Token": config.csrf_token, "Idempotency-Key": pendingSubmission.key }, body: fingerprint });
    if (typeof job.id !== "string") throw new TypeError("Missing job ID");
    pendingSubmission = null;
    if (live) $("confirm-live").checked = false;
    $("submit-status").textContent = "任务已保存。关闭网页不会自动取消；可在执行记录中主动停止。";
    await refreshJobs();
    await selectJob(job.id);
  } catch (error) {
    showNotice("global-error", describeError(error), "error");
    $("submit-status").textContent = "如未确认是否创建成功，请保持选项不变再次提交，将复用同一提交标识。也可以先刷新任务列表。";
  } finally {
    submitting = false;
    updateForm();
  }
}

async function cancelJob() {
  if (!config || !selectedJob || cancelling || TERMINAL.has(selectedJob.status)) return;
  const id = selectedId;
  const epoch = selectionEpoch;
  cancelling = true;
  $("cancel-job").disabled = true;
  try {
    const job = await api(`/api/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST", headers: { "X-CSRF-Token": config.csrf_token } });
    if (epoch === selectionEpoch && id === selectedId) renderJob(job);
    await refreshJobs();
  } catch (error) {
    if (epoch === selectionEpoch) showNotice("task-notice", `未确认取消成功。${describeError(error)}请核对任务最终状态。`, "warning");
  } finally {
    if (epoch === selectionEpoch) {
      cancelling = false;
      if (selectedJob) $("cancel-job").disabled = TERMINAL.has(selectedJob.status) || Boolean(selectedJob.cancel_requested);
    }
  }
}

$("mode").addEventListener("change", () => {
  if ($("mode").value === "demo") {
    liveQuestion = $("question").value;
    $("question").value = config.demo_question;
  } else $("question").value = liveQuestion;
  $("confirm-live").checked = false;
  updateForm();
});
$("confirm-live").addEventListener("change", updateForm);
$("model-profile").addEventListener("change", () => { $("confirm-live").checked = false; updateForm(); });
$("scenario").addEventListener("change", updateForm);
$("research-form").addEventListener("submit", (event) => { void submitJob(event); });
$("cancel-job").addEventListener("click", () => { void cancelJob(); });
$("refresh-jobs").addEventListener("click", () => {
  if (!config) void initialize();
  else {
    void refreshJobs();
    void refreshSelected().catch((error) => showNotice("global-error", describeError(error), "error"));
  }
});

async function initialize() {
  try {
    await loadConfig();
    showNotice("global-error", "");
    await refreshJobs();
  } catch (error) {
    config = null;
    updateForm();
    showNotice("global-error", `${describeError(error)}点击任务记录旁的“刷新”重新连接。`, "error");
    $("service-info").textContent = "尚未连接本地服务";
  }
}

const pollTimer = setInterval(async () => {
  if (disposed || polling || !config || document.hidden) return;
  polling = true;
  try {
    await refreshJobs();
    if (selectedId) await refreshSelected();
  } catch { /* Preserve the last committed state; reconnecting SSE and manual refresh remain available. */ }
  finally { polling = false; }
}, 3000);
window.addEventListener("pagehide", () => { disposed = true; closeStream(); clearInterval(pollTimer); });
void initialize();
