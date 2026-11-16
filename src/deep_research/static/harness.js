"use strict";
const el = id => document.getElementById(id);
let currentStatus = "idle";
const names = {
  ready: "待执行", running: "执行中", interrupted: "已中断，可检查后恢复",
  blocked: "模型结果未知，已阻止恢复", completed: "执行已完成",
  attempt_started: "开始一次执行", action_started: "开始步骤", action_completed: "步骤已保存",
  action_failed: "步骤失败", checkpoint_reused: "复用已保存结果",
  model: "模型处理", search: "检索资料", fetch: "读取资料",
  restore_local: "恢复本地证据", search_new: "补查新来源", revise_existing: "修正已有结论",
  search_alternative: "核查来源冲突", unverified: "待复核", assessed_answered: "支持复核已回答",
  open: "仍未解决", answered: "已有回答", partial: "部分回答", missing: "缺少回答",
  needs_review: "仍需核对", no_conclusion: "尚无关联结论", linked: "已关联候选结论",
  ResearchPlan: "规划研究", ResearchReport: "生成报告", RedReview: "审查报告",
  BlueRepairs: "局部修订", ReferenceRed: "编号审查", ResearchAssessment: "支持复核",
  ShortQueries: "改写检索词", PointAnswers: "按要点生成回答", LocalRepair: "修复缺口"
};
async function read(url) {
  const response = await fetch(url, {cache: "no-store"});
  if (!response.ok) throw new Error("运行记录暂时无法读取。");
  return response.json();
}
function text(parent, value, tag="p") {
  const node = document.createElement(tag); node.textContent = value; parent.append(node); return node;
}
async function show() {
  if (!el("runs").value) return;
  const row = await read(`/api/harness/${encodeURIComponent(el("runs").value)}`);
  currentStatus = row.status;
  el("notice").textContent = names[row.status] || row.status;
  const done = row.actions.filter(a => a.status === "completed").length;
  el("summary").textContent = `已保存 ${done} 个外部步骤；本任务原期限：${new Date(row.deadline*1000).toLocaleString()}`;
  const reused = row.events.filter(e => e.kind === "checkpoint_reused");
  el("reuse").textContent = `累计复用 ${reused.length} 次检查点结果，未因此重新发出对应外部请求。`;
  const cache = row.result?.retrieval_cache;
  el("cache").textContent = cache ? `最近一次执行：检索缓存命中 ${cache.hits} 次，合并等待 ${cache.joined} 次，上游请求 ${cache.upstream} 次。` : "执行结束后显示缓存汇总。";
  el("timeline").replaceChildren();
  const starts = new Map();
  for (const e of row.events) {
    if (e.kind === "action_started") starts.set(e.data.key, e.at);
    const duration = ["action_completed", "action_failed"].includes(e.kind) && starts.has(e.data.key)
      ? ` · 耗时 ${((e.at-starts.get(e.data.key))*1000).toFixed(0)} 毫秒` : "";
    const detail = e.data.query || e.data.title || names[e.data.stage] || e.data.stage || e.data.code || "";
    const pointIds = (e.data.point_ids || []).join("、");
    text(el("timeline"), `${new Date(e.at*1000).toLocaleTimeString()} · ${names[e.kind] || e.kind} · ${names[e.data.kind] || ""} ${detail} ${pointIds}${duration}`, "li");
  }
  el("points").replaceChildren(); el("gaps").replaceChildren();
  const map = row.result?.research_map;
  const points = map?.points || map?.requirements || [];
  if (!points.length) text(el("points"), "尚无已保存的研究要点结果。");
  for (const p of points) {
    const details = document.createElement("details"); el("points").append(details);
    text(details, `${p.id || p.point_id}: ${p.question || p.text || ""} · ${names[p.status] || p.status || "待检查"}`, "summary");
    for (const claim of map.conclusions || []) {
      if (!(claim.point_ids || []).includes(p.id)) continue;
      text(details, `${claim.id}：${claim.text}`);
      for (const eid of claim.evidence_ids || []) {
        const evidence = (map.evidence || []).find(e => e.id === eid);
        if (evidence) text(details, `${eid} · ${evidence.source_id} 原文位置 ${evidence.start}–${evidence.end}：${evidence.quote}`, "blockquote");
      }
    }
    for (const gap of map.gaps || []) if ((gap.point_ids || []).includes(p.id)) text(details, `未解决：${gap.description}`);
  }
  const routing = row.result?.gap_retrieval?.routing;
  for (const t of routing?.targets || []) text(el("gaps"), `${t.point_id} · ${names[t.action] || t.action} · ${names[t.closure] || t.closure}`);
  if (row.result?.review_status) text(el("gaps"), `整份报告审查：${row.result.review_status === "passed" ? "通过" : "未通过或尚未解决"}`);
}
async function refresh() {
  try {
    const previous = el("runs").value;
    const data = await read("/api/harness"); el("runs").replaceChildren();
    for (const r of data.runs) {
      const option = document.createElement("option"); option.value = r.id;
      option.textContent = `${new Date(r.created*1000).toLocaleString()} · ${names[r.status] || r.status} · ${r.id.slice(0,8)}`;
      el("runs").append(option);
    }
    if (data.runs.some(r => r.id === previous)) el("runs").value = previous;
    if (!data.runs.length) el("notice").textContent = "还没有可恢复任务。先运行离线 harness-start 演示。";
    else await show();
  } catch (error) { el("notice").textContent = error.message; }
}
el("runs").addEventListener("change", () => show().catch(e => el("notice").textContent=e.message));
el("refresh").addEventListener("click", refresh);
refresh();
setInterval(() => { if (!document.hidden && currentStatus !== "completed") refresh(); }, 5000);
