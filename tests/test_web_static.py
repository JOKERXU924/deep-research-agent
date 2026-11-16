"""Small static safety/accessibility checks, not a substitute for browser QA."""

import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from typing import get_args

import pytest

from deep_research.web_models import WebRequest

STATIC = Path(__file__).parents[1] / "src" / "deep_research" / "static"


class Elements(HTMLParser):
    def __init__(self, html: str):
        super().__init__()
        self.elements: list[tuple[str, dict[str, str | None]]] = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


@pytest.fixture
def html():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@pytest.fixture
def js():
    return (STATIC / "app.js").read_text(encoding="utf-8")


def test_page_has_unique_ids_and_matching_script_references(html, js):
    ids = [attrs["id"] for _, attrs in Elements(html).elements if "id" in attrs]
    assert len(ids) == len(set(ids))
    referenced = set(re.findall(r'\$\("([a-z-]+)"\)', js))
    assert referenced <= set(ids)


def test_form_controls_have_explicit_labels(html):
    elements = Elements(html).elements
    labels = {attrs.get("for") for tag, attrs in elements if tag == "label"}
    for tag, attrs in elements:
        if tag in {"textarea", "select"} or (tag == "input" and attrs.get("type") == "text"):
            assert attrs["id"] in labels


def test_no_external_assets_or_inline_handlers(html):
    for tag, attrs in Elements(html).elements:
        assert not any(key.lower().startswith("on") for key in attrs)
        for key in ("src", "href"):
            if attrs.get(key):
                assert attrs[key].startswith(("/", "#"))
        if tag == "script":
            assert attrs.get("src") == "/app.js"
            assert "defer" in attrs


def test_demo_default_is_fixed_and_does_not_confirm_real_calls(html):
    controls = {attrs["id"]: attrs for _, attrs in Elements(html).elements if "id" in attrs}
    assert "readonly" in controls["question"]
    assert controls["question"]["maxlength"] == "1000"
    assert "checked" in controls["review"]
    assert "checked" not in controls["compression"]
    assert "checked" not in controls["confirm-live"]
    assert "disabled" in controls["submit-job"]
    assert "固定教学样例" in html


def test_selection_and_local_gap_are_visible_and_not_interpreted_as_markup(js):
    run_ui_checks(
        js,
        """
renderGapAudit({retrieval_quality: {enabled: true, method: "lexical_heuristic",
  selection: {encoder_kind: "semantic", candidates: 50, context_bytes: 2000, max_bytes: 3000,
    targets: ["<img src=x onerror=alert(1)>"]}}, gap_retrieval: {
    status: "completed", search_calls: 0, fetch_calls: 0, local_added_evidence_ids: ["E9"]}});
const text = $("gap-retrieval-details").textContent;
const tags = (element) => [element.tag, ...element.children.flatMap(tags)];
assert.match(text, /关键词 \\+ 本地 E5/);
assert.match(text, /2000 \\/ 3000 字节/);
assert.match(text, /从已保存原文补出的证据：E9/);
assert.match(text, /<img src=x/);
assert.equal(tags($("gap-retrieval-details")).includes("img"), false);
""",
    )


def test_scenario_options_match_server_contract(html):
    options = {
        attrs["value"]
        for tag, attrs in Elements(html).elements
        if tag == "option" and attrs["value"] not in {"demo", "live"}
    }
    assert options == set(get_args(WebRequest.model_fields["scenario"].annotation))


@pytest.mark.parametrize(
    "forbidden", ["innerHTML", "outerHTML", "insertAdjacentHTML", "document.write(", "eval("]
)
def test_model_content_cannot_be_interpreted_as_html(js, forbidden):
    assert forbidden not in js
    assert "textContent" in js
    assert "document.createElement" in js


def test_request_guards_and_reconnect_bookkeeping_are_present(js):
    for required in (
        '"X-CSRF-Token"',
        '"Idempotency-Key"',
        "crypto.randomUUID()",
        "pendingSubmission.fingerprint !== fingerprint",
        "event.seq <= seenSequence",
        "epoch !== selectionEpoch",
        "readSeq !== selectedReadSeq",
        "connection.close",
    ):
        if required == "connection.close":
            assert "stream.close()" in js
        else:
            assert required in js
    assert 'if (live) $("confirm-live").checked = false' in js
    assert '"pagehide"' in js


def test_report_quality_is_not_inferred_from_completion(html, js):
    assert "执行完成 ≠ 研究质量合格" in html
    assert 'review.status !== "passed"' in js
    assert "降级草稿 / 未通过审查" in js
    assert 'job.result?.memory_write?.status === "failed"' in js


def test_css_has_narrow_screen_and_keyboard_focus_support():
    css = (STATIC / "styles.css").read_text(encoding="utf-8")
    assert "@media(max-width:720px)" in css
    assert ":focus-visible" in css
    assert "prefers-reduced-motion" in css
    assert "@import" not in css
    assert "url(" not in css


def run_ui_checks(js: str, checks: str) -> None:
    """Exercise pure DOM behavior without a server, browser, or model calls."""
    executable = shutil.which("node")
    if executable is None:
        pytest.skip("Node.js is optional; required only for static UI behavior checks")
    harness = r"""
const { readFileSync } = require("node:fs");
const vm = require("node:vm");
const assert = require("node:assert/strict");
const { source, checks } = JSON.parse(readFileSync(0, "utf8"));
class Element {
  constructor(tag) {
    this.tag = tag;
    this.children = [];
    this.text = "";
    this.classes = new Set();
    this.classList = {
      add: (name) => this.classes.add(name),
      remove: (name) => this.classes.delete(name),
      contains: (name) => this.classes.has(name),
      toggle: (name, force) => force ? this.classes.add(name) : this.classes.delete(name),
    };
  }
  set textContent(value) { this.text = String(value); this.children = []; }
  get textContent() { return this.text + this.children.map((item) => item.textContent).join(" "); }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.text = ""; this.children = children; }
  addEventListener() {}
  setAttribute() {}
  removeAttribute(name) { delete this[name]; }
}
const elements = new Map();
const context = vm.createContext({
  assert,
  document: {
    getElementById: (id) => {
      if (!elements.has(id)) elements.set(id, new Element("div"));
      return elements.get(id);
    },
    createElement: (tag) => new Element(tag),
    createDocumentFragment: () => new Element("fragment"),
  },
  window: { addEventListener() {} },
  // Startup is deliberately held pending; no network request is ever dispatched.
  fetch: () => new Promise(() => {}),
  setInterval: () => 0,
  clearInterval() {},
});
(async () => {
  vm.runInContext(source, context);
  await vm.runInContext(checks, context);
})().catch((error) => { console.error(error); process.exitCode = 1; });
"""
    result = subprocess.run(
        [executable, "-e", harness],
        input=json.dumps({"source": js, "checks": checks}),
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_model_failure_diagnostics_show_actual_stage_and_allow_legacy_records(js):
    run_ui_checks(
        js,
        """
assert.match(describeError({code: "schema_mismatch"}), /字段结构/);
assert.match(describeError({code: "invalid_json"}), /无法解析/);
assert.equal(renderModelFailures([
  {stage: "report", status: "completed", details: {}},
  {stage: "review.red.0", status: "failed", details: {error_code: "schema_mismatch"}},
  {stage: "review.blue.1", status: "failed", details: {
    error_code: "schema_mismatch",
    validation: {schema: "BlueRepairs", issues: [
      {path: "operations[0]", kind: "value_error"}
    ], truncated: false}
  }}
]), 2);
const text = $("model-audit-failures").textContent;
assert.match(text, /review.red.0/);
assert.match(text, /review.blue.1/);
assert.match(text, /BlueRepairs/);
assert.match(text, /operations\\[0\\]/);
assert.match(text, /未保存字段诊断/);
assert.match(text, /后续格式修复可能成功/);
assert.doesNotMatch(text, /report ·/);
""",
    )


def test_model_failure_diagnostics_are_bounded_text_and_clear_when_absent(js):
    run_ui_checks(
        js,
        """
const payload = "<img src=x onerror=alert(1)>";
renderModelFailures([{stage: "review.blue.1", status: "failed", details: {
  error_code: "schema_mismatch",
  validation: {schema: "BlueRepairs", issues: Array.from({length: 20}, () => ({
    path: payload, kind: "value_error"
  })), truncated: true}
}}]);
const container = $("model-audit-failures");
assert.match(container.textContent, /<img src=x/);
assert.equal((container.textContent.match(/value_error/g) || []).length, 8);
assert.match(container.textContent, /有界诊断摘要/);
function tags(element) { return [element.tag, ...element.children.flatMap(tags)]; }
assert.equal(tags(container).includes("img"), false);
assert.equal(renderModelFailures(undefined), 0);
assert.equal(container.textContent, "");
""",
    )


def test_model_failure_diagnostics_reset_immediately_when_selection_changes(js):
    run_ui_checks(
        js,
        """
renderModelFailures([{stage: "review.blue.1", status: "failed", details: {
  error_code: "schema_mismatch"
}}]);
$("model-audit-profile").textContent = "old profile";
void selectJob("another-job");
assert.equal($("model-audit-failures").textContent, "");
assert.equal($("model-audit-profile").textContent, "");
assert.equal($("model-audit-panel").classList.contains("hidden"), true);
""",
    )


def test_model_failure_diagnostics_ignore_stale_reads_and_clear_on_read_failure(js):
    run_ui_checks(
        js,
        """
(async () => {
  selectedId = "chosen-job";
  selectionEpoch = 3;
  const pending = [];
  fetch = () => new Promise((resolve) => pending.push(resolve));
  const first = loadModelAudit("chosen-job", 3);
  const second = loadModelAudit("chosen-job", 3);
  function response(stage) {
    return {ok: true, json: async () => ({model_budget: null, calls: [{
      stage, status: "failed", details: {error_code: "schema_mismatch"}
    }]})};
  }
  pending[1](response("review.blue.1"));
  await second;
  pending[0](response("review.red.0"));
  await first;
  assert.match($("model-audit-failures").textContent, /review.blue.1/);
  assert.doesNotMatch($("model-audit-failures").textContent, /review.red.0/);
  assert.equal($("model-audit-panel").classList.contains("hidden"), false);
  fetch = async () => { throw new Error("unavailable"); };
  await loadModelAudit("chosen-job", 3);
  assert.equal($("model-audit-failures").textContent, "");
  assert.match($("model-audit-profile").textContent, /暂不可读取/);
})()
""",
    )


def test_review_findings_follow_selected_hash_and_round_not_last_event(js):
    run_ui_checks(
        js,
        """
const chosenHash = "a".repeat(64);
const otherHash = "b".repeat(64);
function event(round, hash, description) {
  return {event: {kind: "red", round, report_hash: hash, review: {
    report_hash: hash,
    scores: {evidence_support: 72, logical_consistency: 85, citation_quality: 78},
    findings: [{id: "F1", category: "coverage", severity: "high", description,
      claim_ids: [], evidence_ids: ["E2"]}]
  }}};
}
renderReviewDetails({
  review: {best_hash: chosenHash, best_round: 0, finding_count: 1, status: "unresolved"},
  review_events: [event(0, chosenHash, "选中版本仍缺少光反应证据"),
    event(1, otherHash, "非选中版本的意见"),
    event(2, chosenHash, "相同哈希但不同轮次的意见")]
});
const text = $("review-details").textContent;
assert.match(text, /展示第 0 轮选中版本/);
assert.match(text, /选中版本仍缺少光反应证据/);
assert.match(text, /证据支持 72\\/100/);
assert.match(text, /关联陈述：未指定；关联证据：E2/);
assert.doesNotMatch(text, /非选中版本的意见|相同哈希但不同轮次的意见/);
""",
    )


def test_review_missing_ambiguous_or_inconsistent_snapshots_are_not_zero_issues(js):
    run_ui_checks(
        js,
        """
const hash = "a".repeat(64);
const review = {best_hash: hash, best_round: 1, finding_count: 0, status: "passed"};
const event = {event: {kind: "red", round: 1, report_hash: hash,
  review: {report_hash: hash, findings: []}}};
for (const bundle of [
  {review},
  {review, review_events: [event, event]},
  {review: {...review, finding_count: 1}, review_events: [event]},
  {review: {status: "passed"}, review_events: [event]},
  {review, review_events: [{event: {...event.event,
    review: {report_hash: "b".repeat(64), findings: []}}}]}
]) {
  renderReviewDetails(bundle);
  assert.match($("review-details").textContent, /审查明细不可用或记录不一致/);
  assert.doesNotMatch($("review-details").textContent, /0 条待核对意见|没有列出的审查问题/);
}
renderReviewDetails({review, review_events: [event]});
assert.match($("review-details").textContent, /0 条待核对意见/);
assert.match($("review-details").textContent, /仍需同时满足评分条件/);
renderReviewDetails({});
assert.equal($("review-details").textContent, "");
""",
    )


def test_review_findings_and_gap_audit_are_bounded_text_not_markup(js):
    run_ui_checks(
        js,
        """
const payload = "<img src=x onerror=alert(1)>";
const hash = "a".repeat(64);
renderReviewDetails({
  review: {best_hash: hash, best_round: 1, finding_count: 1},
  review_events: [{event: {kind: "red", round: 1, report_hash: hash, review: {
    report_hash: hash, findings: [{id: payload, category: "evidence", severity: "medium",
      description: payload + "x".repeat(1000), claim_ids: [payload], evidence_ids: [payload]}]
  }}}]
});
renderGapAudit({gap_retrieval: {status: payload, reason: "budget_exhausted",
  added_source_ids: [payload], added_evidence_ids: ["E5"], search_calls: 2, fetch_calls: 1}});
const container = $("review-details");
assert.match(container.textContent, /<img src=x/);
assert.ok(container.textContent.length < 1000);
function tags(element) { return [element.tag, ...element.children.flatMap(tags)]; }
assert.equal(tags(container).includes("img"), false);
assert.equal(tags($("gap-retrieval-details")).includes("img"), false);
assert.match($("gap-retrieval-details").textContent, /补查完成不代表缺口已解决/);
assert.match($("gap-retrieval-details").textContent, /预算计数：检索 2 次；抓取 1 次/);
assert.match($("gap-retrieval-details").textContent, /计数可能包括未实际发出的尝试/);
assert.match($("gap-retrieval-details").textContent, /失败不退回预算/);
renderGapAudit({gap_retrieval: {status: "skipped"}});
assert.match($("gap-retrieval-details").textContent, /预算计数：检索 未知 次/);
assert.match($("gap-retrieval-details").textContent, /新增来源：未记录/);
renderGapAudit({});
assert.equal($("gap-retrieval-details").textContent, "");
""",
    )


def test_review_details_clear_on_selection_and_report_read_failure(js):
    run_ui_checks(
        js,
        """
(async () => {
  $("review-details").textContent = "previous findings";
  $("gap-retrieval-details").textContent = "previous audit";
  $("download-json").href = "/old-report";
  void selectJob("new-job");
  assert.equal($("review-details").textContent, "");
  assert.equal($("gap-retrieval-details").textContent, "");
  assert.equal($("download-json").href, undefined);
  assert.equal($("report-panel").classList.contains("hidden"), true);
  $("review-details").textContent = "partial findings";
  fetch = async () => { throw new Error("unavailable"); };
  await loadReport("new-job", selectionEpoch);
  assert.equal($("review-details").textContent, "");
  assert.equal($("report-panel").classList.contains("hidden"), true);
  assert.match($("task-notice").textContent, /报告读取失败/);
})()
""",
    )


def test_retrieval_audit_distinguishes_filtering_and_unavailable_counts(js):
    run_ui_checks(
        js,
        """
renderGapAudit({retrieval_quality: {enabled: true, method: "lexical_heuristic",
  accepted_count: 3, filtered_count: 2}, gap_retrieval: {status: "skipped",
  reason: "model_budget", search_calls: 0, fetch_calls: 0,
  added_source_ids: [], added_evidence_ids: [], queries: []}});
let text = $("gap-retrieval-details").textContent;
assert.match(text, /接受候选 3 条；过滤候选 2 条/);
assert.match(text, /不保证语义相关或事实正确/);
assert.match(text, /已跳过（skipped）/);
assert.match(text, /模型预算不足（model_budget）/);
renderGapAudit({retrieval_quality: {enabled: false}});
text = $("gap-retrieval-details").textContent;
assert.match(text, /接受候选 未知 条/);
assert.match(text, /不能据此认定资料相关/);
assert.doesNotMatch(text, /已跳过|model_budget/);
""",
    )


def test_report_late_response_cannot_replace_new_selection_findings(js):
    run_ui_checks(
        js,
        """
(async () => {
  selectedId = "old-job";
  selectionEpoch = 4;
  let resolveOld;
  fetch = () => new Promise((resolve) => { resolveOld = resolve; });
  const oldRead = loadReport("old-job", 4);
  selectedId = "new-job";
  selectionEpoch = 5;
  $("review-details").textContent = "new job findings";
  resolveOld({ok: true, json: async () => ({report: {title: "old report", sections: []},
    review: {status: "passed"}})});
  await oldRead;
  assert.equal($("review-details").textContent, "new job findings");
  assert.doesNotMatch($("report").textContent, /old report/);
})()
""",
    )


def test_report_bundle_contract_retains_warning_after_successful_gap_retrieval(js):
    run_ui_checks(
        js,
        """
const hash = "c".repeat(64);
const finding = {id: "F1", category: "coverage", severity: "medium", claim_ids: [],
  evidence_ids: ["E1"], description: "仍需核对未覆盖问题", search_queries: ["Calvin cycle"]};
renderReport({
  report: {title: "研究报告", sections: [{heading: "结论", claims: [
    {text: "证据有限的结论", citations: ["E1"]}
  ]}], limitations: []},
  execution: {degraded: ["review_unresolved"]},
  review: {status: "unresolved", stop_reason: "max_rounds", best_hash: hash,
    best_round: 0, best_evidence_ids: ["E1"], finding_count: 1,
    final_scores: {evidence_support: 70, logical_consistency: 75, citation_quality: 70}},
  review_events: [
    {seq: 2, event: {kind: "red", round: 0, report_hash: hash, evidence_ids: ["E1"],
      review: {report_hash: hash, findings: [finding], scores: {
        evidence_support: 70, logical_consistency: 75, citation_quality: 70}}}},
    {seq: 5, event: {kind: "red", round: 1, report_hash: hash, evidence_ids: ["E1", "E2"],
      review: {report_hash: hash, findings: [], scores: {
        evidence_support: 50, logical_consistency: 50, citation_quality: 50}}}}
  ],
  retrieval_quality: {enabled: true, method: "lexical_heuristic",
    accepted_count: 3, filtered_count: 1},
  gap_retrieval: {status: "completed", reason: null, queries: ["Calvin cycle"],
    search_calls: 1, fetch_calls: 1, added_source_ids: ["S2"], added_evidence_ids: ["E2"]}
});
assert.match($("report-notice").textContent, /降级草稿 \\/ 未通过审查/);
assert.match($("report-notice").textContent, /max_rounds/);
assert.match($("review-details").textContent, /展示第 0 轮选中版本/);
assert.match($("review-details").textContent, /仍需核对未覆盖问题/);
assert.match($("review-details").textContent, /证据支持 70\\/100/);
assert.match($("gap-retrieval-details").textContent, /已完成补查（completed）/);
assert.match($("gap-retrieval-details").textContent, /新增来源：S2；新增证据：E2/);
assert.match($("gap-retrieval-details").textContent, /补查检索词：Calvin cycle/);
assert.match($("gap-retrieval-details").textContent, /补查完成不代表缺口已解决/);
""",
    )
