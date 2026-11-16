"""Research map DOM behavior including unassigned gaps, XSS and historical reports."""

import pytest
from test_web_static import STATIC, run_ui_checks


@pytest.fixture
def js():
    return (STATIC / "app.js").read_text(encoding="utf-8")


def test_map_shows_relationships_exact_quotes_and_clears_old_selection(js):
    run_ui_checks(
        js,
        r"""
const bundle = {sources: [
  {source_id: 'S1', title: '官方说明', url: 'javascript:alert(1)'}], research_map: {
  version: 1, review_binding: 'matched',
  points: [{id:'P1', question:'写入限制？', status:'review_flagged',
    claim_ids:['C1'], gap_ids:['G1']},
    {id:'P2', question:'部署成本？', status:'no_conclusion', claim_ids:[], gap_ids:['G2']}],
  conclusions:[{id:'C1', text:'候选结论', point_ids:['P1'], evidence_ids:['E1']},
    {id:'C2', text:'尚未归属', point_ids:[], evidence_ids:[]}],
  evidence:[{id:'E1', source_id:'S1', start:2, end:8, quote:'<img src=x onerror=alert(1)>'}],
  gaps:[{id:'G1', kind:'review_finding', priority:'high', point_ids:['P1'],
      description:'条件不足', suggested_queries:['并发条件']},
    {id:'G2', kind:'no_conclusion', priority:'high', point_ids:['P2'], description:'未回答成本'},
    {id:'G3', kind:'review_finding', priority:'medium', point_ids:[], description:'整体缺口'}]
}};
renderResearchMap(bundle);
let text = $('research-map').textContent;
assert.match(text, /P1.*写入限制/);
assert.match(text, /P2.*部署成本/);
assert.match(text, /E1.*官方说明/);
assert.match(text, /字符位置 \[2, 8\)/);
assert.match(text, /G1.*审查问题/);
assert.match(text, /未回答成本/);
assert.match(text, /尚未归入要点的结论/);
assert.match(text, /整体缺口/);
assert.match(text, /尚未据此发起查询/);
const tags = el => [el.tag, ...el.children.flatMap(tags)];
assert.equal(tags($('research-map')).includes('img'), false);
assert.equal(tags($('research-map')).includes('a'), false);
assert.equal(tags($('research-map')).includes('details'), true);
resetReport();
assert.equal($('research-map').textContent, '');
renderResearchMap({});
assert.match($('research-map').textContent, /历史报告未保存/);
assert.doesNotMatch($('research-map').textContent, /条件不足|未回答成本/);
""",
    )


def test_unknown_version_and_missing_review_are_explicit(js):
    run_ui_checks(
        js,
        r"""
renderResearchMap({research_map:{version:99}});
assert.match($('research-map').textContent, /格式不可识别/);
renderResearchMap({research_map:{version:1, review_binding:'unavailable',
  points:[], conclusions:[], evidence:[], gaps:[]}});
assert.match($('research-map').textContent, /审查记录不可用/);
""",
    )
