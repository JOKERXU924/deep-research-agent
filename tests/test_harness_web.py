import httpx

from deep_research.harness_store import HarnessStore
from deep_research.web_app import create_app


async def test_readonly_dashboard_exposes_trace_not_model_payload_or_config(tmp_path):
    store = HarnessStore(tmp_path / "harness-research.harness.sqlite3")
    config = dict(private_setting="private-profile", action_limits=dict(model=9))
    rid = store.create(config)
    owner = store.claim(rid, config)
    key = store.begin(
        rid,
        owner,
        "model",
        dict(ordinal=1, stage="ResearchReport", request={"secret": "sk-test-private"}),
        limits=config["action_limits"],
    )
    store.finish(rid, owner, key, {"content": "private-response"})
    store.release(rid, owner)
    app = create_app(db=tmp_path / "web.sqlite3", output_dir=tmp_path / "reports")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8765"
    ) as client:
        response = await client.get("/harness")
        assert response.status_code == 200 and "harness.js" in response.text
        assert (await client.get("/harness.js")).status_code == 200
        assert (await client.get("/api/harness")).json()["runs"][0]["id"] == rid
        trace = await client.get("/api/harness/" + rid)
        assert trace.status_code == 200
        assert "ResearchReport" in trace.text
        for private in ("private-profile", "private-response", "sk-test-private"):
            assert private not in trace.text
        assert (await client.post("/api/harness/" + rid)).status_code == 403
