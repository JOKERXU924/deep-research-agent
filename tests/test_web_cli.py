"""Loopback-only startup and explicit live opt-in, without starting a server."""

import sys
from pathlib import Path

import pytest

import deep_research.cli as cli
from deep_research.errors import ResearchError


def test_serve_cli_defaults_and_explicit_live_flag(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        cli, "serve_research", lambda **kw: calls.append(kw) or {"status": "stopped"}
    )
    monkeypatch.setattr(sys, "argv", ["research", "serve"])
    assert cli.main() == 0
    assert calls[-1]["port"] == 8765 and calls[-1]["allow_live"] is False
    assert calls[-1]["db"] == Path("data/web.sqlite3")
    assert calls[-1]["evidence_selection"] == "coverage"
    monkeypatch.setattr(sys, "argv", ["research", "serve", "--allow-live", "--port", "8787"])
    assert cli.main() == 0
    assert calls[-1]["allow_live"] is True and calls[-1]["port"] == 8787
    monkeypatch.setattr(sys, "argv", ["research", "serve", "--evidence-selection", "multilingual"])
    assert cli.main() == 0
    assert calls[-1]["evidence_selection"] == "multilingual"


@pytest.mark.parametrize("port", [0, 80, 1023, 65536, True])
def test_invalid_port_fails_before_server_import_or_secret_access(tmp_path, monkeypatch, port):
    monkeypatch.setattr(cli, "load_settings", lambda *_: pytest.fail("Unexpected settings"))
    with pytest.raises(ResearchError) as caught:
        cli.serve_research(
            db=tmp_path / "x.db", output_dir=tmp_path, env_file=tmp_path / "x.env", port=port
        )
    assert caught.value.code == "web_configuration"
    assert not (tmp_path / "x.db").exists()


def test_uvicorn_configuration_is_single_worker_loopback_without_proxy_headers(
    tmp_path, monkeypatch
):
    import uvicorn

    import deep_research.web_app as web

    marker = object()
    app_options, server_options = [], []
    monkeypatch.setattr(web, "create_app", lambda **kw: app_options.append(kw) or marker)
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: server_options.append((app, kw)))
    monkeypatch.setattr(cli, "load_settings", lambda *_: pytest.fail("Unexpected settings"))
    result = cli.serve_research(
        db=tmp_path / "x.db", output_dir=tmp_path, env_file=tmp_path / "x.env"
    )
    assert result == {"status": "stopped", "mode": "local_web"}
    app, options = server_options[0]
    assert app is marker and options["host"] == "127.0.0.1"
    assert options["workers"] == 1 and options["reload"] is False
    assert options["proxy_headers"] is False and options["access_log"] is False
    assert app_options[0]["allow_live"] is False


@pytest.mark.parametrize("flag", ["--host", "--workers", "--reload"])
def test_no_exposed_network_or_multiworker_flags(monkeypatch, flag):
    monkeypatch.setattr(sys, "argv", ["research", "serve", flag, "x"])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
