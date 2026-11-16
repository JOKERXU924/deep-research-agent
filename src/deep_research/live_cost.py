"""Durable experiment-wide CNY reservations at the published Flash peak/miss rates.

Upper accounting estimate, not a provider invoice. Input reserves UTF-8 bytes
plus 8192 framing tokens; unknown attempts retain their full reservation.
"""

import hashlib
import json
import sqlite3
from datetime import UTC, datetime

from deep_research.errors import ResearchError

RATE_SOURCE = "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/"
INPUT_MICRO_PER_TOKEN = 2
OUTPUT_MICRO_PER_TOKEN = 8


class CurrencyLedger:
    def __init__(self, path, cap_micro=None):
        if cap_micro is not None and (
            type(cap_micro) is not int or not 0 < cap_micro <= 10_000_000
        ):
            raise ValueError("Experiment cap must not exceed CNY 10")
        self.path, self.healthy = path, True
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as c:
            c.execute("CREATE TABLE IF NOT EXISTS cost_config(cap INTEGER NOT NULL)")
            rows = c.execute("SELECT cap FROM cost_config").fetchall()
            if cap_micro is None:
                cap_micro = rows[0][0] if len(rows) == 1 else 5_000_000
            if type(cap_micro) is not int or not 0 < cap_micro <= 10_000_000:
                raise ValueError("Invalid persisted monetary cap")
            if rows and rows != [(cap_micro,)]:
                raise ValueError("Cannot change persisted monetary cap")
            if not rows:
                c.execute("INSERT INTO cost_config VALUES(?)", (cap_micro,))
            c.execute("""CREATE TABLE IF NOT EXISTS attempts(
                id INTEGER PRIMARY KEY, created_at TEXT, label TEXT, request_hash TEXT,
                reserved_micro INTEGER, charged_micro INTEGER, status TEXT,
                input_tokens INTEGER, output_tokens INTEGER, error_code TEXT)""")
        self.cap_micro = cap_micro

    def extend_cap(self, new_cap_micro, *, expected_cap_micro, authorization):
        """Explicit audited authorization; never reset or rewrite prior attempts."""
        if (
            type(new_cap_micro) is not int
            or type(expected_cap_micro) is not int
            or not 0 < expected_cap_micro < new_cap_micro <= 10_000_000
            or not isinstance(authorization, str)
            or not 1 <= len(authorization.strip()) <= 500
        ):
            raise ValueError("Invalid explicit budget extension")
        with sqlite3.connect(self.path) as c:
            c.execute("BEGIN IMMEDIATE")
            if c.execute("SELECT cap FROM cost_config").fetchall() != [(expected_cap_micro,)]:
                raise ValueError("Budget extension conflicts with persisted cap")
            c.execute("""CREATE TABLE IF NOT EXISTS cap_extensions(
                id INTEGER PRIMARY KEY, created_at TEXT NOT NULL,
                old_cap INTEGER NOT NULL, new_cap INTEGER NOT NULL,
                authorization TEXT NOT NULL, attempts_before INTEGER NOT NULL,
                charged_before INTEGER NOT NULL)""")
            count, charged = c.execute(
                "SELECT COUNT(*), COALESCE(SUM(charged_micro),0) FROM attempts"
            ).fetchone()
            c.execute(
                "INSERT INTO cap_extensions(created_at,old_cap,new_cap,authorization,"
                "attempts_before,charged_before) VALUES(?,?,?,?,?,?)",
                (
                    datetime.now(UTC).isoformat(),
                    expected_cap_micro,
                    new_cap_micro,
                    authorization.strip(),
                    count,
                    charged,
                ),
            )
            c.execute("UPDATE cost_config SET cap=?", (new_cap_micro,))
        self.cap_micro = new_cap_micro

    def reserve(self, request, label):
        if not self.healthy:
            raise ResearchError("cost_audit", "费用账本不可用，停止外发请求。")
        raw = request.model_dump_json().encode()
        reserve = (len(raw) + 8192) * INPUT_MICRO_PER_TOKEN
        reserve += request.max_output_tokens * OUTPUT_MICRO_PER_TOKEN
        try:
            with sqlite3.connect(self.path) as c:
                c.execute("BEGIN IMMEDIATE")
                spent = c.execute("SELECT COALESCE(SUM(charged_micro),0) FROM attempts").fetchone()[
                    0
                ]
                persisted = c.execute("SELECT cap FROM cost_config").fetchall()
                if len(persisted) != 1 or persisted[0][0] != self.cap_micro:
                    raise ResearchError("cost_audit", "费用上限已变更，请重新打开原账本。")
                if spent + reserve > persisted[0][0]:
                    raise ResearchError("cost_budget", "本轮人民币预算不足，未发送请求。")
                cursor = c.execute(
                    "INSERT INTO attempts(created_at,label,request_hash,"
                    "reserved_micro,charged_micro,status) VALUES(?,?,?,?,?,?)",
                    (
                        datetime.now(UTC).isoformat(),
                        label,
                        hashlib.sha256(raw).hexdigest(),
                        reserve,
                        reserve,
                        "reserved",
                    ),
                )
                return cursor.lastrowid
        except sqlite3.Error:
            self.healthy = False
            raise ResearchError("cost_audit", "费用预留无法保存，未发送请求。") from None

    def finish(self, identifier, usage, error_code=None):
        try:
            with sqlite3.connect(self.path) as c:
                c.execute("BEGIN IMMEDIATE")
                row = c.execute(
                    "SELECT reserved_micro,status FROM attempts WHERE id=?", (identifier,)
                ).fetchone()
                if row is None or row[1] != "reserved":
                    raise ValueError("Missing or completed reservation")
                cost = (
                    row[0]
                    if usage is None
                    else (
                        usage.input_tokens * INPUT_MICRO_PER_TOKEN
                        + usage.output_tokens * OUTPUT_MICRO_PER_TOKEN
                    )
                )
                c.execute(
                    "UPDATE attempts SET charged_micro=?,status=?,input_tokens=?,"
                    "output_tokens=?,error_code=? WHERE id=?",
                    (
                        cost,
                        "unknown" if usage is None else "known",
                        usage.input_tokens if usage else None,
                        usage.output_tokens if usage else None,
                        error_code,
                        identifier,
                    ),
                )
            if cost > row[0]:
                self.healthy = False
                raise ResearchError("cost_bound", "服务用量超过保守预留，停止后续请求。")
        except (sqlite3.Error, ValueError):
            self.healthy = False
            raise ResearchError("cost_audit", "费用结算无法保存，停止后续请求。") from None

    def summary(self):
        with sqlite3.connect(self.path) as c:
            rows = c.execute(
                "SELECT id,label,reserved_micro,charged_micro,status,"
                "input_tokens,output_tokens,error_code FROM attempts ORDER BY id"
            ).fetchall()
        keys = (
            "id",
            "label",
            "reserved_micro",
            "charged_micro",
            "status",
            "input_tokens",
            "output_tokens",
            "error_code",
        )
        return dict(
            cap_cny=self.cap_micro / 1e6,
            accounted_upper_cny=sum(r[3] for r in rows) / 1e6,
            calls=len(rows),
            unknown_calls=sum(r[4] != "known" for r in rows),
            input_cny_per_million=2,
            output_cny_per_million=8,
            rate_source=RATE_SOURCE,
            invoice_verified=False,
            attempts=[dict(zip(keys, r, strict=True)) for r in rows],
        )


class MeteredProvider:
    def __init__(self, provider, ledger, label, capture_dir=None):
        if provider.settings.model != "deepseek-flash":
            raise ResearchError("cost_model", "本轮预算仅允许已核价的 deepseek-flash。")
        self.provider, self.ledger, self.label = provider, ledger, label
        self.capture_dir = capture_dir

    async def complete(self, request):
        identifier = self.ledger.reserve(request, self.label)
        try:
            result = await self.provider.complete(request)
        except BaseException as error:
            try:
                self.ledger.finish(identifier, None, getattr(error, "code", type(error).__name__))
            except BaseException:
                self.ledger.healthy = False
            raise
        self.ledger.finish(identifier, result.usage)
        if self.capture_dir is not None:
            self.capture_dir.mkdir(parents=True, exist_ok=True)
            (self.capture_dir / f"{identifier}.json").write_text(
                json.dumps(
                    dict(
                        label=self.label,
                        request=request.model_dump(),
                        response=result.model_dump(),
                    ),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        return result
