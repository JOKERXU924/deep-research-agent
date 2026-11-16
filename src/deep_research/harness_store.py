"""Fenced single-owner execution journal. Completed actions are immutable checkpoints."""

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from uuid import uuid4

from deep_research.errors import ResearchError


def encode(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def digest(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def fail(code):
    return ResearchError("harness_" + code, "研究运行状态不允许此操作：" + code)


class HarnessStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS harness_runs(
                  id TEXT PRIMARY KEY, config TEXT NOT NULL, fingerprint TEXT NOT NULL,
                  status TEXT NOT NULL, owner TEXT, lease REAL NOT NULL DEFAULT 0,
                  created REAL NOT NULL, deadline REAL NOT NULL, result TEXT, result_hash TEXT);
                CREATE TABLE IF NOT EXISTS harness_actions(
                  run_id TEXT NOT NULL, key TEXT NOT NULL, kind TEXT NOT NULL,
                  args TEXT NOT NULL, status TEXT NOT NULL, output TEXT, output_hash TEXT,
                  attempts INTEGER NOT NULL DEFAULT 1, error TEXT,
                  PRIMARY KEY(run_id,key));
                CREATE TABLE IF NOT EXISTS harness_events(
                  seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
                  kind TEXT NOT NULL, data TEXT NOT NULL, at REAL NOT NULL);
            """)

    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        return db

    def create(self, config, *, seconds=600):
        if not 0 < seconds <= 600:
            raise fail("configuration")
        rid = str(uuid4())
        now = time.time()
        with self.connect() as db:
            db.execute(
                "INSERT INTO harness_runs(id,config,fingerprint,status,created,deadline) "
                "VALUES(?,?,?,?,?,?)",
                (rid, encode(config), digest(config), "ready", now, now + seconds),
            )
        return rid

    def get(self, rid):
        with self.connect() as db:
            row = db.execute("SELECT * FROM harness_runs WHERE id=?", (rid,)).fetchone()
        if row is None:
            raise fail("not_found")
        r = dict(row)
        r["config"] = json.loads(r["config"])
        if digest(r["config"]) != r["fingerprint"]:
            raise fail("corrupt")
        if r["result"] is not None:
            r["result"] = json.loads(r["result"])
            if digest(r["result"]) != r["result_hash"]:
                raise fail("corrupt")
        return r

    def claim(self, rid, config, *, ttl=30):
        owner = str(uuid4())
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            r = db.execute("SELECT * FROM harness_runs WHERE id=?", (rid,)).fetchone()
            if r is None or r["fingerprint"] != digest(config):
                raise fail("configuration_changed")
            if r["status"] == "completed":
                raise fail("completed")
            if r["owner"] and r["lease"] > time.time():
                raise fail("busy")
            # The provider may have executed a pending request. Never silently resend it.
            unknown = db.execute(
                "SELECT 1 FROM harness_actions WHERE run_id=? AND kind='model' "
                "AND status IN ('pending','unknown')",
                (rid,),
            ).fetchone()
            if unknown:
                db.execute(
                    "UPDATE harness_runs SET status='blocked',owner=NULL,lease=0 WHERE id=?", (rid,)
                )
                self.event(db, rid, "recovery_blocked", {"reason": "model_outcome_unknown"})
                db.commit()
                raise fail("model_outcome_unknown")
            if time.time() >= r["deadline"]:
                raise fail("deadline")
            db.execute(
                "UPDATE harness_runs SET owner=?,lease=?,status='running' WHERE id=?",
                (owner, time.time() + ttl, rid),
            )
            self.event(db, rid, "attempt_started", {})
        return owner

    @staticmethod
    def event(db, rid, kind, data):
        db.execute(
            "INSERT INTO harness_events(run_id,kind,data,at) VALUES(?,?,?,?)",
            (rid, kind, encode(data), time.time()),
        )

    @staticmethod
    def guard(db, rid, owner, *, check_deadline=True):
        r = db.execute("SELECT * FROM harness_runs WHERE id=?", (rid,)).fetchone()
        if not r or r["owner"] != owner or r["lease"] <= time.time():
            raise fail("lease_lost")
        if check_deadline and time.time() >= r["deadline"]:
            raise fail("deadline")

    def heartbeat(self, rid, owner):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self.guard(db, rid, owner, check_deadline=False)
            db.execute("UPDATE harness_runs SET lease=? WHERE id=?", (time.time() + 30, rid))

    def action(self, rid, kind, args):
        key = digest(dict(kind=kind, args=args))
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM harness_actions WHERE run_id=? AND key=?", (rid, key)
            ).fetchone()
        if row is None:
            return key, None
        value = dict(row)
        if digest(dict(kind=value["kind"], args=json.loads(value["args"]))) != key:
            raise fail("checkpoint_corrupt")
        if value["status"] == "completed":
            value["output"] = json.loads(value["output"])
            if digest(value["output"]) != value["output_hash"]:
                raise fail("checkpoint_corrupt")
        return key, value

    def begin(self, rid, owner, kind, args, *, limits):
        key, old = self.action(rid, kind, args)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self.guard(db, rid, owner)
            used = db.execute(
                "SELECT COALESCE(SUM(attempts),0) FROM harness_actions WHERE run_id=? AND kind=?",
                (rid, kind),
            ).fetchone()[0]
            if used >= limits[kind]:
                raise fail("budget")
            if old and old["status"] == "completed":
                raise fail("duplicate")
            if old and kind == "model":
                raise fail("model_outcome_unknown")
            db.execute(
                "INSERT INTO harness_actions(run_id,key,kind,args,status) VALUES(?,?,?,?,?) "
                "ON CONFLICT(run_id,key) DO UPDATE SET status='pending',attempts=attempts+1",
                (rid, key, kind, encode(args), "pending"),
            )
            self.event(db, rid, "action_started", dict(key=key, kind=kind, args=args))
        return key

    def finish(self, rid, owner, key, output):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self.guard(db, rid, owner)
            changed = db.execute(
                "UPDATE harness_actions SET status='completed',output=?,"
                "output_hash=?,error=NULL WHERE run_id=? AND key=? "
                "AND status='pending'",
                (encode(output), digest(output), rid, key),
            )
            if changed.rowcount != 1:
                raise fail("state")
            self.event(db, rid, "action_completed", dict(key=key))

    def action_error(self, rid, owner, key, kind, code):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self.guard(db, rid, owner, check_deadline=False)
            db.execute(
                "UPDATE harness_actions SET status=?,error=? WHERE run_id=? AND key=?",
                ("unknown" if kind == "model" else "failed", code, rid, key),
            )
            self.event(db, rid, "action_failed", dict(key=key, kind=kind, code=code))

    def reused(self, rid, owner, key, kind):
        with self.connect() as db:
            self.guard(db, rid, owner)
            self.event(db, rid, "checkpoint_reused", dict(key=key, kind=kind))

    def release(self, rid, owner, *, result=None):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self.guard(db, rid, owner, check_deadline=result is not None)
            unknown = db.execute(
                "SELECT 1 FROM harness_actions WHERE run_id=? AND kind='model' "
                "AND status IN ('pending','unknown')",
                (rid,),
            ).fetchone()
            status = "blocked" if unknown else "completed" if result is not None else "interrupted"
            if result is not None and unknown:
                raise fail("model_outcome_unknown")
            db.execute(
                "UPDATE harness_runs SET owner=NULL,lease=0,status=?,result=?,result_hash=? "
                "WHERE id=?",
                (
                    status,
                    encode(result) if result is not None else None,
                    digest(result) if result is not None else None,
                    rid,
                ),
            )
            self.event(db, rid, status, {})

    def trace(self, rid):
        run = self.get(rid)
        with self.connect() as db:
            events = [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM harness_events WHERE run_id=? ORDER BY seq LIMIT 4000", (rid,)
                )
            ]
            actions = [
                dict(r)
                for r in db.execute(
                    "SELECT key,kind,status,attempts,error FROM harness_actions WHERE run_id=?",
                    (rid,),
                )
            ]
        for e in events:
            e["data"] = json.loads(e["data"])
            # Outgoing model payloads are never sent to the dashboard.
            args = e["data"].pop("args", {})
            if e["kind"] == "action_started":
                e["data"]["stage"] = args.get("stage", e["data"].get("kind"))
                e["data"]["point_ids"] = args.get("point_ids", [])
                if "query" in args:
                    e["data"]["query"] = args["query"][:120]
                if "title" in args:
                    e["data"]["title"] = args["title"][:180]
        result = dict(run["result"]) if run["result"] else None
        if result:
            result.pop("report_path", None)
        return dict(
            id=rid,
            status=run["status"],
            deadline=run["deadline"],
            events=events,
            actions=actions,
            result=result,
        )

    def list_runs(self):
        with self.connect() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT id,status,created,deadline FROM harness_runs "
                    "ORDER BY created DESC LIMIT 100"
                )
            ]
