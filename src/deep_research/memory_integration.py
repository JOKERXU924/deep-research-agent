"""Explicit historical evidence reuse; memory never supplies system instructions."""

from typing import TYPE_CHECKING

from deep_research.errors import ResearchError
from deep_research.research_store import ResearchStore

if TYPE_CHECKING:
    from deep_research.memory_store import MemoryStore


class RunMemory:
    """One bounded recall snapshot, imported only after attempting fresh retrieval."""

    def __init__(
        self,
        memory: "MemoryStore",
        *,
        store: ResearchStore,
        scope: str,
        source_mode: str,
        max_age_days: int,
    ) -> None:
        self.memory = memory
        self.store = store
        self.scope = scope
        self.source_mode = source_mode
        self.max_age_days = max_age_days
        self.selection: dict | None = None
        self.recalled: list[dict] = []
        self.skipped_fresh = 0
        self.merged = False

    def recall(self, question: str) -> None:
        self.selection = self.memory.search(
            question,
            scope=self.scope,
            source_mode=self.source_mode,
            limit=4,
            max_age_days=self.max_age_days,
        )

    def merge(self, run_id: str, sources: list[dict], evidence: list[dict]) -> None:
        if self.merged or self.selection is None:
            return
        fresh_urls = {item["url"] for item in sources}
        fresh_hashes = {item["sha256"] for item in sources}
        imported: dict[tuple[str, str], str] = {}
        for hit in self.selection["hits"]:
            original = hit["source"]
            if original["url"] in fresh_urls or original["sha256"] in fresh_hashes:
                self.skipped_fresh += 1
                continue
            key = (original["url"], original["sha256"])
            new_sources = []
            if key not in imported:
                source = {**original, "source_id": f"S{len(sources) + 1}"}
                new_sources.append(source)
                source_id = source["source_id"]
            else:
                source_id = imported[key]
            item = {
                **hit["evidence"],
                "evidence_id": f"E{len(evidence) + 1}",
                "source_id": source_id,
            }
            # M1's store revalidates hash and exact original character offsets.
            # Do not modify retrieved_at: a recall is not a fresh web observation.
            self.store.save_sources(run_id, new_sources, [item])
            imported[key] = source_id
            sources.extend(new_sources)
            evidence.append(item)
            self.recalled.append(
                {
                    "evidence_id": item["evidence_id"],
                    "source_id": source_id,
                    "memory_id": hit["memory_id"],
                    "origin_run_id": hit["origin_run_id"],
                    "origin_evidence_id": hit["origin_evidence_id"],
                    "score": hit["score"],
                    "retrieved_at": original["retrieved_at"],
                    "conflict_notes": hit.get("conflict_notes", []),
                }
            )
        self.merged = True

    def metadata(self) -> dict:
        selection = self.selection or {}
        return {
            "scope": self.scope,
            "source_mode": self.source_mode,
            "encoder_id": selection.get("encoder_id"),
            "max_age_days": self.max_age_days,
            "selected_count": len(selection.get("hits", [])),
            "recalled_count": len(self.recalled),
            "excluded_conflicts": selection.get("excluded_conflicts", 0),
            "excluded_stale": selection.get("excluded_stale", 0),
            "skipped_fresh": self.skipped_fresh,
            "recalled": list(self.recalled),
        }

    def retains_conflicts(self) -> bool:
        return any(
            note.get("resolution", {}).get("policy") == "keep_both"
            for item in self.recalled
            for note in item["conflict_notes"]
            if isinstance(note.get("resolution"), dict)
        )

    def prompt_metadata(self) -> dict:
        """Keep verbose human audit notes out of the bounded model context."""
        metadata = self.metadata()
        metadata["recalled"] = []
        for item in self.recalled:
            compact = dict(item)
            notes = item["conflict_notes"]
            if notes:
                compact["conflict_notes"] = [
                    {
                        "id": note["id"],
                        "reason": note["reason"],
                        "policy": note["resolution"]["policy"],
                    }
                    for note in notes[:4]
                ]
                compact["conflict_count"] = len(notes)
                compact["conflict_details_truncated"] = len(notes) > 4
            metadata["recalled"].append(compact)
        return metadata

    def publish(self, run_id: str, *, degraded: bool) -> dict:
        """Post-completion publication has its own result; never erase a good report."""
        if degraded:
            return {"status": "skipped", "reason": "degraded_run"}
        try:
            result = self.memory.ingest_run(run_id, scope=self.scope)
        except ResearchError as error:
            return {"status": "failed", "code": error.code}
        except Exception:
            return {"status": "failed", "code": "memory_unexpected"}
        return {**result, "status": "saved"}
