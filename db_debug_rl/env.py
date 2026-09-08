"""The db-debug-rl environment: a multi-turn debugging loop over a real DB.

Actions (JSON dicts):
  {"action": "sql", "sql": "SELECT ..."}           — read-only probe on the trap DB
  {"action": "hypothesis", "table": ..., "columns": [...], "claim": str}
  {"action": "localize", "root_cause": {table, columns, transformation},
   "diagnostic_sql": "SELECT ..."}                 — terminal attempt

The observation contains the real schema, the pipeline lineage narrative, the
suspect row of the final reporting table and the remaining query budget.
Rewards are execution-verifiable (see reward.py); hypotheses give shaping
feedback only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from db_debug_rl import db as dbh
from db_debug_rl.pipelines import PIPELINES
from db_debug_rl.defects import DEFECTS_BY_ID
from db_debug_rl.oracle import EnvData, build_env_data, grade_localization, suspect_row
from db_debug_rl.reward import compute_reward

SYSTEM_PROMPT = (
    "You are a senior data-quality debugger. A reporting pipeline materialized a "
    "final table over a production database, and ONE row of that table is suspect: "
    "its value disagrees with an independent recomputation of the same business "
    "metric. You have read-only SQL access to the whole working database (source "
    "tables + staging tables + the final reported table). Investigate, form "
    "hypotheses about which table/field/transformation is wrong, then finish with "
    "a localisation and a diagnostic SQL query that flags every row with the same "
    "incoherence (reference the suspect columns; do NOT filter by the suspect's "
    "primary key; the query must return 0 rows on a healthy database).\n"
    "Emit actions as a single JSON object:\n"
    '  {"action": "sql", "sql": "..."}  — probe the DB (SELECT/WITH only)\n'
    '  {"action": "hypothesis", "table": "...", "columns": ["..."], "claim": "..."}\n'
    '  {"action": "localize", "root_cause": {"table": "...", "columns": ["..."], '
    '"transformation": "..."}, "diagnostic_sql": "..."}\n'
    "The reported table carries independently recomputed reference columns; the "
    "cleanest diagnostic is a row-level invariant reconciling the suspect field "
    "against its reference column (or an anti-join against the source table "
    "when whole rows are missing).")


@dataclass
class EpisodeSpec:
    pipeline_id: str
    defect_id: str
    seed: int = 0
    k_planted: int = 3
    max_queries: int = 12
    max_turns: int = 24


@dataclass
class DbgRLEnv:
    spec: EpisodeSpec
    provided_env: EnvData | None = field(default=None, repr=False, compare=False)
    env: EnvData | None = field(default=None, init=False)
    turns: int = field(default=0, init=False)
    queries: int = field(default=0, init=False)
    n_correct_hypotheses: int = field(default=0, init=False)
    done: bool = field(default=False, init=False)

    def reset(self) -> dict[str, Any]:
        pipeline = PIPELINES[self.spec.pipeline_id]
        defect = DEFECTS_BY_ID[self.spec.defect_id]
        if self.env is not None and self.env is not self.provided_env:
            self.env.close()
        if self.provided_env is not None:
            self.env = self.provided_env
        else:
            self.env = build_env_data(pipeline, defect, seed=self.spec.seed,
                                      k_planted=self.spec.k_planted)
            dbh.readonly_guard(self.env.trap)
        self.turns = self.queries = self.n_correct_hypotheses = 0
        self.done = False
        return self._obs()

    def _obs(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        pipeline = self.env.pipeline
        schema = dbh.schema_summary(self.env.trap)
        obs: dict[str, Any] = {
            "db": pipeline.db,
            "domain": pipeline.domain,
            "pipeline_id": pipeline.pipeline_id,
            "final_table": pipeline.final_table,
            "lineage": pipeline.lineage,
            "schema": dbh.schema_text(schema),
            "suspect_pk": self.env.suspect_pk,
            "suspect_row": suspect_row(self.env),
            "queries_used": self.queries,
            "queries_left": self.spec.max_queries - self.queries,
        }
        if extra:
            obs.update(extra)
        return obs

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict]:
        if self.done or self.env is None:
            raise RuntimeError("episode finished — call reset()")
        self.turns += 1
        kind = (action or {}).get("action")
        if kind == "sql":
            return self._step_sql(action)
        if kind == "hypothesis":
            return self._step_hypothesis(action)
        if kind == "localize":
            return self._step_localize(action)
        return self._obs({"error": f"unknown action {kind!r}"}), 0.0, False, {}

    def _maybe_force_done(self, info: dict[str, Any]) -> bool:
        if self.queries >= self.spec.max_queries:
            self.done = True
            info["reason"] = "query budget exhausted without localisation"
            info["success"] = False
            return True
        if self.turns >= self.spec.max_turns:
            self.done = True
            info["reason"] = "turn budget exhausted without localisation"
            info["success"] = False
            return True
        return False

    def _step_sql(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict]:
        status, res = dbh.exec_sql(self.env.trap, str(action.get("sql", "")))
        if status == "err":
            obs = self._obs({"sql_error": res})
            info = {"valid_sql": False}
            done = self._maybe_force_done(info)
            return obs, 0.0, done, info
        self.queries += 1
        obs = self._obs({"sql_result": res})
        info = {"valid_sql": True}
        done = self._maybe_force_done(info)
        return obs, 0.0, done, info

    def _step_hypothesis(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict]:
        hyp = {"table": action.get("table"),
               "columns": action.get("columns") or []}
        grade = grade_localization(hyp, self.env.defect)
        if grade["correct"]:
            self.n_correct_hypotheses += 1
        obs = self._obs({"hypothesis_feedback": {
            "table_ok": grade["table_ok"], "primary_identified": grade["primary_identified"]}})
        info = {"hypothesis_correct": grade["correct"]}
        done = self._maybe_force_done(info)
        return obs, (0.05 if grade["correct"] else 0.0), done, info

    def _step_localize(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict]:
        root_cause = action.get("root_cause") or {}
        diag_sql = str(action.get("diagnostic_sql") or "")
        self.done = True
        result = compute_reward(self.env, root_cause, diag_sql,
                                self.n_correct_hypotheses)
        obs = self._obs({"verdict": result})
        info = {"success": bool(result["sql_grade"].get("valid")
                                and result["sql_grade"].get("catches")
                                and not result["sql_grade"].get("false_positives")
                                and result["loc_grade"]["correct"]),
                "reward_detail": result}
        return obs, result["reward"], True, info
