"""Rollout utilities for db-debug-rl (GRPO-stage rollouts, gated).

A policy is any callable ``act(obs, history) -> action dict``. ``rollout``
drives one episode and returns the transcript with per-turn rewards;
``batch_rollout`` runs many specs. The actual GRPO launch stays gated behind
``slm.grpo_config`` (see TRAINING_REPORT.md) — this module only produces the
reward-labelled transcripts a trainer would consume.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from db_debug_rl.env import DbgRLEnv, EpisodeSpec

Policy = Callable[[dict[str, Any], list[dict[str, Any]]], dict[str, Any]]


def rollout(spec: EpisodeSpec, policy: Policy) -> dict[str, Any]:
    env = DbgRLEnv(spec)
    obs = env.reset()
    history: list[dict[str, Any]] = []
    transcript: list[dict[str, Any]] = []
    total, success = 0.0, False
    while not env.done:
        action = policy(obs, history)
        try:
            action_parsed = action if isinstance(action, dict) else json.loads(action)
        except (json.JSONDecodeError, TypeError):
            action_parsed = {"action": "sql", "sql": ""}
        obs, reward, done, info = env.step(action_parsed)
        total += reward
        success = success or bool(info.get("success"))
        transcript.append({"action": action_parsed, "reward": reward, "done": done,
                           "info": {k: v for k, v in info.items() if k != "reward_detail"}})
        history.append({"obs": obs, "action": action_parsed})
    env.env.close()
    return {"spec": spec.__dict__, "total_reward": round(total, 4), "success": success,
            "turns": len(transcript), "transcript": transcript}


def batch_rollout(specs: list[EpisodeSpec], policy: Policy) -> list[dict[str, Any]]:
    return [rollout(spec, policy) for spec in specs]
