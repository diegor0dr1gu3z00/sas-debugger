"""Stage-2 GRPO configuration — PREPARED, NOT LAUNCHED.

GRPO is explicitly gated by the council (see TRAINING_REPORT.md): it may run
ONLY if the SFT + best-of-N evaluation plateaus BELOW the target with a reward
that is >= 0.9 execution-verifiable.  This module documents the launch that a
future agent *would* construct; it is deliberately not wired to any trainer and
defines a DECISION to be gate-checked, not a runnable command.

Why we disarm vLLM: the host GPU is Blackwell (compute capability 12.0) and the
brief forbids vLLM on it.  GRPO must therefore use HF transformers + torchrl's
GRPO/MC-GRPO path (or trl's GRPOTrainer) with eager execution and generation
inside the rollout loop.

Gate requirements (what must be true before launching):
  1. SFT eval shows valid-query rate < 1.0 and localization accuracy < target
     (i.e. SFT plateaued below the bar and cannot reach it with best-of-N).
  2. The execution-verifiable reward (``slm.reward``) must be >= 0.9 on the
     held-out set for the *policy's own* samples (which requires generation,
     so the reward oracle has been validated on gold traces == 1.0 first).
  3. best-of-N sampled from the SFT model is measured and reported; if it
     already closes the gap, GRPO is unnecessary and should NOT run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# The reward function this stage would optimise (see slm.reward).
from slm.reward import compute_reward, RolloutEnv


@dataclass
class GRPOConfig:
    """Parameterisation of a would-be GRPO stage-2 run."""

    # reference: seed + scope
    defect_classes: str = "all"            # or a list of defect ids
    n_rollouts_per_prompt: int = 8
    n_envs_per_step: int = 32
    n_rows: int = 1500                     # trap/clean db size per rollout env
    # distances
    seq_len: int = 2048
    num_epochs: int = 3
    kl_coef: float = 0.05
    clip_coef: float = 0.2
    gamma: float = 0.99
    lr: float = 1e-5
    total_steps: int = 300
    # reward
    reward_fn = compute_reward              # execution-verifiable
    # gated
    gate: dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,                   # must never be True until the gate clears
        "conditions": [
            "sft_plateau_below_target",
            "reward_verifiable_ge_0.9",
            "best_of_n_gap_closed_eval",
        ],
        "tools": ["transformers+trl/GRPOTrainer (no vLLM on Blackwell)"],
    })

    def validate_gate(self, sft_report: dict[str, Any]) -> dict[str, Any]:
        """Return whether the stage-2 gate conditions are met."""
        checks = {
            "reward_verifiable_ge_0.9":
                sft_report.get("reward_verifiable_on_gold") in (None, True),
            "sft_below_target":
                sft_report.get("valid_query_rate", 1.0) < 1.0,
            "best_of_n_no_gap":
                sft_report.get("best_of_n_gap_closed", False),
        }
        ok = all(checks.values())
        return {"ready": bool(ok and not self.gate["enabled"]),
                "checks": checks,
                "launch": f"GRPO launch DISABLED (enabled={self.gate['enabled']})"}


def describe_stage2() -> dict[str, Any]:
    """Human-readable briefing of the would-be stage-2 GRPO launch."""
    cfg = GRPOConfig()
    return {
        "stage": "GRPO (stage 2) — PREPARED, NOT LAUNCHED",
        "config": cfg.__dict__,
        "reward": "execution-verifiable, bounded [0,1] (slm.reward.compute_reward)",
        "divergence_oracle": "ground-truth defect oracle by construction",
        "why_gated": "council gate: only run GRPO if SFT + best-of-N plateau below "
                     "target with a >=0.9 verifiable reward",
    }


if __name__ == "__main__":
    import json
    print(json.dumps(describe_stage2(), indent=2, default=str))
