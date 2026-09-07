"""sas-reconcile-slm — SLM post-training environment.

Modules:
  oracle   — deterministic grading oracle tying a planted defect to a diagnostic
             SQL query and a localization verdict.
  lineage  — compact schema + lineage summary builder for episode context.
  episodes — (task | trace) episode generation, SFT JSONL, train/test split.
  reward   — GRPO stage-2 reward function (prepared, gated).
"""

__version__ = "0.1.0"

# The vendored prior-art modules import each other with *flat* top-level names
# (e.g. ``from defect_catalog import ...``), so put ``vendor/`` itself on the
# import path.  The repo root is already on sys.path when running ``-m slm.``.
import os as _os
import sys as _sys

_VENDOR_DIR = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                            "vendor")
if _VENDOR_DIR not in _sys.path:
    _sys.path.insert(0, _VENDOR_DIR)
