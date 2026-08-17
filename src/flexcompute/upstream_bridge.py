"""Bridge to the vendored Newkirk reference model (``upstream/``).

The reference project is used **unmodified**. It has two properties that make
it awkward to import as a library, and this module isolates both:

1.  **Flat imports.** Modules inside ``upstream/src`` import each other by bare
    name (``from config import Config``), so ``upstream/src`` must be on
    ``sys.path``.
2.  **Root-relative data paths.** Several defaults are relative strings
    (``"output_tables/hourly_load_data.csv"``, ``"output_tables/fade_surrogate.pkl"``),
    so the process working directory must be the upstream repo root at the
    moment those files are read.

Everything here is import/path plumbing only. No modelling behaviour lives in
this module, and nothing in ``upstream/`` is patched or rewritten.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Iterator

# Repo layout: <project_root>/src/flexcompute/upstream_bridge.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_ROOT = PROJECT_ROOT / "upstream"
UPSTREAM_SRC = UPSTREAM_ROOT / "src"
UPSTREAM_TABLES = UPSTREAM_ROOT / "output_tables"


def ensure_upstream_importable() -> None:
    """Put ``upstream/src`` on ``sys.path`` (idempotent)."""
    if not UPSTREAM_SRC.is_dir():
        raise FileNotFoundError(
            f"Reference model not found at {UPSTREAM_SRC}. Clone it with:\n"
            "  git clone https://github.com/acnewkirk/AI-Datacenter-Microgrid-Analysis.git upstream"
        )
    path = str(UPSTREAM_SRC)
    if path not in sys.path:
        sys.path.insert(0, path)


@contextlib.contextmanager
def upstream_workdir() -> Iterator[Path]:
    """Run a block with the process CWD set to the upstream repo root.

    Needed only for upstream call paths that read their data files through
    relative default paths and expose no override -- currently the hourly IT
    load CSV reached via ``DatacenterAnalyzer.calculate_facility_load``.

    Not thread-safe (``os.chdir`` is process-global). This harness is
    single-threaded by design; if that ever changes, replace this with explicit
    path injection.
    """
    previous = Path.cwd()
    os.chdir(UPSTREAM_ROOT)
    try:
        yield UPSTREAM_ROOT
    finally:
        os.chdir(previous)


def load_upstream_config():
    """Return an upstream ``Config`` with file paths made absolute.

    The only mutation is ``degradation.fade_model_path``: upstream defaults it
    to a root-relative string. Making it absolute lets the battery fade
    surrogate load regardless of CWD. No numeric parameter is touched.
    """
    ensure_upstream_importable()
    from config import load_config  # type: ignore[import-not-found]

    cfg = load_config()
    cfg.degradation.fade_model_path = str(UPSTREAM_TABLES / "fade_surrogate.pkl")
    return cfg


def upstream_commit() -> str:
    """Short git SHA of the vendored reference model, for provenance stamping."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "-C", str(UPSTREAM_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # pragma: no cover - provenance is best-effort
        return "unknown"
