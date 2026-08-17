"""Test configuration: make both ``src/`` and the vendored upstream importable."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flexcompute.scenario import Scenario  # noqa: E402
from flexcompute.upstream_bridge import ensure_upstream_importable  # noqa: E402
from flexcompute.weather import CACHE_DIR  # noqa: E402

ensure_upstream_importable()


BASELINE_SCENARIO = Scenario()  # dallas / 10k GPUs / ac_coupled / mv_coupled / pvgis


def _weather_is_cached(scenario: Scenario) -> bool:
    stem = f"{scenario.weather_source}_tmy_{scenario.latitude:.4f}_{scenario.longitude:.4f}"
    return (CACHE_DIR / f"{stem}.parquet").exists()


requires_weather = pytest.mark.skipif(
    not _weather_is_cached(BASELINE_SCENARIO),
    reason=(
        "No cached TMY for the baseline scenario. Run "
        "`python scripts/run_baseline.py` once (needs network) to warm the cache."
    ),
)


@pytest.fixture(scope="session")
def site():
    """The baseline Dallas site. Built once; every test shares it."""
    return BASELINE_SCENARIO.build()


@pytest.fixture(scope="session")
def cfg():
    from flexcompute.upstream_bridge import load_upstream_config

    return load_upstream_config()
