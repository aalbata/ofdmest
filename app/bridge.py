"""Glue between the page and the ofdmest package.

The page sends one request per SNR as JSON and receives the result of
``ofdmest.report.compare_from_specs`` as JSON. All numerical work is done by the package.
"""
from __future__ import annotations

import json
import math

import numpy as np

from ofdmest import DelayProfile, __version__
from ofdmest.report import compare_from_specs


def _plain(value):
    """Replace values that JSON cannot carry (infinities, not-a-number) by None."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def version() -> str:
    return __version__


def run_point(request_json: str) -> str:
    """Evaluate one operating point; errors of the inputs are returned under the key ``error``."""
    q = json.loads(request_json)
    try:
        profile = q["profile"]
        if isinstance(profile, dict):
            profile = DelayProfile(np.array(profile["delay_s"], dtype=float), np.array(profile["power"], dtype=float), name=profile["name"])
        out = compare_from_specs(
            q["grid"],
            profile,
            float(q["snr_db"]),
            map_spec=q.get("map") or None,
            window=q.get("window") or None,
            design_spread_s=q.get("design_spread_s") or None,
            cap_at_max_delay=bool(q.get("cap_at_max_delay")),
            search_best=bool(q.get("search_best")),
        )
    except (ValueError, KeyError, IndexError, SyntaxError, NameError, ZeroDivisionError) as err:
        return json.dumps({"error": f"{type(err).__name__}: {err}"})
    return json.dumps(_plain(out))
