"""Reference results for the browser application, computed by the package under CPython.

Writes ``reference.json`` next to this file: the versions of the interpreter and of the
numerical libraries, and a list of requests in the form that the page sends, each with the
result of ``bridge.run_point``. ``check_page.mjs`` replays the requests in the page and
compares the numbers.
Run from the repository root after installing the package: python app/check/make_reference.py
"""
from __future__ import annotations

import csv
import json
import platform
import sys
from pathlib import Path

import numpy
import scipy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import bridge  # noqa: E402


def measured(site: str) -> dict:
    path = HERE.parents[1] / "results" / "nist_profiles" / f"nist_profile_{site}.csv"
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {"delay_s": [float(r["delay_s"]) for r in rows], "power": [float(r["power"]) for r in rows], "name": f"NIST {site}, measured"}


REQUESTS = [
    {"grid": "lte:100", "profile": "exp:131.8ns", "snr_db": 15.0},
    {"grid": "nr:51:30:1", "profile": "exp:36.1ns", "snr_db": 5.0},
    {"grid": "nr:51:30:2", "profile": measured("machine_shop"), "snr_db": 25.0, "cap_at_max_delay": True},
    {"grid": "lte:100", "profile": measured("steam_plant"), "snr_db": 20.0, "design_spread_s": 36.1e-9},
    {"grid": "dvbt2:8k:PP3:1/8", "map": "comb:6", "window": 1024, "profile": "exp:74.3ns", "snr_db": 15.0},
    {"grid": "custom:64:312500:64:0.8us", "map": "centred:8", "profile": "exp:74.3ns", "snr_db": 10.0, "search_best": True},
    {"grid": "nr:51:30:1", "profile": "inf:14400:6000:nlos", "snr_db": 30.0},
]


def main() -> None:
    cases = [{"request": q, "result": json.loads(bridge.run_point(json.dumps(q)))} for q in REQUESTS]
    for case in cases:
        if "error" in case["result"]:
            raise SystemExit(f"request failed: {case['result']['error']}")
    environment = {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__, "ofdmest": bridge.version()}
    out = {"environment": environment, "cases": cases}
    (HERE / "reference.json").write_text(json.dumps(out, indent=1) + chr(10), encoding="utf-8", newline=chr(10))
    print(f"wrote reference.json with {len(cases)} cases")


if __name__ == "__main__":
    main()
