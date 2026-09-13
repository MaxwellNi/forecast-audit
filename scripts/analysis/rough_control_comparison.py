"""Public rough-control extension of the fixed canonical comparison.

The recorded run compares prespecified rough-control regimes using the same
canonical methods and keeps every result. Use a new output directory.
"""
from __future__ import annotations
import hashlib, importlib.metadata, inspect, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import canonical_baselines as M  # noqa: E402

ORIGINAL_DRAW_SAMPLE = M.draw_sample
E_ABS_Z = float(np.sqrt(2.0 / np.pi))
def draw_sample_ext(regime, alternative, seed, n):
    if regime in ("linear", "smooth", "nonlinear", "heavy_tail"):
        return ORIGINAL_DRAW_SAMPLE(regime, alternative, seed, n)
    rng = np.random.default_rng(seed)             # identical draw order to the harness
    z = rng.normal(size=n); u = rng.normal(size=n); ex, ey = rng.normal(size=(2, n))
    if regime == "kink":        g = 0.8 * (np.abs(z) - E_ABS_Z)
    elif regime == "step":      g = 0.8 * np.sign(z)
    elif regime == "oscillation": g = 0.8 * np.sqrt(2.0) * np.sin(3.0 * z)
    else: raise ValueError(regime)
    if alternative == "positive_covariance":   x, y = g + ex + 0.5 * u, g + ey + 0.5 * u
    elif alternative == "zero_covariance_dependence": x, y = g + ex, g + (ex * ex - 1) / np.sqrt(2) + ey
    elif alternative == "null":                x, y = g + ex, g + ey
    else: raise ValueError(alternative)
    return x, y, z
M.draw_sample = draw_sample_ext

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    from causallearn.utils.KCI import KCI
    doc = M.protocol()
    doc.update({"experiment": "independent_rough_control_regimes", "seed_start": 720260907,
                "regimes": ["kink", "step", "oscillation"], "alternatives": ["null", "positive_covariance"],
                "extra_cell": ["kink", "zero_covariance_dependence"],
                "dgp": "As the harness, with g(Z)=0.8(|Z|-sqrt(2/pi)) kink; 0.8 sign(Z) step; 0.8 sqrt(2) sin(3Z) oscillation; constants fixed in the frozen preregistration before any run.",
                "prereg": "protocols/rough_control_design.md",
                "prereg_sha256": M.sha256(ROOT / "protocols/rough_control_design.md"),
                "wrapper_sha256": M.sha256(Path(__file__)),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "script_sha256": M.sha256(Path(M.__file__)),
                "dependencies": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn", "causal-learn")},
                "kci_source_sha256": M.sha256(inspect.getfile(KCI)),
                "freeze_status": "Regimes, constants and decision rule frozen in the preregistration; protocol written locally immediately before the single run."})
    proto = out / "protocol.json"
    with proto.open("x", encoding="utf-8") as h: json.dump(doc, h, indent=2); h.write("\n")
    print("protocol", proto, M.sha256(proto), flush=True)
    M.run(proto, out)

if __name__ == "__main__":
    main()
