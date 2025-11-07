"""
DEPRECATED wrapper: routes anomaly filtering to featuresBuilder.py
so you get unified features, RV detection, and anomaly flags.
"""

import argparse
import os
import sys
import subprocess


def main():
    ap = argparse.ArgumentParser(description="[Deprecated] Use featuresBuilder.py; this forwards to it.")
    ap.add_argument("--input", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                   "reactor_readings_cleaned.csv"))
    ap.add_argument("--out-flags", default=None, help="Output CSV with flags (default: *_with_flags.csv)")
    ap.add_argument("--out-filtered", default=None, help="Output filtered CSV (default: *_filtered.csv)")
    ap.add_argument("--slope-window", type=int, default=3)
    ap.add_argument("--temp-rise-k-per-s", type=float, default=10.0)
    ap.add_argument("--temp-dip-k-per-s", type=float, default=10.0)
    ap.add_argument("--min-fuel", type=float, default=75.0)
    ap.add_argument("--rod", type=float, default=55.0)
    ap.add_argument("--rod-tol", type=float, default=5.0)
    ap.add_argument("--require-coolant-open", action="store_true", default=True)
    ap.add_argument("--no-require-coolant-open", dest="require_coolant_open", action="store_false")
    ap.add_argument("--dpdt-max", type=float, default=None)
    ap.add_argument("--interp-limit", type=int, default=0)
    args = ap.parse_args()

    in_path = args.input
    if not os.path.isfile(in_path):
        raise SystemExit(f"Input not found: {in_path}")

    base, ext = os.path.splitext(in_path)
    out_features = args.out_flags or f"{base}_with_flags.csv"
    out_filtered = args.out_filtered or f"{base}_filtered.csv"

    fb = os.path.join(os.path.dirname(__file__), "featuresBuilder.py")
    if not os.path.isfile(fb):
        raise SystemExit("featuresBuilder.py not found; cannot delegate.")

    cmd = [
        sys.executable, fb,
        "--input", in_path,
        "--out-features", out_features,
        "--out-filtered", out_filtered,
        "--slope-window", str(int(args.slope_window)),
        "--rise-k-per-s", str(float(args.temp_rise_k_per_s)),
        "--dip-k-per-s", str(float(args.temp_dip_k_per_s)),
        "--min-fuel", str(float(args.min_fuel)),
        "--rod", str(float(args.rod)),
        "--rod-tol", str(float(args.rod_tol)),
        "--rv-k-per-s", "7.5",
    ]
    if args.require_coolant_open:
        cmd.append("--require-coolant-open")
    else:
        cmd.append("--no-require-coolant-open")
    if args.dpdt_max is not None:
        cmd.extend(["--dpdt-max", str(float(args.dpdt_max))])
    if int(args.interp_limit) > 0:
        cmd.extend(["--interp-limit", str(int(args.interp_limit))])

    print("[anomalyFilter] Forwarding to featuresBuilder.py ...")
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
