import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd


def _pick_default_input() -> str:
    # Always use the default cleaned CSV as the canonical source
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "reactor_readings_cleaned.csv")


def _cast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in [
        "timestamp",
        "temperature",
        "pressure",
        "fuel",
        "rod_insertion",
        "coolant",
        "feedwater",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _rolling_median(x: pd.Series, win: int) -> pd.Series:
    if win <= 1:
        return x
    return x.rolling(int(win), center=True, min_periods=1).median()


def _slope_per_second(y: pd.Series, t: pd.Series) -> pd.Series:
    dy = y.diff()
    dt = t.diff()
    with np.errstate(divide="ignore", invalid="ignore"):
        s = dy / dt
    return s.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_features_and_flags(
    df: pd.DataFrame,
    *,
    slope_window: int = 3,
    min_fuel: float = 75.0,
    rod_nominal: float = 55.0,
    rod_tol: float = 5.0,
    require_coolant_open: bool = True,
    dpdt_max: Optional[float] = None,
    # Relief valve parameters
    rv_k_per_s: float = 7.5,
    rv_match_tol: float = 1.5,
    # Anomaly thresholds (positive/negative not accounted by RV)
    rise_k_per_s: float = 10.0,
    dip_k_per_s: float = 10.0,
    pad_rows: int = 1,
):
    df = _cast_numeric(df.copy())
    n = len(df)
    out = pd.DataFrame(index=df.index)
    if n == 0:
        return out, out

    # Copy base signals
    for c in [
        "timestamp",
        "temperature",
        "pressure",
        "fuel",
        "rod_insertion",
        "coolant",
        "feedwater",
    ]:
        if c in df.columns:
            out[c] = df[c]

    # Smooth and derive slopes
    t_s = _rolling_median(df["temperature"], slope_window)
    p_s = _rolling_median(df["pressure"], slope_window) if "pressure" in df else None
    out["dTdt"] = _slope_per_second(t_s, df["timestamp"]).astype(float)
    if p_s is not None:
        out["dPdt"] = _slope_per_second(p_s, df["timestamp"]).astype(float)

    # Controls steady: avoid attributing to RV during adjustments
    rods = df.get("rod_insertion", pd.Series(np.nan, index=df.index))
    rods_prev = rods.shift(1)
    coolant = df.get("coolant", pd.Series(np.nan, index=df.index))
    coolant_prev = coolant.shift(1)
    controls_steady = (
        (rods.notna() & rods_prev.notna() & (rods - rods_prev).abs() <= 0.2)
        & (coolant.notna() & coolant_prev.notna() & (coolant == coolant_prev))
    )
    out["controls_steady"] = controls_steady.fillna(False)

    # Stable regime (feedwater assumed working; do not require feedwater state)
    stable = pd.Series(True, index=df.index)
    if require_coolant_open and "coolant" in df.columns:
        stable &= (df["coolant"].fillna(0) >= 1)
    if "fuel" in df.columns:
        stable &= (df["fuel"].fillna(-np.inf) >= float(min_fuel))
    if "rod_insertion" in df.columns:
        r = df["rod_insertion"].fillna(np.inf)
        stable &= r.between(float(rod_nominal - rod_tol), float(rod_nominal + rod_tol))
    if dpdt_max is not None and "dPdt" in out.columns:
        stable &= (out["dPdt"].abs() <= float(dpdt_max))
    out["stable"] = stable.fillna(False)

    # Relief valve estimate: negative slope magnitude / 7.5 K/s under stable and controls steady
    cond_rv = out["stable"] & out["controls_steady"] & (out["dTdt"] < 0)
    mag = (-out["dTdt"].clip(upper=0)).where(cond_rv, other=0.0)
    with np.errstate(invalid="ignore"):
        rv_est = np.rint(mag / float(rv_k_per_s)).astype(int)
    rv_est = np.clip(rv_est, 0, 4)
    out["rv_est"] = rv_est

    # Anomaly labelling
    exclude = np.zeros(n, dtype=bool)
    reason = np.array([""] * n, dtype=object)

    # Rows that look like deliberate RV pulls: do NOT exclude, just label
    if rv_k_per_s > 0:
        # residual to nearest rv_est multiple
        target = -rv_est.astype(float) * float(rv_k_per_s)
        resid = (out["dTdt"] - target).abs()
        rv_like = cond_rv & (rv_est > 0) & (resid <= float(rv_match_tol))
        out["rv_detected"] = rv_like.astype(bool)
        for i in np.where(rv_like.to_numpy())[0]:
            reason[i] = (reason[i] + "," if reason[i] else "") + f"rv_{int(rv_est.iloc[i])}"
    else:
        out["rv_detected"] = False

    # Rising temperature under stable + steady controls → likely coolant disabled externally
    rise_hits = np.where((out["stable"]) & (out["controls_steady"]) & (out["dTdt"] >= float(rise_k_per_s)))[0]
    for i in rise_hits:
        exclude[i] = True
        reason[i] = (reason[i] + "," if reason[i] else "") + "coolant_off_external"

    # Strong dips not explained by RV
    dip_hits = np.where(
        (out["stable"]) & (out["controls_steady"]) & (out["dTdt"] <= -float(dip_k_per_s)) & (~out["rv_detected"]))[0]
    for i in dip_hits:
        exclude[i] = True
        reason[i] = (reason[i] + "," if reason[i] else "") + "unexplained_temp_dip"

    # Pad around excluded indices
    if int(pad_rows) > 0 and (exclude.any()):
        pad = int(pad_rows)
        marks = np.zeros(n, dtype=bool)
        idxs = np.where(exclude)[0]
        for j in idxs:
            lo = max(0, j - pad)
            hi = min(n - 1, j + pad)
            marks[lo : hi + 1] = True
        exclude = marks

    out["_exclude"] = exclude
    out["anomaly_reason"] = reason

    # Filtered view (exclude anomalies but keep RV rows)
    filtered = out.loc[~out["_exclude"]].copy()
    return out, filtered


def main():
    ap = argparse.ArgumentParser(
        description="Build features (dTdt/dPdt), detect RV events, and flag/exclude anomalies in one pass"
    )
    ap.add_argument("--input", default=_pick_default_input(), help="Input CSV (resuscitated preferred)")
    ap.add_argument("--out-features", default=None, help="Output CSV with features+flags (default: *_features.csv)")
    ap.add_argument("--out-filtered", default=None, help="Output CSV with excluded rows removed (default: *_filtered.csv)")

    # Stability and slope params
    ap.add_argument("--slope-window", type=int, default=3)
    ap.add_argument("--min-fuel", type=float, default=75.0)
    ap.add_argument("--rod", type=float, default=55.0)
    ap.add_argument("--rod-tol", type=float, default=5.0)
    ap.add_argument("--require-coolant-open", action="store_true", default=True)
    ap.add_argument("--no-require-coolant-open", dest="require_coolant_open", action="store_false")
    ap.add_argument("--dpdt-max", type=float, default=None)

    # RV + anomaly thresholds
    ap.add_argument("--rv-k-per-s", type=float, default=7.5)
    ap.add_argument("--rv-match-tol", type=float, default=1.5)
    ap.add_argument("--rise-k-per-s", type=float, default=10.0)
    ap.add_argument("--dip-k-per-s", type=float, default=10.0)
    ap.add_argument("--pad-rows", type=int, default=1)

    # Optional: interpolate small gaps in filtered output
    ap.add_argument("--interp-limit", type=int, default=0, help="If >0, interpolate up to N-row gaps in numeric cols")

    args = ap.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit(f"Input not found: {args.input}")

    df = pd.read_csv(args.input)

    feats, filt = build_features_and_flags(
        df,
        slope_window=int(args.slope_window),
        min_fuel=float(args.min_fuel),
        rod_nominal=float(args.rod),
        rod_tol=float(args.rod_tol),
        require_coolant_open=bool(args.require_coolant_open),
        dpdt_max=(float(args.dpdt_max) if args.dpdt_max is not None else None),
        rv_k_per_s=float(args.rv_k_per_s),
        rv_match_tol=float(args.rv_match_tol),
        rise_k_per_s=float(args.rise_k_per_s),
        dip_k_per_s=float(args.dip_k_per_s),
        pad_rows=int(args.pad_rows),
    )

    base, ext = os.path.splitext(args.input)
    out_features = args.out_features or f"{base}_features.csv"
    out_filtered = args.out_filtered or f"{base}_filtered.csv"

    # Optional interpolation on filtered numeric columns
    if int(args.interp_limit) > 0 and len(filt) > 0:
        num_cols = [c for c in ["temperature", "pressure", "fuel", "rod_insertion"] if c in filt.columns]
        for c in num_cols:
            filt[c] = pd.to_numeric(filt[c], errors="coerce")
            filt[c] = filt[c].interpolate(method="linear", limit=int(args.interp_limit), limit_direction="both").round(1)

    feats.to_csv(out_features, index=False)
    filt.to_csv(out_filtered, index=False)

    excluded = int(feats["_exclude"].sum()) if "_exclude" in feats.columns else 0
    rv_hits = int(feats.get("rv_detected", pd.Series(False)).sum())
    print(
        f"Wrote {out_features} and {out_filtered} | rows={len(feats)} | excluded={excluded} | rv_events={rv_hits}"
    )


if __name__ == "__main__":
    main()
