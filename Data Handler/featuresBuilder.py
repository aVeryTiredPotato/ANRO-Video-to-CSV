import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd


def _pickDefaultInput() -> str:
    # Always use the default cleaned CSV as the canonical source
    baseDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(baseDir, "reactor_readings_cleaned.csv")


keyNumeric = [
    "timestamp",
    "temperature",
    "pressure",
    "fuel",
    "rodInsertion",
    "coolant",
    "feedwater",
    "waterLevel",
    "feedwaterFlow",
    "fwpFlowRate1",
    "fwpUtilization1",
    "fwpRpm1",
    "fwpFlowRate2",
    "fwpUtilization2",
    "fwpRpm2",
    "totalOutput",
    "currentPowerOrder",
    "marginOfError",
    "flowRate1",
    "flowRate2",
    "rpm1",
    "rpm2",
    "valvesPct1",
    "valvesPct2",
    "vibration1",
    "vibration2",
]
keyState = ["coolant", "feedwater"]

def _castNumeric(df: pd.DataFrame) -> pd.DataFrame:
    # Cast known numeric columns and any other column that already looks numeric
    for c in df.columns:
        if (c in keyNumeric) or (pd.api.types.is_numeric_dtype(df[c])):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _rollingMedian(x: pd.Series, win: int) -> pd.Series:
    if win <= 1:
        return x
    return x.rolling(int(win), center=True, min_periods=1).median()


def _slopePerSecond(y: pd.Series, t: pd.Series) -> pd.Series:
    dy = y.diff()
    dt = t.diff()
    with np.errstate(divide="ignore", invalid="ignore"):
        s = dy / dt
    return s.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def buildFeaturesAndFlags(
    df: pd.DataFrame,
    *,
    slopeWindow: int = 3,
    minFuel: float = 75.0,
    rodNominal: float = 55.0,
    rodTol: float = 5.0,
    requireCoolantOpen: bool = True,
    dpdtMax: Optional[float] = None,
    # Relief valve parameters
    rvKPerS: float = 7.5,
    rvMatchTol: float = 1.5,
    # Anomaly thresholds (positive/negative not accounted by RV)
    riseKPerS: float = 10.0,
    dipKPerS: float = 10.0,
    padRows: int = 0,
    minAnomalyRun: int = 2,
):
    df = _castNumeric(df.copy())
    n = len(df)
    out = pd.DataFrame(index=df.index)
    if n == 0:
        return out, out
    if "timestamp" not in df.columns or "temperature" not in df.columns:
        # Cannot derive slopes without time and temperature
        return out, out

    # Copy all input columns through to the features output for flexibility
    for c in df.columns:
        out[c] = df[c]

    # Smooth and derive slopes
    tS = _rollingMedian(df["temperature"], slopeWindow)
    pS = _rollingMedian(df["pressure"], slopeWindow) if "pressure" in df else None
    out["dTdt"] = _slopePerSecond(tS, df["timestamp"]).astype(float)
    if pS is not None:
        out["dPdt"] = _slopePerSecond(pS, df["timestamp"]).astype(float)

    # Controls steady: avoid attributing to RV during adjustments
    # Controls steady when rods near nominal, feedwater at 2/2, coolant enabled, and fuel above threshold
    controlsSteady = pd.Series(True, index=df.index)
    rods = df.get("rodInsertion", pd.Series(np.nan, index=df.index))
    if "rodInsertion" in df.columns:
        controlsSteady &= np.isclose(rods, float(rodNominal), atol=1e-6)
    if "feedwater" in df.columns:
        fw = pd.to_numeric(df["feedwater"], errors="coerce")
        controlsSteady &= (fw >= 2)
    if "coolant" in df.columns:
        controlsSteady &= (pd.to_numeric(df["coolant"], errors="coerce") >= 1)
    if "fuel" in df.columns:
        controlsSteady &= (pd.to_numeric(df["fuel"], errors="coerce") > float(minFuel))
    out["controls_steady"] = controlsSteady.fillna(False)

    # Stable regime (feedwater assumed working; do not require feedwater state)
    stable = pd.Series(True, index=df.index)
    if requireCoolantOpen and "coolant" in df.columns:
        stable &= (df["coolant"].fillna(0) >= 1)
    if "fuel" in df.columns:
        stable &= (df["fuel"].fillna(-np.inf) >= float(minFuel))
    if "rodInsertion" in df.columns:
        r = df["rodInsertion"].fillna(np.inf)
        stable &= r.between(float(rodNominal - rodTol), float(rodNominal + rodTol))
    if dpdtMax is not None and "dPdt" in out.columns:
        stable &= (out["dPdt"].abs() <= float(dpdtMax))
    out["stable"] = stable.fillna(False)

    # Relief valve estimate: negative slope magnitude / 7.5 K/s under stable and controls steady
    condRv = out["stable"] & out["controls_steady"] & (out["dTdt"] < 0)
    mag = (-out["dTdt"].clip(upper=0)).where(condRv, other=0.0)
    with np.errstate(invalid="ignore"):
        rvEst = np.rint(mag / float(rvKPerS)).astype(int)
    rvEst = np.clip(rvEst, 0, 4)
    out["rv_est"] = rvEst

    # Anomaly labelling
    exclude = np.zeros(n, dtype=bool)
    reason = np.array([""] * n, dtype=object)

    # Rows that look like deliberate RV pulls: do NOT exclude, just label
    if rvKPerS > 0:
        # residual to nearest rv_est multiple
        target = -rvEst.astype(float) * float(rvKPerS)
        resid = (out["dTdt"] - target).abs()
        rvLike = condRv & (rvEst > 0) & (resid <= float(rvMatchTol))
        out["rv_detected"] = rvLike.astype(bool)
        for i in np.where(rvLike.to_numpy())[0]:
            reason[i] = (reason[i] + "," if reason[i] else "") + f"rv_{int(rvEst.iloc[i])}"
    else:
        out["rv_detected"] = False


    def _enforceMinRun(mask: np.ndarray, minRun: int) -> np.ndarray:
        if int(minRun) <= 1:
            return mask
        outMask = np.zeros_like(mask, dtype=bool)
        runStart = None
        for idx, val in enumerate(mask):
            if val and runStart is None:
                runStart = idx
            elif not val and runStart is not None:
                if idx - runStart >= int(minRun):
                    outMask[runStart:idx] = True
                runStart = None
        if runStart is not None and (len(mask) - runStart) >= int(minRun):
            outMask[runStart:] = True
        return outMask

    # Rising temperature under stable + steady controls -> likely coolant disabled externally
    riseMask = (out["stable"] & out["controls_steady"] & (out["dTdt"] >= float(riseKPerS))).to_numpy()
    riseMask = _enforceMinRun(riseMask, minAnomalyRun)
    riseHits = np.where(riseMask)[0]
    for i in riseHits:
        exclude[i] = True
        reason[i] = (reason[i] + "," if reason[i] else "") + "coolant_off_external"

    # Strong dips not explained by RV
    dipMask = (out["stable"] & out["controls_steady"] & (out["dTdt"] <= -float(dipKPerS)) & (~out["rv_detected"]))
    dipMask = _enforceMinRun(dipMask.to_numpy(), minAnomalyRun)
    dipHits = np.where(dipMask)[0]
    for i in dipHits:
        exclude[i] = True
        reason[i] = (reason[i] + "," if reason[i] else "") + "unexplained_temp_dip"

    # Pad around excluded indices
    if int(padRows) > 0 and (exclude.any()):
        pad = int(padRows)
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
    ap.add_argument("--input", default=_pickDefaultInput(), help="Input CSV (resuscitated preferred)")
    ap.add_argument("--out-features", dest="outFeatures", default=None, help="Output CSV with features+flags (default: *_features.csv)")
    ap.add_argument("--out-filtered", dest="outFiltered", default=None, help="Output CSV with excluded rows removed (default: *_filtered.csv)")

    # Stability and slope params
    ap.add_argument("--slope-window", dest="slopeWindow", type=int, default=3)
    ap.add_argument("--min-fuel", dest="minFuel", type=float, default=75.0)
    ap.add_argument("--rod", dest="rod", type=float, default=55.0)
    ap.add_argument("--rod-tol", dest="rodTol", type=float, default=5.0)
    ap.add_argument("--require-coolant-open", dest="requireCoolantOpen", action="store_true", default=True)
    ap.add_argument("--no-require-coolant-open", dest="requireCoolantOpen", action="store_false")
    ap.add_argument("--dpdt-max", dest="dpdtMax", type=float, default=None)

    # RV + anomaly thresholds
    ap.add_argument("--rv-k-per-s", dest="rvKPerS", type=float, default=7.5)
    ap.add_argument("--rv-match-tol", dest="rvMatchTol", type=float, default=1.5)
    ap.add_argument("--rise-k-per-s", dest="riseKPerS", type=float, default=10.0)
    ap.add_argument("--dip-k-per-s", dest="dipKPerS", type=float, default=10.0)
    ap.add_argument("--pad-rows", dest="padRows", type=int, default=0)
    ap.add_argument("--min-anomaly-run", dest="minAnomalyRun", type=int, default=2, help="Min consecutive rows to flag an anomaly")

    # Optional: interpolate small gaps in filtered output
    ap.add_argument("--interp-limit", dest="interpLimit", type=int, default=0, help="If >0, interpolate up to N-row gaps in numeric cols")

    args = ap.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit(f"Input not found: {args.input}")

    df = pd.read_csv(args.input)

    feats, filt = buildFeaturesAndFlags(
        df,
        slopeWindow=int(args.slopeWindow),
        minFuel=float(args.minFuel),
        rodNominal=float(args.rod),
        rodTol=float(args.rodTol),
        requireCoolantOpen=bool(args.requireCoolantOpen),
        dpdtMax=(float(args.dpdtMax) if args.dpdtMax is not None else None),
        rvKPerS=float(args.rvKPerS),
        rvMatchTol=float(args.rvMatchTol),
        riseKPerS=float(args.riseKPerS),
        dipKPerS=float(args.dipKPerS),
        padRows=int(args.padRows),
        minAnomalyRun=int(args.minAnomalyRun),
    )

    base, ext = os.path.splitext(args.input)
    outFeatures = args.outFeatures or f"{base}_features.csv"
    outFiltered = args.outFiltered or f"{base}_filtered.csv"

    # Optional interpolation on filtered numeric columns
    if int(args.interpLimit) > 0 and len(filt) > 0:
        numCols = [c for c in filt.columns if c not in keyState and c != "timestamp" and pd.api.types.is_numeric_dtype(filt[c])]
        for c in numCols:
            filt[c] = pd.to_numeric(filt[c], errors="coerce")
            filt[c] = filt[c].interpolate(method="linear", limit=int(args.interpLimit), limit_direction="both").round(1)

    feats.to_csv(outFeatures, index=False)
    filt.to_csv(outFiltered, index=False)

    excluded = int(feats["_exclude"].sum()) if "_exclude" in feats.columns else 0
    rvHits = int(feats.get("rv_detected", pd.Series(False)).sum())
    print(
        f"Wrote {outFeatures} and {outFiltered} | rows={len(feats)} | excluded={excluded} | rv_events={rvHits}"
    )


if __name__ == "__main__":
    main()
