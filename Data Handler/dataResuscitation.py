import os
import sys
import argparse
import pandas as pd
import numpy as np
from pandas.api.types import is_object_dtype as isObjectDtype
import cv2
import easyocr
import torch
from typing import Dict, Tuple, List
from tqdm import tqdm


ranges = {
    "temperature": (300.0, 4000.0),
    "pressure": (500.0, 10000.0),
    "fuel": (0.0, 100.0),
    "rodInsertion": (0.0, 100.0),
    "totalOutput": (0.0, 50000.0),
    "currentPowerOrder": (0.0, 50000.0),
    "marginOfError": (1000.0, 1500.0),
    "fwpFlowRate1": (0.0, 1.5),
    "fwpUtilization1": (0.0, 100.0),
    "fwpRpm1": (0.0, 5000.0),
    "fwpFlowRate2": (0.0, 1.5),
    "fwpUtilization2": (0.0, 100.0),
    "fwpRpm2": (0.0, 5000.0),
    "flowRate1": (0.0, 15.0),
    "flowRate2": (0.0, 15.0),
    "rpm1": (0.0, 5000.0),
    "rpm2": (0.0, 5000.0),
    "valvesPct1": (0.0, 100.0),
    "valvesPct2": (0.0, 100.0),
    "vibration1": (100.0, 500.0),
    "vibration2": (100.0, 500.0),
}
decimalKeys = {
    "flowRate1",
    "flowRate2",
    "valvesPct1",
    "valvesPct2",
    "fwpFlowRate1",
    "fwpFlowRate2",
    "fwpUtilization1",
    "fwpUtilization2",
}

# Default jump thresholds per frame (can be overridden)
defaultMaxJump = {
    "temperature": 120.0,
    "pressure": 120.0,
    "fuel": 0.10,
    "rodInsertion": 1.5,
}

# Equality tolerance for stuck-run detection
defaultEps = {
    "temperature": 0.1,
    "pressure": 0.1,
    "fuel": 0.05,
    "rodInsertion": 0.05,
}


def loadCsv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize expected numeric columns
    for col in [
        "temperature",
        "pressure",
        "fuel",
        "rodInsertion",
        "totalOutput",
        "currentPowerOrder",
        "marginOfError",
        "fwpFlowRate1",
        "fwpUtilization1",
        "fwpRpm1",
        "fwpFlowRate2",
        "fwpUtilization2",
        "fwpRpm2",
        "flowRate1",
        "flowRate2",
        "rpm1",
        "rpm2",
        "valvesPct1",
        "valvesPct2",
        "vibration1",
        "vibration2",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def detectRangeViolations(s: pd.Series, lo: float, hi: float) -> pd.Series:
    return (s < lo) | (s > hi)


def detectBigJumps(s: pd.Series, threshold: float) -> pd.Series:
    # Compare to previous non-NaN value
    prev = s.shift(1)
    return (s.notna() & prev.notna() & ((s - prev).abs() > threshold))


def detectStuckRuns(values: pd.Series, times: pd.Series, eps: float,
                      minFrames: int = 15, minSeconds: float = 1.0) -> pd.Series:
    v = values.copy()
    t = times.copy()
    # Start a new group where change exceeds tolerance or when NaN state changes
    change = v.diff().abs().fillna(np.inf) > eps
    nanFlip = v.isna() ^ v.shift(1).isna()
    startNew = change | nanFlip
    gid = startNew.cumsum()
    # Compute run lengths and durations
    counts = gid.map(gid.value_counts())
    # Duration per group
    firstTime = t.groupby(gid).transform('first')
    lastTime = t.groupby(gid).transform('last')
    durations = (lastTime - firstTime).fillna(0.0)
    # Mark stuck if run is long enough by frames or seconds and value is not NaN
    return (counts >= int(minFrames)) | (durations >= float(minSeconds))


def attachConfidenceFlags(df: pd.DataFrame, confThresh: dict) -> pd.DataFrame:
    out = {}
    for key, th in confThresh.items():
        col = f"_conf_{key}"
        if col in df.columns:
            out[f"lowconf_{key}"] = (pd.to_numeric(df[col], errors="coerce") < th)
    return pd.DataFrame(out, index=df.index)


def main():
    ap = argparse.ArgumentParser(description="Re-OCR frames flagged by low confidence (_conf_*) in reactor CSV data")
    ap.add_argument("--cleaned", default="reactor_readings_cleaned.csv", help="Path to cleaned CSV")
    ap.add_argument("--raw", default=None, help="Optional path to raw CSV (no smoothing); not required for confidence-driven flow")
    ap.add_argument("--video", required=True, help="Path to source video for re-OCR")
    ap.add_argument("--outdir", default=None, help="Directory for output files")
    ap.add_argument("--fps", type=float, default=None, help="Override video FPS (use exact framerate if known)")
    ap.add_argument("--force-cpu", dest="forceCpu",
                    action="store_true", default=False, help="Force EasyOCR CPU mode")

    ap.add_argument("--max-jump-temperature", type=float, default=defaultMaxJump["temperature"]) 
    ap.add_argument("--max-jump-pressure", type=float, default=defaultMaxJump["pressure"]) 
    ap.add_argument("--max-jump-fuel", type=float, default=defaultMaxJump["fuel"]) 
    ap.add_argument("--max-jump-rod", type=float, default=defaultMaxJump["rodInsertion"]) 

    ap.add_argument("--stuck-frames", type=int, default=15, help="Min frames to consider a stuck run")
    ap.add_argument("--stuck-seconds", type=float, default=1.0, help="Min seconds to consider a stuck run")
    ap.add_argument("--eps-temperature", type=float, default=defaultEps["temperature"]) 
    ap.add_argument("--eps-pressure", type=float, default=defaultEps["pressure"]) 
    ap.add_argument("--eps-fuel", type=float, default=defaultEps["fuel"]) 
    ap.add_argument("--eps-rod", type=float, default=defaultEps["rodInsertion"]) 

    # ROI controls (defaults mirror current dataGrabber configuration)
    ap.add_argument("--roi-temperature", dest="roiTemperature",
                    default="2007,607,2105,584", help="x1,y1,x2,y2 for temperature")
    ap.add_argument("--roi-pressure", dest="roiPressure",
                    default="2001,564,2117,540", help="x1,y1,x2,y2 for pressure")
    ap.add_argument("--roi-fuel", dest="roiFuel",
                    default="1932,470,2004,445", help="x1,y1,x2,y2 for fuel")
    ap.add_argument("--roi-rod", dest="roiRod",
                    default="1967,272,2033,245", help="x1,y1,x2,y2 for rod insertion")
    ap.add_argument("--roi-coolant", dest="roiCoolant",
                    default="2098,182,2176,152", help="x1,y1,x2,y2 for coolant state (optional)")
    ap.add_argument("--roi-feedwater", dest="roiFeedwater",
                    default="1921,381,2039,353", help="x1,y1,x2,y2 for feedwater (optional)")
    ap.add_argument("--roi-fwp-flow-rate1", dest="roiFwpFlowRate1",
                    default="0,0,0,0", help="x1,y1,x2,y2 for FWP flow rate 1 (optional)")
    ap.add_argument("--roi-fwp-utilization1", dest="roiFwpUtilization1",
                    default="0,0,0,0", help="x1,y1,x2,y2 for FWP utilization 1 (optional)")
    ap.add_argument("--roi-fwp-rpm1", dest="roiFwpRpm1",
                    default="0,0,0,0", help="x1,y1,x2,y2 for FWP RPM 1 (optional)")
    ap.add_argument("--roi-fwp-flow-rate2", dest="roiFwpFlowRate2",
                    default="0,0,0,0", help="x1,y1,x2,y2 for FWP flow rate 2 (optional)")
    ap.add_argument("--roi-fwp-utilization2", dest="roiFwpUtilization2",
                    default="0,0,0,0", help="x1,y1,x2,y2 for FWP utilization 2 (optional)")
    ap.add_argument("--roi-fwp-rpm2", dest="roiFwpRpm2",
                    default="0,0,0,0", help="x1,y1,x2,y2 for FWP RPM 2 (optional)")
    ap.add_argument("--roi-total-output", dest="roiTotalOutput",
                    default="0,0,0,0", help="x1,y1,x2,y2 for total output (optional)")
    ap.add_argument("--roi-current-power-order", dest="roiCurrentPowerOrder",
                    default="0,0,0,0", help="x1,y1,x2,y2 for current power order (optional)")
    ap.add_argument("--roi-margin-of-error", dest="roiMarginOfError",
                    default="0,0,0,0", help="x1,y1,x2,y2 for margin of error (optional)")
    ap.add_argument("--roi-flow-rate", "--roi-flow-rate1", dest="roiFlowRate1",
                    default="0,0,0,0", help="x1,y1,x2,y2 for flow rate 1 (optional)")
    ap.add_argument("--roi-flow-rate2", dest="roiFlowRate2",
                    default="0,0,0,0", help="x1,y1,x2,y2 for flow rate 2 (optional)")
    ap.add_argument("--roi-rpm", "--roi-rpm1", dest="roiRpm1",
                    default="0,0,0,0", help="x1,y1,x2,y2 for RPM 1 (optional)")
    ap.add_argument("--roi-rpm2", dest="roiRpm2",
                    default="0,0,0,0", help="x1,y1,x2,y2 for RPM 2 (optional)")
    ap.add_argument("--roi-valves-pct", "--roi-valves-pct1", dest="roiValvesPct1",
                    default="0,0,0,0", help="x1,y1,x2,y2 for valves percent 1 (optional)")
    ap.add_argument("--roi-valves-pct2", dest="roiValvesPct2",
                    default="0,0,0,0", help="x1,y1,x2,y2 for valves percent 2 (optional)")
    ap.add_argument("--roi-vibration1", dest="roiVibration1",
                    default="0,0,0,0", help="x1,y1,x2,y2 for vibration 1 (optional)")
    ap.add_argument("--roi-vibration2", dest="roiVibration2",
                    default="0,0,0,0", help="x1,y1,x2,y2 for vibration 2 (optional)")

    # Re-OCR behavior
    ap.add_argument(
        "--keys",
        default="temperature,pressure,fuel,rodInsertion,coolant,feedwater,totalOutput,currentPowerOrder,marginOfError,fwpFlowRate1,fwpUtilization1,fwpRpm1,fwpFlowRate2,fwpUtilization2,fwpRpm2,flowRate1,flowRate2,rpm1,rpm2,valvesPct1,valvesPct2,vibration1,vibration2",
        help="Comma list of keys to re-OCR (must have _conf_* columns present)"
    )
    ap.add_argument("--strict-conf-temperature", type=float, default=0.30)
    ap.add_argument("--strict-conf-pressure", type=float, default=0.30)
    ap.add_argument("--strict-conf-fuel", type=float, default=0.50)
    ap.add_argument("--strict-conf-rod", type=float, default=0.40)
    ap.add_argument("--strict-conf-state", type=float, default=0.30, help="Strict confidence for state reads (coolant/feedwater)")
    ap.add_argument("--save-roi", dest="saveRoi",
                    action="store_true", help="Save ROI crops and best variants for debugging")
    ap.add_argument("--roi-debug-dir", dest="roiDebugDir",
                    default="roi_debug/resuscitated", help="Where to save ROI debug images")

    args = ap.parse_args()

    cleaned = loadCsv(args.cleaned)
    raw = loadCsv(args.raw) if args.raw else None

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.cleaned))
    os.makedirs(outdir, exist_ok=True)

    # Key metadata: determines how to OCR each column
    keyTypes = {
        "temperature": "numeric",
        "pressure": "numeric",
        "fuel": "numeric",
        "rodInsertion": "numeric",
        "waterLevel": "numeric",
        "feedwaterFlow": "numeric",
        "fwpFlowRate1": "numeric",
        "fwpUtilization1": "numeric",
        "fwpRpm1": "numeric",
        "fwpFlowRate2": "numeric",
        "fwpUtilization2": "numeric",
        "fwpRpm2": "numeric",
        "totalOutput": "numeric",
        "currentPowerOrder": "numeric",
        "marginOfError": "numeric",
        "flowRate1": "numeric",
        "flowRate2": "numeric",
        "rpm1": "numeric",
        "rpm2": "numeric",
        "valvesPct1": "numeric",
        "valvesPct2": "numeric",
        "vibration1": "numeric",
        "vibration2": "numeric",
        "coolant": "state",
        "feedwater": "feed",
    }
    strictConf = {
        "temperature": args.strict_conf_temperature,
        "pressure": args.strict_conf_pressure,
        "fuel": args.strict_conf_fuel,
        "rodInsertion": args.strict_conf_rod,
    }

    # Use low-confidence _conf_* markers from cleaned CSV to decide what to re-OCR
    report = cleaned.copy()
    ts = pd.to_numeric(report.get("timestamp", pd.Series(range(len(report)), dtype=float)), errors="coerce").fillna(0.0)
    confMarkerCols = [c for c in report.columns if c.startswith("_conf_")]

    keysRequested = [k.strip() for k in args.keys.split(',') if k.strip()]

    def confThreshForKey(key: str) -> float:
        if keyTypes.get(key) in ("state", "feed"):
            return float(args.strict_conf_state)
        return float(strictConf.get(key, 0.30))

    if not confMarkerCols:
        print("ERROR: No _conf_* columns found; cannot determine low-confidence frames.")
        return

    perKeyIndices = {}
    # Build per-key low-confidence indices from available _conf_* columns
    keyToConf = {c.replace("_conf_", ""): c for c in confMarkerCols}
    keysAvailable = [k for k in keysRequested if k in keyToConf]
    keysMissingConf = [k for k in keysRequested if k not in keyToConf]
    if not keysAvailable:
        print("Nothing to do: no _conf_* columns found for requested keys.")
        if keysMissingConf:
            print("Missing _conf_* columns:", ", ".join(keysMissingConf))
        return
    for key in keysAvailable:
        confCol = keyToConf[key]
        confVals = pd.to_numeric(report[confCol], errors="coerce")
        lowConf = confVals.isna() | (confVals < confThreshForKey(key))
        perKeyIndices[key] = list(report.index[lowConf])
    planSource = "low-confidence _conf_* markers"

    # --- High-scrutiny re-OCR on low-confidence frames ---
    # Print plan summary per key
    print(f"Re-OCR plan (from {planSource}):")
    for k in keysRequested:
        if k in keysMissingConf:
            print(f"- {k}: missing _conf_ column")
        else:
            print(f"- {k}: {len(perKeyIndices.get(k, []))} items")
    if keysMissingConf:
        print("Missing _conf_* columns:", ", ".join(keysMissingConf))
    flaggedCount = sum(len(v) for v in perKeyIndices.values())
    print(f"Total flagged items: {flaggedCount}")
    if flaggedCount == 0:
        print("No low-confidence frames/keys; skipping re-OCR.")
        return

    # Parse ROI strings
    def parseRoi(s: str) -> Tuple[int, int, int, int]:
        parts = [int(p.strip()) for p in s.split(',')]
        if len(parts) != 4:
            raise ValueError(f"ROI must have 4 integers: {s}")
        return parts[0], parts[1], parts[2], parts[3]

    # Only attach ROI entries for keys where an ROI was provided
    def _maybeParseRoi(name: str, val: str):
        if val is None or str(val).strip() == "":
            return None
        try:
            roi = parseRoi(val)
            if roi[0] == roi[1] == roi[2] == roi[3] == 0:
                return None
            return roi
        except Exception:
            print(f"Warning: skipping ROI for {name}, could not parse: {val}")
            return None

    rois: Dict[str, Tuple[int, int, int, int]] = {
        k: v for k, v in {
            "temperature": _maybeParseRoi("temperature", args.roiTemperature),
            "pressure": _maybeParseRoi("pressure", args.roiPressure),
            "fuel": _maybeParseRoi("fuel", args.roiFuel),
            "rodInsertion": _maybeParseRoi("rodInsertion", args.roiRod),
            "coolant": _maybeParseRoi("coolant", args.roiCoolant),
            "feedwater": _maybeParseRoi("feedwater", args.roiFeedwater),
            "fwpFlowRate1": _maybeParseRoi("fwpFlowRate1", args.roiFwpFlowRate1),
            "fwpUtilization1": _maybeParseRoi("fwpUtilization1", args.roiFwpUtilization1),
            "fwpRpm1": _maybeParseRoi("fwpRpm1", args.roiFwpRpm1),
            "fwpFlowRate2": _maybeParseRoi("fwpFlowRate2", args.roiFwpFlowRate2),
            "fwpUtilization2": _maybeParseRoi("fwpUtilization2", args.roiFwpUtilization2),
            "fwpRpm2": _maybeParseRoi("fwpRpm2", args.roiFwpRpm2),
            "totalOutput": _maybeParseRoi("totalOutput", args.roiTotalOutput),
            "currentPowerOrder": _maybeParseRoi("currentPowerOrder", args.roiCurrentPowerOrder),
            "marginOfError": _maybeParseRoi("marginOfError", args.roiMarginOfError),
            "flowRate1": _maybeParseRoi("flowRate1", args.roiFlowRate1),
            "flowRate2": _maybeParseRoi("flowRate2", args.roiFlowRate2),
            "rpm1": _maybeParseRoi("rpm1", args.roiRpm1),
            "rpm2": _maybeParseRoi("rpm2", args.roiRpm2),
            "valvesPct1": _maybeParseRoi("valvesPct1", args.roiValvesPct1),
            "valvesPct2": _maybeParseRoi("valvesPct2", args.roiValvesPct2),
            "vibration1": _maybeParseRoi("vibration1", args.roiVibration1),
            "vibration2": _maybeParseRoi("vibration2", args.roiVibration2),
        }.items() if v is not None
    }

    keys = [k.strip() for k in args.keys.split(',') if k.strip()]

    # Initialize EasyOCR reader with optional CPU force
    _useGpu = (torch.cuda.is_available() and not args.forceCpu)
    reader = easyocr.Reader(['en'], gpu=_useGpu)
    print(f"EasyOCR (resuscitation) GPU={'ON' if _useGpu else 'OFF'}")

    # Preprocessing variants for high-scrutiny OCR
    def superPreprocess(roiImg) -> List[np.ndarray]:
        # Pruned variant set for speed: CLAHE + Gaussian blur + (Otsu, Adaptive Gaussian),
        # plus a single MORPH_CLOSE. No mean threshold, no bitwise_not, minimal duplication.
        arr = []
        if roiImg is None or getattr(roiImg, 'size', 0) == 0:
            return arr
        try:
            gray = cv2.cvtColor(roiImg, cv2.COLOR_BGR2GRAY)
        except Exception:
            gray = roiImg
        scales = [3.0]  # single scale suffices for most cases
        blurs = [(3, 3)]
        for s in scales:
            g = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gc = clahe.apply(g)
            for b in blurs:
                gb = cv2.GaussianBlur(gc, b, 0)
                vGauss = cv2.adaptiveThreshold(gb, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7)
                _, vOtsu = cv2.threshold(gb, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                kern = np.ones((2, 2), np.uint8)
                arr.append(cv2.morphologyEx(vGauss, cv2.MORPH_CLOSE, kern, iterations=1))
                arr.append(cv2.morphologyEx(vOtsu, cv2.MORPH_CLOSE, kern, iterations=1))
        # Deduplicate by shape + coarse checksum
        uniq = []
        seen = set()
        for a in arr:
            key = (a.shape, int(a.mean()))
            if key not in seen:
                uniq.append(a)
                seen.add(key)
        return uniq[:12]

    digitFixSafe = str.maketrans({'O':'0','o':'0','S':'5','s':'5','I':'1','l':'1','B':'8'})
    import re as _re
    _patterns = {
        'temperature': _re.compile(r"^(?P<VAL>\d{3,4}\.\d)K$"),
        'pressure': _re.compile(r"^(?P<VAL>\d{3,5}\.\d)KPA$"),
        'percent': _re.compile(r"^(?P<VAL>\d{1,3}(?:\.\d)?)%$"),
        'percent_opt': _re.compile(r"^(?P<VAL>\d{1,3}(?:\.\d)?)%?$"),
        'kw': _re.compile(r"^(?P<VAL>\d{1,5}(?:\.\d)?)(?:KW)?$"),
        'rpm': _re.compile(r"^(?P<VAL>\d{1,5}(?:\.\d)?)(?:RPM)?$"),
        'flow': _re.compile(r"^(?P<VAL>\d{1,3}(?:\.\d{1,2})?)(?:L/S|LS)?$"),
        'plain': _re.compile(r"^(?P<VAL>\d{1,4}(?:\.\d)?)$"),
    }
    def _normalizeText(s: str) -> str:
        return s.replace(',', '.').replace(' ', '').upper()

    def _isNaPowerOrder(text: str) -> bool:
        t = _normalizeText(text)
        t = _re.sub(r"[^A-Z/]", "", t).replace("\\", "/").replace("-", "/")
        return t in ("NA", "N/A")

    def _coerceDecimalForRange(key: str, val: float, text: str):
        if val is None or key not in decimalKeys:
            return val
        t = _normalizeText(text)
        if "." in t:
            return val
        lo, hi = ranges.get(key, (-float("inf"), float("inf")))
        if val <= hi:
            return val
        for div in (10.0, 100.0, 1000.0):
            scaled = val / div
            if lo <= scaled <= hi:
                return scaled
        return val

    def _extractValueForKey(key: str, text: str):
        t = _normalizeText(text)
        if key in ("fuel", "rodInsertion"):
            m = _patterns['percent'].match(t)
            return float(m.group('VAL')) if m else None
        if key == 'temperature':
            m = _patterns['temperature'].match(t)
            return float(m.group('VAL')) if m else None
        if key == 'pressure':
            m = _patterns['pressure'].match(t)
            return float(m.group('VAL')) if m else None
        if key in ("totalOutput", "currentPowerOrder", "marginOfError"):
            m = _patterns['kw'].match(t)
            val = float(m.group('VAL')) if m else None
            return _coerceDecimalForRange(key, val, t)
        if key in ("rpm1", "rpm2"):
            m = _patterns['rpm'].match(t)
            val = float(m.group('VAL')) if m else None
            return _coerceDecimalForRange(key, val, t)
        if key in ("fwpRpm1", "fwpRpm2"):
            m = _patterns['rpm'].match(t)
            val = float(m.group('VAL')) if m else None
            return _coerceDecimalForRange(key, val, t)
        if key in ("flowRate1", "flowRate2"):
            m = _patterns['flow'].match(t)
            val = float(m.group('VAL')) if m else None
            return _coerceDecimalForRange(key, val, t)
        if key in ("fwpFlowRate1", "fwpFlowRate2"):
            m = _patterns['flow'].match(t)
            val = float(m.group('VAL')) if m else None
            return _coerceDecimalForRange(key, val, t)
        if key in ("valvesPct1", "valvesPct2"):
            m = _patterns['percent_opt'].match(t)
            val = float(m.group('VAL')) if m else None
            return _coerceDecimalForRange(key, val, t)
        if key in ("fwpUtilization1", "fwpUtilization2"):
            m = _patterns['percent_opt'].match(t)
            val = float(m.group('VAL')) if m else None
            return _coerceDecimalForRange(key, val, t)
        if key in ("vibration1", "vibration2"):
            m = _patterns['plain'].match(t)
            return float(m.group('VAL')) if m else None
        return None

    def _readtextMulti(images, allow=None):
        try:
            if hasattr(reader, 'readtext_batched'):
                if allow is not None:
                    return reader.readtext_batched(list(images), detail=1, paragraph=False, allowlist=allow)
                return reader.readtext_batched(list(images), detail=1, paragraph=False)
        except Exception:
            pass
        out = []
        for img in images:
            try:
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                out.append(reader.readtext(img, detail=1, paragraph=False, allowlist=allow))
            except Exception:
                out.append([])
        return out

    def ocrNumericStrict(key: str, roiImg) -> Tuple[float, float, str]:
        allow = "0123456789.%KkPpAaWwRrMmLlSs/Nn"
        variants = superPreprocess(roiImg)
        buckets = {}
        bestConfFor = {}
        bestTxtFor = {}
        bestSeen = 0.0
        bestNaConf, bestNaTxt = 0.0, ""
        resList = _readtextMulti(list(variants), allow)
        for res in resList:
            for (_box, txt, conf) in res:
                txt = txt.translate(digitFixSafe)
                if key == "currentPowerOrder" and _isNaPowerOrder(txt):
                    raw = float(conf)
                    if raw > bestNaConf:
                        bestNaConf, bestNaTxt = raw, txt
                    if raw > bestSeen:
                        bestSeen = raw
                    continue
                val = _extractValueForKey(key, txt)
                if val is None:
                    continue
                val = round(float(val), 1)
                raw = float(conf)
                score = raw + min(len(txt), 12) * 0.01
                buckets[val] = buckets.get(val, 0.0) + score
                if val not in bestConfFor or raw > bestConfFor[val]:
                    bestConfFor[val] = raw
                    bestTxtFor[val] = txt
                if raw > bestSeen:
                    bestSeen = raw
            if bestSeen > 0.85:
                break
        if not buckets:
            if bestNaConf > 0.0:
                return None, min(0.99, bestNaConf), bestNaTxt
            return None, 0.0, ""
        bestVal = max(buckets.items(), key=lambda kv: kv[1])[0]
        bestConf = min(0.99, bestConfFor.get(bestVal, 0.0))
        if bestNaConf > bestConf:
            return None, min(0.99, bestNaConf), bestNaTxt
        return bestVal, bestConf, bestTxtFor.get(bestVal, "")

    def ocrCoolantOpen(roiImg) -> Tuple[int, float, str]:
        # Returns (1 if OPEN detected else 0, confidence, raw_text)
        allow = "ABCDEFGHIJKLMNOPQRSTUVWXYZ/ "
        variants = superPreprocess(roiImg)
        bestC = 0.0
        bestT = ""
        resList = _readtextMulti(list(variants), allow)
        for res in resList:
            for (_box, txt, conf) in res:
                t = (txt or "").upper()
                if "OPEN" in t and float(conf) > bestC:
                    bestC, bestT = float(conf), t
            if bestC > 0.85:
                break
        return (1 if bestC > 0 else 0), min(0.99, bestC), bestT

    def ocrFeedwaterCountStrict(roiImg) -> Tuple[int, float, str]:
        # Returns (count 0..2 or None->-1, conf, raw_text)
        import re as _re2
        pat = _re2.compile(r"([0-2])\s*/\s*2")
        allow = "0123456789/ "
        # In addition to full ROI, also try left-ratio crop
        variants = superPreprocess(roiImg)
        try:
            h, w = roiImg.shape[:2]
            left = roiImg[:, : max(1, int(w * 0.65))]
            variants += superPreprocess(left)
        except Exception:
            pass
        byVal = {0: 0.0, 1: 0.0, 2: 0.0}
        bestC = 0.0
        bestT = ""
        resList = _readtextMulti(list(variants), allow)
        for res in resList:
            for (_box, txt, conf) in res:
                t = "" if txt is None else str(txt)
                t2 = "".join(ch for ch in t if ch.isdigit() or ch == '/')
                m = pat.search(t2)
                if not m:
                    continue
                n = int(m.group(1))
                if 0 <= n <= 2:
                    c = float(conf)
                    byVal[n] += c
                    if c > bestC:
                        bestC, bestT = c, t2
            if bestC > 0.85:
                break
        picked = max(byVal.items(), key=lambda kv: kv[1])[0]
        if byVal[picked] == 0.0:
            return -1, 0.0, ""
        return picked, min(0.99, bestC), bestT

    # Prepare capture
    cap = cv2.VideoCapture(args.video)
    if not cap or not cap.isOpened():
        print(f"ERROR: cannot open video: {args.video}")
        return
    fps = float(args.fps) if args.fps else (cap.get(cv2.CAP_PROP_FPS) or 60.0)

    replaceLog = []
    updated = cleaned.copy()

    # Ensure _raw_* columns can store string payloads safely (avoid float dtype warnings)
    for col in [c for c in updated.columns if c.startswith("_raw_")]:
        if not isObjectDtype(updated[col]):
            updated[col] = updated[col].astype("object")
    os.makedirs(args.roiDebugDir, exist_ok=True) if args.saveRoi else None

    # Safety banner for VRAM mode
    if _useGpu:
        print("GPU mode enabled for EasyOCR (batched).")
    else:
        print("CPU fallback mode (expect slower runtime).")

    # Helper: one fast variant (CLAHE+Gaussian+Otsu) for batching
    def fastVariant(img):
        try:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception:
            g = img
        g = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        g = cv2.GaussianBlur(g, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        g = clahe.apply(g)
        _, vOtsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kern = np.ones((2, 2), np.uint8)
        return cv2.morphologyEx(vOtsu, cv2.MORPH_CLOSE, kern, iterations=1)

    # Build task list grouped by frame index
    tasks = []  # (frame_idx, idx_row, key)
    for key in keysAvailable:
        if key not in rois:
            continue
        for idx in perKeyIndices.get(key, []):
            t = float(ts.iloc[idx]) if idx < len(ts) else (idx / fps)
            frameIdx = int(round(t * fps))
            tasks.append((frameIdx, idx, key))
    tasks.sort(key=lambda x: x[0])

    processedPairs = 0
    outStream = sys.stdout if getattr(sys.stdout, "write", None) else sys.__stdout__
    pbar = tqdm(total=len(tasks), desc="Resuscitating", ncols=80, file=outStream) if tasks else None
    if tasks:
        tasksByFrame = {}
        for frameIdx, idxRow, key in tasks:
            tasksByFrame.setdefault(frameIdx, []).append((frameIdx, idxRow, key))
        frameIndices = sorted(tasksByFrame.keys())
        frameStart = frameIndices[0]
        frameEnd = frameIndices[-1]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameStart)
        currentFrame = frameStart

        while currentFrame <= frameEnd:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            group = tasksByFrame.get(currentFrame)
            if not group:
                currentFrame += 1
                continue
            frameIdx = currentFrame
            h, w = frame.shape[:2]

            # Per-allowlist batches
            batchNum, ctxNum = [], []
            batchState, ctxState = [], []
            batchFeed, ctxFeed = [], []

            # Prepare primary ROI variants for batch
            for (_f, idxRow, key) in group:
                if key not in rois:
                    continue
                x1, y1, x2, y2 = rois[key]
                xLo, xHi = (x1, x2) if x1 <= x2 else (x2, x1)
                yLo, yHi = (y1, y2) if y1 <= y2 else (y2, y1)
                xLo, yLo = max(0, xLo), max(0, yLo)
                xHi, yHi = min(w, xHi), min(h, yHi)
                baseRoi = frame[yLo:yHi, xLo:xHi]
                v = fastVariant(baseRoi)
                ktype = keyTypes.get(key, "numeric")
                if ktype == "numeric":
                    batchNum.append(v)
                    ctxNum.append((key, idxRow, frameIdx, baseRoi))
                elif ktype == "state":
                    batchState.append(v)
                    ctxState.append((key, idxRow, frameIdx, baseRoi))
                elif ktype == "feed":
                    batchFeed.append(v)
                    ctxFeed.append((key, idxRow, frameIdx, baseRoi))

            # Process each batch type with appropriate allowlists
            def handleNumeric(ctx, res):
                # Aggregate best per context
                bestVal, bestConf, bestTxt = None, 0.0, ""
                bestNaConf, bestNaTxt = 0.0, ""
                key = ctx[0]
                for (_box, txt, conf) in res:
                    txt2 = txt.translate(digitFixSafe)
                    if key == "currentPowerOrder" and _isNaPowerOrder(txt2):
                        raw = float(conf)
                        if raw > bestNaConf:
                            bestNaConf, bestNaTxt = raw, txt
                        continue
                    val = _extractValueForKey(key, txt2)
                    if val is None:
                        continue
                    raw = float(conf)
                    if raw > bestConf:
                        bestConf, bestVal, bestTxt = raw, round(float(val), 1), txt
                if bestVal is None and bestNaConf > 0.0:
                    return None, min(0.99, bestNaConf), bestNaTxt
                if bestNaConf > bestConf:
                    return None, min(0.99, bestNaConf), bestNaTxt
                return bestVal, bestConf, bestTxt

            def handleState(res):
                bestC, bestT = 0.0, ""
                for (_box, txt, conf) in res:
                    t = (txt or "").upper()
                    if "OPEN" in t and float(conf) > bestC:
                        bestC, bestT = float(conf), t
                return (1 if bestC > 0 else 0), min(0.99, bestC), bestT

            def handleFeed(res):
                import re as _re2
                pat = _re2.compile(r"([0-2])\s*/\s*2")
                bestC, bestT, bestV = 0.0, "", -1
                for (_box, txt, conf) in res:
                    t2 = "".join(ch for ch in (txt or "") if ch.isdigit() or ch == '/')
                    m = pat.search(t2)
                    if not m:
                        continue
                    n = int(m.group(1))
                    if 0 <= n <= 2 and float(conf) > bestC:
                        bestC, bestT, bestV = float(conf), t2, n
                return (bestV if bestV >= 0 else None), min(0.99, bestC), bestT

            # Run batched OCR per type
            if batchNum:
                resList = _readtextMulti(batchNum, "0123456789.%KkPpAaWwRrMmLlSs/Nn")
                for ctx, res in zip(ctxNum, resList):
                    key, idxRow, fidx, baseRoi = ctx
                    val, conf, txt = handleNumeric(ctx, res)
                    # Fallback to strict if weak (numeric keys only)
                    if keyTypes.get(key, "numeric") == "numeric" and (val is None or conf < strictConf.get(key, 0.3)):
                        # Try padded strict search
                        pads = [(0,0,0,0), (0,0,8,0), (0,0,0,4), (4,2,8,2)]
                        bestVal, bestConf, bestTxt = None, 0.0, ""
                        x1, y1, x2, y2 = rois[key]
                        xLo, xHi = (x1, x2) if x1 <= x2 else (x2, x1)
                        yLo, yHi = (y1, y2) if y1 <= y2 else (y2, y1)
                        xLo, yLo = max(0, xLo), max(0, yLo)
                        xHi, yHi = min(w, xHi), min(h, yHi)
                        for (lp, tp, rp, bp) in pads:
                            xl = max(0, xLo - lp); yl = max(0, yLo - tp)
                            xr = min(w, xHi + rp); yr = min(h, yHi + bp)
                            vimg = frame[yl:yr, xl:xr]
                            vv, cc, tt = ocrNumericStrict(key, vimg)
                            if cc > bestConf:
                                bestVal, bestConf, bestTxt = vv, cc, tt
                            processedPairs += 1
                        val, conf, txt = bestVal, bestConf, bestTxt
                    # Apply acceptance/update and always set conf/raw
                    if keyTypes.get(key, "numeric") == "numeric":
                        accept = (val is not None and conf >= strictConf.get(key, 0.3))
                        if accept and key in ranges:
                            old = updated.at[idxRow, key] if key in updated.columns else None
                            lo, hi = ranges[key]
                            if val is not None and (lo <= val <= hi) and ((pd.isna(old)) or (str(old) != str(val))):
                                updated.at[idxRow, key] = float(val)
                                replaceLog.append({"index": int(idxRow), "time": float(ts.iloc[idxRow]) if idxRow < len(ts) else (idxRow / fps),
                                                    "key": key, "old": None if pd.isna(old) else old, "new": float(val),
                                                    "conf": float(conf), "text": txt, "frame": int(frameIdx)})
                                if args.saveRoi:
                                    try:
                                        cv2.imwrite(os.path.join(args.roiDebugDir, f"{idxRow:06d}_{key}.png"), baseRoi)
                                    except Exception:
                                        pass
                    updated.at[idxRow, f"_raw_{key}"] = txt
                    updated.at[idxRow, f"_conf_{key}"] = float(conf)

            if batchState:
                resList = _readtextMulti(batchState, "ABCDEFGHIJKLMNOPQRSTUVWXYZ/ ")
                for ctx, res in zip(ctxState, resList):
                    key, idxRow, fidx, baseRoi = ctx
                    val, conf, txt = handleState(res)
                    accept = (conf >= args.strict_conf_state)
                    if accept:
                        old = updated.at[idxRow, key] if key in updated.columns else None
                        if (pd.isna(old)) or (str(old) != str(val)):
                            updated.at[idxRow, key] = int(val)
                            replaceLog.append({"index": int(idxRow), "time": float(ts.iloc[idxRow]) if idxRow < len(ts) else (idxRow / fps),
                                                "key": key, "old": None if pd.isna(old) else old, "new": int(val),
                                                "conf": float(conf), "text": txt, "frame": int(frameIdx)})
                            if args.saveRoi:
                                try:
                                    cv2.imwrite(os.path.join(args.roiDebugDir, f"{idxRow:06d}_{key}.png"), baseRoi)
                                except Exception:
                                    pass
                    updated.at[idxRow, f"_raw_{key}"] = txt
                    updated.at[idxRow, f"_conf_{key}"] = float(conf)

            if batchFeed:
                resList = _readtextMulti(batchFeed, "0123456789/ ")
                for ctx, res in zip(ctxFeed, resList):
                    key, idxRow, fidx, baseRoi = ctx
                    val, conf, txt = handleFeed(res)
                    accept = (val is not None and conf >= args.strict_conf_state)
                    if accept:
                        old = updated.at[idxRow, key] if key in updated.columns else None
                        if (pd.isna(old)) or (str(old) != str(val)):
                            updated.at[idxRow, key] = int(val)
                            replaceLog.append({"index": int(idxRow), "time": float(ts.iloc[idxRow]) if idxRow < len(ts) else (idxRow / fps),
                                                "key": key, "old": None if pd.isna(old) else old, "new": int(val),
                                                "conf": float(conf), "text": txt, "frame": int(frameIdx)})
                            if args.saveRoi:
                                try:
                                    cv2.imwrite(os.path.join(args.roiDebugDir, f"{idxRow:06d}_{key}.png"), baseRoi)
                                except Exception:
                                    pass
                    updated.at[idxRow, f"_raw_{key}"] = txt
                    updated.at[idxRow, f"_conf_{key}"] = float(conf)

            # Final flush per frame done; continue to next frame
            if pbar:
                pbar.update(len(group))
            currentFrame += 1


    if pbar:
        pbar.close()

    # Save updated CSV without adding smoothing/interpolation
    # Final cleanup/interpolation and write directly to the cleaned CSV path
    for col in [c for c, t in keyTypes.items() if t == "numeric"]:
        if col in updated.columns:
            updated[col] = pd.to_numeric(updated[col], errors="coerce")
            updated[col] = updated[col].interpolate(method="linear", limit_direction="both").round(2)
    cleanedOut = os.path.abspath(args.cleaned)
    updated.to_csv(cleanedOut, index=False)
    print(f"Resuscitated CSV written to cleaned output: {cleanedOut}")
    print(f"Processed {processedPairs} flagged key/frame pairs.")

    # Save replacement log
    if replaceLog:
        replDf = pd.DataFrame(replaceLog)
        replPath = os.path.join(outdir, "resuscitation_replacements.csv")
        replDf.to_csv(replPath, index=False)
        print(f"Replacements written: {replPath} ({len(replaceLog)} updates)")
    else:
        print("No replacements made (nothing met strict acceptance criteria).")


if __name__ == "__main__":
    main()
