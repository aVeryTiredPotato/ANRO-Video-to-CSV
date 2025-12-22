import os
import argparse
import pandas as pd
import numpy as np
from pandas.api.types import is_object_dtype
import cv2
import easyocr
import torch
from typing import Dict, Tuple, List
from tqdm import tqdm


RANGES = {
    "temperature": (300.0, 4000.0),
    "pressure": (500.0, 10000.0),
    "fuel": (0.0, 100.0),
    "rod_insertion": (0.0, 100.0),
}

# Default jump thresholds per frame (can be overridden)
DEFAULT_MAX_JUMP = {
    "temperature": 120.0,
    "pressure": 120.0,
    "fuel": 0.10,
    "rod_insertion": 1.5,
}

# Equality tolerance for stuck-run detection
DEFAULT_EPS = {
    "temperature": 0.1,
    "pressure": 0.1,
    "fuel": 0.05,
    "rod_insertion": 0.05,
}


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize expected numeric columns
    for col in ["temperature", "pressure", "fuel", "rod_insertion"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def detect_range_violations(s: pd.Series, lo: float, hi: float) -> pd.Series:
    return (s < lo) | (s > hi)


def detect_big_jumps(s: pd.Series, threshold: float) -> pd.Series:
    # Compare to previous non-NaN value
    prev = s.shift(1)
    return (s.notna() & prev.notna() & ((s - prev).abs() > threshold))


def detect_stuck_runs(values: pd.Series, times: pd.Series, eps: float,
                      min_frames: int = 15, min_seconds: float = 1.0) -> pd.Series:
    v = values.copy()
    t = times.copy()
    # Start a new group where change exceeds tolerance or when NaN state changes
    change = v.diff().abs().fillna(np.inf) > eps
    nan_flip = v.isna() ^ v.shift(1).isna()
    start_new = change | nan_flip
    gid = start_new.cumsum()
    # Compute run lengths and durations
    counts = gid.map(gid.value_counts())
    # Duration per group
    first_time = t.groupby(gid).transform('first')
    last_time = t.groupby(gid).transform('last')
    durations = (last_time - first_time).fillna(0.0)
    # Mark stuck if run is long enough by frames or seconds and value is not NaN
    return (counts >= int(min_frames)) | (durations >= float(min_seconds))


def attach_confidence_flags(df: pd.DataFrame, conf_thresh: dict) -> pd.DataFrame:
    out = {}
    for key, th in conf_thresh.items():
        col = f"_conf_{key}"
        if col in df.columns:
            out[f"lowconf_{key}"] = (pd.to_numeric(df[col], errors="coerce") < th)
    return pd.DataFrame(out, index=df.index)


def main():
    ap = argparse.ArgumentParser(description="Re-OCR frames flagged as suspicious (_sus_*) in reactor CSV data")
    ap.add_argument("--cleaned", default="reactor_readings_cleaned.csv", help="Path to cleaned CSV")
    ap.add_argument("--raw", default=None, help="Optional path to raw CSV (no smoothing); not required for _sus_-driven flow")
    ap.add_argument("--video", required=True, help="Path to source video for re-OCR")
    ap.add_argument("--outdir", default=None, help="Directory for output files")
    ap.add_argument("--fps", type=float, default=None, help="Override video FPS (use exact framerate if known)")
    ap.add_argument("--force-cpu", action="store_true", default=False, help="Force EasyOCR CPU mode")

    ap.add_argument("--max-jump-temperature", type=float, default=DEFAULT_MAX_JUMP["temperature"]) 
    ap.add_argument("--max-jump-pressure", type=float, default=DEFAULT_MAX_JUMP["pressure"]) 
    ap.add_argument("--max-jump-fuel", type=float, default=DEFAULT_MAX_JUMP["fuel"]) 
    ap.add_argument("--max-jump-rod", type=float, default=DEFAULT_MAX_JUMP["rod_insertion"]) 

    ap.add_argument("--stuck-frames", type=int, default=15, help="Min frames to consider a stuck run")
    ap.add_argument("--stuck-seconds", type=float, default=1.0, help="Min seconds to consider a stuck run")
    ap.add_argument("--eps-temperature", type=float, default=DEFAULT_EPS["temperature"]) 
    ap.add_argument("--eps-pressure", type=float, default=DEFAULT_EPS["pressure"]) 
    ap.add_argument("--eps-fuel", type=float, default=DEFAULT_EPS["fuel"]) 
    ap.add_argument("--eps-rod", type=float, default=DEFAULT_EPS["rod_insertion"]) 

    # ROI controls (defaults mirror current dataGrabber configuration)
    ap.add_argument("--roi-temperature", default="2007,607,2105,584", help="x1,y1,x2,y2 for temperature")
    ap.add_argument("--roi-pressure", default="2001,564,2117,540", help="x1,y1,x2,y2 for pressure")
    ap.add_argument("--roi-fuel", default="1932,470,2004,445", help="x1,y1,x2,y2 for fuel")
    ap.add_argument("--roi-rod", default="1967,272,2033,245", help="x1,y1,x2,y2 for rod insertion")
    ap.add_argument("--roi-coolant", default="2098,182,2176,152", help="x1,y1,x2,y2 for coolant state (optional)")
    ap.add_argument("--roi-feedwater", default="1921,381,2039,353", help="x1,y1,x2,y2 for feedwater (optional)")

    # Re-OCR behavior
    ap.add_argument("--keys", default="temperature,pressure,fuel,rod_insertion,coolant,feedwater", help="Comma list of keys to re-OCR (must have _sus_* column present)")
    ap.add_argument("--strict-conf-temperature", type=float, default=0.30)
    ap.add_argument("--strict-conf-pressure", type=float, default=0.30)
    ap.add_argument("--strict-conf-fuel", type=float, default=0.50)
    ap.add_argument("--strict-conf-rod", type=float, default=0.40)
    ap.add_argument("--strict-conf-state", type=float, default=0.30, help="Strict confidence for state reads (coolant/feedwater)")
    ap.add_argument("--save-roi", action="store_true", help="Save ROI crops and best variants for debugging")
    ap.add_argument("--roi-debug-dir", default="roi_debug/resuscitated", help="Where to save ROI debug images")

    args = ap.parse_args()

    cleaned = load_csv(args.cleaned)
    raw = load_csv(args.raw) if args.raw else None

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.cleaned))
    os.makedirs(outdir, exist_ok=True)

    # Key metadata: determines how to OCR each column
    KEY_TYPES = {
        "temperature": "numeric",
        "pressure": "numeric",
        "fuel": "numeric",
        "rod_insertion": "numeric",
        "water_level": "numeric",
        "feedwater_flow": "numeric",
        "coolant": "state",
        "feedwater": "feed",
    }

    # Use only _sus_* markers from cleaned CSV to decide what to re-OCR
    report = cleaned.copy()
    ts = pd.to_numeric(report.get("timestamp", pd.Series(range(len(report)), dtype=float)), errors="coerce").fillna(0.0)
    sus_marker_cols = [c for c in report.columns if c.startswith("_sus_")]
    if not sus_marker_cols:
        print("ERROR: No _sus_* columns found. Re-run dataGrabber.py with DEBUG_TOGGLE=True to include _sus_ markers.")
        return
    # Build per-key suspect indices dynamically from available _sus_* columns
    key_to_marker = {c.replace("_sus_", ""): c for c in sus_marker_cols}
    keys_requested = [k.strip() for k in args.keys.split(',') if k.strip()]
    keys_available = [k for k in keys_requested if key_to_marker.get(k, None) in report.columns]
    if not keys_available:
        print("Nothing to do: none of the requested keys have _sus_* markers present in the CSV.")
        return
    per_key_indices = {k: list(report.index[report[key_to_marker[k]].astype(bool)]) for k in keys_available}

    # --- High-scrutiny re-OCR on suspect frames ---
    flagged_count = sum(len(v) for v in per_key_indices.values())
    if flagged_count == 0:
        print("No _sus_-flagged frames/keys; skipping re-OCR.")
        return
    # Print plan summary per key
    print("Re-OCR plan (from _sus_* markers):")
    for k in keys_available:
        print(f"- {k}: {len(per_key_indices.get(k, []))} items")
    print(f"Total flagged items: {flagged_count}")

    # Parse ROI strings
    def parse_roi(s: str) -> Tuple[int, int, int, int]:
        parts = [int(p.strip()) for p in s.split(',')]
        if len(parts) != 4:
            raise ValueError(f"ROI must have 4 integers: {s}")
        return parts[0], parts[1], parts[2], parts[3]

    # Only attach ROI entries for keys where an ROI was provided
    def _maybe_parse_roi(name: str, val: str):
        if val is None or str(val).strip() == "":
            return None
        try:
            return parse_roi(val)
        except Exception:
            print(f"Warning: skipping ROI for {name}, could not parse: {val}")
            return None

    rois: Dict[str, Tuple[int, int, int, int]] = {
        k: v for k, v in {
            "temperature": _maybe_parse_roi("temperature", args.roi_temperature),
            "pressure": _maybe_parse_roi("pressure", args.roi_pressure),
            "fuel": _maybe_parse_roi("fuel", args.roi_fuel),
            "rod_insertion": _maybe_parse_roi("rod_insertion", args.roi_rod),
            "coolant": _maybe_parse_roi("coolant", args.roi_coolant),
            "feedwater": _maybe_parse_roi("feedwater", args.roi_feedwater),
        }.items() if v is not None
    }

    keys = [k.strip() for k in args.keys.split(',') if k.strip()]
    strict_conf = {
        "temperature": args.strict_conf_temperature,
        "pressure": args.strict_conf_pressure,
        "fuel": args.strict_conf_fuel,
        "rod_insertion": args.strict_conf_rod,
    }

    # Initialize EasyOCR reader with optional CPU force
    _use_gpu = (torch.cuda.is_available() and not args.force_cpu)
    reader = easyocr.Reader(['en'], gpu=_use_gpu)
    print(f"EasyOCR (resuscitation) GPU={'ON' if _use_gpu else 'OFF'}")

    # Preprocessing variants for high-scrutiny OCR
    def super_preprocess(roi_img) -> List[np.ndarray]:
        # Pruned variant set for speed: CLAHE + Gaussian blur + (Otsu, Adaptive Gaussian),
        # plus a single MORPH_CLOSE. No mean threshold, no bitwise_not, minimal duplication.
        arr = []
        if roi_img is None or getattr(roi_img, 'size', 0) == 0:
            return arr
        try:
            gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        except Exception:
            gray = roi_img
        scales = [3.0]  # single scale suffices for most cases
        blurs = [(3, 3)]
        for s in scales:
            g = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gc = clahe.apply(g)
            for b in blurs:
                gb = cv2.GaussianBlur(gc, b, 0)
                v_gauss = cv2.adaptiveThreshold(gb, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7)
                _, v_otsu = cv2.threshold(gb, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                kern = np.ones((2, 2), np.uint8)
                arr.append(cv2.morphologyEx(v_gauss, cv2.MORPH_CLOSE, kern, iterations=1))
                arr.append(cv2.morphologyEx(v_otsu, cv2.MORPH_CLOSE, kern, iterations=1))
        # Deduplicate by shape + coarse checksum
        uniq = []
        seen = set()
        for a in arr:
            key = (a.shape, int(a.mean()))
            if key not in seen:
                uniq.append(a)
                seen.add(key)
        return uniq[:12]

    DIGIT_FIX_SAFE = str.maketrans({'O':'0','o':'0','S':'5','s':'5','I':'1','l':'1','B':'8'})
    import re as _re
    _PATTERNS = {
        'temperature': _re.compile(r"^(?P<VAL>\d{3,4}\.\d)K$"),
        'pressure': _re.compile(r"^(?P<VAL>\d{3,5}\.\d)KPA$"),
        'percent': _re.compile(r"^(?P<VAL>\d{1,3}(?:\.\d)?)%$"),
    }
    def _normalize_text(s: str) -> str:
        return s.replace(',', '.').replace(' ', '').upper()

    def _extract_value_for_key(key: str, text: str):
        t = _normalize_text(text)
        if key in ("fuel", "rod_insertion"):
            m = _PATTERNS['percent'].match(t)
            return float(m.group('VAL')) if m else None
        if key == 'temperature':
            m = _PATTERNS['temperature'].match(t)
            return float(m.group('VAL')) if m else None
        if key == 'pressure':
            m = _PATTERNS['pressure'].match(t)
            return float(m.group('VAL')) if m else None
        return None

    def _readtext_multi(images, allow=None):
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

    def ocr_numeric_strict(key: str, roi_img) -> Tuple[float, float, str]:
        allow = "0123456789.%KkPpAa"
        variants = super_preprocess(roi_img)
        buckets = {}
        best_conf_for = {}
        best_txt_for = {}
        best_seen = 0.0
        res_list = _readtext_multi(list(variants), allow)
        for res in res_list:
            for (_box, txt, conf) in res:
                txt = txt.translate(DIGIT_FIX_SAFE)
                val = _extract_value_for_key(key, txt)
                if val is None:
                    continue
                val = round(float(val), 1)
                raw = float(conf)
                score = raw + min(len(txt), 12) * 0.01
                buckets[val] = buckets.get(val, 0.0) + score
                if val not in best_conf_for or raw > best_conf_for[val]:
                    best_conf_for[val] = raw
                    best_txt_for[val] = txt
                if raw > best_seen:
                    best_seen = raw
            if best_seen > 0.85:
                break
        if not buckets:
            return None, 0.0, ""
        best_val = max(buckets.items(), key=lambda kv: kv[1])[0]
        best_conf = min(0.99, best_conf_for.get(best_val, 0.0))
        return best_val, best_conf, best_txt_for.get(best_val, "")

    def ocr_coolant_open(roi_img) -> Tuple[int, float, str]:
        # Returns (1 if OPEN detected else 0, confidence, raw_text)
        allow = "ABCDEFGHIJKLMNOPQRSTUVWXYZ/ "
        variants = super_preprocess(roi_img)
        best_c = 0.0
        best_t = ""
        res_list = _readtext_multi(list(variants), allow)
        for res in res_list:
            for (_box, txt, conf) in res:
                t = (txt or "").upper()
                if "OPEN" in t and float(conf) > best_c:
                    best_c, best_t = float(conf), t
            if best_c > 0.85:
                break
        return (1 if best_c > 0 else 0), min(0.99, best_c), best_t

    def ocr_feedwater_count_strict(roi_img) -> Tuple[int, float, str]:
        # Returns (count 0..2 or None->-1, conf, raw_text)
        import re as _re2
        pat = _re2.compile(r"([0-2])\s*/\s*2")
        allow = "0123456789/ "
        # In addition to full ROI, also try left-ratio crop
        variants = super_preprocess(roi_img)
        try:
            h, w = roi_img.shape[:2]
            left = roi_img[:, : max(1, int(w * 0.65))]
            variants += super_preprocess(left)
        except Exception:
            pass
        by_val = {0: 0.0, 1: 0.0, 2: 0.0}
        best_c = 0.0
        best_t = ""
        res_list = _readtext_multi(list(variants), allow)
        for res in res_list:
            for (_box, txt, conf) in res:
                t = "" if txt is None else str(txt)
                t2 = "".join(ch for ch in t if ch.isdigit() or ch == '/')
                m = pat.search(t2)
                if not m:
                    continue
                n = int(m.group(1))
                if 0 <= n <= 2:
                    c = float(conf)
                    by_val[n] += c
                    if c > best_c:
                        best_c, best_t = c, t2
            if best_c > 0.85:
                break
        picked = max(by_val.items(), key=lambda kv: kv[1])[0]
        if by_val[picked] == 0.0:
            return -1, 0.0, ""
        return picked, min(0.99, best_c), best_t

    # Prepare capture
    cap = cv2.VideoCapture(args.video)
    if not cap or not cap.isOpened():
        print(f"ERROR: cannot open video: {args.video}")
        return
    fps = float(args.fps) if args.fps else (cap.get(cv2.CAP_PROP_FPS) or 60.0)

    replace_log = []
    updated = cleaned.copy()

    # Ensure _raw_* columns can store string payloads safely (avoid float dtype warnings)
    for col in [c for c in updated.columns if c.startswith("_raw_")]:
        if not is_object_dtype(updated[col]):
            updated[col] = updated[col].astype("object")
    os.makedirs(args.roi_debug_dir, exist_ok=True) if args.save_roi else None

    # Safety banner for VRAM mode
    if _use_gpu:
        print("GPU mode enabled for EasyOCR (batched).")
    else:
        print("CPU fallback mode (expect slower runtime).")

    # Helper: one fast variant (CLAHE+Gaussian+Otsu) for batching
    def fast_variant(img):
        try:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception:
            g = img
        g = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        g = cv2.GaussianBlur(g, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        g = clahe.apply(g)
        _, v_otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kern = np.ones((2, 2), np.uint8)
        return cv2.morphologyEx(v_otsu, cv2.MORPH_CLOSE, kern, iterations=1)

    # Build task list grouped by frame index
    tasks = []  # (frame_idx, idx_row, key)
    for key in keys_available:
        if key not in rois:
            continue
        for idx in per_key_indices.get(key, []):
            t = float(ts.iloc[idx]) if idx < len(ts) else (idx / fps)
            frame_idx = int(round(t * fps))
            tasks.append((frame_idx, idx, key))
    tasks.sort(key=lambda x: x[0])

    processed_pairs = 0
    pbar = tqdm(total=len(tasks), desc="Resuscitating", ncols=80) if tasks else None
    i = 0
    while i < len(tasks):
        frame_idx = tasks[i][0]
        # Collect all tasks for this frame
        j = i
        group = []
        while j < len(tasks) and tasks[j][0] == frame_idx:
            group.append(tasks[j])
            j += 1
        i = j

        # Read frame once
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            if pbar:
                pbar.update(len(group))
            continue
        h, w = frame.shape[:2]

        # Per-allowlist batches
        batch_num, ctx_num = [], []
        batch_state, ctx_state = [], []
        batch_feed, ctx_feed = [], []

        # Prepare primary ROI variants for batch
        for (_f, idx_row, key) in group:
            if key not in rois:
                continue
            x1, y1, x2, y2 = rois[key]
            x_lo, x_hi = (x1, x2) if x1 <= x2 else (x2, x1)
            y_lo, y_hi = (y1, y2) if y1 <= y2 else (y2, y1)
            x_lo, y_lo = max(0, x_lo), max(0, y_lo)
            x_hi, y_hi = min(w, x_hi), min(h, y_hi)
            base_roi = frame[y_lo:y_hi, x_lo:x_hi]
            v = fast_variant(base_roi)
            ktype = KEY_TYPES.get(key, "numeric")
            if ktype == "numeric":
                batch_num.append(v)
                ctx_num.append((key, idx_row, frame_idx, base_roi))
            elif ktype == "state":
                batch_state.append(v)
                ctx_state.append((key, idx_row, frame_idx, base_roi))
            elif ktype == "feed":
                batch_feed.append(v)
                ctx_feed.append((key, idx_row, frame_idx, base_roi))

        # Process each batch type with appropriate allowlists
        def handle_numeric(ctx, res):
            # Aggregate best per context
            best_val, best_conf, best_txt = None, 0.0, ""
            for (_box, txt, conf) in res:
                txt2 = txt.translate(DIGIT_FIX_SAFE)
                val = _extract_value_for_key(ctx[0], txt2)
                if val is None:
                    continue
                raw = float(conf)
                if raw > best_conf:
                    best_conf, best_val, best_txt = raw, round(float(val), 1), txt
            return best_val, best_conf, best_txt

        def handle_state(res):
            best_c, best_t = 0.0, ""
            for (_box, txt, conf) in res:
                t = (txt or "").upper()
                if "OPEN" in t and float(conf) > best_c:
                    best_c, best_t = float(conf), t
            return (1 if best_c > 0 else 0), min(0.99, best_c), best_t

        def handle_feed(res):
            import re as _re2
            pat = _re2.compile(r"([0-2])\s*/\s*2")
            best_c, best_t, best_v = 0.0, "", -1
            for (_box, txt, conf) in res:
                t2 = "".join(ch for ch in (txt or "") if ch.isdigit() or ch == '/')
                m = pat.search(t2)
                if not m:
                    continue
                n = int(m.group(1))
                if 0 <= n <= 2 and float(conf) > best_c:
                    best_c, best_t, best_v = float(conf), t2, n
            return (best_v if best_v >= 0 else None), min(0.99, best_c), best_t

        # Run batched OCR per type
        if batch_num:
            res_list = _readtext_multi(batch_num, "0123456789.%KkPpAa")
            for ctx, res in zip(ctx_num, res_list):
                key, idx_row, fidx, base_roi = ctx
                val, conf, txt = handle_numeric(ctx, res)
                # Fallback to strict if weak (numeric keys only)
                if KEY_TYPES.get(key, "numeric") == "numeric" and (val is None or conf < strict_conf.get(key, 0.3)):
                    # Try padded strict search
                    pads = [(0,0,0,0), (0,0,8,0), (0,0,0,4), (4,2,8,2)]
                    best_val, best_conf, best_txt = None, 0.0, ""
                    x1, y1, x2, y2 = rois[key]
                    x_lo, x_hi = (x1, x2) if x1 <= x2 else (x2, x1)
                    y_lo, y_hi = (y1, y2) if y1 <= y2 else (y2, y1)
                    x_lo, y_lo = max(0, x_lo), max(0, y_lo)
                    x_hi, y_hi = min(w, x_hi), min(h, y_hi)
                    for (lp, tp, rp, bp) in pads:
                        xl = max(0, x_lo - lp); yl = max(0, y_lo - tp)
                        xr = min(w, x_hi + rp); yr = min(h, y_hi + bp)
                        vimg = frame[yl:yr, xl:xr]
                        vv, cc, tt = ocr_numeric_strict(key, vimg)
                        if cc > best_conf:
                            best_val, best_conf, best_txt = vv, cc, tt
                        processed_pairs += 1
                    val, conf, txt = best_val, best_conf, best_txt
                # Apply acceptance/update and always set conf/raw
                if KEY_TYPES.get(key, "numeric") == "numeric":
                    accept = (val is not None and conf >= strict_conf.get(key, 0.3))
                    if accept and key in RANGES:
                        old = updated.at[idx_row, key] if key in updated.columns else None
                        lo, hi = RANGES[key]
                        if val is not None and (lo <= val <= hi) and ((pd.isna(old)) or (str(old) != str(val))):
                            updated.at[idx_row, key] = float(val)
                            replace_log.append({"index": int(idx_row), "time": float(ts.iloc[idx_row]) if idx_row < len(ts) else (idx_row / fps),
                                                "key": key, "old": None if pd.isna(old) else old, "new": float(val),
                                                "conf": float(conf), "text": txt, "frame": int(frame_idx)})
                            if args.save_roi:
                                try:
                                    cv2.imwrite(os.path.join(args.roi_debug_dir, f"{idx_row:06d}_{key}.png"), base_roi)
                                except Exception:
                                    pass
                updated.at[idx_row, f"_raw_{key}"] = txt
                updated.at[idx_row, f"_conf_{key}"] = float(conf)
                sc = f"_sus_{key}"
                if sc in updated.columns and val is not None and conf >= (strict_conf.get(key, 0.3) if key in strict_conf else args.strict_conf_state):
                    updated.at[idx_row, sc] = 0

        if batch_state:
            res_list = _readtext_multi(batch_state, "ABCDEFGHIJKLMNOPQRSTUVWXYZ/ ")
            for ctx, res in zip(ctx_state, res_list):
                key, idx_row, fidx, base_roi = ctx
                val, conf, txt = handle_state(res)
                accept = (conf >= args.strict_conf_state)
                if accept:
                    old = updated.at[idx_row, key] if key in updated.columns else None
                    if (pd.isna(old)) or (str(old) != str(val)):
                        updated.at[idx_row, key] = int(val)
                        replace_log.append({"index": int(idx_row), "time": float(ts.iloc[idx_row]) if idx_row < len(ts) else (idx_row / fps),
                                            "key": key, "old": None if pd.isna(old) else old, "new": int(val),
                                            "conf": float(conf), "text": txt, "frame": int(frame_idx)})
                        if args.save_roi:
                            try:
                                cv2.imwrite(os.path.join(args.roi_debug_dir, f"{idx_row:06d}_{key}.png"), base_roi)
                            except Exception:
                                pass
                updated.at[idx_row, f"_raw_{key}"] = txt
                updated.at[idx_row, f"_conf_{key}"] = float(conf)
                sc = f"_sus_{key}"
                if sc in updated.columns and conf >= args.strict_conf_state:
                    updated.at[idx_row, sc] = 0

        if batch_feed:
            res_list = _readtext_multi(batch_feed, "0123456789/ ")
            for ctx, res in zip(ctx_feed, res_list):
                key, idx_row, fidx, base_roi = ctx
                val, conf, txt = handle_feed(res)
                accept = (val is not None and conf >= args.strict_conf_state)
                if accept:
                    old = updated.at[idx_row, key] if key in updated.columns else None
                    if (pd.isna(old)) or (str(old) != str(val)):
                        updated.at[idx_row, key] = int(val)
                        replace_log.append({"index": int(idx_row), "time": float(ts.iloc[idx_row]) if idx_row < len(ts) else (idx_row / fps),
                                            "key": key, "old": None if pd.isna(old) else old, "new": int(val),
                                            "conf": float(conf), "text": txt, "frame": int(frame_idx)})
                        if args.save_roi:
                            try:
                                cv2.imwrite(os.path.join(args.roi_debug_dir, f"{idx_row:06d}_{key}.png"), base_roi)
                            except Exception:
                                pass
                updated.at[idx_row, f"_raw_{key}"] = txt
                updated.at[idx_row, f"_conf_{key}"] = float(conf)
                sc = f"_sus_{key}"
                if sc in updated.columns and accept:
                    updated.at[idx_row, sc] = 0

        # Final flush per frame done; continue to next frame
        if pbar:
            pbar.update(len(group))

    if pbar:
        pbar.close()

    # Save updated CSV without adding smoothing/interpolation
    # Final cleanup/interpolation and write directly to the cleaned CSV path
    for col in [c for c, t in KEY_TYPES.items() if t == "numeric"]:
        if col in updated.columns:
            updated[col] = pd.to_numeric(updated[col], errors="coerce")
            updated[col] = updated[col].interpolate(method="linear", limit_direction="both").round(2)
    cleaned_out = os.path.abspath(args.cleaned)
    updated.to_csv(cleaned_out, index=False)
    print(f"Resuscitated CSV written to cleaned output: {cleaned_out}")
    print(f"Processed {processed_pairs} flagged key/frame pairs.")

    # Save replacement log
    if replace_log:
        repl_df = pd.DataFrame(replace_log)
        repl_path = os.path.join(outdir, "resuscitation_replacements.csv")
        repl_df.to_csv(repl_path, index=False)
        print(f"Replacements written: {repl_path} ({len(replace_log)} updates)")
    else:
        print("No replacements made (nothing met strict acceptance criteria).")


if __name__ == "__main__":
    main()
