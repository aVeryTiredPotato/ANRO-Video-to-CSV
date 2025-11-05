import os
import re
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import easyocr
import torch
import moviepy as mp

# --- GPU Initialization --- (delete this line if you want to use your cpu for some reason)
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# --- Paths ---
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
video_path = os.path.join(base_dir, "2025-11-04 17-15-01.mkv") # Ensure that you change the name of this file

output_csv = "reactor_readings_cleaned.csv"
error_log = "ocr_errors.log"

# Debug output for ROI snapshots (set False to disable extra images and conf columns)
DEBUG_TOGGLE = False
ROI_DEBUG_RATE = 50  # save every Nth frame
ROI_DEBUG_DIR = "roi_debug"
if DEBUG_TOGGLE:
    os.makedirs(ROI_DEBUG_DIR, exist_ok=True)
    # Create a small test clip, to avoid checking ENTIRE video every time
    clip = mp.VideoFileClip(video_path).subclipped(60, 70)
    clip.write_videofile(os.path.join(base_dir, "test_clip.mp4"), codec="libx264")
    video_path = os.path.join(base_dir, "test_clip.mp4")

# --- Regions of Interest (ROIs) --- (ensure that you change these values per video, no two videos will have the same coordinates)
regions = {
    "coolant": (2098, 182, 2176, 152),
    "rod_insertion": (1967, 272, 2033, 245),
    "feedwater": (1921, 381, 2039, 353),
    "fuel": (1932, 470, 2004, 445),
    "pressure": (2001, 564, 2117, 540),
    "temperature": (2007, 607, 2105, 584),
}

# Optional ROI paddings (l, t, r, b) to reduce tight crops
ROI_PAD = {
    # Fuel tends to clip the last digit when animating --> add right padding
    "fuel": (0, 0, 10, 0),
    # Rod insertion can lose bottom strokes --> add bottom padding
    "rod_insertion": (0, 0, 0, 4),
}

# --- Sanity Ranges ---
RANGES = {
    "temperature": (300, 4000),
    "pressure": (500, 10000),
    "fuel": (0, 100),
    "rod_insertion": (0, 100),
}

# --- Confidence + constraints ---
# Minimum confidence to accept a fresh value per signal
CONF_THRESH = {
    "temperature": 0.20,
    "pressure": 0.20,
    "fuel": 0.40,
    "rod_insertion": 0.25,
}
# Max allowed change per frame for constrained signals
MAX_DELTA_PER_FRAME = {"fuel": 0.1}
# EMA for confidence readout smoothing (display only)
CONF_EMA_ALPHA = 0.4

# --- Helper Functions ---
def clamp_or_nan(v, key):
    if v is None:
        return None
    lo, hi = RANGES.get(key, (-float("inf"), float("inf")))
    return v if lo <= v <= hi else None

def prefer_near(prev, cand, max_jump):
    if cand is None or prev is None:
        return cand
    return cand if abs(cand - prev) <= max_jump else prev

# --- OCR preprocessing ---
def preprocess_variants_simple(roi):
    roi = cv2.copyMakeBorder(roi, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g = clahe.apply(g)

    b1 = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7)
    b2 = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35, 5)
    b3 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kern = np.ones((2, 2), np.uint8)
    variants = [cv2.morphologyEx(v, cv2.MORPH_CLOSE, kern, iterations=1) for v in (b1, b2, b3)]
    variants.append(cv2.bitwise_not(variants[0]))  # non-inverted fallback
    return variants

def read_with_easyocr_simple(img, allow):
    res = reader.readtext(img, detail=1, paragraph=False, allowlist=allow)
    if not res:
        return "", 0.0
    best = max(res, key=lambda x: x[2])
    return best[1], float(best[2])

DIGIT_FIX_SAFE = str.maketrans({'O':'0','o':'0','S':'5','s':'5','I':'1','l':'1','B':'8'})

# --- Field-aware numeric OCR ---
def _normalize_text(s: str) -> str:
    s = s.replace(',', '.')
    s = s.replace(' ', '')
    s = s.upper()
    return s

_PATTERNS = {
    # Anchor full token: exactly NNNN.NK
    'temperature': re.compile(r"^(?P<VAL>\d{3,4}\.\d)K$"),
    # Exactly NNNN.NKPA
    'pressure': re.compile(r"^(?P<VAL>\d{3,5}\.\d)KPA$"),
    # Exactly NN(.N)%
    'percent': re.compile(r"^(?P<VAL>\d{1,3}(?:\.\d)?)%$"),
}

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

def _unit_bonus(key: str, text: str) -> float:
    t = _normalize_text(text).upper()
    if key == 'temperature' and 'K' in t:
        return 0.15
    if key == 'pressure' and 'KPA' in t.upper():
        return 0.15
    if key in ('fuel','rod_insertion') and '%' in t:
        return 0.10
    return 0.0

def _quantize_value(key: str, v: float) -> float:
    if v is None:
        return None
    if key in ("temperature", "pressure", "fuel", "rod_insertion"):
        return round(v, 1)
    return v

def ocr_numeric_for_key(key: str, variants):
    # Aggregate all detections across variants, bucketed by quantized value
    allow = "0123456789.%KkPpAa"
    buckets = {}  # val_bucket -> sum_score
    conf_sum = {}  # val_bucket -> combined confidence (noisy-or)
    best_txt_for = {}  # representative text per bucket
    best_conf_for = {}  # val_bucket -> best single conf

    for img in variants:
        results = reader.readtext(img, detail=1, paragraph=False, allowlist=allow)
        for (_box, txt, conf) in results:
            txt = txt.translate(DIGIT_FIX_SAFE)
            tnorm = _normalize_text(txt)
            # Reject mixed junk: disallow letters not in the expected unit set
            if key in ("fuel", "rod_insertion"):
                allowed_letters = {"%"}
            elif key == "temperature":
                allowed_letters = {"K"}
            elif key == "pressure":
                allowed_letters = {"K", "P", "A"}
            else:
                allowed_letters = set()
            if any(ch.isalpha() and ch not in allowed_letters for ch in tnorm):
                continue
            val = _extract_value_for_key(key, txt)
            if val is None:
                continue
            vb = _quantize_value(key, val)
            raw = float(conf)
            score = raw + _unit_bonus(key, txt) + min(len(txt), 10) * 0.01
            buckets[vb] = buckets.get(vb, 0.0) + score
            prev = conf_sum.get(vb, 0.0)
            conf_sum[vb] = 1.0 - (1.0 - prev) * (1.0 - raw)
            if vb not in best_conf_for or raw > best_conf_for[vb]:
                best_conf_for[vb] = raw
                best_txt_for[vb] = txt

    if not buckets:
        return None, 0.0, ""

    # Pick the bucket with the highest aggregate score
    best_val = max(buckets.items(), key=lambda kv: kv[1])[0]
    # Report confidence as the best single detection for that bucket (not summed)
    best_conf = min(0.99, best_conf_for.get(best_val, 0.0))
    best_txt = best_txt_for.get(best_val, "")
    return best_val, best_conf, best_txt

# --- State OCR ---
def ocr_state_variants_simple(variants, keywords):
    # Aggregate confidence across variants where any keyword appears
    best_t, best_c = "", 0.0
    agg = 0.0
    for img in variants:
        t, c = read_with_easyocr_simple(img, allow="ABCDEFGHIJKLMNOPQRSTUVWXYZ/ ")
        t_up = t.upper().strip()
        if any(k in t_up for k in keywords):
            agg += float(c)
        if float(c) > best_c:
            best_t, best_c = t_up, float(c)
    # Report confidence as best single detection (avoid inflated sums)
    found = any(k in best_t for k in keywords)
    return found, min(0.99, best_c), best_t
_FEED_RE = re.compile(r"([0-2])\s*/\s*2")
def parse_feedwater_count(text: str):
    if not text:
        return None
    # Normalize and keep only digits and '/'
    t = re.sub(r"[^0-9/]", "", text.upper())
    if not t:
        return None
    m = _FEED_RE.search(t)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None



def ocr_feedwater_count_from_variants(roi, variants):
    # Crop to where the ratio is (left side) to avoid letters in ACTIVE
    h, w = roi.shape[:2]
    ratio_roi = roi[:, : max(1, int(w * 0.60))]  # slightly wider to avoid cutting digits
    # Build variants from the ratio crop
    ratio_variants = preprocess_variants_simple(ratio_roi)
    # Also consider the full-ROI variants as a fallback
    all_variants = list(ratio_variants) + list(variants)
    by_val = {0: 0.0, 1: 0.0, 2: 0.0}  # aggregate score (sum)
    conf_comb = {0: 0.0, 1: 0.0, 2: 0.0}  # combined confidence via noisy-or
    best_txt, best_conf = "", 0.0
    for img in all_variants:
        res = reader.readtext(img, detail=1, paragraph=False, allowlist="0123456789/ ")
        for (_box, txt, conf) in res:
            # Keep only digits and '/'; tolerate stray characters
            t = re.sub(r"[^0-9/]", "", txt)
            if not t:
                continue
            m = _FEED_RE.search(t)
            if not m:
                continue
            try:
                num = int(m.group(1))
            except ValueError:
                continue
            if 0 <= num <= 2:
                raw = float(conf)
                by_val[num] += raw
                prev = conf_comb.get(num, 0.0)
                conf_comb[num] = 1.0 - (1.0 - prev) * (1.0 - raw)
                if raw > best_conf:
                    best_conf, best_txt = float(conf), t

    picked = max(by_val.items(), key=lambda kv: kv[1])[0]
    if by_val[picked] == 0.0:
        return None, 0.0, ""
    # Return a normalized raw display with ACTIVE appended
    norm_raw = f"{picked}/2 ACTIVE"
    return picked, min(0.99, best_conf), norm_raw

# --- Stability buffers ---
STATE_WINDOW = 5  # smoothing window for states
state_buffers = {"coolant": [], "feedwater": []}

# Pending change confirmation (e.g., require 2 consecutive frames)
pending_updates = {"fuel": {"cand": None, "count": 0}}

# Confidence EMA memory (display only)
conf_ema = {}

def smooth_state(key: str, value: float):
    # Binary majority smoothing (returns 0/1)
    if key not in state_buffers:
        state_buffers[key] = []
    state_buffers[key].append(1 if value else 0)
    state_buffers[key] = state_buffers[key][-STATE_WINDOW:]
    needed = (STATE_WINDOW // 2) + 1
    return 1 if sum(state_buffers[key]) >= needed else 0


def smooth_multistate(key, value):
    state_buffers[key].append(value)
    state_buffers[key] = state_buffers[key][-STATE_WINDOW:]
    vals = state_buffers[key]
    counts = {v: vals.count(v) for v in set(vals)}
    max_count = max(counts.values())
    candidates = [v for v, c in counts.items() if c == max_count]
    for v in reversed(vals):
        if v in candidates:
            return v
    return value

# --- OCR Loop ---
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

data_rows, errors = [], []
last_ok = {k: None for k in ["temperature", "pressure", "fuel", "rod_insertion"]}

print(f"Processing {total_frames} frames from {video_path}...")

for frame_idx in tqdm(range(total_frames), desc="Extracting OCR Data", ncols=80):
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = frame_idx / fps
    row = {"timestamp": timestamp}

    for key, (x1, y1, x2, y2) in regions.items():
        try:
            # Normalize coordinates (handle reversed x1/x2 or y1/y2) and apply padding
            h, w = frame.shape[:2]
            x_lo, x_hi = (x1, x2) if x1 <= x2 else (x2, x1)
            y_lo, y_hi = (y1, y2) if y1 <= y2 else (y2, y1)
            lpad, tpad, rpad, bpad = ROI_PAD.get(key, (0, 0, 0, 0))
            x1p = max(0, x_lo - lpad)
            y1p = max(0, y_lo - tpad)
            x2p = min(w, x_hi + rpad)
            y2p = min(h, y_hi + bpad)
            if x2p <= x1p or y2p <= y1p:
                raise ValueError(f"Invalid ROI for {key}: {(x1,y1,x2,y2)} -> {(x1p,y1p,x2p,y2p)}")
            roi = frame[y1p:y2p, x1p:x2p]

            variants = preprocess_variants_simple(roi)

            # Optional debug: save raw and some processed variants (guard empty crops)
            if DEBUG_TOGGLE and (frame_idx % ROI_DEBUG_RATE == 0):
                try:
                    if roi is not None and getattr(roi, 'size', 0) > 0:
                        cv2.imwrite(os.path.join(ROI_DEBUG_DIR, f"{frame_idx:04d}_{key}_raw.png"), roi)
                        for vi, vimg in enumerate(variants[:3]):
                            if vimg is not None and getattr(vimg, 'size', 0) > 0:
                                cv2.imwrite(os.path.join(ROI_DEBUG_DIR, f"{frame_idx:04d}_{key}_proc{vi}.png"), vimg)
                except Exception as dbg_e:
                    errors.append(f"Frame {frame_idx} | {key}: debug save failed: {dbg_e}")

            conf = 0.0
            v = None

            if key == "coolant":
                state, conf, rawt = ocr_state_variants_simple(variants, ["OPEN"]) 
                v = smooth_state("coolant", 1 if state else 0)
                row["_raw_" + key] = rawt
                row["_sus_" + key] = 0

            elif key == "feedwater":
                count, conf, rawt = ocr_feedwater_count_from_variants(roi, variants)
                if count is None:
                    count = state_buffers["feedwater"][-1] if state_buffers["feedwater"] else 0
                v = smooth_multistate("feedwater", int(max(0, min(2, count))))
                # Normalize raw to "<n>/2 ACTIVE" based on the final (possibly smoothed) value
                row["_raw_" + key] = f"{v}/2 ACTIVE"
                row["_sus_" + key] = 1 if conf < 0.20 else 0

            else:
                v, conf, rawn = ocr_numeric_for_key(key, variants)
                v = clamp_or_nan(v, key)
                row["_raw_" + key] = rawn
                suspect = 0

                # Gate acceptance on confidence
                th = CONF_THRESH.get(key, 0.0)
                if conf < th:
                    suspect = 1
                    v = None

                if key == "pressure" and v is not None:
                    if v < 1000:
                        v *= 10
                    v = round(v, 1)
                if key == "temperature" and v is not None:
                    v = round(v, 1)
                if key in ["fuel", "rod_insertion"] and v is not None:
                    v = round(v, 1)

                # Limit rod insertion jumps
                if key == "rod_insertion" and last_ok[key] is not None and v is not None:
                    if abs(v - last_ok[key]) > 1.5:
                        v = last_ok[key]

                # Temperature/pressure jump limit
                if key in ("temperature", "pressure"):
                    v = prefer_near(last_ok[key], v, max_jump=120)

                # Fuel-specific strict gating: require small delta or two-frame confirmation
                if key == "fuel":
                    prev = last_ok[key]
                    if v is not None and prev is not None and abs(v - prev) > MAX_DELTA_PER_FRAME["fuel"]:
                        suspect = 1
                        pu = pending_updates["fuel"]
                        if pu["cand"] is not None and abs(v - pu["cand"]) <= 0.05:
                            pu["count"] += 1
                        else:
                            pu["cand"], pu["count"] = v, 1
                        # Accept only after 2 consecutive frames agreeing
                        if pu["count"] >= 2:
                            prev = v
                            pending_updates["fuel"] = {"cand": None, "count": 0}
                        v = prev
                    else:
                        pending_updates["fuel"] = {"cand": None, "count": 0}

                if v is None:
                    v = last_ok[key]
                last_ok[key] = v
                row["_sus_" + key] = suspect

            row[key] = v
            if DEBUG_TOGGLE:
                # Smooth confidence visibility to reduce frame-to-frame whiplash
                prev_c = conf_ema.get(key, conf)
                sm_c = (1 - CONF_EMA_ALPHA) * prev_c + CONF_EMA_ALPHA * conf
                conf_ema[key] = sm_c
                row["_conf_" + key] = float(sm_c)
        except Exception as e:
            errors.append(f"Frame {frame_idx} | {key}: {e}")
            row[key] = last_ok.get(key, None)

    data_rows.append(row)

cap.release()

# --- Log errors ---
if errors:
    with open(error_log, "w", encoding="utf-8") as f:
        f.write("\n".join(errors))
    print(f"\n{len(errors)} OCR errors logged to {error_log}")

# --- Data Cleanup ---
df = pd.DataFrame(data_rows)
for col in ["temperature", "pressure", "fuel", "rod_insertion"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

for key, (lo, hi) in RANGES.items():
    if key in df:
        df.loc[(df[key] < lo) | (df[key] > hi), key] = np.nan

for col in ["temperature", "pressure"]:
    if col in df.columns:
        df[col] = df[col].rolling(5, min_periods=1, center=True).median()
for col in ["fuel", "rod_insertion"]:
    if col in df.columns:
        df[col] = df[col].rolling(3, min_periods=1, center=True).median()
if not DEBUG_TOGGLE:
    df = df[[c for c in df.columns if not c.startswith(("_raw_", "_conf_", "_sus_"))]]

# Tone down interpolation to avoid propagating errors too far
# - Do not interpolate fuel; allow short forward/back fills only
if "fuel" in df.columns:
    df["fuel"] = df["fuel"].ffill(limit=2).bfill(limit=1)

# - Temperature/pressure: allow short linear interpolation for gaps up to 2
for col in ["temperature", "pressure"]:
    if col in df.columns:
        df[col] = df[col].interpolate(method="linear", limit=2, limit_direction="forward").ffill(limit=2).bfill(limit=1)

# - Rod insertion: keep small smoothing already applied; avoid bridging long NaN runs
if "rod_insertion" in df.columns:
    df["rod_insertion"] = df["rod_insertion"].ffill(limit=2).bfill(limit=1)

# --- Save Clean Data ---
output_path = os.path.join(os.path.dirname(video_path), output_csv)
df.to_csv(output_path, index=False)

print(f"\nCleaned data saved to {output_path}")
