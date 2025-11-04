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
video_path = os.path.join(base_dir, "anroShiftNov3rd.mkv") # Ensure that you change the name of this file

# Create a 10s test clip (60s–70s) (completely optional, just used for testing. if youre confident in the results, you can remove these lines)
clip = mp.VideoFileClip(video_path).subclipped(60, 70)
clip.write_videofile(os.path.join(base_dir, "test_clip.mp4"), codec="libx264")
video_path = os.path.join(base_dir, "test_clip.mp4")

output_csv = "reactor_readings_cleaned.csv"
error_log = "ocr_errors.log"

# Debug output for ROI snapshots (will make copies of the RIO snapshots into its own folder. Set this to false if not in use, as it will make a lot of images)
DEBUG_SAVE_ROI = True
ROI_DEBUG_RATE = 50  # save every Nth frame
ROI_DEBUG_DIR = "roi_debug"
if DEBUG_SAVE_ROI:
    os.makedirs(ROI_DEBUG_DIR, exist_ok=True)

# --- Regions of Interest (ROIs) --- (ensure that you change these values per video, no two videos will have the same coordinates)
regions = {
    "coolant": (2237, 151, 2329, 185),
    "rod_insertion": (2092, 255, 2165, 287),
    "feedwater": (2039, 372, 2182, 406),
    "fuel": (2057, 467, 2149, 497),
    "pressure": (2149, 568, 2282, 595),
    "temperature": (2164, 617, 2269, 642),
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
    s = s.replace('KPA', 'kPa').replace('Kpa', 'kPa').replace('k pa', 'kPa')
    s = s.replace(' ', '')
    return s

_PATTERNS = {
    'temperature': re.compile(r"(?P<val>\d{3,4}\.\d)K", re.IGNORECASE),
    'pressure': re.compile(r"(?P<val>\d{3,5}\.\d)kPa", re.IGNORECASE),
    'percent': re.compile(r"(?P<val>\d{1,3}(?:\.\d)?)%"),
}

def _extract_value_for_key(key: str, text: str):
    t = _normalize_text(text)
    if key in ("fuel", "rod_insertion"):
        m = _PATTERNS['percent'].search(t)
        if m:
            return float(m.group('val'))
        m2 = re.search(r"\d{1,3}(?:\.\d)?", t)
        return float(m2.group()) if m2 else None
    if key == 'temperature':
        m = _PATTERNS['temperature'].search(t)
        if m:
            return float(m.group('val'))
        m2 = re.search(r"\d{3,4}\.\d", t)
        return float(m2.group()) if m2 else None
    if key == 'pressure':
        m = _PATTERNS['pressure'].search(t)
        if m:
            return float(m.group('val'))
        m2 = re.search(r"\d{3,5}\.\d", t)
        return float(m2.group()) if m2 else None
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

def ocr_numeric_for_key(key: str, variants):
    allow = "0123456789.%KkPpAa"
    best_val, best_score, best_conf, best_txt = None, -1.0, 0.0, ""
    for img in variants:
        results = reader.readtext(img, detail=1, paragraph=False, allowlist=allow)
        for (_box, txt, conf) in results:
            txt = txt.translate(DIGIT_FIX_SAFE)
            val = _extract_value_for_key(key, txt)
            if val is None:
                continue
            score = float(conf) + _unit_bonus(key, txt) + min(len(txt), 10) * 0.01
            if score > best_score:
                best_val, best_score, best_conf, best_txt = val, score, float(conf), txt
    return best_val, best_conf, best_txt

# --- State OCR ---
def ocr_state_variants_simple(variants, keywords):
    best_t, best_conf = "", 0.0
    for img in variants:
        t, c = read_with_easyocr_simple(img, allow="ABCDEFGHIJKLMNOPQRSTUVWXYZ/ ")
        t = t.upper().strip()
        if c > best_conf:
            best_t, best_conf = t, c
    return any(k in best_t for k in keywords), best_conf, best_t

_FEED_RE = re.compile(r"([0-2])\s*/\s*2")

def parse_feedwater_count(text: str):
    if not text:
        return None
    t = text.upper().replace(' ', '')
    m = _FEED_RE.search(t)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    if 'ACTIVE' in t:
        return 1
    if 'INACTIVE' in t or 'OFF' in t:
        return 0
    return None

def ocr_feedwater_count_from_variants(roi, variants):
    # Crop to where the ratio is (left side) to avoid letters in ACTIVE
    h, w = roi.shape[:2]
    ratio_roi = roi[:, : max(1, int(w * 0.55))]
    # Build variants from the ratio crop
    ratio_variants = preprocess_variants_simple(ratio_roi)

    by_val = {0: 0.0, 1: 0.0, 2: 0.0}
    best_txt, best_conf = "", 0.0
    for img in ratio_variants:
        res = reader.readtext(img, detail=1, paragraph=False, allowlist="0123456789/")
        for (_box, txt, conf) in res:
            t = txt.upper().replace(" ", "")
            m = _FEED_RE.search(t)
            if not m:
                continue
            try:
                num = int(m.group(1))
            except ValueError:
                continue
            if 0 <= num <= 2:
                by_val[num] += float(conf)
                if float(conf) > best_conf:
                    best_conf, best_txt = float(conf), t

    picked = max(by_val.items(), key=lambda kv: kv[1])[0]
    if by_val[picked] == 0.0:
        return None, 0.0, ""
    return picked, best_conf, best_txt

# --- Stability buffers ---
STATE_WINDOW = 5  # smoothing window for states
state_buffers = {"coolant": [], "feedwater": []}

def smooth_state(key: str, value: float):
    if key not in state_buffers:
        state_buffers[key] = []
    state_buffers[key].append(value)
    state_buffers[key] = state_buffers[key][-STATE_WINDOW:]
    avg = sum(state_buffers[key]) / len(state_buffers[key])
    return round(avg, 2)


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
            # Apply per-field ROI padding safely within frame bounds
            h, w = frame.shape[:2]
            lpad, tpad, rpad, bpad = ROI_PAD.get(key, (0, 0, 0, 0))
            x1p = max(0, x1 - lpad)
            y1p = max(0, y1 - tpad)
            x2p = min(w, x2 + rpad)
            y2p = min(h, y2 + bpad)
            roi = frame[y1p:y2p, x1p:x2p]

            variants = preprocess_variants_simple(roi)

            # Optional debug: save raw and some processed variants
            if DEBUG_SAVE_ROI and (frame_idx % ROI_DEBUG_RATE == 0):
                cv2.imwrite(os.path.join(ROI_DEBUG_DIR, f"{frame_idx:04d}_{key}_raw.png"), roi)
                for vi, vimg in enumerate(variants[:3]):
                    cv2.imwrite(os.path.join(ROI_DEBUG_DIR, f"{frame_idx:04d}_{key}_proc{vi}.png"), vimg)

            conf = 0.0
            v = None

            if key == "coolant":
                state, conf, rawt = ocr_state_variants_simple(variants, ["OPEN"]) 
                v = smooth_state("coolant", 1 if state else 0)
                row["_raw_" + key] = rawt

            elif key == "feedwater":
                count, conf, rawt = ocr_feedwater_count_from_variants(roi, variants)
                if count is None:
                    count = state_buffers["feedwater"][-1] if state_buffers["feedwater"] else 0
                v = smooth_multistate("feedwater", int(max(0, min(2, count))))
                row["_raw_" + key] = rawt

            else:
                v, conf, rawn = ocr_numeric_for_key(key, variants)
                v = clamp_or_nan(v, key)
                row["_raw_" + key] = rawn

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

                if v is None:
                    v = last_ok[key]
                last_ok[key] = v

            row[key] = v
            row["_conf_" + key] = conf

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

df.interpolate(method="linear", limit=6, limit_direction="both", inplace=True)
df.ffill(inplace=True)
df.bfill(inplace=True)

# --- Save Clean Data ---
output_path = os.path.join(os.path.dirname(video_path), output_csv)
df.to_csv(output_path, index=False)

print(f"\nCleaned data saved to {output_path}")
