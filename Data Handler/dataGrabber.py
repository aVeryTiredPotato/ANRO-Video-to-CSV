import os
import re
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import easyocr
import torch
import moviepy as mp
import sys
import subprocess

# --- GPU Initialization ---
# Allow forcing CPU via CLI flag --force-cpu to avoid CUDA hangs
FORCE_CPU = any(arg.lower() == "--force-cpu" for arg in sys.argv)
_use_gpu = (torch.cuda.is_available() and not FORCE_CPU)
reader = easyocr.Reader(['en'], gpu=_use_gpu)
print(f"EasyOCR initialized. GPU={'ON' if _use_gpu else 'OFF'}")
try:
    # Enable cuDNN autotune to pick optimal algorithms for current shapes
    import torch.backends.cudnn as cudnn
    if _use_gpu and torch.backends.cudnn.is_available():
        cudnn.benchmark = True
        print("cuDNN benchmark autotune: ON")
    else:
        print("cuDNN benchmark autotune: OFF")
except Exception as _cudnn_e:
    print(f"cuDNN setup skipped: {_cudnn_e}")
if _use_gpu:
    reader.detector.to(torch.device('cuda'))
    reader.recognizer.to(torch.device('cuda'))
    print("EasyOCR detector and recognizer moved to GPU.")
    print("Device:", next(reader.recognizer.parameters()).device)

# --- Paths ---
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
video_path = os.path.join(base_dir, "2025-11-04 17-16-32.mkv") # Ensure that you change the name of this file

output_csv = "reactor_readings_cleaned.csv"
error_log = "ocr_errors.log"

# --- Sampling Control ---
# Set how many frames per second to OCR (e.g., 60, 30, 15). None means process all frames.
SAMPLE_FPS = 15

# Debug output for ROI snapshots (set False to disable extra images and conf columns)
DEBUG_TOGGLE = True
ROI_DEBUG_RATE = 50  # save every Nth frame
ROI_DEBUG_DIR = "roi_debug"
if DEBUG_TOGGLE:
    os.makedirs(ROI_DEBUG_DIR, exist_ok=True)
    # Create a small test clip, to avoid checking ENTIRE video every time
    # Disable audio to reduce I/O and encoding time
    clip = mp.VideoFileClip(video_path, audio=False).subclipped(0, 90)
    clip.write_videofile(os.path.join(base_dir, "test_clip.mp4"), codec="libx264", audio=False, logger='bar')
    clip.close()
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
    # Pruned variants: CLAHE + Gaussian + (Otsu, Adaptive Gaussian)
    roi = cv2.copyMakeBorder(roi, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g = clahe.apply(g)

    v_gauss = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7)
    v_otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kern = np.ones((2, 2), np.uint8)
    variants = [
        cv2.morphologyEx(v_gauss, cv2.MORPH_CLOSE, kern, iterations=1),
        cv2.morphologyEx(v_otsu, cv2.MORPH_CLOSE, kern, iterations=1),
    ]
    return variants

def fast_variant(roi):
    # Single fast variant for batching: CLAHE + Gaussian + Otsu + CLOSE
    roi = cv2.copyMakeBorder(roi, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g = clahe.apply(g)
    v_otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kern = np.ones((2, 2), np.uint8)
    return cv2.morphologyEx(v_otsu, cv2.MORPH_CLOSE, kern, iterations=1)

def read_with_easyocr_simple(img, allow):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    res = reader.readtext(img, detail=1, paragraph=False, allowlist=allow)
    if not res:
        return "", 0.0
    best = max(res, key=lambda x: x[2])
    return best[1], float(best[2])

DIGIT_FIX_SAFE = str.maketrans({'O':'0','o':'0','S':'5','s':'5','I':'1','l':'1','B':'8'})

# --- Batched OCR helper ---
def readtext_multi(images, allow=None):
    try:
        if hasattr(reader, 'readtext_batched'):
            return reader.readtext_batched(list(images), detail=1, paragraph=False, allowlist=allow)
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


if _use_gpu:
    print("CUDA available:", torch.cuda.is_available())
    print("Device name:", torch.cuda.get_device_name(0))

def ocr_numeric_for_key(key: str, variants):
    # Aggregate all detections across variants, bucketed by quantized value
    allow = "0123456789.%KkPpAa"
    buckets = {}  # val_bucket -> sum_score
    conf_sum = {}  # val_bucket -> combined confidence (noisy-or)
    best_txt_for = {}  # representative text per bucket
    best_conf_for = {}  # val_bucket -> best single conf

    # Batch across variants in a single EasyOCR call to reduce overhead
    best_seen_conf = 0.0
    for img in list(variants):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
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
            if raw > best_seen_conf:
                best_seen_conf = raw
        # Early-exit if we already have a strong hit
        if best_seen_conf > 0.85:
            break

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
    # Robust OPEN/CLOSED detection across variants with basic normalization
    def _norm(s: str) -> str:
        t = (s or "").upper().replace('0', 'O').replace('1', 'I')
        t = t.replace('CIOSED', 'CLOSED').replace('CLO5ED', 'CLOSED')
        t = t.replace('OPFN', 'OPEN').replace('0PEN', 'OPEN')
        return t.strip()

    def looks_open(tu: str) -> bool:
        if 'OPEN' in tu:
            return True
        # Handle partials like 'PEN', 'QPEN'
        if tu.startswith('PEN') or ' QPEN' in (' ' + tu) or ' PEN' in (' ' + tu):
            return True
        return False

    best_t, best_c = "", 0.0
    open_sum, closed_sum = 0.0, 0.0
    # Batch across variants
    allow = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/ "
    for img in list(variants):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        results = reader.readtext(img, detail=1, paragraph=False, allowlist=allow)
        for (_box, t, c) in results:
            tu = _norm(t)
            cu = float(c)
            if looks_open(tu):
                open_sum += cu
            if 'CLOSED' in tu:
                closed_sum += cu
            if cu > best_c:
                best_t, best_c = tu, cu
        if best_c > 0.85:
            break
    # Prefer the class with more aggregated evidence; tie favors OPEN
    found_open = (open_sum >= closed_sum) and (open_sum > 0.0)
    return found_open, min(0.99, best_c), best_t
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
    allow = "0123456789/ "
    for img in list(all_variants):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        res = reader.readtext(img, detail=1, paragraph=False, allowlist=allow)
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
        if best_conf > 0.85:
            break

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

# Pending large-jump acceptance to avoid lock-in
MAX_JUMP_PER_FRAME = {"temperature": 120.0, "pressure": 120.0, "rod_insertion": 2.0}
JUMP_STABILITY_THRESH = {"temperature": 30.0, "pressure": 50.0, "rod_insertion": 1.0}  # how close consecutive jump candidates must be
pending_jumps = {"temperature": {"cand": None, "count": 0},
                 "pressure": {"cand": None, "count": 0},
                 "rod_insertion": {"cand": None, "count": 0}}

# Raw CSV output (no smoothing/interpolation)
RAW_OUTPUT_CSV = "reactor_readings_raw.csv"

# Confidence EMA memory (display only)
conf_ema = {}

def smooth_state(key: str, value: float):
    # Binary majority smoothing (returns 0/1)
    if key not in state_buffers:
        state_buffers[key] = []
    state_buffers[key].append(1 if value else 0)
    state_buffers[key] = state_buffers[key][-STATE_WINDOW:]
    # Dynamic majority threshold so early frames don't default to 0
    needed = (len(state_buffers[key]) // 2) + 1
    return 1 if sum(state_buffers[key]) >= needed else 0


def smooth_multistate(key, value):
    # Clamp to expected finite set and append
    try:
        v = int(value)
    except Exception:
        v = value
    if key == "feedwater":
        try:
            v = max(0, min(2, int(v)))
        except Exception:
            # leave as-is if cannot coerce; will be ignored by counts logic
            pass
    state_buffers[key].append(v)
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
if not cap.isOpened():
    print(f"Error: failed to open video: {video_path}")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video opened: {vw}x{vh} @ {fps:.3f} fps, frames={total_frames}")

def get_sample_indices(total_frames: int, fps: float, sample_fps) -> list:
    if not sample_fps or sample_fps >= fps:
        return list(range(total_frames))
    # Time-based selection to distribute samples evenly across non-integer FPS
    indices = []
    t = 0.0
    step_t = 1.0 / float(sample_fps)
    while True:
        idx = int(round(t * fps))
        if idx >= total_frames:
            break
        if not indices or idx != indices[-1]:
            indices.append(idx)
        t += step_t
    return indices

sample_indices = get_sample_indices(total_frames, fps, SAMPLE_FPS)

data_rows, errors = [], []
last_ok = {k: None for k in ["temperature", "pressure", "fuel", "rod_insertion"]}

print(f"Processing {len(sample_indices)} sampled frames (of {total_frames}) from {video_path}...")

def iter_sampled_frames(cap, sample_indices):
    if not sample_indices:
        return
    ptr = 0
    target = sample_indices[ptr]
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx == target:
            yield frame_idx, frame
            ptr += 1
            if ptr >= len(sample_indices):
                break
            target = sample_indices[ptr]
        frame_idx += 1

for frame_idx, frame in tqdm(iter_sampled_frames(cap, sample_indices), total=len(sample_indices), desc="Extracting OCR Data", ncols=80):

    # Frame hash skip: avoid reprocessing identical frames
    try:
        _fh = hash(frame.tobytes())
        if '_last_frame_hash' in globals() and _fh == globals().get('_last_frame_hash'):
            continue
        globals()['_last_frame_hash'] = _fh
    except Exception:
        pass

    timestamp = frame_idx / fps
    row = {"timestamp": timestamp}
    raw_vals = {"timestamp": timestamp}

    # First: build ROIs and fast variants per key (heavy variants built lazily on demand)
    roi_map, fast_map, var_map = {}, {}, {}
    for key, (x1, y1, x2, y2) in regions.items():
        try:
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
            roi_map[key] = roi
            # Defer heavy variants; compute only if fast pass is weak
            fast_map[key] = fast_variant(roi)
            if DEBUG_TOGGLE and (frame_idx % ROI_DEBUG_RATE == 0):
                try:
                    if roi is not None and getattr(roi, 'size', 0) > 0:
                        cv2.imwrite(os.path.join(ROI_DEBUG_DIR, f"{frame_idx:04d}_{key}_raw.png"), roi)
                        for vi, vimg in enumerate(var_map.get(key, [])[:2]):
                            if vimg is not None and getattr(vimg, 'size', 0) > 0:
                                cv2.imwrite(os.path.join(ROI_DEBUG_DIR, f"{frame_idx:04d}_{key}_proc{vi}.png"), vimg)
                except Exception as dbg_e:
                    errors.append(f"Frame {frame_idx} | {key}: debug save failed: {dbg_e}")
        except Exception as e:
            errors.append(f"Frame {frame_idx} | {key}: ROI build failed: {e}")

    # Second: batched fast OCR per group
    numeric_keys = [k for k in ("temperature","pressure","fuel","rod_insertion") if k in roi_map]
    state_keys = [k for k in ("coolant",) if k in roi_map]
    feed_keys = [k for k in ("feedwater",) if k in roi_map]

    pre = {}
    # Numeric batch (all numeric keys in one call)
    if numeric_keys:
        allow_num = "0123456789.%KkPpAa"
        num_imgs = []
        for k in numeric_keys:
            img = fast_map[k]
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            num_imgs.append(img)
        num_res_lists = readtext_multi(num_imgs, allow=allow_num)
        for k, res in zip(numeric_keys, num_res_lists):
            best_val, best_conf, best_txt = None, 0.0, ""
            for (_box, txt, conf) in res:
                txt2 = (txt or "").translate(DIGIT_FIX_SAFE)
                v = _extract_value_for_key(k, txt2)
                if v is None:
                    continue
                if float(conf) > best_conf:
                    best_conf, best_val, best_txt = float(conf), round(float(v), 1), txt
            pre[k] = (best_val, best_conf, best_txt)

    # State batch (coolant open/closed) in one call
    if state_keys:
        allow_state = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/ "
        st_imgs = []
        for k in state_keys:
            img = fast_map[k]
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            st_imgs.append(img)
        st_res_lists = readtext_multi(st_imgs, allow=allow_state)
        for k, res in zip(state_keys, st_res_lists):
            open_sum, closed_sum, best_c, best_t = 0.0, 0.0, 0.0, ""
            for (_box, txt, conf) in res:
                t = (txt or "").upper().replace('0','O').replace('1','I')
                if 'OPEN' in t or t.startswith('PEN') or ' QPEN' in (' ' + t) or ' PEN' in (' ' + t):
                    open_sum += float(conf)
                if 'CLOSED' in t:
                    closed_sum += float(conf)
                if float(conf) > best_c:
                    best_c, best_t = float(conf), t
            found_open = (open_sum >= closed_sum) and (open_sum > 0.0)
            pre[k] = (1 if found_open else 0, min(0.99, best_c), best_t)

    # Feedwater batch (0/1/2) in one call
    if feed_keys:
        import re as _re2
        pat = _re2.compile(r"([0-2])\s*/\s*2")
        allow_feed = "0123456789/ "
        fw_imgs = []
        for k in feed_keys:
            img = fast_map[k]
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            fw_imgs.append(img)
        fw_res_lists = readtext_multi(fw_imgs, allow=allow_feed)
        for k, res in zip(feed_keys, fw_res_lists):
            best_c, best_t, best_n = 0.0, "", None
            for (_box, txt, conf) in res:
                t2 = "".join(ch for ch in (txt or "") if ch.isdigit() or ch == '/')
                m = pat.search(t2)
                if not m:
                    continue
                n = int(m.group(1))
                if 0 <= n <= 2 and float(conf) > best_c:
                    best_c, best_t, best_n = float(conf), t2, n
            pre[k] = (best_n, min(0.99, best_c), best_t)

    # Third: per-key acceptance + fallbacks and per-field confidence capture
    for key in regions.keys():
        try:
            roi = roi_map.get(key)
            variants = var_map.get(key, [])
            conf = 0.0
            v = None
            rawt = ""

            if key == "coolant":
                state_open, conf, rawt = pre.get(key, (None, 0.0, ""))
                if state_open is None or conf < 0.20:
                    if not variants:
                        variants = preprocess_variants_simple(roi)
                        var_map[key] = variants
                    state_open, conf, rawt = ocr_state_variants_simple(variants, ["OPEN","PEN","QPEN"]) 
                raw_vals["coolant"] = 1 if state_open else 0
                prev_state = state_buffers.get("coolant", [])[-1] if state_buffers.get("coolant") else 1
                input_state = (1 if state_open else 0) if conf >= 0.20 else prev_state
                v = smooth_state("coolant", input_state)
                row["_raw_" + key] = rawt
                row["_sus_" + key] = 1 if conf < 0.20 else 0

            elif key == "feedwater":
                count, conf, rawt = pre.get(key, (None, 0.0, ""))
                if count is None or conf < 0.20:
                    if not variants:
                        variants = preprocess_variants_simple(roi)
                        var_map[key] = variants
                    count, conf, rawt = ocr_feedwater_count_from_variants(roi, variants)
                raw_vals["feedwater"] = count
                if count is None or conf < 0.20:
                    prev_fw = state_buffers["feedwater"][-1] if state_buffers["feedwater"] else 0
                    count_use = int(prev_fw)
                else:
                    count_use = int(max(0, min(2, count)))
                v = smooth_multistate("feedwater", count_use)
                row["_raw_" + key] = f"{v}/2 ACTIVE"
                row["_sus_" + key] = 1 if conf < 0.20 else 0

            else:
                v_parsed, conf, rawt = pre.get(key, (None, 0.0, ""))
                # If fast pass failed or is below per-field threshold, build heavy variants lazily
                if v_parsed is None or conf < CONF_THRESH.get(key, 0.0):
                    if not variants:
                        variants = preprocess_variants_simple(roi)
                        var_map[key] = variants
                    v_parsed, conf, rawt = ocr_numeric_for_key(key, variants)
                v_raw = clamp_or_nan(v_parsed, key)
                if key in ["fuel", "rod_insertion"] and v_raw is not None and 100.0 < v_raw < 200.0:
                    v_raw = v_raw - 100.0
                v = v_raw
                raw_vals[key] = v_raw
                row["_raw_" + key] = rawt
                suspect = 0

                th = CONF_THRESH.get(key, 0.0)
                if conf < th:
                    suspect = 1
                    v = None

                if key == "pressure" and v is not None:
                    v = round(v, 1)
                if key == "temperature" and v is not None:
                    v = round(v, 1)
                if key in ["fuel", "rod_insertion"] and v is not None:
                    v = round(v, 1)

                if key in ("temperature", "pressure", "rod_insertion"):
                    prev = last_ok[key]
                    if v is not None and prev is not None and key in MAX_JUMP_PER_FRAME and abs(v - prev) > MAX_JUMP_PER_FRAME[key]:
                        pj = pending_jumps[key]
                        if pj["cand"] is not None and abs(v - pj["cand"]) <= JUMP_STABILITY_THRESH[key]:
                            pj["count"] += 1
                        else:
                            pj["cand"], pj["count"] = v, 1
                        if pj["count"] >= 2 and conf >= th:
                            prev = v
                            pending_jumps[key] = {"cand": None, "count": 0}
                        v = prev
                    else:
                        pending_jumps[key] = {"cand": None, "count": 0}

                if key == "fuel":
                    prev = last_ok[key]
                    if v is not None and prev is not None and abs(v - prev) > MAX_DELTA_PER_FRAME["fuel"]:
                        suspect = 1
                        pu = pending_updates["fuel"]
                        if pu["cand"] is not None and abs(v - pu["cand"]) <= 0.05:
                            pu["count"] += 1
                        else:
                            pu["cand"], pu["count"] = v, 1
                        if pu["count"] >= 2:
                            prev = v
                            pending_updates["fuel"] = {"cand": None, "count": 0}
                        v = prev
                    else:
                        pending_updates["fuel"] = {"cand": None, "count": 0}

                if v is not None:
                    last_ok[key] = v
                row["_sus_" + key] = suspect

            row[key] = v
            row["_conf_" + key] = float(conf)
        except Exception as e:
            errors.append(f"Frame {frame_idx} | {key}: {e}")
            row[key] = last_ok.get(key, None)
            row["_conf_" + key] = 0.0

    data_rows.append(row)
    # Also accumulate a raw (no smoothing/interpolation) row for export
    if 'raw_rows' not in globals():
        raw_rows = []
    raw_rows.append(raw_vals)

cap.release()

# --- Log errors ---
if errors:
    with open(error_log, "w", encoding="utf-8") as f:
        f.write("\n".join(errors))
    print(f"\n{len(errors)} OCR errors logged to {error_log}")

# --- Data Cleanup ---
df = pd.DataFrame(data_rows)
for col in ["temperature", "pressure", "fuel", "rod_insertion", "feedwater", "coolant"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

for key, (lo, hi) in RANGES.items():
    if key in df:
        df.loc[(df[key] < lo) | (df[key] > hi), key] = np.nan

"""
Preserve exact reads where not flagged suspicious. We compute smoothing
and limited interpolation, then restore original values on rows where
the corresponding _sus_ flag is 0.
"""

# Capture originals and masks before smoothing
orig_series = {}
sus_masks = {}
for key in ["temperature", "pressure", "fuel", "rod_insertion"]:
    if key in df.columns:
        orig_series[key] = df[key].copy()
        sus_col = f"_sus_{key}"
        sus_masks[key] = df[sus_col].astype(bool) if sus_col in df.columns else pd.Series(False, index=df.index)

for col in ["temperature", "pressure"]:
    if col in df.columns:
        df[col] = df[col].rolling(5, min_periods=1, center=True).median()
for col in ["fuel", "rod_insertion"]:
    if col in df.columns:
        df[col] = df[col].rolling(3, min_periods=1, center=True).median()

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

# If the last fuel value is missing, carry forward the last known value
if "fuel" in df.columns and len(df) > 0:
    try:
        if pd.isna(df["fuel"].iloc[-1]):
            df.loc[df.index[-1], "fuel"] = df["fuel"].ffill().iloc[-1]
    except Exception:
        pass

# Restore original values for non-suspicious rows (where available)
for key in ["temperature", "pressure", "fuel", "rod_insertion"]:
    if key in df.columns and key in orig_series:
        mask_ok = (~sus_masks.get(key, pd.Series(False, index=df.index))) & orig_series[key].notna()
        df.loc[mask_ok, key] = orig_series[key][mask_ok]

# Enforce 0.1 quantization
for key in ["temperature", "pressure", "fuel", "rod_insertion"]:
    if key in df.columns:
        df[key] = df[key].round(1)

# --- Save Clean Data ---
output_path = os.path.join(os.path.dirname(video_path), output_csv)
df.to_csv(output_path, index=False)

print(f"\nCleaned data saved to {output_path}")

# --- Save Raw Data (no smoothing/interpolation) ---
try:
    if 'raw_rows' in globals() and raw_rows:
        df_raw = pd.DataFrame(raw_rows)
        for col in ["temperature", "pressure", "fuel", "rod_insertion", "feedwater", "coolant"]:
            if col in df_raw.columns:
                df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
        raw_output_path = os.path.join(os.path.dirname(video_path), RAW_OUTPUT_CSV)
        df_raw.to_csv(raw_output_path, index=False)
    print(f"Raw data (no interpolation) saved to {raw_output_path}")
except Exception as e:
    print(f"Warning: failed to write raw CSV: {e}")

# --- Optional: Invoke resuscitation on _sus_-flagged fields ---
try:
    script_path = os.path.join(os.path.dirname(__file__), "dataResuscitation.py")
    if os.path.isfile(script_path):
        # Build ROI args from current config
        def _fmt_roi(t):
            return ",".join(str(int(v)) for v in t)
        args = [
            sys.executable, script_path,
            "--video", video_path,
            "--cleaned", output_path,
            "--fps", str(float(fps)),
            "--roi-temperature", _fmt_roi(regions.get("temperature", (0,0,0,0))),
            "--roi-pressure", _fmt_roi(regions.get("pressure", (0,0,0,0))),
            "--roi-fuel", _fmt_roi(regions.get("fuel", (0,0,0,0))),
            "--roi-rod", _fmt_roi(regions.get("rod_insertion", (0,0,0,0))),
            "--roi-coolant", _fmt_roi(regions.get("coolant", (0,0,0,0))),
            "--roi-feedwater", _fmt_roi(regions.get("feedwater", (0,0,0,0))),
        ]
        # Pass raw CSV if available
        try:
            if 'raw_output_path' in locals() and os.path.isfile(raw_output_path):
                args += ["--raw", raw_output_path]
        except Exception:
            pass
        print("\nLaunching resuscitation on _sus_-flagged fields...\n")
        subprocess.run(args, check=False)

        # dataResuscitation now writes directly to cleaned CSV.
        # Strip debug columns from cleaned output if DEBUG_TOGGLE is off.
        if not DEBUG_TOGGLE:
            try:
                df_clean = pd.read_csv(output_path)
                # Keep per-field confidence columns, drop raw/sus only
                df_clean = df_clean[[c for c in df_clean.columns if not c.startswith(("_raw_", "_sus_"))]]
                df_clean.to_csv(output_path, index=False)
                print("Kept _conf_* columns; stripped _raw_/_sus_ (DEBUG_TOGGLE=False).")
            except Exception as e:
                print(f"Warning: failed to strip debug columns: {e}")
    else:
        print("Note: dataResuscitation.py not found; skipping automatic re-OCR step.")
        # If debug is off, strip debug columns from cleaned CSV now
        if not DEBUG_TOGGLE:
            try:
                df_clean = pd.read_csv(output_path)
                df_clean = df_clean[[c for c in df_clean.columns if not c.startswith(("_raw_", "_conf_", "_sus_"))]]
                df_clean.to_csv(output_path, index=False)
                print("Stripped debug columns from cleaned CSV (DEBUG_TOGGLE=False).")
            except Exception as e:
                print(f"Warning: failed to strip debug columns: {e}")
except Exception as e:
    print(f"Warning: failed to launch resuscitation: {e}")

# --- Auto-run features builder (unified features + anomaly flags) ---
try:
    fb_script = os.path.join(os.path.dirname(__file__), "featuresBuilder.py")
    if os.path.isfile(fb_script):
        args_fb = [
            sys.executable, fb_script,
            "--input", output_path,
            "--slope-window", "3",
            "--min-fuel", "75",
            "--rod", "55",
            "--rod-tol", "5",
            "--rv-k-per-s", "7.5",
            # Keep reasonable anomaly thresholds; can be tuned later
            "--rise-k-per-s", "10.0",
            "--dip-k-per-s", "10.0",
        ]
        # Require coolant open by default (aligned with stability def)
        args_fb.append("--require-coolant-open")
        print("\nBuilding features and anomaly flags via featuresBuilder.py...\n")
        subprocess.run(args_fb, check=False)
    else:
        print("Note: featuresBuilder.py not found; skipping features build.")
except Exception as e:
    print(f"Warning: failed to run features builder: {e}")
