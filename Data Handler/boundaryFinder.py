import os
import cv2
import random
import torch
import sys

# Ensure line-buffered output for GUI log streaming
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

# Paths
scriptDir = os.path.dirname(os.path.abspath(__file__))
baseDir = os.path.abspath(os.path.join(scriptDir, ".."))
regionsOutPath = os.path.join(scriptDir, "roi_regions.txt")

def pickLatestVideo(rootDir: str) -> str:
    override = os.environ.get("ANRO_VIDEO")
    if override and os.path.isfile(override):
        return override
    exts = {".mp4", ".mkv", ".avi", ".mov", ".wmv"}
    candidates = []
    for name in os.listdir(rootDir):
        path = os.path.join(rootDir, name)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(name)[1].lower() in exts:
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No video files found in {rootDir}")
    return max(candidates, key=lambda p: os.path.getmtime(p))

videoPath = pickLatestVideo(baseDir)
print(f"Using video: {videoPath}")

print("Current working directory:", os.getcwd())

cap = cv2.VideoCapture(videoPath)
ret, frame = cap.read()
cap.release()

if not ret:
    raise ValueError(f"Failed to read frame from {videoPath}")

cv2.imwrite("sample_frame.png", frame)

imagePath = "sample_frame.png"
img = cv2.imread(imagePath)
clone = img.copy()

# List keys in the order you want to fill them (press SPACE to skip a key).
roiKeys = [
    # Primary systems
    "coolant",
    "rodInsertion",
    "feedwater",
    # Primary readings
    "fuel",
    "pressure",
    "temperature",
    # Feedwater readings
    "waterLevel",
    "feedwaterFlow",
    # Feedwater pump 1
    "fwpFlowRate1",
    "fwpUtilization1",
    "fwpRpm1",
    # Feedwater pump 2
    "fwpFlowRate2",
    "fwpUtilization2",
    "fwpRpm2",
    # Power output readings
    "totalOutput",
    "currentPowerOrder",
    "marginOfError",
    # Turbine 1
    "flowRate1",
    "rpm1",
    "valvesPct1",
    "vibration1",
    # Turbine 2
    "flowRate2",
    "rpm2",
    "valvesPct2",
    "vibration2",
]

roiSections = [
    ("primarySystems", ["coolant", "rodInsertion", "feedwater"]),
    ("primaryReadings", ["fuel", "pressure", "temperature"]),
    ("feedwaterReadings", ["waterLevel", "feedwaterFlow"]),
    ("feedwaterPump1", ["fwpFlowRate1", "fwpUtilization1", "fwpRpm1"]),
    ("feedwaterPump2", ["fwpFlowRate2", "fwpUtilization2", "fwpRpm2"]),
    ("powerOutput", ["totalOutput", "currentPowerOrder", "marginOfError"]),
    ("turbine1", ["flowRate1", "rpm1", "valvesPct1", "vibration1"]),
    ("turbine2", ["flowRate2", "rpm2", "valvesPct2", "vibration2"]),
]

keyToSection = {}
sectionOrder = [name for name, _ in roiSections]
sectionToKeys = {}
for name, keys in roiSections:
    sectionToKeys[name] = keys
    for key in keys:
        keyToSection[key] = name

sectionIndexByName = {name: i for i, name in enumerate(sectionOrder)}

# Storage for marked regions
regions = []
currentPoints = []
roiMap = {}
keyIdx = 0

def announceKey():
    if keyIdx < len(roiKeys):
        print(f"Current key: {roiKeys[keyIdx]} ({keyIdx + 1}/{len(roiKeys)})")
    else:
        print("All keys filled. Press ESC to quit or right-click to reset.")

def skipSectionForKey(currentKey):
    section = keyToSection.get(currentKey)
    if not section:
        return
    keys = sectionToKeys.get(section, [])
    for k in keys:
        if k not in roiMap:
            roiMap[k] = (0, 0, 0, 0)
            print(f"{k} skipped: 0,0,0,0 (section {section})")

def advanceKeyIndex():
    global keyIdx
    while keyIdx < len(roiKeys) and roiKeys[keyIdx] in roiMap:
        keyIdx += 1

def clickEvent(event, x, y, flags, param):
    global currentPoints, img, keyIdx
    # Left click - record corner
    if event == cv2.EVENT_LBUTTONDOWN:
        if keyIdx >= len(roiKeys):
            print("All keys filled. Press ESC to quit or right-click to reset.")
            return
        currentPoints.append((x, y))
        if len(currentPoints) == 2:
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            cv2.rectangle(img, currentPoints[0], currentPoints[1], color, 2)
            label = f"{len(regions) + 1}"
            midX = int((currentPoints[0][0] + currentPoints[1][0]) / 2)
            midY = int((currentPoints[0][1] + currentPoints[1][1]) / 2)
            cv2.putText(
                img,
                label,
                (midX - 10, midY + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
            regions.append(tuple(currentPoints))
            x1, y1 = currentPoints[0]
            x2, y2 = currentPoints[1]
            key = roiKeys[keyIdx]
            roiMap[key] = (x1, y1, x2, y2)
            print(f"{key} saved: {x1},{y1},{x2},{y2}")
            keyIdx += 1
            advanceKeyIndex()
            currentPoints = []
            announceKey()

    # Right click - reset
    elif event == cv2.EVENT_RBUTTONDOWN:
        img[:] = clone.copy()
        regions.clear()
        currentPoints.clear()
        roiMap.clear()
        keyIdx = 0
        print("Reset all regions")
        announceKey()


# Create window (center it so it doesn't appear off-screen)
cv2.namedWindow("Mark ROIs", cv2.WINDOW_NORMAL)
try:
    import tkinter as tk
    _root = tk.Tk()
    _root.withdraw()
    screenW = _root.winfo_screenwidth()
    screenH = _root.winfo_screenheight()
    _root.destroy()
    winW = img.shape[1]
    winH = img.shape[0]
    posX = max(0, int((screenW - winW) / 2))
    posY = max(0, int((screenH - winH) / 2))
    cv2.moveWindow("Mark ROIs", posX, posY)
except Exception:
    pass
cv2.setMouseCallback("Mark ROIs", clickEvent)

print("Left-click twice to draw ROI boxes. Right-click to reset. SPACE to skip a key (0,0,0,0).")
announceKey()
while True:
    cv2.imshow("Mark ROIs", img)
    key = cv2.waitKey(1) & 0xFF
    if key == 32:  # SPACE to skip current key
        if keyIdx < len(roiKeys):
            skipKey = roiKeys[keyIdx]
            roiMap[skipKey] = (0, 0, 0, 0)
            print(f"{skipKey} skipped: 0,0,0,0")
            # Skip the rest of the section
            skipSectionForKey(skipKey)
            keyIdx += 1
            advanceKeyIndex()
            announceKey()
        else:
            print("All keys filled. Press ESC to quit or right-click to reset.")
    if key == 27:  # ESC to quit
        break

cv2.destroyAllWindows()

# Print regions for copy-pasting
print("\nPaste into regions = {")
for key in roiKeys:
    x1, y1, x2, y2 = roiMap.get(key, (0, 0, 0, 0))
    print(f"    \"{key}\": ({x1},{y1},{x2},{y2}),")
print("}")

# Write regions to a text file for dataGrabber.py
try:
    with open(regionsOutPath, "w", encoding="utf-8") as f:
        f.write("# ROI regions for dataGrabber.py\n")
        f.write("# format: key=x1,y1,x2,y2\n")
        for key in roiKeys:
            x1, y1, x2, y2 = roiMap.get(key, (0, 0, 0, 0))
            f.write(f"{key}={x1},{y1},{x2},{y2}\n")
    print(f"\nSaved regions to: {regionsOutPath}")
except Exception as e:
    print(f"\nWarning: failed to write regions file: {e}")
