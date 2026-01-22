import os
import cv2
import random
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

# Paths
scriptDir = os.path.dirname(os.path.abspath(__file__))
baseDir = os.path.abspath(os.path.join(scriptDir, ".."))
regionsOutPath = os.path.join(scriptDir, "roi_regions.txt")

def pickLatestVideo(rootDir: str) -> str:
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
    # Power output readings
    "totalOutput",
    "currentPowerOrder",
    "marginOfError",
    # Turbine 1
    "flowRate1",
    "rpm1",
    "valvesPct1",
    # Turbine 2
    "flowRate2",
    "rpm2",
    "valvesPct2",
]

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


# Create window
cv2.namedWindow("Mark ROIs")
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
            keyIdx += 1
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
