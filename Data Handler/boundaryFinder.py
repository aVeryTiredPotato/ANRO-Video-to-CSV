import os, cv2, random
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")


# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, "..", "2025-11-04 17-16-32.mkv")
video_path = os.path.abspath(video_path)

print("Current working directory:", os.getcwd())

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    raise ValueError(f"Failed to read frame from {video_path}")
cv2.imwrite("sample_frame.png", frame)

image_path = "sample_frame.png"
img = cv2.imread(image_path)
clone = img.copy()

# Storage for marked regions
regions = []
current_points = []

def click_event(event, x, y, flags, param):
    global current_points, img
    # Left click → record corner
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        if len(current_points) == 2:
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
            cv2.rectangle(img, current_points[0], current_points[1], color, 2)
            label = f"{len(regions)+1}"
            mid_x = int((current_points[0][0] + current_points[1][0]) / 2)
            mid_y = int((current_points[0][1] + current_points[1][1]) / 2)
            cv2.putText(img, label, (mid_x-10, mid_y+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            regions.append(tuple(current_points))
            print(f"Region {len(regions)} saved: {current_points[0]} -> {current_points[1]}")
            current_points = []

    # Right click → reset
    elif event == cv2.EVENT_RBUTTONDOWN:
        img[:] = clone.copy()
        regions.clear()
        current_points.clear()
        print("Reset all regions")

# Create window
cv2.namedWindow("Mark ROIs")
cv2.setMouseCallback("Mark ROIs", click_event)

print("Left-click twice to draw ROI boxes. Right-click to reset.")
while True:
    cv2.imshow("Mark ROIs", img)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to quit
        break

cv2.destroyAllWindows()

# Print regions for copy-pasting
print("\nFinal ROI coordinates:")
for i, (p1, p2) in enumerate(regions, 1):
    print(f"ROI {i}: {p1} -> {p2}")
