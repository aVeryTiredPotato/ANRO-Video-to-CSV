import argparse
import json
import os
import random

import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VIDEO_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "2025-11-16 14-18-57.mp4"))
DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "roi_config.json")

# When the user clicks through, ROIs are labeled in this order to match dataGrabberModular.py.
ROI_KEYS = ["coolant", "rod_insertion", "feedwater", "fuel", "pressure", "temperature"]


def load_frame(video_path: str, frame_index: int = 0):
    """Grab a single frame from the chosen video."""
    cap = cv2.VideoCapture(video_path)
    if frame_index:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Failed to read frame {frame_index} from {video_path}")
    return frame


def _label_for_index(idx: int) -> str:
    return ROI_KEYS[idx] if idx < len(ROI_KEYS) else f"roi_{idx+1}"


def collect_regions(frame):
    """
    Display the frame and let the user mark ROIs.

    Left click twice to draw a box. Right click resets everything.
    ROIs are auto-labeled in ROI_KEYS order for dataGrabberModular.py.
    """
    display = frame.copy()
    clone = frame.copy()
    regions = {}
    current_points = []

    def on_mouse(event, x, y, flags, param):
        nonlocal display
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append((x, y))
            if len(current_points) == 2:
                (x1, y1), (x2, y2) = current_points
                tl = (min(x1, x2), min(y1, y2))
                br = (max(x1, x2), max(y1, y2))
                label = _label_for_index(len(regions))
                color = (
                    random.randint(50, 255),
                    random.randint(50, 255),
                    random.randint(50, 255),
                )
                cv2.rectangle(display, tl, br, color, 2)
                mid = ((tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2)
                cv2.putText(display, label, (mid[0] - 10, mid[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                regions[label] = (tl[0], tl[1], br[0], br[1])
                print(f"Saved ROI '{label}': {tl[0]},{tl[1]},{br[0]},{br[1]}")
                current_points.clear()
        elif event == cv2.EVENT_RBUTTONDOWN:
            display = clone.copy()
            regions.clear()
            current_points.clear()
            print("Reset all regions.")

    cv2.namedWindow("Mark ROIs")
    cv2.setMouseCallback("Mark ROIs", on_mouse)

    print("\nInstructions:")
    print("- Left-click twice to draw a box.")
    print("- ROIs will be labeled in this order: " + ", ".join(ROI_KEYS) + ".")
    print("- Right-click anywhere to clear and restart.")
    print("- Press ESC or Q when you are finished.\n")

    while True:
        cv2.imshow("Mark ROIs", display)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    cv2.destroyAllWindows()
    return regions


def save_config(video_path: str, rois: dict, config_path: str = DEFAULT_CONFIG_PATH):
    """Write ROI coordinates and the chosen video path to a JSON config."""
    payload = {
        "video_path": os.path.abspath(video_path),
        "regions": {k: list(v) for k, v in rois.items()},
        "roi_order": ROI_KEYS,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nConfig saved to {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Mark ROIs on a frame and save them for dataGrabberModular.py")
    parser.add_argument("-v", "--video", default=DEFAULT_VIDEO_PATH, help="Path to the source video")
    parser.add_argument("-f", "--frame", type=int, default=0, help="Frame index to display for ROI marking")
    parser.add_argument("-o", "--output", default=DEFAULT_CONFIG_PATH, help="Where to write roi_config.json")
    parser.add_argument("--snapshot", default=None, help="Optional path to save the frame being marked")
    args = parser.parse_args()

    frame = load_frame(args.video, args.frame)
    if args.snapshot:
        cv2.imwrite(args.snapshot, frame)
        print(f"Saved snapshot to {args.snapshot}")

    regions = collect_regions(frame)
    if not regions:
        print("No ROIs were captured. Exiting without writing a config.")
        return

    save_config(args.video, regions, args.output)

    print("\nFinal ROI coordinates:")
    for name, (x1, y1, x2, y2) in regions.items():
        print(f"- {name}: {x1},{y1},{x2},{y2}")


if __name__ == "__main__":
    main()
