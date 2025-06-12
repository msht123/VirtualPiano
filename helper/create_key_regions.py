import cv2
import json
import os

# --- Configuration ---
SAMPLE_VIDEO_PATH = "./piano_notes/tap_C.mov"
OUTPUT_JSON_PATH = "./piano_notes/key_regions.json"

# Target dimensions (MUST MATCH inference script resolution!)
TARGET_WIDTH = 640
TARGET_HEIGHT = 480

KEY_NAMES_TO_DEFINE = [
    "C4", "Csharp4", "D4", "Dsharp4", "E4", "F4",
    "Fsharp4", "G4", "Gsharp4", "A4", "Asharp4", "B4"
]

points = []
current_frame = None

def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(current_frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Define Key Regions - Press 'n' for next key, 'q' to quit", current_frame)

def create_key_regions_interactive():
    global points, current_frame

    cap = cv2.VideoCapture(SAMPLE_VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {SAMPLE_VIDEO_PATH}")
        return

    success, frame_to_show = cap.read()
    cap.release()
    if not success:
        print("Error: Could not read frame from video.")
        return

    # Resize and flip to match inference pipeline
    frame_to_show = cv2.resize(frame_to_show, (TARGET_WIDTH, TARGET_HEIGHT))
    frame_to_show = cv2.flip(frame_to_show, -1)
    current_frame = frame_to_show.copy()

    cv2.namedWindow("Define Key Regions - Press 'n' for next key, 'q' to quit")
    cv2.setMouseCallback("Define Key Regions - Press 'n' for next key, 'q' to quit", mouse_callback)

    key_regions_data = {}
    key_index = 0

    print("\n--- Interactive Key Region Definition ---")
    print("Click Top-Left then Bottom-Right for each key.")
    print("Press 'n' to confirm, 'r' to reset, 'q' to quit.")

    while key_index < len(KEY_NAMES_TO_DEFINE):
        current_key_name = KEY_NAMES_TO_DEFINE[key_index]
        points = []

        display_frame = current_frame.copy()
        cv2.putText(display_frame, f"Click Top-Left, then Bottom-Right for: {current_key_name}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Define Key Regions - Press 'n' for next key, 'q' to quit", display_frame)

        while True:
            key_press = cv2.waitKey(0) & 0xFF
            if key_press == ord('q'):
                print("Quitting.")
                cv2.destroyAllWindows()
                return
            elif key_press == ord('r'):
                points = []
                current_frame = frame_to_show.copy()
                print(f"Reset points for {current_key_name}")
                break
            elif key_press == ord('n'):
                if len(points) == 2:
                    x_coords = sorted([points[0][0], points[1][0]])
                    y_coords = sorted([points[0][1], points[1][1]])
                    key_regions_data[current_key_name] = {
                        "x_min": x_coords[0],
                        "y_min": y_coords[0],
                        "x_max": x_coords[1],
                        "y_max": y_coords[1]
                    }
                    print(f"Saved region for {current_key_name}: {key_regions_data[current_key_name]}")
                    key_index += 1
                    current_frame = frame_to_show.copy()
                    break
                else:
                    print("Please click exactly 2 points.")
            else:
                print("Unrecognized key. Use 'n', 'r', or 'q'.")

    cv2.destroyAllWindows()
    if key_regions_data:
        os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
        with open(OUTPUT_JSON_PATH, 'w') as f:
            json.dump(key_regions_data, f, indent=4)
        print(f"Saved key regions to: {OUTPUT_JSON_PATH}")
    else:
        print("No regions defined.")

if __name__ == "__main__":
    create_key_regions_interactive()
