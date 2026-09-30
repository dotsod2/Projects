import cv2
import numpy as np
from pathlib import Path

# Set up paths relative to the script location
BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "input_images" / "plot1" / "plot1.jpg"
GROUND_TRUTH_PATH = BASE_DIR / "ground_truth" / "plot1_truth.jpg"
OUTPUT_PATH = BASE_DIR / "output_images" / "aligned_output.jpg"

# Global state variables
selected_pt_idx = -1
radius = 64  # Click target radius for dragging points
def create_anaglyph_overlay(src_img, dst_img, src_pts):
    """
    Warps src_img to match dst_img dimensions and creates a Red/Cyan false-color
    anaglyph overlay to highlight alignment errors.
    """
    h, w, _ = dst_img.shape
    
    # Destination corners
    dst_pts = np.float32([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ])
    
    # Compute Homography
    H, _ = cv2.findHomography(src_pts, dst_pts)
    
    if H is None or H.shape != (3, 3):
        return dst_img.copy()

    # Warp input image to ground truth size
    warped_src = cv2.warpPerspective(src_img, H, (w, h))

    # Convert both to grayscale
    gray_src = cv2.cvtColor(warped_src, cv2.COLOR_BGR2GRAY)
    gray_dst = cv2.cvtColor(dst_img, cv2.COLOR_BGR2GRAY)

    # Build Anaglyph (BGR format in OpenCV)
    # Channel 0: Blue  (Ground Truth)
    # Channel 1: Green (Ground Truth)
    # Channel 2: Red   (Warped Input)
    anaglyph = np.zeros((h, w, 3), dtype=np.uint8)
    anaglyph[:, :, 0] = gray_dst
    anaglyph[:, :, 1] = gray_dst
    anaglyph[:, :, 2] = gray_src

    return anaglyph
def create_overlay(src_img, dst_img, src_pts, alpha=0.5):
    """
    Warps src_img using 4 source points to fit dst_img dimensions,
    then overlays it on dst_img at specified alpha opacity.
    """
    h, w, _ = dst_img.shape
    
    # Target points are the 4 corners of the ground truth image
    dst_pts = np.float32([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ])
    
    # Calculate Homography Matrix
    H, _ = cv2.findHomography(src_pts, dst_pts)
    
    if H is None or H.shape != (3, 3):
        return dst_img.copy()

    # Warp input image to ground truth size
    warped_src = cv2.warpPerspective(src_img, H, (w, h))

    # Blend warped image with ground truth (50% opacity)
    # Formula: overlay = (alpha * warped_src) + ((1 - alpha) * dst_img)
    blended = cv2.addWeighted(warped_src, alpha, dst_img, 1.0 - alpha, 0)
    
    return blended

def mouse_handler(event, x, y, flags, param):
    global selected_pt_idx
    src_pts = param["pts"]

    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if click is near any of the 4 points
        for i, pt in enumerate(src_pts):
            if np.linalg.norm(pt - np.array([x, y])) < radius * 2:
                selected_pt_idx = i
                break

    elif event == cv2.EVENT_MOUSEMOVE and selected_pt_idx != -1:
        # Drag point
        src_pts[selected_pt_idx] = [x, y]

    elif event == cv2.EVENT_LBUTTONUP:
        selected_pt_idx = -1

def main():
    src_img = cv2.imread(str(INPUT_PATH))
    dst_img = cv2.imread(str(GROUND_TRUTH_PATH))

    if src_img is None or dst_img is None:
        print("Error: Could not load input or ground truth image. Check file paths.")
        return

    h_src, w_src, _ = src_img.shape

    # Initial 4 corner points [top-left, top-right, bottom-right, bottom-left]
    src_pts = np.float32([
        [100, 100],
        [w_src - 100, 100],
        [w_src - 100, h_src - 100],
        [100, h_src - 100]
    ])

    param = {"pts": src_pts}
    
    cv2.namedWindow("Interactive Homography Controls", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Interactive Homography Controls", mouse_handler, param)

    print("\n--- Controls ---")
    print("• Red = Warped Input | Cyan = Ground Truth Reference")
    print("• Click and drag red circles on the left window to eliminate color fringes.")
    print("• Press 's' to save the clean aligned result.")
    print("• Press 'q' or ESC to exit.\n")

    while True:
        # Generate Red/Cyan overlay
        anaglyph_preview = create_anaglyph_overlay(src_img, dst_img, param["pts"])

        # Render control points on the original source image
        src_display = src_img.copy()
        for i, pt in enumerate(param["pts"]):
            cv2.circle(src_display, (int(pt[0]), int(pt[1])), radius, (0, 0, 255), -1)
            cv2.putText(src_display, str(i + 1), (int(pt[0]) + 10, int(pt[1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Match heights for side-by-side display
        h_dst, w_dst, _ = dst_img.shape
        src_resized = cv2.resize(src_display, (int(w_src * (h_dst / h_src)), h_dst))
        combined_view = np.hstack((src_resized, anaglyph_preview))

        cv2.imshow("Interactive Homography Controls", combined_view)

        key = cv2.waitKey(15) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            # Save clean, full-resolution warped result (without red/cyan filter or points)
            dst_corner_pts = np.float32([[0, 0], [w_dst - 1, 0], [w_dst - 1, h_dst - 1], [0, h_dst - 1]])
            H, _ = cv2.findHomography(param["pts"], dst_corner_pts)
            aligned_final = cv2.warpPerspective(src_img, H, (w_dst, h_dst))
            
            cv2.imwrite(str(OUTPUT_PATH), aligned_final)
            print(f"Saved aligned output to: {OUTPUT_PATH}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()