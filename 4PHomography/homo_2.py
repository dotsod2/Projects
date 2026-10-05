import numpy as np
import cv2
import glob
import os

# -----------------------------
# Global Interactive State
# -----------------------------
points = []  # Display-scale source points [[x,y], ...]
selected_idx = -1  # Currently dragged point index (-1 if none)
dragging = False  # Dragging state flag
needs_update = False  # Flag indicating points moved and preview needs refresh


def draw_points(img_copy, pts, active_idx=-1):
    """Draw points, labels, and bounding polygon on image copy."""
    for i, pt in enumerate(pts):
        color = (0, 255, 0) if i == active_idx else (0, 0, 255)
        cv2.circle(img_copy, tuple(pt), 6, color, -1)
        cv2.circle(img_copy, tuple(pt), 8, (255, 255, 255), 1)
        cv2.putText(
            img_copy, f"P{i + 1}", (pt[0] + 10, pt[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
        )
    if len(pts) == 4:
        poly_pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_copy, [poly_pts], True, (255, 0, 0), 1)


def mouse_callback(event, x, y, flags, param):
    """Handles Shift+Click initial selection and dedicated point dragging."""
    global points, selected_idx, dragging, needs_update

    # --- Phase 1: Initial Selection (Requires SHIFT + Left Click) ---
    if len(points) < 4:
        if event == cv2.EVENT_LBUTTONDOWN and (flags & cv2.EVENT_FLAG_SHIFTKEY):
            points.append([x, y])
            print(f"Point {len(points)} set to: ({x}, {y})")
            if len(points) == 4:
                needs_update = True
        return

    # --- Phase 2: Point Adjustment (Click & Drag) ---
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if click is specifically ON an existing point (within 25px radius)
        distances = [np.hypot(pt[0] - x, pt[1] - y) for pt in points]
        if len(distances) > 0:
            min_idx = int(np.argmin(distances))
            if distances[min_idx] <= 25:
                selected_idx = min_idx
                dragging = True

    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        if selected_idx != -1:
            points[selected_idx] = [x, y]
            needs_update = True  # Marks that points have moved

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        selected_idx = -1
        # NOTE: Automatic update on mouse release is removed.
        # Update occurs ONLY when pressing 'U' or 'Space'.


def create_color_diff_overlay(warped_img, gt_img):
    """
    Creates a Red/Green visual alignment overlay.
    - Ground Truth = Green Channel
    - Warped Image = Red Channel
    - Perfectly aligned matching regions = White / Yellow
    """
    gray_warp = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    gray_gt = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)

    height, width = gray_warp.shape
    overlay = np.zeros((height, width, 3), dtype=np.uint8)

    # Channel 2 (Red)   <- Warped Image
    # Channel 1 (Green) <- Ground Truth
    # Channel 0 (Blue)  <- Minimum overlap (turns overlap into crisp White)
    overlay[:, :, 2] = gray_warp
    overlay[:, :, 1] = gray_gt
    overlay[:, :, 0] = cv2.min(gray_warp, gray_gt)

    return overlay


# -----------------------------
# Main Execution Workflow
# -----------------------------
def main():
    global points, selected_idx, dragging, needs_update

    input_plot = input("Enter the plot number: ").strip()

    input_folder = f"input_images/plot{input_plot}"
    gt_folder = f"ground_truth/plot{input_plot}"
    output_folder = f"output_images/plot{input_plot}_aligned"

    os.makedirs(output_folder, exist_ok=True)

    # 1. Load Ground Truth Image
    gt_candidates = glob.glob(f"{gt_folder}/*.jpg") + glob.glob(f"{gt_folder}/*.png")
    if not gt_candidates:
        print(f"Error: Could not find any ground truth images in '{gt_folder}'.")
        print(f"Please ensure your ground truth image is located in '{gt_folder}/'.")
        return

    gt_path = gt_candidates[0]
    gt_image = cv2.imread(gt_path)
    if gt_image is None:
        print(f"Failed to read ground truth image at: {gt_path}")
        return

    print(f"Loaded Ground Truth: {gt_path}")

    dst_h, dst_w = gt_image.shape[:2]
    dst_pts = np.float32([
        [0, 0],  # Top Left
        [dst_w, 0],  # Top Right
        [dst_w, dst_h],  # Bottom Right
        [0, dst_h]  # Bottom Left
    ])

    # 2. Get Input Images
    image_files = glob.glob(f"{input_folder}/*.jpg") + glob.glob(f"{input_folder}/*.png")

    if not image_files:
        print(f"No input images found in '{input_folder}'.")
        return

    print(f"Found {len(image_files)} image(s) to process.")
    print("\nControls:")
    print("  SHIFT + Left Click : Select initial 4 points")
    print("  Click & Drag Point : Move existing corner point")
    print("  [U] or [Space]     : Update / Refresh Alignment Preview")
    print("  [Enter]            : Save Alignment & Next Image")
    print("  [R]                : Reset Points")
    print("  [ESC]              : Exit Script\n")

    display_width = 900
    display_gt = cv2.resize(gt_image, (display_width, int(dst_h * (display_width / dst_w))))

    for image_path in image_files:
        print(f"--- Processing: {os.path.basename(image_path)} ---")

        original_image = cv2.imread(image_path)
        if original_image is None:
            print("Could not load image. Skipping...")
            continue

        orig_h, orig_w = original_image.shape[:2]
        scale_factor = display_width / orig_w
        display_h = int(orig_h * scale_factor)

        display_source = cv2.resize(original_image, (display_width, display_h))

        # Reset global state for this image
        points = []
        selected_idx = -1
        dragging = False
        needs_update = False
        preview_ready = False

        win_source = "Source Image (Shift+Click 4 Corners / Drag to Adjust)"
        win_diff = "Alignment Overlay"

        cv2.namedWindow(win_source, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win_source, mouse_callback)

        while True:
            canvas_source = display_source.copy()

            # Phase 1: Selecting 4 initial points
            if len(points) < 4:
                draw_points(canvas_source, points)
                cv2.putText(
                    canvas_source, f"HOLD SHIFT + Click point {len(points) + 1}/4",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )
                cv2.imshow(win_source, canvas_source)

            # Phase 2: Points selected, allow adjustments
            else:
                draw_points(canvas_source, points, active_idx=selected_idx)

                if needs_update:
                    status_msg = "Points moved! Press [U] or [SPACE] to update preview"
                    color = (0, 0, 255)  # Red warning
                else:
                    status_msg = "Press [U]/[SPACE] to update | [ENTER] to Save"
                    color = (0, 255, 0)  # Green ready

                cv2.putText(
                    canvas_source, status_msg,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
                )
                cv2.imshow(win_source, canvas_source)

            key = cv2.waitKey(20) & 0xFF

            # [ESC] - Exit script
            if key == 27:
                print("Exiting...")
                cv2.destroyAllWindows()
                return

            # [R] - Reset points
            elif key in (ord('r'), ord('R')):
                points = []
                needs_update = False
                preview_ready = False
                if cv2.getWindowProperty(win_diff, cv2.WND_PROP_VISIBLE) >= 1:
                    cv2.destroyWindow(win_diff)

            # [U] or [SPACE] - Explicit Update Trigger
            elif key in (ord('u'), ord('U'), 32):
                if len(points) == 4:
                    src_pts = np.float32(points) / scale_factor
                    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    warped_preview = cv2.warpPerspective(original_image, H, (dst_w, dst_h))

                    disp_warped = cv2.resize(warped_preview, (display_gt.shape[1], display_gt.shape[0]))
                    diff_overlay = create_color_diff_overlay(disp_warped, display_gt)

                    cv2.putText(
                        diff_overlay, "Red: Image | Green: Ground Truth | White: Match",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                    )
                    cv2.imshow(win_diff, diff_overlay)
                    needs_update = False
                    preview_ready = True

            # [ENTER] - Confirm & Save aligned image
            elif key == 13:
                if len(points) == 4:
                    src_pts = np.float32(points) / scale_factor
                    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    aligned_image = cv2.warpPerspective(original_image, H, (dst_w, dst_h))

                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    output_path = os.path.join(output_folder, f"{base_name}_aligned.jpg")
                    cv2.imwrite(output_path, aligned_image)

                    print(f"Saved aligned image: {output_path}\n")

                    cv2.destroyAllWindows()
                    break
                else:
                    print("Please select all 4 points before proceeding.")

    cv2.destroyAllWindows()
    print("Successfully processed all images!")


if __name__ == "__main__":
    main()