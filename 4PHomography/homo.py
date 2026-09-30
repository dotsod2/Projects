import numpy as np
import cv2
import glob
import os

# -----------------------------
# Settings
# -----------------------------

input_plot = input("Enter the plot number: ")

input_folder = "input_images/plot" + input_plot
output_folder = "output_images/plot" + input_plot + "_aligned"

os.makedirs(output_folder, exist_ok=True)

# Find all JPG images in the input folder
image_files = glob.glob(input_folder + "/*.jpg")

print(f"Found {len(image_files)} images.")

# -----------------------------
# Click function
# -----------------------------

clicked_points = []

def click_event(event, x, y, flags, param):
    global image

    if event == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_SHIFTKEY:

        clicked_points.append([x, y])

        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

        cv2.putText(
            image,
            f"P{len(clicked_points)}",
            (x + 10, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        cv2.imshow("Select 4 Corners", image)

        print(f"Point {len(clicked_points)}: ({x}, {y})")


# -----------------------------
# Destination points
# -----------------------------

width = 9504
height = 6336

dst_pts = np.float32([
    [0, 0],                 # Top Left
    [width, 0],             # Top Right
    [width, height],        # Bottom Right
    [0, height]             # Bottom Left
])


# -----------------------------
# Process every image
# -----------------------------

for image_path in image_files:

    print("\nProcessing:", image_path)

    # Reset clicked points for this image
    clicked_points = []

    # Load image
    image = cv2.imread(image_path)

    if image is None:
        print("Could not load:", image_path)
        continue

    # -------------------------
    # Resize for display
    # -------------------------

    display_width = 900

    scale_factor = display_width / image.shape[1]

    display_height = int(image.shape[0] * scale_factor)

    display_image = cv2.resize(
        image,
        (display_width, display_height)
    )

    # We want the callback to modify the DISPLAY image
    image = display_image

    # -------------------------
    # Display image
    # -------------------------

    cv2.namedWindow("Select 4 Corners", cv2.WINDOW_NORMAL)

    cv2.imshow("Select 4 Corners", image)

    cv2.setMouseCallback("Select 4 Corners", click_event)

    # -------------------------
    # Wait for 4 points or ESC
    # -------------------------

    while len(clicked_points) < 4:

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            print("Stopping.")
            cv2.destroyAllWindows()
            raise SystemExit

    # -------------------------
    # Convert points back to
    # original image coordinates
    # -------------------------

    raw_clicked_points = np.float32(clicked_points)

    src_pts = raw_clicked_points / scale_factor

    # -------------------------
    # Calculate homography
    # -------------------------

    H = cv2.getPerspectiveTransform(
        src_pts,
        dst_pts
    )

    # -------------------------
    # Warp ORIGINAL image
    # -------------------------

    original_image = cv2.imread(image_path)

    aligned_image = cv2.warpPerspective(
        original_image,
        H,
        (width, height)
    )

    # -------------------------
    # Save
    # -------------------------

    filename = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    output_path = (
        output_folder
        + "/"
        + filename
        + "_SORTED.jpg"
    )

    cv2.imwrite(output_path, aligned_image)

    print("Saved:", output_path)

    cv2.destroyWindow("Select 4 Corners")


cv2.destroyAllWindows()

print("\nFinished processing all images!")
