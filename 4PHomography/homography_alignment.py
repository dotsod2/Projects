# This notebook is a first attempt at homography 4 point alignment
#This cell is just imports and installs
import numpy as np
import cv2
import glob
import os
# Defining a function to get user clicks of 4 corners
#list to store user's clicked coordinates
clicked_points = []

# defining the function
def click_event(event, x, y, flags, param):
    global image

    # Check if the user performed a left mouse button click
    if event == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_SHIFTKEY:
        # Append the [x,y] coordinates to our global tracking list
        clicked_points.append([x,y])

        # Draw a solid red dot and label on the image where clicked
        cv2.circle(image, (x,y), 5, (0,0,255), -1)
        cv2.putText(image, f"P{len(clicked_points)}", (x + 10, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0,6, (0,0,255), 2)
        cv2.imshow("Select 4 Corners", image)

        # Closes window once 4 coordinates are gathered
        if len(clicked_points) == 4:
            print("Captured Coordinates:", clicked_points)
            cv2.destroyWindow("Select 4 Corners")

# Loading in the image
input_plot = input("Enter the plot number: ")
input_name = input("Enter the image name: ")
image = cv2.imread("plot" + input_plot + "/" + input_name + ".jpg")

# Resizing display window
display_width = 900 
scale_factor = display_width / image.shape[1] 
display_height = int(image.shape[0] * scale_factor)

display_image = cv2.resize(image, (display_width, display_height))
# Define 4 corner points in source image via user input
#print("Please click the 4 corners from top left to bottom left in a clockwise fashion")
cv2.imshow("Select 4 corners", display_image)
cv2.setMouseCallback("Select 4 corners", click_event)

# wait until window closes
while True:
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

cv2.destroyAllWindows()
raw_clicked_points = np.float32(clicked_points)
src_pts = raw_clicked_points / scale_factor
# Define the desired 4 point destination points
# Defining output resolution
width = 9504
height = 6336
dst_pts = np.float32([
    [0, 0],  # Top Left
    [width, 0],  # Top Right
    [width, height],  # Bottom Right
    [0, height]  # Bottom Left
])

# Calculating 3x3 Homography Matrix
# cv2.getPerspectiveTransform works specifically for exactly 4 points
H = cv2.getPerspectiveTransform(src_pts, dst_pts)

# 5. Warp the image perspective to align it
aligned_image = cv2.warpPerspective(image, H, (width, height))

# 6. Save or display the result
cv2.imwrite("plot" + input_plot + "_aligned/" + input_name + "_SORTED.jpg", aligned_image)
#cv2.imshow("Original", image)
cv2.imshow("Aligned", aligned_image)
cv2.waitKey(0)
cv2.destroyAllWindows()