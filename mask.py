# File to see how a mask is made

import cv2
import numpy as np

# read the image
image = cv2.imread("Images/FTP.6.62.34_Fruit Quality Fruit3_1_2021-12-07-10-19-18.jpg")

# create a copy of the image to draw the mask on
mask = np.zeros(image.shape[:2], dtype=np.uint8)

# function to draw the mask
def draw_mask(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(mask, (x, y), 10, (255, 255, 255), -1)
        apply_mask()
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(mask, (x, y), 10, (0, 0, 0), -1)
        apply_mask()

def apply_mask():
    # apply the mask to the image
    image_masked = cv2.bitwise_and(image, image, mask=mask)

    # display the image and the mask
    cv2.imshow("image", image)
    cv2.imshow("mask", image_masked)

# create a window to draw the mask
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("image", 600, 600)
cv2.setMouseCallback("image", draw_mask)

while True:
    key = cv2.waitKey(1)

    # press 'q' to exit
    if key == ord('q'):
        break

cv2.destroyAllWindows()
