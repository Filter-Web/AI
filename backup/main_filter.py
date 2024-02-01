from face_detector import YoloDetector
import numpy as np
from PIL import Image
import cv2

def overlay(image, x, y, w, h, overlay_image):
    alpha = overlay_image[:, :, 3]
    masked_image = alpha / 255
    for c in range(0, 3): # BGR
        image[y-h:y+h, x-w:x+w, c] = (overlay_image[:, :, c] * masked_image) + (image[y-h:y+h, x-w:x+w, c] * (1-masked_image))

left = cv2.imread('filter_left.png', cv2.IMREAD_UNCHANGED)
left = cv2.resize(left, (100, 100))
right = cv2.imread('filter_right.png', cv2.IMREAD_UNCHANGED)
right = cv2.resize(right, (100, 100))

model = YoloDetector(target_size=720, device="cuda:0", min_face=90)
img = cv2.imread('capture_2.jpg')
bboxes,points = model.predict(img)

for box,landmark in zip(bboxes,points):
    for lm in landmark:
            left_eye, right_eye, nose, left_mouse, right_mouse = lm
            left_loc = left_eye
            left_loc[0] = (int)((left_eye[0] + left_mouse[0]) / 2) - 40
            left_loc[1] = (int)((left_eye[1] + left_mouse[1]) / 2)
            right_loc = right_eye
            right_loc[0] = (int)((right_eye[0] + right_mouse[0]) / 2) + 40
            right_loc[1] = (int)((right_eye[1] + right_mouse[1]) / 2)
            overlay(img, *left_loc, 50, 50, left)
            overlay(img, *right_loc, 50, 50, right)

cv2.imwrite("Test_Filter.jpg", img)
cv2.imshow("Test", img)
cv2.waitKey(0)
