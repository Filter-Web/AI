from face_detector import YoloDetector
import numpy as np
from PIL import Image
import cv2

def overlay(image, x, y, w, h, overlay_image):
    alpha = overlay_image[:, :, 3]
    masked_image = alpha / 255
    for c in range(0, 3): # BGR
        image[y-h:y+h, x-w:x+w, c] = (overlay_image[:, :, c] * masked_image) + (image[y-h:y+h, x-w:x+w, c] * (1-masked_image))

left_init = cv2.imread('filter/filter_left.png', cv2.IMREAD_UNCHANGED)
left_init = cv2.resize(left_init, (100, 100))
right_init = cv2.imread('filter/filter_right.png', cv2.IMREAD_UNCHANGED)
right_init = cv2.resize(right_init, (100, 100))

model = YoloDetector(target_size=720, device="cuda:0", min_face=90)
# img = cv2.imread('capture_2.jpg')
img = cv2.imread('49_Greeting_peoplegreeting_49_25.jpg')
bboxes,points = model.predict(img)
print(bboxes, points)
for box,landmark in zip(bboxes[0], points[0]): # rkr tkfka djfrnfakek filter size ekfma
        x1, y1, x2, y2 = box
        left_eye, right_eye, nose, left_mouse, right_mouse = landmark
        face_aspect_ratio = np.linalg.norm(right_eye[0] - left_eye[0]) / np.linalg.norm(y2 - y1) #  Euclidean distance is the l2 norm 
        # print(face_aspect_ratio)
        if face_aspect_ratio > 0.2 or (right_eye[0] <= left_eye[0]):
            img_size = int((x2 - x1) / 2 / 2) # set size through test
            if img_size < 2:
                img_size = 2
            
            left_init = cv2.resize(left_init, (img_size * 2, img_size * 2))
            right_init = cv2.resize(right_init, (img_size * 2, img_size * 2))
            left_loc = [(int)(left_eye[0] - img_size), # (box[0] + (int)((box[0] - left_eye[0]) * 0.8)), # soqnswja
                        (int)(left_eye[1] - img_size/4)]
            right_loc = [(int)(right_eye[0] + img_size), # (box[0] + (int)((box[0] - right_eye[0]) * 0.8)), # soqnswja
                        (int)(right_eye[1] - img_size/4)]
            overlay(img, *left_loc, img_size, img_size, left_init)
            overlay(img, *right_loc, img_size, img_size, right_init)


cv2.imwrite("Test_Filter_axisZ.jpg", img)
cv2.imshow("Test", img)
cv2.waitKey(0)