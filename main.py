from face_detector import YoloDetector
import numpy as np
import cv2

# Issue Solved: occlusion(indirectly using aspect ratio),
#               two or more objects,
#               size (using aspect ratio)
#               range over (crop overlay_image)
# Issue Solving: rotation

def overlay(image, x, y, w, h, overlay_image):
    # image's cordinate
    y_start, y_end = max(0, y - h), min(image.shape[0], y + h) # image's shape !!: y, x
    x_start, x_end = max(0, x - w), min(image.shape[1], x + w) # image's shape !!: y, x
    # crop overlay image
    overlay_y_start, overlay_y_end = max(0, 0 - (y - h)), min(h * 2, h * 2 + (image.shape[0] - (y + h)))
    overlay_x_start, overlay_x_end = max(0, 0 - (x - w)), min(w * 2, w * 2 + (image.shape[1] - (x + w)))
    overlay_image = overlay_image[overlay_y_start:overlay_y_end, overlay_x_start:overlay_x_end, :]
    # print(overlay_image.shape, w) # Debug
    # for alpha
    masked_image = overlay_image[:, :, 3] / 255
    for c in range(0, 3): # BGR
        image[y_start:y_end, x_start:x_end, c] = (overlay_image[:, :, c] * masked_image) + (image[y_start:y_end, x_start:x_end, c] * (1-masked_image))

# upload all related filters
left_init = cv2.imread('filter/filter_left.png', cv2.IMREAD_UNCHANGED)
right_init = cv2.imread('filter/filter_right.png', cv2.IMREAD_UNCHANGED)
center_init = cv2.imread('filter/filter_center.png', cv2.IMREAD_UNCHANGED)

video_capture = cv2.VideoCapture(0)
video_capture.set(3, 640)
video_capture.set(4, 480)

def process_base(frame, bboxes, points):
    for box,landmark in zip(bboxes, points): # rkr tkfka djfrnfakek filter size ekfma
        x1, y1, x2, y2 = box
        frame = cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0), 3)
        for x, y in landmark:
            frame = cv2.circle(frame, (x, y), 3, (0,255,0), 1)

# reference: https://medium.com/analytics-vidhya/eye-aspect-ratio-ear-and-drowsiness-detector-using-dlib-a0b2c292d706
def process_side(frame, bboxes, points):
    # rotation wjrdyd
    for box,landmark in zip(bboxes, points): # rkr tkfka djfrnfakek filter size ekfma
        x1, y1, x2, y2 = box
        left_eye, right_eye, nose, left_mouse, right_mouse = landmark
        face_aspect_ratio = np.linalg.norm(right_eye[0] - left_eye[0]) / np.linalg.norm(y2 - y1) #  Euclidean distance is the l2 norm 
        # print(face_aspect_ratio)
        if face_aspect_ratio > 0.2 or (right_eye[0] <= left_eye[0]):
            img_size = int((x2 - x1) / 2 / 2) # set size through test
            if img_size < 2:
                img_size = 2
            
            left = cv2.resize(left_init, (img_size * 2, img_size * 2))
            right = cv2.resize(right_init, (img_size * 2, img_size * 2))
            left_loc = [(int)(left_eye[0] - img_size), # (box[0] + (int)((box[0] - left_eye[0]) * 0.8)), # soqnswja
                        (int)(left_eye[1] - img_size/4)]
            right_loc = [(int)(right_eye[0] + img_size), # (box[0] + (int)((box[0] - right_eye[0]) * 0.8)), # soqnswja
                        (int)(right_eye[1] - img_size/4)]
            overlay(frame, *left_loc, img_size, img_size, left)
            overlay(frame, *right_loc, img_size, img_size, right)

def process_center(frame, bboxes, points):
    for box,landmark in zip(bboxes, points):
        x1, y1, x2, y2 = box
        left_eye, right_eye, nose, left_mouse, right_mouse = landmark
        face_aspect_ratio = np.linalg.norm(right_eye[0] - left_eye[0]) / np.linalg.norm(y2 - y1) #  Euclidean distance is the l2 norm 
        # print(face_aspect_ratio)
        if face_aspect_ratio > 0.2 or (right_eye[0] <= left_eye[0]):
            img_size = int((x2 - x1) / 4 / 2)
            if img_size < 2:
                img_size = 2

            center = cv2.resize(center_init, (img_size * 2, img_size * 2))
            center_loc = [(int)((left_eye[0] + right_eye[0]) / 2), 
                        (int)(y1 - img_size)]
            overlay(frame, *center_loc, img_size, img_size, center)

while True:
    ret, frame = video_capture.read()

    model = YoloDetector(target_size=640, device="cuda:0", min_face=90)
    bboxes, points = model.predict(frame)

    process_base(frame, bboxes[0], points[0])
    # process_side(frame, bboxes[0], points[0])
    # process_center(frame, bboxes[0], points[0])

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
