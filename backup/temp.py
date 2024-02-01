from face_detector import YoloDetector
import numpy as np
import cv2

# cascPath = sys.argv[1]
# faceCascade = cv2.CascadeClassifier(cascPath)

def overlay(image, x, y, w, h, overlay_image):
    alpha = overlay_image[:, :, 3]
    masked_image = alpha / 255
    for c in range(0, 3): # BGR
        image[y-h:y+h, x-w:x+w, c] = (overlay_image[:, :, c] * masked_image) + (image[y-h:y+h, x-w:x+w, c] * (1-masked_image))

# left = cv2.imread('filter_left.png', cv2.IMREAD_UNCHANGED)
# left = cv2.resize(left, (100, 100))
# right = cv2.imread('filter_right.png', cv2.IMREAD_UNCHANGED)
# right = cv2.resize(right, (100, 100))
center = cv2.imread('filter_head.png', cv2.IMREAD_UNCHANGED)
head = cv2.resize(center, (100, 100))

video_capture = cv2.VideoCapture(0)
video_capture.set(3, 640)
video_capture.set(4, 480)

while True:
    ret, frame = video_capture.read()

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    model = YoloDetector(target_size=720, device="cuda:0", min_face=90)
    # orgimg = cv2.imread('49_Greeting_peoplegreeting_49_25.jpg')
    bboxes,points = model.predict(frame)

    for box,landmark in zip(bboxes,points): # occlusion > issue, 2 objects, rotation
        for lm in landmark:
            left_eye, right_eye, nose, left_mouse, right_mouse = lm
            center_loc = left_eye
            center_loc[0] = (int)((left_eye[0] + right_eye[0]) / 2) #  + left_mouse[0]) / 2) - 40
            center_loc[1] = (int)(box[1]) # + left_mouse[1]) / 2)
            overlay(frame, *center_loc, 50, 50, center)
            # whdghldql

    # for box,landmark in zip(bboxes,points):
    #     for x1,y1,x2,y2 in box :
    #         frame = cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0), 3)
    #         for lm in landmark:
    #             for x, y in lm:
    #                 frame = cv2.circle(frame, (x, y), 3, (0,255,0), 1)

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
