# app.py
from flask import Flask, render_template, Response, request
from flask_cors import CORS
import cv2
from face_detector import YoloDetector
import numpy as np
from PIL import Image

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)/
model = YoloDetector(target_size=720, device="cuda:0", min_face=90)

video_path = 0 # "/dev/video0" # 0 # 1 + cv2.CAP_DSHOW # "/home/lee52/Desktop/capstone/yoloface/flask/video/web_test.avi"
# video_path = "./flask/video/web_test.avi"

results = None

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

def gen_frames(frame):
    # print(frame)
    bboxes, points = model(frame)
    process_base(frame, bboxes[0], points[0])
    processed_pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
    frame = processed_pil_image.tobytes()
    yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def root():
    return 'welcome to flask'

@app.route('/api/receiveWebcamStream', methods=['POST'])
def receive_webcam_stream():
    webcam_data = request.data
    # webcam_data를 처리하는 로직 추가
    pil_image = Image.frombytes('RGBA', (1280, 720), webcam_data)
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGR)
    image_path = 'received_image.png'
    cv2.imwrite(image_path, image)

    # generate
    return Response(gen_frames(image), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)