from flask import Flask, request, render_template, Response
from flask_cors import CORS
import numpy as np
import cv2
import numpy as np
from PIL import Image

from face_detector import YoloDetector

app = Flask(__name__)

model = YoloDetector(target_size=720, device="cuda:0", min_face=90)

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

def gen_frames(frame):
    # print(frame)
    bboxes, points = model(frame)
    process_base(frame, bboxes[0], points[0])
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame = pil_image.tobytes()
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
    bboxes, points = model(image)
    process_base(image, bboxes[0], points[0])
    processed_pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGBA))
    processed_frame = processed_pil_image.tobytes()
    # return processed_pil_image # Response(processed_pil_image, mimetype='multipart/x-mixed-replace; boundary=frame')
    # image.save(image_path)
    # return Response(gen_frames(image), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)