# app.py
from flask import Flask, render_template, Response
import cv2
from face_detector import YoloDetector

app = Flask(__name__)

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)/
model = YoloDetector(target_size=720, device="cuda:0", min_face=90)

video_path = 1 + cv2.CAP_DSHOW # "/home/lee52/Desktop/capstone/yoloface/flask/video/web_test.avi"
# video_path = "./flask/video/web_test.avi"

@app.route('/')
def video_show():
    return render_template('video_show.html')

def gen_frames(cap):
    while True:
        _, frame = cap.read()
        if not _:
            break
        else:
            bboxes, points = model(frame)
            for bbox,landmark in zip(bboxes,points):
                for x1,y1,x2,y2 in bbox :
                    results = cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0), 3)
                    for lm in landmark:
                        for x, y in lm:
                            results = cv2.circle(results, (x, y), 3, (0,255,0), 1)
            annotated_frame = results.render()
            
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    cap = cv2.VideoCapture(video_path)
    return Response(gen_frames(cap), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)