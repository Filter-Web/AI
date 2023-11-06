# app.py
# https://ejleep1.tistory.com/1003
from flask import Flask, render_template, Response
import cv2
from face_detector import YoloDetector
import time
import datetime
import sys

app = Flask(__name__)

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)/
model = YoloDetector(target_size=720, device="cuda:0", min_face=90)

video_path = 0 # "/dev/video0" # 0 # 1 + cv2.CAP_DSHOW # "/home/lee52/Desktop/capstone/yoloface/flask/video/web_test.avi"
# video_path = "./flask/video/web_test.avi"

results = None

@app.route('/')
def index():
    """Video streaming home page."""
    now = datetime.datetime.now()
    timeString = now.strftime("%Y-%m-%d %H:%M")
    templateData = {
        'title':'Image Streaming',
        'time': timeString
        }
    return render_template('index.html', **templateData)

def gen_frames():
    camera = cv2.VideoCapture(0)
    time.sleep(0.2)
    lastTime = time.time()*1000.0

    while True:
        _, frame = camera.read()
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bboxes, points = model(frame)
        print(bboxes)
        delt = time.time()*1000.0-lastTime
        s = str(int(delt))
        #print (delt," Found {0} faces!".format(len(faces)) )
        lastTime = time.time()*1000.0
        # Draw a rectangle around the faces
        for bbox,landmark in zip(bboxes,points):
                for x1,y1,x2,y2 in bbox :
                    frame = cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0), 3)
                    for lm in landmark:
                        for x, y in lm:
                            frame = cv2.circle(frame, (x, y), 3, (0,255,0), 1)

        now = datetime.datetime.now()
        timeString = now.strftime("%Y-%m-%d %H:%M")
        cv2.putText(frame, timeString, (10, 45),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        # cv2.imshow("Frame", frame)
        # key = cv2.waitKey(1) & 0xFF # if the `q` key was pressed, break from the loop
        # if key == ord("q"):
        #     break

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0')