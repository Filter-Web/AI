from flask import Flask, request, render_template, Response
import cv2
from face_detector import YoloDetector

app = Flask(__name__)

@app.route('/index')
def index():
  return render_template('index_2.html')

@app.route('/cam_post')
def cam_load():
    return render_template('cam.html')

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


@app.route('/cam')
def cam_post():
    cap = cv2.VideoCapture(1 + cv2.CAP_DSHOW) 
    return Response(gen_frames(cap), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__' :   
  app.run(host='0.0.0.0', port=8000, debug=True)
