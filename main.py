from face_detector import YoloDetector
import numpy as np
import cv2

# cascPath = sys.argv[1]
# faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
video_capture.set(3, 640)
video_capture.set(4, 480)

while True:
    ret, frame = video_capture.read()

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    model = YoloDetector(target_size=720, device="cuda:0", min_face=90)
    # orgimg = cv2.imread('49_Greeting_peoplegreeting_49_25.jpg')
    bboxes,points = model.predict(frame)

    for box,landmark in zip(bboxes,points):
        for x1,y1,x2,y2 in box :
            frame = cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0), 3)
            for lm in landmark:
                for x, y in lm:
                    frame = cv2.circle(frame, (x, y), 3, (0,255,0), 1)

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
