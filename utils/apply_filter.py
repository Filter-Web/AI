# https://ejleep1.tistory.com/1003
# https://velog.io/@bangsy/Python-OpenCV3
# https://tech.kakaopay.com/post/image-processing-server-framework/
# https://wikidocs.net/215181
import cv2
from face_detector import YoloDetector
import numpy as np
import imutils

def overlay(image, x, y, w, h, overlay_image, alpha):
    # image's cordinate
    y_start, y_end = max(0, y - h), min(image.shape[0], y + h) # image's shape !!: y, x
    x_start, x_end = max(0, x - w), min(image.shape[1], x + w) # image's shape !!: y, x
    # crop overlay image
    overlay_y_start, overlay_y_end = max(0, 0 - (y - h)), min(h * 2, h * 2 + (image.shape[0] - (y + h)))
    overlay_x_start, overlay_x_end = max(0, 0 - (x - w)), min(w * 2, w * 2 + (image.shape[1] - (x + w)))
    overlay_image = overlay_image[overlay_y_start:overlay_y_end, overlay_x_start:overlay_x_end, :]
    # print(overlay_image.shape, w) # Debug
    # for alpha
    masked_image = (overlay_image[:, :, 3] / 255) * alpha
    for c in range(0, 3): # BGR
        image[y_start:y_end, x_start:x_end, c] = (overlay_image[:, :, c] * masked_image) + (image[y_start:y_end, x_start:x_end, c] * (1-masked_image))

# upload all related filters
leftBeard_init = cv2.imread('filter/leftBeard.png', cv2.IMREAD_UNCHANGED)
rightBeard_init = cv2.imread('filter/rightBeard.png', cv2.IMREAD_UNCHANGED)
center_init = cv2.imread('filter/SET_suryongFace.png', cv2.IMREAD_UNCHANGED)
starMoon_init = cv2.imread('filter/starMoon.png', cv2.IMREAD_UNCHANGED)
footPrint_init = imutils.rotate_bound(cv2.imread('filter/footPrint.png', cv2.IMREAD_UNCHANGED), 20)
rainbow_init = cv2.imread('filter/rainbow.png', cv2.IMREAD_UNCHANGED)
crystalBall_init = cv2.imread('filter/crystalBall.png', cv2.IMREAD_UNCHANGED)
brightSungshin_init = imutils.rotate_bound(cv2.imread('filter/brightSungshin.png', cv2.IMREAD_UNCHANGED), -20)
leftHeart_init = cv2.imread('filter/SET_heart.png', cv2.IMREAD_UNCHANGED)
rightSuryong_init = cv2.imread('filter/SET_suryongFace.png', cv2.IMREAD_UNCHANGED)
cloudSuryong_init = cv2.imread('filter/cloudSuryong.png', cv2.IMREAD_UNCHANGED)

def process_base(frame, bboxes, points):
    for box,landmark in zip(bboxes, points): # rkr tkfka djfrnfakek filter size ekfma
        x1, y1, x2, y2 = box
        frame = cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0), 3)
        for x, y in landmark:
            frame = cv2.circle(frame, (x, y), 3, (0,255,0), 1)

# reference: https://medium.com/analytics-vidhya/eye-aspect-ratio-ear-and-drowsiness-detector-using-dlib-a0b2c292d706
def process_side(frame, bboxes, points): # outside of two ears
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

            left = cv2.resize(leftBeard_init, (img_size * 2, img_size * 2))
            right = cv2.resize(rightBeard_init, (img_size * 2, img_size * 2))
            left_loc = [(int)(left_eye[0] - img_size), # (box[0] + (int)((box[0] - left_eye[0]) * 0.8)), # soqnswja
                        (int)(left_eye[1] - img_size/4)]
            right_loc = [(int)(right_eye[0] + img_size), # (box[0] + (int)((box[0] - right_eye[0]) * 0.8)), # soqnswja
                        (int)(right_eye[1] - img_size/4)]
            overlay(frame, *left_loc, img_size, img_size, left, 0.8)
            overlay(frame, *right_loc, img_size, img_size, right, 0.8)

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
            overlay(frame, *center_loc, img_size, img_size, center, 0.8)

def process_starMoon(frame, bboxes, points):
    for box,landmark in zip(bboxes, points):
        x1, y1, x2, y2 = box
        left_eye, right_eye, nose, left_mouse, right_mouse = landmark
        face_aspect_ratio = np.linalg.norm(right_eye[0] - left_eye[0]) / np.linalg.norm(y2 - y1) #  Euclidean distance is the l2 norm
        # print(face_aspect_ratio)
        if face_aspect_ratio > 0.2 or (right_eye[0] <= left_eye[0]):
            img_size = int((x2 - x1) / 4)
            if img_size < 2:
                img_size = 2

            center = cv2.resize(starMoon_init, (img_size * 2, img_size * 2))
            center_loc = [(int)(right_eye[0] + img_size),
                        (int)((right_eye[1] + img_size/1.2))]
            overlay(frame, *center_loc, img_size, img_size, center, 0.8)

def process_footPrint(frame, bboxes, points):
    for box,landmark in zip(bboxes, points):
        x1, y1, x2, y2 = box
        left_eye, right_eye, nose, left_mouse, right_mouse = landmark
        face_aspect_ratio = np.linalg.norm(right_eye[0] - left_eye[0]) / np.linalg.norm(y2 - y1) #  Euclidean distance is the l2 norm
        # print(face_aspect_ratio)
        if face_aspect_ratio > 0.2 or (right_eye[0] <= left_eye[0]):
            img_size = int((x2 - x1) / 1.5)
            if img_size < 2:
                img_size = 2

            center = cv2.resize(footPrint_init, (img_size * 2, img_size * 2))
            center_loc = [(int)(x1 + img_size * 2),
                        (int)(right_eye[1] - img_size / 4)]
            overlay(frame, *center_loc, img_size, img_size, center, 0.8)

def process_rainbow(frame, bboxes, points):
    for box,landmark in zip(bboxes, points):
        x1, y1, x2, y2 = box
        left_eye, right_eye, nose, left_mouse, right_mouse = landmark
        face_aspect_ratio = np.linalg.norm(right_eye[0] - left_eye[0]) / np.linalg.norm(y2 - y1) #  Euclidean distance is the l2 norm
        # print(face_aspect_ratio)
        if face_aspect_ratio > 0.2 or (right_eye[0] <= left_eye[0]):
            img_size = int((x2 - x1) / 9)
            if img_size < 2:
                img_size = 2

            center = cv2.resize(rainbow_init, (img_size * 2, img_size * 2))
            center_loc = [(int)(right_eye[0] + img_size),
                        (int)(right_eye[1] + img_size * 1.5)]
            overlay(frame, *center_loc, img_size, img_size, center, 0.6)

def process_crystalBall(frame, bboxes, points):
    for box,landmark in zip(bboxes, points):
        x1, y1, x2, y2 = box
        left_eye, right_eye, nose, left_mouse, right_mouse = landmark
        face_aspect_ratio = np.linalg.norm(right_eye[0] - left_eye[0]) / np.linalg.norm(y2 - y1) #  Euclidean distance is the l2 norm
        # print(face_aspect_ratio)
        if face_aspect_ratio > 0.3 or (right_eye[0] <= left_eye[0]):
            img_size = int((x2 - x1) / 6)
            if img_size < 2:
                img_size = 2

            center = cv2.resize(crystalBall_init, (img_size * 2, img_size * 2))
            center_loc = [(int)(right_eye[0] + img_size/6),
                        (int)(right_eye[1] + img_size * 1.2)]
            overlay(frame, *center_loc, img_size, img_size, center, 1)

def process_brightSungshin(frame, bboxes, points):
    for box,landmark in zip(bboxes, points):
        x1, y1, x2, y2 = box
        left_eye, right_eye, nose, left_mouse, right_mouse = landmark
        face_aspect_ratio = np.linalg.norm(right_eye[0] - left_eye[0]) / np.linalg.norm(y2 - y1) #  Euclidean distance is the l2 norm
        # print(face_aspect_ratio)
        if face_aspect_ratio > 0.2 or (right_eye[0] <= left_eye[0]):
            img_size = int((x2 - x1) * 1.2)
            if img_size < 2:
                img_size = 2

            center = cv2.resize(brightSungshin_init, (img_size * 2, img_size * 2))
            center_loc = [(int)(left_eye[0] + img_size/8),
                        (int)(y1 - img_size/4)]
            overlay(frame, *center_loc, img_size, img_size, center, 0.8)

def process_eyes(frame, bboxes, points): # outside of two ears
    # rotation wjrdyd
    for box,landmark in zip(bboxes, points): # rkr tkfka djfrnfakek filter size ekfma
        x1, y1, x2, y2 = box
        left_eye, right_eye, nose, left_mouse, right_mouse = landmark
        face_aspect_ratio = np.linalg.norm(right_eye[0] - left_eye[0]) / np.linalg.norm(y2 - y1) #  Euclidean distance is the l2 norm
        # print(face_aspect_ratio)
        if face_aspect_ratio > 0.2 or (right_eye[0] <= left_eye[0]):
            img_size = int((x2 - x1) / 20) # set size through test
            if img_size < 2:
                img_size = 2

            left = cv2.resize(leftHeart_init, (img_size * 2, img_size * 2))
            right = cv2.resize(leftHeart_init, (img_size * 2, img_size * 2))
            left_loc = [(int)(left_eye[0]), # (box[0] + (int)((box[0] - left_eye[0]) * 0.8)), # soqnswja
                        (int)(left_eye[1])]
            right_loc = [(int)(right_eye[0]), # (box[0] + (int)((box[0] - right_eye[0]) * 0.8)), # soqnswja
                        (int)(right_eye[1])]
            overlay(frame, *left_loc, img_size, img_size, left, 0.5)
            overlay(frame, *right_loc, img_size, img_size, right, 0.5)

def process_cloudSuryong(frame, bboxes, points):
    for box,landmark in zip(bboxes, points):
        x1, y1, x2, y2 = box
        left_eye, right_eye, nose, left_mouse, right_mouse = landmark
        face_aspect_ratio = np.linalg.norm(right_eye[0] - left_eye[0]) / np.linalg.norm(y2 - y1) #  Euclidean distance is the l2 norm
        # print(face_aspect_ratio)
        if face_aspect_ratio > 0.2 or (right_eye[0] <= left_eye[0]):
            img_size = int((x2 - x1) * 0.9)
            if img_size < 2:
                img_size = 2

            center = cv2.resize(cloudSuryong_init, (img_size * 2, img_size * 2))
            center_loc = [(int)((x1 + x2) / 2),
                          (int)(y1 - img_size/2)]
            # center_loc = [(int)(left_eye[0] + img_size/8),
            #             (int)(y1 - img_size/4)]
            overlay(frame, *center_loc, img_size, img_size, center, 0.8)


