import io
import time

import numpy as numpy
from edgetpu.detection.engine import DetectionEngine


import cv2
from config import Config

class VideoCamera(object):
    def __init__(self):
        with open(Config.LABEL_PATH, 'r', encoding="utf-8") as f:
            pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
            self.labels = dict((int(k), v) for k, v in pairs)
        self.engine = DetectionEngine(Config.MODEL_PATH)

        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        if self.video:
            self.video.set(3, 640)
            self.video.set(4, 480)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 470)
        fontScale              = 0.6
        fontColor              = (255,255,255)
        lineType               = 2

        annotate_text = ""
        annotate_text_time = time.time()
        time_to_show_prediction = 1.0

        _, width, height, channels = self.engine.get_input_tensor_shape()
        if not self.video.isOpened():
            print('Camera is not opened')
        ret, img = self.video.read()
        if not ret:
            print('Camera is not read')
        input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input = cv2.resize(input, (width, height))
        input = input.reshape((width * height * channels))
        rows = img.shape[0]
        cols = img.shape[1]
        start_ms = time.time()
        #############
        # Run inference.
        ans = self.engine.DetectWithInputTensor(input, threshold=Config.DETECT_THRESHOLD,
            top_k=Config.TOP_K)
        elapsed_ms = time.time() - start_ms
        # Display result.
        if ans:
            for obj in ans:
                box = obj.bounding_box.flatten().tolist()
                #print ('id=', obj.label_id, 'score = ', obj.score, 'box = ', box)
                # Draw a rectangle.
                x = box[0] * cols
                y = box[1] * rows
                right = box[2] * cols
                bottom = box[3] * rows
                if obj.score > Config.DETECT_THRESHOLD:
                    cv2.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
                    annotate_text = "%s %.2f  %.2fms" % (
                        self.labels[obj.label_id], obj.score, elapsed_ms*1000.0)
                    annotate_text_time = time.time()
                    cv2.putText(img, annotate_text, 
                        (int(x), int(bottom)), 
                        font, 
                        fontScale,
                        fontColor,
                        lineType)
                        
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()
