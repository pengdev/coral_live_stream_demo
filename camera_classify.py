import io
import time

import numpy as numpy
import edgetpu.classification.engine

import cv2
from config import Config

class VideoCamera(object):
    def __init__(self):
        with open(Config.LABEL_PATH, 'r', encoding="utf-8") as f:
            pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
            self.labels = dict((int(k), v) for k, v in pairs)
        self.engine = edgetpu.classification.engine.ClassificationEngine(Config.MODEL_PATH)

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
        print('closing camera')
        self.video.release()
    
    def get_frame(self):
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 470)
        fontScale              = 0.6
        fontColor              = (255,255,255)
        lineType               = 2

        annotate_text = ""

        _, width, height, channels = self.engine.get_input_tensor_shape()
        if not self.video.isOpened():
            print('Camera is not opened')
        ret, img = self.video.read()
        if not ret:
            print('Camera is not read')
        input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input = cv2.resize(input, (width, height))
        input = input.reshape((width * height * channels))
        start_ms = time.time()
        results = self.engine.ClassifyWithInputTensor(input, top_k=Config.TOP_K)
        elapsed_ms = time.time() - start_ms
        # if results:
        #     print( "%s %.2f\n%.2fms" % (self.labels[results[0][0]], results[0][1], elapsed_ms*1000.0))
        if results and\
                    results[0][1] > Config.DETECT_THRESHOLD:
            annotate_text = "%s %.2f  %.2fms" % (
                self.labels[results[0][0]], results[0][1], elapsed_ms*1000.0)
                    
            cv2.putText(img, annotate_text, 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()
