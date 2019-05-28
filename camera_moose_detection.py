import io
import time
import random
import requests
import numpy as numpy
import subprocess
import threading
from edgetpu.detection.engine import DetectionEngine


import cv2
from config import Config

class VideoCamera(object):
    MOOSE_REPORT_URL = 'https://mooseetws.herokuapp.com/api/pi/v1/'
    LIGHT_POLE_ID = 1
    last_report = 0

    def __init__(self):
        print('starting camera')
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
        print('closing camera')
        self.video.release()

    def report_db(self, name, score):
        print('#### report to db: ', name, score, self.LIGHT_POLE_ID)
        payload = {'objectType': name,
                'confidence': float(score), 'poleId': self.LIGHT_POLE_ID}
        start = time.perf_counter()
        res = requests.post(self.MOOSE_REPORT_URL, json=payload)
        print('Request took', time.perf_counter()-start, 's')
        print('Response:', res.json())

    def report_moose(self, name, score):
        if time.time() - self.last_report > 10:
            report_db_thread = threading.Thread(target=self.report_db, args=(name, score))
            report_db_thread.start()
            subprocess.Popen(['python','grove_led_blink.py'])
            self.last_report = time.time()

    def get_frame(self):
        start_time = time.time()
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        topLeftCornerOfText = (10, 20)
        bottomLeftCornerOfText = (10, 470)
        fontScale              = 0.6
        fontColorWhite              = (160,160,160)
        fontColorRed              = (0,0,255)
        fontColorGreen              = (0,255,0)
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
        rows = img.shape[0]
        cols = img.shape[1]
        record_time = time.time()
        elapsed_record_ms = record_time - start_time
        #############
        # Run inference.
        ans = self.engine.DetectWithInputTensor(input, threshold=Config.DETECT_THRESHOLD,
            top_k=Config.TOP_K)
        detection_time = time.time()
        elapsed_detection_ms = detection_time - record_time
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
                    if obj.label_id > 16 and obj.label_id < 26:
                        cv2.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), 
                            fontColorRed, thickness=1)
                        self.report_moose(self.labels[obj.label_id], obj.score)
                    else:
                        cv2.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), 
                            fontColorGreen, thickness=1)
                    annotate_text = "%s %.2f" % (
                        self.labels[obj.label_id], obj.score)
                    annotate_text_time = time.time()
                    cv2.putText(img, annotate_text, 
                        (int(x), int(bottom)), 
                        font, 
                        fontScale,
                        fontColorWhite,
                        lineType)
        elapsed_frame_ms = (time.time() - start_time) * 1000.0
        frame_rate_text = "FPS: %.2f record: %.2fms detection: %.2fms" % (1000.0/elapsed_frame_ms,
            elapsed_record_ms * 1000.0, elapsed_detection_ms * 1000.0)
        cv2.putText(img, frame_rate_text,
                topLeftCornerOfText,
                font, fontScale,
                fontColorWhite, lineType)
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()
