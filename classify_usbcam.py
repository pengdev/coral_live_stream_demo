# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A demo to classify Raspberry Pi camera stream."""

import argparse
import io
import time

import numpy as np
import cv2

import edgetpu.classification.engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--model', help='File path of Tflite model.', required=True)
    parser.add_argument(
      '--label', help='File path of label file.', required=True)
    args = parser.parse_args()

    with open(args.label, 'r', encoding="utf-8") as f:
        pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
        labels = dict((int(k), v) for k, v in pairs)

    engine = edgetpu.classification.engine.ClassificationEngine(args.model)
    camera = cv2.VideoCapture(0)

    if camera:
        camera.set(3, 640)
        camera.set(4, 480)


        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 470)
        fontScale              = 0.6
        fontColor              = (255,255,255)
        lineType               = 2

        annotate_text = ""
        annotate_text_time = time.time()
        time_to_show_prediction = 3.0
        min_confidence = 0.2

        _, width, height, channels = engine.get_input_tensor_shape()
        try:
            while True:
                if not camera.isOpened():
                    continue
                ret, img = camera.read()
                if not ret:
                    continue
                input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                input = cv2.resize(input, (width, height))
                input = input.reshape((width * height * channels))
                start_ms = time.time()
                results = engine.ClassifyWithInputTensor(input, top_k=1)
                elapsed_ms = time.time() - start_ms

                if results:
                    print( "%s %.2f\n%.2fms" % (labels[results[0][0]], results[0][1], elapsed_ms*1000.0))
        finally:
            camera.release()

if __name__ == '__main__':
    main()