## Coral object classification model live stream

### Description
Stream USB camera with Coral Edge TPU based object detection/classification over network.
The project is dependent on Google's Coral USB Accelerator.

### Credits
Base code took from log0's video_streaming_with_flask_example. 
Coral object classification originally from Google LLC.

### Usage
0. Install the Edge TPU runtime and Python library following 
https://coral.withgoogle.com/docs/accelerator/get-started/
1. Install Python dependencies: cv2, flask.
2. Run "python3 main.py", or run "sh ./start.sh".
3. Navigate the browser to the http://ip-adress:5000.

### Reference
1. https://github.com/log0/video_streaming_with_flask_example
2. https://coral.withgoogle.com/docs/edgetpu/api-intro
