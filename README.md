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
2. Download pretrained Edge TPU sample models using: sh download_models.sh
3. Run "python3 main.py", or run "sh ./start.sh" in the background.
4. Navigate the browser to the http://ip-adress:5000.

### Switch models
1. Switch between detection/classification by modifying main.py imports
2. Change the model used for detection/classification by modifying the config.py

### Reference
1. https://github.com/log0/video_streaming_with_flask_example
2. https://coral.withgoogle.com/docs/edgetpu/api-intro
