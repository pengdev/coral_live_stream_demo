class Config:
    DATA_DIR ='./all_models/'
#    MODEL_PATH = DATA_DIR + 'mobilenet_v2_1.0_224_quant_edgetpu.tflite'
    MODEL_PATH = DATA_DIR + 'mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite'
    LABEL_PATH = DATA_DIR + 'imagenet_labels.txt'
    DETECT_THRESHOLD = 0.2
    TOP_K = 3
