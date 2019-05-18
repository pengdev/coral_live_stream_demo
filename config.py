class Config:
    DATA_DIR ='./all_models/'
#   For classification
#    MODEL_PATH = DATA_DIR + 'mobilenet_v2_1.0_224_quant_edgetpu.tflite'
#    LABEL_PATH = DATA_DIR + 'imagenet_labels.txt'

#   For face detection
#    MODEL_PATH = DATA_DIR + 'mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite'

#   For object detection
#    MODEL_PATH = DATA_DIR + 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
#    LABEL_PATH = DATA_DIR + 'coco_labels.txt'

#   For classify plants
#    MODEL_PATH = DATA_DIR + 'mobilenet_v2_1.0_224_inat_plant_quant_edgetpu.tflite'
#    LABEL_PATH = DATA_DIR + 'inat_plant_labels.txt'
    
    MODEL_PATH = DATA_DIR + 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    LABEL_PATH = DATA_DIR + 'coco_labels.txt'
    
    DETECT_THRESHOLD = 0.1
    TOP_K = 1
