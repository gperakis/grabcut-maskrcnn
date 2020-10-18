from pathlib import Path


class ModelConf:
    MODEL_DIR = Path('model')

    LABELS_FILENAME = 'object_detection_classes_coco.txt'

    # filenames to the Mask R-CNN weights and model configuration
    WEIGHTS_FILENAME = 'frozen_inference_graph.pb'
    CONFIG_FILENAME = 'mask_rcnn_inception_v2_coco.pbtxt'

    # model output layer names
    BOXES_OUTPUT_LAYER = 'detection_out_final'
    MASKS_OUTPUT_LAYER = 'detection_masks'

    WEIGHTS_PATH = str(MODEL_DIR.joinpath(WEIGHTS_FILENAME))
    CONFIG_PATH = str(MODEL_DIR.joinpath(CONFIG_FILENAME))

    LABELS_PATH = MODEL_DIR.joinpath(LABELS_FILENAME)
