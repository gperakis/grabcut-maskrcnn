from typing import Optional, NoReturn, Tuple

import cv2
import imutils
import numpy as np

from .config import ModelConf


class InstanceSegmenter:
    def __init__(self, use_gpu: bool = False):
        """
        :param use_gpu: False
            Whether to use GPU during the inference phase
        """
        self.use_gpu = use_gpu

        self._model: Optional[cv2.dnn_Net] = None
        self.labels_ = None

    @property
    def model(self) -> cv2.dnn_Net:
        if self._model is None:
            self._model = self.load_model()

        return self._model

    @property
    def labels(self):
        if self.labels_ is None:
            # load the COCO class labels our Mask R-CNN was trained on
            with open(ModelConf.LABELS_PATH) as f:
                self.labels_ = f.read().strip().split("\n")

        return self.labels_

    def load_model(self) -> cv2.dnn_Net:
        """
        Loads the pre-trained Mask R-CNN model
        :return:
        """
        # load Mask R-CNN trained on the COCO dataset (90 classes) from disk
        print("[INFO] loading Mask R-CNN from disk...")
        net = cv2.dnn.readNetFromTensorflow(ModelConf.WEIGHTS_PATH,
                                            ModelConf.CONFIG_PATH)

        # check if we are going to use GPU
        if self.use_gpu:
            # set CUDA as the preferable backend and target
            print("[INFO] setting preferable backend and target to CUDA...")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        return net

    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        constructs a blob from the input image and then perform a forward pass
        of the Mask R-CNN, giving us
        (1) the bounding box coordinates of the objects in the image along with
        (2) the pixel-wise segmentation for each specific object

        :param image: np.ndarray
            An image as a numpy multidimensional array

        :return: Tuple[np.ndarray, np.ndarray]
            A tuple containing the predicted boxes and their masks.
        """
        image_blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)

        self.model.setInput(image_blob)
        boxes, masks = self.model.forward([ModelConf.BOXES_OUTPUT_LAYER,
                                           ModelConf.MASKS_OUTPUT_LAYER])

        return boxes, masks

    def segment_image(self, image: str, confidence: float = 0.5,
                      threshold: float = 0.3, grabcut_iter: int = 10,
                      wait: int = 6) -> NoReturn:
        """
        :param image: str
            path to input image
        :param confidence:
            minimum probability to filter weak detections. Default is 0.5
        :param threshold: float
            minimum threshold for pixel-wise mask segmentation. Default is 0.3
        :param grabcut_iter: int
            number of GrabCut iterations (larger value => slower runtime)
        :param wait: int
            The time in seconds to weight to show the next predicted label

        :return: NoReturn
        """
        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(self.labels), 3),
                                   dtype="uint8")

        # load our input image from disk and display it to our screen
        image = cv2.imread(image)
        image = imutils.resize(image, width=600)
        cv2.imshow("Input", image)

        boxes, masks = self.predict(image)

        n_boxes = boxes.shape[2]

        # loop over the number of detected objects
        for i in range(0, n_boxes):
            # extract the class ID of the detection along with the confidence
            # (i.e., probability) associated with the prediction
            class_id = int(boxes[0, 0, i, 1])
            pred_confidence = boxes[0, 0, i, 2]

            # filter out weak predictions by ensuring the detected probability
            # is greater than the minimum probability
            if pred_confidence > confidence:
                # show the class label
                print(f"[INFO] showing output for '{self.labels[class_id]}'")
                print(f'[INFO] label confidence: {pred_confidence}',
                      end='\n\n')

                # scale the bounding box coordinates back relative to the size
                # of the image and then compute the width and the height of the
                # bounding box
                (h, w) = image.shape[:2]
                box = boxes[0, 0, i, 3:7] * np.array([w, h, w, h])
                (start_x, start_y, end_x, end_y) = box.astype("int")
                box_width = end_x - start_x
                box_height = end_y - start_y

                # extract the pixel-wise segmentation for the object, resize
                # the mask such that it's the same dimensions as the bounding
                # box, and then finally threshold to create a *binary* mask
                mask = masks[i, class_id]
                mask = cv2.resize(mask, (box_width, box_height),
                                  interpolation=cv2.INTER_CUBIC)
                mask = (mask > threshold).astype("uint8") * 255

                # allocate a memory for our output Mask R-CNN mask and store
                # the predicted Mask R-CNN mask in the GrabCut mask
                rcnn_mask = np.zeros(image.shape[:2], dtype="uint8")
                rcnn_mask[start_y:end_y, start_x:end_x] = mask

                # apply a bitwise AND to the input image to show the output
                # of applying the Mask R-CNN mask to the image
                rcnn_output = cv2.bitwise_and(image, image, mask=rcnn_mask)

                # show the output of the Mask R-CNN and bitwise AND operation
                cv2.imshow("R-CNN Mask", rcnn_mask)
                cv2.imshow("R-CNN Output", rcnn_output)
                cv2.waitKey(wait * 1000)

                # clone the Mask R-CNN mask (so we can use it when applying
                # GrabCut) and set any mask values greater than zero to be
                # "probable foreground"
                # (otherwise they are "definite background")
                gc_mask = rcnn_mask.copy()
                gc_mask[gc_mask > 0] = cv2.GC_PR_FGD
                gc_mask[gc_mask == 0] = cv2.GC_BGD

                # allocate memory for two arrays that the GrabCut algorithm
                # internally uses when segmenting the foreground from the
                # background and then apply GrabCut using the mask segmentation
                # method
                print(f"[INFO] applying GrabCut "
                      f"to '{self.labels[class_id]}' ROI...")
                fg_model = np.zeros((1, 65), dtype="float")
                bg_model = np.zeros((1, 65), dtype="float")

                (gc_mask, bg_model, fg_model) = cv2.grabCut(
                    image, gc_mask, None, bg_model, fg_model,
                    iterCount=grabcut_iter, mode=cv2.GC_INIT_WITH_MASK)

                # set all definite background and probable background pixels to
                # 0 while definite foreground and probable foreground pixels
                # are set to 1, then scale the mask from the range [0, 1] to
                # [0, 255]
                output_mask = np.where(
                    (gc_mask == cv2.GC_BGD) | (gc_mask == cv2.GC_PR_BGD), 0, 1)
                output_mask = (output_mask * 255).astype("uint8")

                # apply a bitwise AND to the image using our mask generated
                # by GrabCut to generate our final output image
                output = cv2.bitwise_and(image, image, mask=output_mask)

                # show the output GrabCut mask as well as the output of
                # applying the GrabCut mask to the original input image
                cv2.imshow("GrabCut Mask", output_mask)
                cv2.imshow("Output", output)
                cv2.waitKey(wait * 1000)
