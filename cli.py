import argparse
from src.segment import InstanceSegmenter

# construct the argument parse and parse the arguments
parser = argparse.ArgumentParser()

parser.add_argument("-i", "--image", required=True,
                    help="path to input image")

parser.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")

parser.add_argument("-t", "--threshold", type=float, default=0.3,
                    help="minimum threshold for pixel-wise mask segmentation")
parser.add_argument("-u", "--use-gpu", type=bool, default=0,
                    help="boolean indicating if CUDA GPU should be used")

parser.add_argument("-e", "--iter", type=int, default=10,
                    help="# of GrabCut iterations "
                         "(larger value => slower runtime)")

args = vars(parser.parse_args())

use_gpu = args["use_gpu"]
image = args["image"]
confidence = args["confidence"]
threshold = args["threshold"]
grabcut_iter = args["iter"]

if __name__ == "__main__":
    segmenter = InstanceSegmenter(use_gpu)
    segmenter.segment_image(image, confidence, threshold, grabcut_iter)
