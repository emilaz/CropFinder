# *******************************************************************
#
# Author : Thanh Nguyen, 2018
# Email  : sthanhng@gmail.com
# Github : https://github.com/sthanhng
#
# BAP, AI Team
# Face detection using the YOLOv3 algorithm
#
# Description : dont_use_me.py
# The main code of the Face detection using the YOLOv3 algorithm
#
# *******************************************************************

# Usage example:  python dont_use_me.py --image samples/outside_000001.jpg \
#                                    --output-dir outputs/
#                 python dont_use_me.py --video samples/subway.mp4 \
#                                    --output-dir outputs/
#                 python dont_use_me.py --src 1 --output-dir outputs/


import argparse
import sys
import os
from time import time

from utils import *


def init_model():
    model_cfg = './yolov3-face.cfg'
    model_weights = './yolov3-wider_16000.weights'
    # init the network
    net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def get_face_coordinates(images):
    start = time()
    net = init_model()

    if not len(os.listdir(os.path.dirname(images))):
        print("[!] ==> Input images {} don't exist".format(images))
        sys.exit(1)
    cap = cv2.VideoCapture(images)
    all_faces = []
    while True:
        has_frame, frame = cap.read()

        # Stop the program if reached end of video
        if not has_frame:
            break
        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                     [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(get_outputs_names(net))

        # Remove the bounding boxes with low confidence
        # returns [left, top, width, height] coordinates
        faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
        # THIS IS ALL WE NEED I THINK
        # print('[i] ==> # detected faces: {}'.format(len(faces)))
        # print('#' * 60)
        if len(faces)>0:
            all_faces.append(faces[0])

    # print('==> Done with obtaining face coordinates! Elapsed:{}'.format(time() - start))
    if len(all_faces) == 0:
        print('No faces found. Returning standard parameters')
        all_faces = [[50, 0, 540, 380]]
    return all_faces


if __name__ == '__main__':
    get_face_coordinates(sys.argv)
