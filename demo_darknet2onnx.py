import sys
import onnx
import os
import argparse
import numpy as np
import cv2
import onnxruntime

from tool.utils import *
from tool.darknet2onnx import *


def main(cfg_file, weight_file, image_path, batch_size, namesfile, num_classes):

    if batch_size <= 0:
        onnx_path_demo = transform_to_onnx(cfg_file, weight_file, batch_size)
    else:
        # Transform to onnx as specified batch size
        transform_to_onnx(cfg_file, weight_file, batch_size)
        # Transform to onnx as demo
        onnx_path_demo = transform_to_onnx(cfg_file, weight_file, 1)

    session = onnxruntime.InferenceSession(onnx_path_demo)
    # session = onnx.load(onnx_path)
    print("The model expects input shape: ", session.get_inputs()[0].shape)

    image_src = cv2.imread(image_path)
    detect(session, image_src, namesfile, num_classes)



def detect(session, image_src, namesfile, num_classes):
    IN_IMAGE_H = session.get_inputs()[0].shape[2]
    IN_IMAGE_W = session.get_inputs()[0].shape[3]

    # Input
    resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.array(img_in) / 255.0
    print('cvt color shape:{}'.format(img_in.shape))
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0
    print("Shape of the network input: ", img_in.shape)

    # Compute
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    for i in range(10):
        print('warming up')
        outputs = session.run(None, {input_name: np.array(img_in)})
        

    start = time.time()
    outputs = session.run(None, {input_name: np.array(img_in)})
    end = time.time()
    print('elapse {}'.format(end - start))
    print(outputs)
    boxes = post_processing(img_in, 0.4, 0.6, outputs)

    num_classes = 80
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = namesfile

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(image_src, boxes[0], savename='predictions_onnx.jpg', class_names=class_names)



if __name__ == '__main__':
    print("Converting to onnx and running demo ...")
    if len(sys.argv) == 7:
        cfg_file = sys.argv[1]
        weight_file = sys.argv[2]
        image_path = sys.argv[3]
        batch_size = int(sys.argv[4])
        namesfile = sys.argv[5]
        num_classes = int(sys.argv[6])
        main(cfg_file, weight_file, image_path, batch_size, namesfile, num_classes)
    else:
        print('Please run this way:\n')
        print('  python demo_darknet2onnx.py <cfgFile> <weightFile> <imageFile> <batchSize> <namesfile> <num_classes>')
