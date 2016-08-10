#!/usr/bin/env python
import sys
import time
import argparse
import lmdb
import os
import math

import cv2
import numpy as np

from caffe.proto import caffe_pb2
import caffe
import pandas as pd

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--inputFile',
                        required=True,
                        help='Fold file with labels.')
    parser.add_argument('-id', '--inputDirectory',
                        required=True,
                        help='Directry with images.')
    parser.add_argument('-o', '--outputFile',
                        required=True,
                        help='Output file.')

    args = parser.parse_args()
    return args

def createDescFile(inputFile, inputDirectory, outputFile):

    output=open(os.path.join(os.getcwd(), outputFile), 'w+')

    images = os.listdir(inputDirectory)
    with open(inputFile) as f:
        input = f.read().splitlines()

    for line in input:
        line = line.split()
        image = line[0].split('/')
        image = image[2].replace('jpg', 'png')
        if image in images:
            output.write('{}\n'.format(' '.join(line)))

    output.close()

def main():
    args = parse_args()
    createDescFile(args.inputFile, args.inputDirectory, args.outputFile)

if __name__ == "__main__":
    main()
