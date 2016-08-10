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
    parser = argparse.ArgumentParser(epilog="Classify Emotions")

    parser.add_argument('-i', '--inputFile',
                        required=True,
                        help='Fold file with labels.')
    parser.add_argument('-o', '--outputFile',
                        required=True,
                        help='Output file.')

    args = parser.parse_args()
    return args

def createDescFile(inputFile, outputFile):

    output=open(os.path.join(os.getcwd(), outputFile), 'a')

    with open(inputFile) as f:
        input = f.read().splitlines()
    del input[0]

    for line in input:
        line = line.split()
        genre = 0
        if line[5] == 'f':
            genre = 1

        ages = ['(4,', '(8,', '(15,', '(25,', '(38,', '(48,', '(60,']
        age = 0

        for index in range(0, len(ages)):
            if ages[index] == line[3]:
                age = index + 1

        output.write('aligned/{}/landmark_aligned_face.{}.{} {} {}\n'.format(line[0], line[2], line[1], genre, age))

    output.close()


def main():
    args = parse_args()
    createDescFile(args.inputFile, args.outputFile)

if __name__ == "__main__":
    main()
