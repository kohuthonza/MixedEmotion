#!/usr/bin/env python

import sys
import numpy as np
import shutil
import caffe
import re

import cv2
import os

def parse_args():
    print( ' '.join(sys.argv))
    import argparse
    parser = argparse.ArgumentParser(epilog="Classify Emotions")

    parser.add_argument('-n', '--net',
                        required=True,
                        help='Caffe net model.')
    parser.add_argument('-d', '--deploy',
                        required=True,
                        help='Deploy for caffe net.')
    parser.add_argument('-ic', '--in-dir-Class',
                        required=True,
                        help='Input directory for classification.')
    parser.add_argument('-fc', '--file-Class',
                        required=True,
                        help='Output file for emotions predictions.')
    parser.add_argument('-fl', '--file-Land',
                        required=True,
                        help='Landmark file.')


    args = parser.parse_args()
    return args


def classifyEmotions(net, inputDirectory, outputFile, landmarksFile):

    if os.path.isfile(os.path.join(os.getcwd(), outputFile + '.txt')):
        os.remove(os.path.join(os.getcwd(), outputFile + '.txt'))
    f=open(os.path.join(os.getcwd(), outputFile + '.txt'), 'w+')

    with open(landmarksFile) as l:
        landmarks = l.read().splitlines()

    batchSize = net.blobs['data'].data.shape[0]

    imagesList = []
    for imageName in natural_sort(os.listdir(inputDirectory)):
        if imageName.endswith(".png"):
            imagesList.append(imageName)

    counter = 0

    while (counter < len(imagesList)):

        if not ((len(imagesList) - counter) >= batchSize):
            sizeOfRange = len(imagesList) - counter
        else:
            sizeOfRange = batchSize

        for index in range(0, sizeOfRange):
            image = cv2.imread(os.path.join(inputDirectory, imagesList[index + counter]), 0)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image = (image - 127.0)/127.0
            image = np.rollaxis(image, 2, 0)
            net.blobs['data'].data[index] = image

        out = net.forward()

        for index in range(0, sizeOfRange):
            labels = landmarks[index + counter].split()
            f.write("{} {} {}\n".format(labels[0], labels[1],' '.join(map(str, out['prob'][index]))))

        print("Batch DONE.")

        counter += sizeOfRange

    f.close()

#--------
#Mark Byers
#http://stackoverflow.com/questions/4836710/does-python-have-a-built-in-function-for-string-natural-sort

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)
#--------


def main():
    args = parse_args()

    caffe.set_mode_gpu()
    net = caffe.Net(args.deploy, args.net, caffe.TEST)
    classifyEmotions(net, args.in_dir_Class, args.file_Class, args.file_Land)


if __name__ == "__main__":
    main()
