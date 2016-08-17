#!/usr/bin/env python

import sys
import numpy as np
import shutil
import caffe
import re
import pandas as pd
import argparse
import itertools
from collections import Counter

import cv2
import os

def parse_args():

    parser = argparse.ArgumentParser(epilog="Classify audience")

    parser.add_argument('-n', '--net',
                        required=True,
                        help='Caffe net model.')
    parser.add_argument('-d', '--deploy',
                        required=True,
                        help='Deploy for caffe net.')
    parser.add_argument('-id', '--inputDirectory',
                        required=True,
                        help='Input directory with images for classification.')
    parser.add_argument('-it', '--inputTestFile',
                        required=True,
                        help='File with labels.')
    parser.add_argument('-ir', '--inputTrainFile',
                        required=True,
                        help='File with labels.')
    parser.add_argument('-ox', '--outputXLSXFile',
                        required=True,
                        help='XLSX file fro confusion matrix.')


    args = parser.parse_args()
    return args


def classify(net, inputDirectory, inputTestFile, inputTrainFile, outputXlsxFile):


    with open(inputTestFile) as f:
        linesTest = f.read().splitlines()

    with open(inputTrainFile) as l:
        linesTrain = l.read().splitlines()


    batchSize = net.blobs['data'].data.shape[0]

    score = np.zeros(2)
    ageConfusionMatrix = np.zeros((8, 8))
    genderConfusionMatrix = np.zeros((2, 2))
    genderError = np.zeros(8)

    counter = 0

    while (counter < len(linesTest)):

        if not ((len(linesTest) - counter) >= batchSize):
            sizeOfRange = len(linesTest) - counter
        else:
            sizeOfRange = batchSize

        for index in range(0, sizeOfRange):
            image = cv2.imread(os.path.join(inputDirectory, linesTest[index + counter].split()[0].split('/')[2].replace('jpg', 'png')), 1)
            image = image[3:67,3:67]
            #image = image[4:132,4:132]
            image = (image - 127.0)/127.0
            image = np.rollaxis(image, 2, 0)
            net.blobs['data'].data[index] = image

        out = net.forward()

        for index in range(0, sizeOfRange):
            for label in range(0, 2):
                if ((out['reshape'][index][:, label].argmax()) == int(linesTest[index + counter].split()[label + 1])):
                    score[label] += 1
                else:
                    if label == 0:
                        genderError[int(linesTest[index + counter].split()[2])] += 1

            print out['reshape'][index][:, 1]
            ageConfusionMatrix[out['reshape'][index][:, 1].argmax(), int(linesTest[index + counter].split()[2])] += 1
            genderConfusionMatrix[out['reshape'][index][:, 0].argmax(), int(linesTest[index + counter].split()[1])] += 1


        print("Batch DONE.")

        counter += sizeOfRange

    score = score/len(linesTest)



    pdConfusionMatrix = []

    ageIndexes = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)','(48, 53)', '(60, 100)']
    genderIndexes = ['male', 'female']


    for matrix, indexes in itertools.izip([ageConfusionMatrix, genderConfusionMatrix], [ageIndexes, genderIndexes]):
        for index in range(0, matrix.shape[0]):
            matrix[index, :] = matrix[index, :]/np.sum(matrix[index, :])
            tmpRow = {}
            for value, index in itertools.izip(matrix[index, :], indexes):
                tmpRow[index] = value
            pdConfusionMatrix.append(tmpRow)

    indexes = ageIndexes + genderIndexes

    pdConfusionMatrix = pd.DataFrame(pdConfusionMatrix, index=indexes)
    pdConfusionMatrix = pdConfusionMatrix.round(4)
    pdConfusionMatrix = pdConfusionMatrix[indexes]
    pdConfusionMatrix.to_excel(outputXlsxFile + '.xlsx')


    print np.around(ageConfusionMatrix, decimals=3)
    print np.around(genderConfusionMatrix, decimals=3)
    print score
    print genderError/np.sum(genderError)

def main():
    args = parse_args()

    caffe.set_mode_gpu()
    net = caffe.Net(args.deploy, args.net, caffe.TEST)
    classify(net, args.inputDirectory, args.inputTestFile, args.inputTrainFile, args.outputXLSXFile)


if __name__ == "__main__":
    main()
