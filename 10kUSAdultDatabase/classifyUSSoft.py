#!/usr/bin/env python

import sys
import numpy as np
import shutil
import caffe
import re
import pandas as pd
import argparse

import cv2
import os

def parse_args():

    parser = argparse.ArgumentParser(epilog="Classify Emotions")

    parser.add_argument('-n', '--net',
                        required=True,
                        help='Caffe net model.')
    parser.add_argument('-d', '--deploy',
                        required=True,
                        help='Deploy for caffe net.')
    parser.add_argument('-i', '--inputDirectory',
                        required=True,
                        help='Input directory with images for classification.')
    parser.add_argument('-ix', '--inputXLSXFile',
                        required=True,
                        help='XLSX file with labels.')
    parser.add_argument('-is', '--inputXLSXFileStatistic',
                        required=True,
                        help='XLSX file for statistic.')
    parser.add_argument('-it', '--inputXLSXFileTrain',
                        required=True,
                        help='XLSX file with train labels.')


    args = parser.parse_args()
    return args


def classify(net, inputDirectory, xlsxTable, statisticXlsxTable, trainXlsxTable):


    statisticTable = pd.read_excel(statisticXlsxTable)
    trainTable = pd.read_excel(trainXlsxTable)

    USTable = pd.read_excel(xlsxTable)
    imagesNames = USTable.index
    images = os.listdir(inputDirectory)
    imagesList = []

    for image in imagesNames:
        if image.replace('_oval.jpg', '.png') in images:
            imagesList.append(image)

    meanValues = np.zeros(trainTable.shape[1])
    mostFrequent =  np.zeros(trainTable.shape[1])
    for columnIndex in range(0 , trainTable.shape[1]):
        meanValues[columnIndex] = np.mean(trainTable[trainTable.columns[columnIndex]].values)
        mostFrequent[columnIndex] = np.bincount(trainTable[trainTable.columns[columnIndex]].values.astype(int)).argmax()
    meanValues = np.rint(meanValues).astype(int)



    batchSize = net.blobs['data'].data.shape[0]

    score = np.zeros(USTable.shape[1])
    scoreMean = np.zeros(USTable.shape[1])
    scoreMostFrequent = np.zeros(USTable.shape[1])


    counter = 0

    while (counter < len(imagesList)):

        if not ((len(imagesList) - counter) >= batchSize):
            sizeOfRange = len(imagesList) - counter
        else:
            sizeOfRange = batchSize

        for index in range(0, sizeOfRange):
            image = cv2.imread(os.path.join(inputDirectory, imagesList[index + counter].replace('_oval.jpg', '.png')), 1)
            image = image[3:67,3:67]
            image = (image - 127.0)/127.0
            image = np.rollaxis(image, 2, 0)
            net.blobs['data'].data[index] = image

        out = net.forward()

        for index in range(0, sizeOfRange):
            for label in range(0, USTable.shape[1]):
                if ((out['reshape'][index][:, label].argmax() - 1) == round(USTable.loc[imagesList[index + counter]].values[label], 0)):
                    score[label] += 1
                if (meanValues[label] == round(USTable.loc[imagesList[index + counter]].values[label], 0)):
                    scoreMean[label] += 1
                if (mostFrequent[label] == round(USTable.loc[imagesList[index + counter]].values[label], 0)):
                    scoreMostFrequent[label] += 1




        print("Batch DONE.")

        counter += sizeOfRange

    score = score/counter
    scoreMean = scoreMean/counter
    scoreMostFrequent = scoreMostFrequent/counter


    statisticIndex = 0
    if (USTable.shape[1] == 50):
        statisticIndex = 1
    if (USTable.shape[1] == 70):
        statisticIndex = 2

    for index in range(0, len(score)):
        statisticTable.loc[USTable.columns[index]][statisticIndex*3] = score[index]
        statisticTable.loc[USTable.columns[index]][statisticIndex*3 + 1] = scoreMean[index]/score[index]
        statisticTable.loc[USTable.columns[index]][statisticIndex*3 + 2] = scoreMostFrequent[index]/score[index]



    print score
    print scoreMean
    print scoreMostFrequent

    statisticTable = statisticTable.round(4)
    statisticTable.to_excel(statisticXlsxTable)

def main():
    args = parse_args()

    caffe.set_mode_gpu()
    net = caffe.Net(args.deploy, args.net, caffe.TEST)
    classify(net, args.inputDirectory, args.inputXLSXFile, args.inputXLSXFileStatistic, args.inputXLSXFileTrain)


if __name__ == "__main__":
    main()
