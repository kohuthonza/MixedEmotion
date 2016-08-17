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

    normalizeValues = np.zeros(trainTable.shape[1])
    meanValues = np.zeros(trainTable.shape[1])
    for columnIndex in range(0 , trainTable.shape[1]):
        normalizeValues[columnIndex] = np.amax(trainTable[trainTable.columns[columnIndex]].values)
        meanValues[columnIndex] = np.mean(trainTable[trainTable.columns[columnIndex]].values)
    normalizeValues = normalizeValues.astype(float)
    meanValues = meanValues.astype(float)

    meanValues = meanValues/normalizeValues


    batchSize = net.blobs['data'].data.shape[0]

    MSE = np.zeros(USTable.shape[1])
    RMSE = np.zeros(USTable.shape[1])
    MeanMSE = np.zeros(USTable.shape[1])
    MeanRMSE = np.zeros(USTable.shape[1])
    PercentageDif = np.zeros(USTable.shape[1])
    accuracy = np.zeros(USTable.shape[1])
    MeanAccuracy = np.zeros(USTable.shape[1])


    counter = 0

    while (counter < len(imagesList)):

        if not ((len(imagesList) - counter) >= batchSize):
            sizeOfRange = len(imagesList) - counter
        else:
            sizeOfRange = batchSize

        for index in range(0, sizeOfRange):
            image = cv2.imread(os.path.join(inputDirectory, imagesList[index + counter].replace('_oval.jpg', '.png')), 0)
            image = image[3:67,3:67]
            image = (image - 127.0)/127.0
            #image = np.rollaxis(image, 2, 0)
            net.blobs['data'].data[index] = image

        out = net.forward()

        for index in range(0, sizeOfRange):
            absDif = np.absolute(USTable.loc[imagesList[index + counter]].values/normalizeValues - out['ip2'][index])
            absDifMean = np.absolute(USTable.loc[imagesList[index + counter]].values/normalizeValues - meanValues)
            RMSE += absDif
            MSE += absDif**2
            MeanRMSE += absDifMean
            MeanMSE += absDifMean**2
            PercentageDif += np.absolute(USTable.loc[imagesList[index + counter]].values/normalizeValues - out['ip2'][index])/normalizeValues
            for columnIndex in range(0, len(out['ip2'][index])):
                if (round(out['ip2'][index][columnIndex], 0) == round((USTable.loc[imagesList[index + counter]].values/normalizeValues)[columnIndex], 0)):
                    accuracy[columnIndex] += 1
                if (round(meanValues[columnIndex], 0) == round((USTable.loc[imagesList[index + counter]].values/normalizeValues)[columnIndex], 0)):
                    MeanAccuracy[columnIndex] += 1

        print("Batch DONE.")

        counter += sizeOfRange


    MSE = MSE/counter
    RMSE = RMSE/counter
    MeanMSE = MeanMSE/counter
    MeanRMSE = MeanRMSE/counter
    PercentageDif = PercentageDif/counter
    accuracy = accuracy/counter
    MeanAccuracy = MeanAccuracy/counter


    statisticIndex = 0
    if (USTable.shape[1] == 50):
        statisticIndex = 1
    elif (USTable.shape[1] == 70):
        statisticIndex = 2

    for index in range(0, len(MSE)):
        statisticTable.loc[USTable.columns[index]][statisticIndex*7] = PercentageDif[index]*100
        statisticTable.loc[USTable.columns[index]][statisticIndex*7 + 1] = MSE[index]
        statisticTable.loc[USTable.columns[index]][statisticIndex*7 + 2] = RMSE[index]
        statisticTable.loc[USTable.columns[index]][statisticIndex*7 + 3] = MSE[index]/MeanMSE[index]
        statisticTable.loc[USTable.columns[index]][statisticIndex*7 + 4] = RMSE[index]/MeanRMSE[index]
        statisticTable.loc[USTable.columns[index]][statisticIndex*7 + 5] = accuracy[index]
        statisticTable.loc[USTable.columns[index]][statisticIndex*7 + 6] = MeanAccuracy[index]/accuracy[index]






    print accuracy
    print('Loss: {}'.format(np.sum(MSE)))
    statisticTable = statisticTable.round(4)
    statisticTable.to_excel(statisticXlsxTable)


def main():
    args = parse_args()

    caffe.set_mode_gpu()
    net = caffe.Net(args.deploy, args.net, caffe.TEST)
    classify(net, args.inputDirectory, args.inputXLSXFile, args.inputXLSXFileStatistic, args.inputXLSXFileTrain)


if __name__ == "__main__":
    main()
