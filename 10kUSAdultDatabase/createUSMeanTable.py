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

    parser.add_argument('-i', '--inputXLSXFile',
                        required=True,
                        help='XLSX file with labels.')
    parser.add_argument('-o', '--outputXLSXFile',
                        required=True,
                        help='Output XLSX file.')

    args = parser.parse_args()
    return args

def countMean(array):

    division = 0
    sum = 0

    for value in array:
        if not np.isnan(value):
            division += 1
            sum  += value

    return sum/division

def findMostFrequentValue(array):

    tmpArray = np.zeros(0, dtype=int)

    for value in array:
        if not np.isnan(value):
            tmpArray = np.append(tmpArray, int(value))

    return float(np.bincount(tmpArray).argmax())

def countUSMeanTableRow(imageMatrix, USTableColumns):

    #means = [True, True, False, True, True, False, False, False, False,
# False, True, False, False, True, False, True, False, True, True, False]

    USMeanTableRow = {}
    #for i, mean in enumerate(means):
        #if mean:
    for i in range(len(USTableColumns) - 2):
        USMeanTableRow[USTableColumns[i + 2]] = countMean(imageMatrix[:,i])
        #else:
        #    USMeanTableRow[USTableColumns[i + 2]] = findMostFrequentValue(imageMatrix[:,i])

    return USMeanTableRow

def createRegressionTable(inputXlsxTable, outputXlsxFile):

    USTable = pd.read_excel(inputXlsxTable, sheetname='All Data')

    USMeanTable = []
    imagesNames = []
    imageName = ''
    imageMatrix = np.zeros(USTable.shape[1] - 2, dtype=float)

    for indexRow in range(0, USTable.shape[0]):
        if imageName != USTable.iloc[indexRow].values[0]:
            if indexRow != 0:
                imagesNames.append(imageName)
                USMeanTable.append(countUSMeanTableRow(imageMatrix, USTable.columns))
                

            imageName = USTable.iloc[indexRow].values[0]
            imageMatrix = USTable.iloc[indexRow].values[2:]
        else:
            imageMatrix = np.vstack([imageMatrix, USTable.iloc[indexRow].values[2:].astype(float)])

    imagesNames.append(imageName)
    USMeanTable.append(countUSMeanTableRow(imageMatrix, USTable.columns))


    pdUSMeanTable = pd.DataFrame(USMeanTable, index=imagesNames)
    pdUSMeanTable.to_excel(outputXlsxFile + '.xlsx')


def main():
    args = parse_args()
    createRegressionTable(args.inputXLSXFile, args.outputXLSXFile)

if __name__ == "__main__":
    main()
