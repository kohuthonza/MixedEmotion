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

    parser.add_argument('-ix', '--inputXLSXFile',
                        required=True,
                        help='XLSX file with labels.')
    parser.add_argument('-i', '--inputDirectory',
                    required=True,
                    help='Input directory with images.')
    parser.add_argument('-o', '--outputDirectory',
                        required=True,
                        help='Output directory for lmdb database.')

    args = parser.parse_args()
    return args



def createImages(imagesIndexes, xlsxTable, inputDirectory, outputDirectory):

    print('Creating images...')

    USTable = pd.read_excel(xlsxTable)

    env_out = lmdb.open('Images' + outputDirectory, map_size=10000000000)

    counter = 0

    t1 = time.time()
    with env_out.begin(write=True) as txn_out:

        c_out = txn_out.cursor()
        for indexRow in range(0, len(imagesIndexes)):
            imageName = USTable.iloc[indexRow].values[0].replace('_oval.jpg', '.png')
            image = cv2.imread(os.path.join(inputDirectory, imageName).replace('_oval.jpg', '.png'), 1)
            if image is not None:
                image = np.rollaxis(image, 2, 0)
                #image = np.resize(image, (1,image.shape[0],image.shape[0]))
                datum = caffe.io.array_to_datum(image)
                key = "%07d" % counter
                c_out.put(key, datum.SerializeToString())
                counter += 1
                if counter % 100 == 0:
                    print "DONE %d in %f s" % (counter, time.time() - t1)



def createSoftLabels(xlsxTable, inputDirectory, outputDirectory):

    print('Creating labels...')

    USTable = pd.read_excel(xlsxTable)
    imagesIndexes = USTable.index.values
    np.random.shuffle(imagesIndexes)

    env_out = lmdb.open('Labels' + outputDirectory, map_size=10000000000)

    imagesNames = os.listdir(inputDirectory)

    counter = 0
    maxValue = 0

    t1 = time.time()
    with env_out.begin(write=True) as txn_out:

        c_out = txn_out.cursor()
        for indexRow in range(0, len(imagesIndexes)):
            if (USTable.iloc[indexRow].values[0].replace('_oval.jpg', '.png') in imagesNames):
                label = np.zeros(USTable.shape[1] - 2)
                for i, value in enumerate(USTable.iloc[imagesIndexes[indexRow]].values[2:]):
                    if np.isnan(value):
                        label[i] = 0
                    else:
                        label[i] = value + 1
                if maxValue < np.amax(label):
                    maxValue = np.amax(label)
                label = np.resize(label, (1, len(label), 1)).astype(int)
                datum = caffe.io.array_to_datum(label)
                key = "%07d" % counter
                c_out.put(key, datum.SerializeToString())
                counter += 1
                if counter % 100 == 0:
                    print "DONE %d in %f s" % (counter, time.time() - t1)

    print maxValue
    return imagesIndexes

def main():
    args = parse_args()
    imagesIndexes = createSoftLabels(args.inputXLSXFile, args.inputDirectory, args.outputDirectory)
    createImages(imagesIndexes, args.inputXLSXFile, args.inputDirectory, args.outputDirectory)

if __name__ == "__main__":
    main()
