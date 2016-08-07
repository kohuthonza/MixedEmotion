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


def createImages(xlsxTable, inputDirectory, outputDirectory):

    print('Creating images...')

    UStable = pd.read_excel(xlsxTable)
    imagesNames = UStable.index

    env_out = lmdb.open('Images' + outputDirectory, map_size=10000000000)

    counter = 0

    t1 = time.time()
    with env_out.begin(write=True) as txn_out:

        c_out = txn_out.cursor()
        for imageName in imagesNames:
            image = cv2.imread(os.path.join(inputDirectory, imageName).replace('_oval.jpg', '.png'), 0)
            if image is not None:
                #image = np.rollaxis(image, 2, 0)
                image = np.resize(image, (1,image.shape[0],image.shape[0]))
                datum = caffe.io.array_to_datum(image)

                key = "%07d" % counter
                c_out.put(key, datum.SerializeToString())

                counter += 1
                if counter % 100 == 0:
                    print "DONE %d in %f s" % (counter, time.time() - t1)



def createRegressionLabels(xlsxTable, inputDirectory, outputDirectory):

    print('Creating labels...')
    UStable = pd.read_excel(xlsxTable)
    print(UStable.shape[1])
    normalizeValues = np.zeros(UStable.shape[1])
    for columnIndex in range(0 , UStable.shape[1]):
        normalizeValues[columnIndex] = np.amax(UStable[UStable.columns[columnIndex]].values)
    normalizeValues = normalizeValues.astype(float)


    env_out = lmdb.open('Labels' + outputDirectory, map_size=10000000000)

    imagesNames = os.listdir(inputDirectory)

    counter = 0

    t1 = time.time()
    with env_out.begin(write=True) as txn_out:

        c_out = txn_out.cursor()
        for indexRow in range(0, UStable.shape[0]):
            if (UStable.index[indexRow].replace('_oval.jpg', '.png') in imagesNames):
                label = UStable.iloc[indexRow].values
                label = label.astype(float)
                label = label/normalizeValues
                label.resize((len(label), 1, 1))

                datum = caffe.io.array_to_datum(label)

                key = "%07d" % counter
                c_out.put(key, datum.SerializeToString())
                counter += 1
                if counter % 100 == 0:
                    print "DONE %d in %f s" % (counter, time.time() - t1)

def main():
    args = parse_args()
    createRegressionLabels(args.inputXLSXFile, args.inputDirectory, args.outputDirectory)
    createImages(args.inputXLSXFile, args.inputDirectory, args.outputDirectory)

if __name__ == "__main__":
    main()
