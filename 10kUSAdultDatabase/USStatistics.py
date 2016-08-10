#!/usr/bin/env python
import sys
import time
import argparse
import os
import math
from collections import Counter

import cv2
import numpy as np

import pandas as pd

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(epilog="10k Statistics")

    parser.add_argument('-i', '--inputXLSXFile',
                        required=True,
                        help='XLSX file with labels.')
    parser.add_argument('-o', '--outputXLSXFile',
                        required=True,
                        help='XLSX file for statistics.')

    args = parser.parse_args()
    return args

def countStatistics(inputXlsxTable, outputXlsxFile):

    USTable = pd.read_excel(inputXlsxTable)
    columns = []

    for index in range(2, len(USTable.columns)):
        array = USTable[USTable.columns[index]].values
        array = array[~np.isnan(array)]
        array = np.rint(array).astype(int)
        columns.append(array)

    statisticsMatrix = []

    for column in columns:
        row = Counter(column)
        sum = 0
        for key, value in row.iteritems():
            sum += value
        for key, value in row.iteritems():
            row[key] = round(value/float(sum), 3)
        row['Mean'] = round(np.mean(column), 3)
        statisticsMatrix.append(row)

    pdStatisticsMatrix = pd.DataFrame(statisticsMatrix, index=USTable.columns[2:])
    pdStatisticsMatrix = pdStatisticsMatrix.sort_index()
    pdStatisticsMatrix.to_excel(outputXlsxFile + '.xlsx')



def main():
    args = parse_args()
    countStatistics(args.inputXLSXFile, args.outputXLSXFile)

if __name__ == "__main__":
    main()
