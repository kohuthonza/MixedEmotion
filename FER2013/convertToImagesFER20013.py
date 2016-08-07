#!/usr/bin/env python
import sys
import time
import argparse
import lmdb
import os
import re
import pandas as pd
import shutil

import cv2
import numpy as np

from caffe.proto import caffe_pb2
import caffe


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="csv file.")
    parser.add_argument("-o", "--output", help="Output directory.")
    parser.add_argument("-t", "--type", help="Type of dataset (Training, PublicTest, PrivateTest).")
    parser.add_argument('-m', "--mirror", help = 'Add mirror images', action="count", default=0, required=False)
    args = parser.parse_args()


    counter = 0
    t1 = time.time()

    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    else:
        shutil.rmtree(args.output)
        os.mkdir(args.output)

    ferDatabase = pd.read_csv(args.input)
    filteredFerDatabase = ferDatabase.loc[ferDatabase['Usage'] == args.type]

    for pixels in filteredFerDatabase.pixels:

        image = np.resize(np.fromstring(pixels, dtype=int, sep=' '),(48,48))
        cv2.imwrite(os.path.join(args.output, 'test' + str(counter) + '.png'), image)

        counter += 1
        if counter % 1000 == 0:
            print "DONE %d in %f s" % (counter, time.time() - t1)
    
    if args.mirror: 
        
        for pixels in filteredFerDatabase.pixels:

            image = np.resize(np.fromstring(pixels, dtype=int, sep=' '),(48,48))
            image = image[:,::-1]
            cv2.imwrite(os.path.join(args.output, 'test' + str(counter) + '.png'), image)

            counter += 1
            if counter % 1000 == 0:
                print "DONE %d in %f s" % (counter, time.time() - t1)
            
            
if __name__ == '__main__':
    main(sys.argv)
 
