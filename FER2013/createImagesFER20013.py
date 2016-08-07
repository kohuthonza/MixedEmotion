#!/usr/bin/env python
import sys
import time
import argparse
import lmdb
import os
import re
import pandas as pd

import cv2
import numpy as np

from caffe.proto import caffe_pb2
import caffe


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="csv file.")
    parser.add_argument("-o", "--output", help="Output LMDB directory.")
    parser.add_argument("-t", "--type", help="Type of dataset (Training, PublicTest, PrivateTest).")
    parser.add_argument('-m', "--mirror", help = 'Add mirror images', action="count", default=0, required=False)
    args = parser.parse_args()

    env_out = lmdb.open(args.output, map_size=30000000000)

    counter = 0
    t1 = time.time()

    ferDatabase = pd.read_csv(args.input)
    filteredFerDatabase = ferDatabase.loc[ferDatabase['Usage'] == args.type]

    
    with env_out.begin(write=True) as txn_out:
        
        c_out = txn_out.cursor()
        for pixels in filteredFerDatabase.pixels:

            image = np.resize(np.fromstring(pixels, dtype=int, sep=' '),(1,48,48))
            image = image/255.0

            datum = caffe.io.array_to_datum(image)

            key = "%07d" % counter
            c_out.put(key, datum.SerializeToString())

            counter += 1
            if counter % 1000 == 0:
                print "DONE %d in %f s" % (counter, time.time() - t1)
        
        if args.mirror:
                 
            for pixels in filteredFerDatabase.pixels:

                image = np.resize(np.fromstring(pixels, dtype=int, sep=' '),(1,48,48))
                image = image[:,::-1]
                image = image/255.0
                
                datum = caffe.io.array_to_datum(image)

                key = "%07d" % counter
                c_out.put(key, datum.SerializeToString())

                counter += 1
                if counter % 1000 == 0:
                    print "DONE %d in %f s" % (counter, time.time() - t1)
            
            
if __name__ == '__main__':
    main(sys.argv)
 
