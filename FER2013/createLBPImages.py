#!/usr/bin/env python
import sys
import time
import argparse
import lmdb
import os
import re

import cv2
import numpy as np

from caffe.proto import caffe_pb2
import caffe

#--------
#Mark Byers
#http://stackoverflow.com/questions/4836710/does-python-have-a-built-in-function-for-string-natural-sort

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)
#--------

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Directory with images.")
    parser.add_argument("-o", "--output", help="Output LMDB directory.")
    args = parser.parse_args()

    env_out = lmdb.open(args.output, map_size=30000000000)

    counter = 0
    t1 = time.time()
    with env_out.begin(write=True) as txn_out:

        c_out = txn_out.cursor()
        for image in natural_sort(os.listdir(args.input)):
		
            a = cv2.imread(os.path.join(args.input,image), 0)
            a = np.asarray(a)
            a = np.pad(a, 1, 'constant', constant_values=0)
        
            lbp = np.zeros((48, 48))

            x = 1
            y = 1

            for y in range(1, a.shape[0]-1):
                for x in range(1, a.shape[0]-1):

                    tmpPixel = 0

                    if a[x][y] <= a[x - 1][y - 1]:   
                         tmpPixel += 1
                    if a[x][y] <= a[x][y - 1]:
                         tmpPixel += 2
                    if a[x][y] <= a[x + 1][y - 1]:
                         tmpPixel += 4
                    if a[x][y] <= a[x + 1][y]:
                         tmpPixel += 8
                    if a[x][y] <= a[x + 1][y + 1]:
                         tmpPixel += 16
                    if a[x][y] <= a[x][y + 1]:
                         tmpPixel += 32
                    if a[x][y] <= a[x - 1][y + 1]:
                         tmpPixel += 64
                    if a[x][y] <= a[x - 1][y]:
                         tmpPixel += 128
                    
                    lbp[x - 1][y - 1] = tmpPixel

            lbp = np.resize(lbp, (1,48,48))            

            datum = caffe.io.array_to_datum(lbp)

            key = "%07d" % counter
            c_out.put(key, datum.SerializeToString())

            counter += 1
            if counter % 1000 == 0:
                print "DONE %d in %f s" % (counter, time.time() - t1)
            
if __name__ == '__main__':
    main(sys.argv)
 











