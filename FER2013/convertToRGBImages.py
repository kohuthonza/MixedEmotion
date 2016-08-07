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
    parser.add_argument("-o", "--output", help="Output directory.")
    args = parser.parse_args()

    
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    else:
        shutil.rmtree(args.output)
        os.mkdir(args.output)

    counter = 0
    t1 = time.time()

    for image in natural_sort(os.listdir(args.input)):
		
        a = cv2.imread(os.path.join(args.input,image), 1)
        a = a - 127.0
        cv2.imwrite(os.path.join(args.output, 'test' + str(counter) + '.png'), a)

        counter += 1
        if counter % 1000 == 0:
            print "DONE %d in %f s" % (counter, time.time() - t1)
            
if __name__ == '__main__':
    main(sys.argv)
 
