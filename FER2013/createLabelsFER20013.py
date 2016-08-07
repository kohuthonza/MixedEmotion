import pandas as pd
import os
import sys
import argparse
import time

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="csv file.")
    parser.add_argument("-o", "--output", help="Output txt file.")
    parser.add_argument("-t", "--type", help="Type of dataset (Training, PublicTest, PrivateTest).")
    parser.add_argument('-m', "--mirror", help = 'Add mirror images', action="count", default=0, required=False)
    args = parser.parse_args()

    counter = 0
    t1 = time.time()

    ferDatabase = pd.read_csv(args.input)
    filteredFerDatabase = ferDatabase.loc[ferDatabase['Usage'] == args.type]
     
    if os.path.isfile(os.path.join(os.getcwd(), args.output + '.txt')):
        os.remove(os.path.join(os.getcwd(), args.output + '.txt'))
    f=open(os.path.join(os.getcwd(), args.output + '.txt'), 'w+')


    for emotion in filteredFerDatabase.emotion:
        
        f.write('test%d.png %d\n' % (counter, int(emotion))) 

        counter += 1
        if counter % 1000 == 0:
            print "DONE %d in %f s" % (counter, time.time() - t1)

    if args.mirror:
        
        for emotion in filteredFerDatabase.emotion:
        
            f.write('test%d.png %d\n' % (counter, int(emotion))) 

            counter += 1
            if counter % 1000 == 0:
                print "DONE %d in %f s" % (counter, time.time() - t1)


    
    f.close() 
            
            
if __name__ == '__main__':
    main(sys.argv)
 
