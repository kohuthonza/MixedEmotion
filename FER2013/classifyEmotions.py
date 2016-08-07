#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import cv2
import sys

import argparse
import caffe
import random
import math
import re
import pandas as pd

import os
import shutil
import itertools

global net


#--------
#Mark Byers
#http://stackoverflow.com/questions/4836710/does-python-have-a-built-in-function-for-string-natural-sort

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)
#--------


def PrintConfusionMatrix():

    """
    Values for confusion matrix
    """
    Angry = 0
    AngryDisgust = 0
    AngryFear = 0
    AngryHappy = 0
    AngrySad = 0
    AngrySurprise = 0
    AngryNeutral = 0

    DisgustAngry = 0
    Disgust = 0
    DisgustFear = 0
    DisgustHappy = 0
    DisgustSad = 0
    DisgustSurprise = 0
    DisgustNeutral = 0

    FearAngry = 0
    FearDisgust = 0
    Fear = 0
    FearHappy = 0
    FearSad = 0
    FearSurprise = 0
    FearNeutral = 0

    HappyAngry = 0
    HappyDisgust = 0
    HappyFear = 0
    Happy = 0
    HappySad = 0
    HappySurprise = 0
    HappyNeutral = 0

    SadAngry = 0
    SadDisgust = 0
    SadFear = 0
    SadHappy = 0
    Sad = 0
    SadSurprise = 0
    SadNeutral = 0

    SurpriseAngry = 0
    SurpriseDisgust = 0
    SurpriseFear = 0
    SurpriseHappy = 0
    SurpriseSad = 0
    Surprise = 0
    SurpriseNeutral = 0

    NeutralAngry = 0
    NeutralDisgust = 0
    NeutralFear = 0
    NeutralHappy = 0
    NeutralSad = 0
    NeutralSurprise = 0
    Neutral = 0

    AngryCount = 0
    DisgustCount = 0
    FearCount = 0
    HappyCount = 0
    SadCount = 0
    SurpriseCount = 0
    NeutralCount = 0

    counter = 0

    for image, line in itertools.izip(natural_sort(os.listdir(args['directory'])), open('test.txt', 'r')):

        image = cv2.imread(os.path.join(args['directory'],image), 1)
        image = (image - 127.0)/127.0
        image = np.rollaxis(image, 2, 0)
        net.blobs['data'].data[0] = image
        out = net.forward()
        prediction = out['prob'][0].argmax()
        label = int(line.split()[1])

        if (label == 0):
            AngryCount += 1
            if (prediction == 0):
                Angry += 1
            elif (prediction == 1):
                AngryDisgust += 1
            elif (prediction == 2):
                AngryFear += 1
            elif (prediction == 3):
                AngryHappy += 1
            elif (prediction == 4):
                AngrySad += 1
            elif (prediction == 5):
                AngrySurprise += 1
            else:
                AngryNeutral += 1

        elif (label == 1):
            DisgustCount += 1
            if (prediction == 0):
                DisgustAngry += 1
            elif (prediction == 1):
                Disgust += 1
            elif (prediction == 2):
                DisgustFear += 1
            elif (prediction == 3):
                DisgustHappy += 1
            elif (prediction == 4):
                DisgustSad += 1
            elif (prediction == 5):
                DisgustSurprise += 1
            else:
                DisgustNeutral += 1


        elif (label == 2):
            FearCount += 1
            if (prediction == 0):
                FearAngry += 1
            elif (prediction == 1):
                FearDisgust += 1
            elif (prediction == 2):
                Fear += 1
            elif (prediction == 3):
                FearHappy += 1
            elif (prediction == 4):
                FearSad += 1
            elif (prediction == 5):
                FearSurprise += 1
            else:
                FearNeutral += 1


        elif (label == 3):
            HappyCount += 1
            if (prediction == 0):
                HappyAngry += 1
            elif (prediction == 1):
                HappyDisgust += 1
            elif (prediction == 2):
                HappyFear += 1
            elif (prediction == 3):
                Happy += 1
            elif (prediction == 4):
                HappySad += 1
            elif (prediction == 5):
                HappySurprise += 1
            else:
                HappyNeutral += 1


        elif (label == 4):
            SadCount += 1
            if (prediction == 0):
                SadAngry += 1
            elif (prediction == 1):
                SadDisgust += 1
            elif (prediction == 2):
                SadFear += 1
            elif (prediction == 3):
                SadHappy += 1
            elif (prediction == 4):
                Sad += 1
            elif (prediction == 5):
                SadSurprise += 1
            else:
                SadNeutral += 1


        elif (label == 5):
            SurpriseCount += 1
            if (prediction == 0):
                SurpriseAngry += 1
            elif (prediction == 1):
                SurpriseDisgust += 1
            elif (prediction == 2):
                SurpriseFear += 1
            elif (prediction == 3):
                SurpriseHappy += 1
            elif (prediction == 4):
                SurpriseSad += 1
            elif (prediction == 5):
                Surprise += 1
            else:
                SurpriseNeutral += 1


        else:
            NeutralCount += 1
            if (prediction == 0):
                NeutralAngry += 1
            elif (prediction == 1):
                NeutralDisgust += 1
            elif (prediction == 2):
                NeutralFear += 1
            elif (prediction == 3):
                NeutralHappy += 1
            elif (prediction == 4):
                NeutralSad += 1
            elif (prediction == 5):
                NeutralSurprise += 1
            else:
                Neutral += 1


        print("Predicted class is #{}. Actual class is #{}".format(prediction, label))


        counter += 1

        if counter % 100 == 0:
            print('{} images processed'.format(counter))

        if (args['numberImages']):
            if counter == args['numberImages']:
                break;

    confusionMatrix = [
{'Angry' : Angry/float(AngryCount), 'Disgust' : AngryDisgust/float(AngryCount), 'Fear' : AngryFear/float(AngryCount), 'Happy' : AngryHappy/float(AngryCount), 'Neutral' : AngryNeutral/float(AngryCount), 'Sad' : AngrySad/float(AngryCount), 'Surprise' : AngrySurprise/float(AngryCount)},
{'Angry' : DisgustAngry/float(DisgustCount), 'Disgust' : Disgust/float(DisgustCount), 'Fear' : DisgustFear/float(DisgustCount), 'Happy' : DisgustHappy/float(DisgustCount), 'Neutral' : DisgustNeutral/float(DisgustCount), 'Sad' : DisgustSad/float(DisgustCount), 'Surprise' : DisgustSurprise/float(DisgustCount)},
{'Angry' : FearAngry/float(FearCount), 'Disgust' : FearDisgust/float(FearCount), 'Fear' : Fear/float(FearCount), 'Happy' : FearHappy/float(FearCount), 'Neutral' : FearNeutral/float(FearCount), 'Sad' : FearSad/float(FearCount), 'Surprise' : FearSurprise/float(FearCount)},
{'Angry' : HappyAngry/float(HappyCount), 'Disgust' : HappyDisgust/float(HappyCount), 'Fear' : HappyFear/float(HappyCount), 'Happy' : Happy/float(HappyCount), 'Neutral' : HappyNeutral/float(HappyCount), 'Sad' : HappySad/float(HappyCount), 'Surprise' : HappySurprise/float(HappyCount)},
{'Angry' : NeutralAngry/float(NeutralCount), 'Disgust' : NeutralDisgust/float(NeutralCount), 'Fear' : NeutralFear/float(NeutralCount), 'Happy' : NeutralHappy/float(NeutralCount), 'Neutral' : Neutral/float(NeutralCount), 'Sad' : NeutralSad/float(NeutralCount), 'Surprise' : NeutralSurprise/float(NeutralCount)},
{'Angry' : SadAngry/float(SadCount), 'Disgust' : SadDisgust/float(SadCount), 'Fear' : SadFear/float(SadCount), 'Happy' : SadHappy/float(SadCount), 'Neutral' : SadNeutral/float(SadCount), 'Sad' : Sad/float(SadCount), 'Surprise' : SadSurprise/float(SadCount)},
{'Angry' : SurpriseAngry/float(SurpriseCount), 'Disgust' : SurpriseDisgust/float(SurpriseCount), 'Fear' : SurpriseFear/float(SurpriseCount), 'Happy' : SurpriseHappy/float(SurpriseCount), 'Neutral' : SurpriseNeutral/float(SurpriseCount), 'Sad' : SurpriseSad/float(SurpriseCount), 'Surprise' : Surprise/float(SurpriseCount)}]

    pdConfusionMatrix = pd.DataFrame(confusionMatrix, index=['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'])
    pdConfusionMatrix = pdConfusionMatrix.round(3)

    print(AngryCount)
    print(DisgustCount)
    print(FearCount)
    print(HappyCount)
    print(NeutralCount)
    print(SadCount)
    print(SurpriseCount)

    pdConfusionMatrix.to_csv(args['csvFile'] + '.csv')
    print(pdConfusionMatrix)



args_parser = argparse.ArgumentParser(description='Classify emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)')
args_parser.add_argument('-c','--confusion', help = '', action="count", default=0, required=False)
args_parser.add_argument('-di','--directory', type = str, help = '', required=False)
args_parser.add_argument('-n','--net', type = str, help = '', required=True)
args_parser.add_argument('-d','--deploy', type = str, help = 'Deploy', required=True)
args_parser.add_argument('-ni','--numberImages', type = int, help = 'Number of images to classify', required=False)
args_parser.add_argument('-f','--csvFile', type = str, help = 'Store confusionMatrix to csv file', required=False)



try:
    args = vars(args_parser.parse_args())
except:
    sys.exit(1)


if (args['confusion']):
    caffe.set_mode_gpu()
    net = caffe.Net(args['deploy'],
                args['net'],
                caffe.TEST)


if (args['confusion']):
    PrintConfusionMatrix()
