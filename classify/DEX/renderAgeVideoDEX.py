#!/usr/bin/env python

import sys
import numpy as np
import shutil
import caffe
import re

import cv2
import os

def parse_args():
    print( ' '.join(sys.argv))
    import argparse
    parser = argparse.ArgumentParser(epilog="Render Emotions To Video")

    parser.add_argument('-iv', '--input-video',
                        required=True,
                        help='Input video.')
    parser.add_argument('-ov', '--output-video',
                        required=True,
                        help='Output video.')
    parser.add_argument('-fc', '--file-Class',
                        required=True,
                        help='File with age predictions.')
    parser.add_argument('-fl', '--file-Land',
                        required=True,
                        help='Landmark file.')
    parser.add_argument('-nf', '--number-Frames',
                        required=True,
                        type=int,
                        help='Number of frames for prediction.')
    parser.add_argument('-b', '--render-Box',
                        action='store_true',
                        default=False)
    parser.add_argument('-l', '--render-Landmarks',
                        action='store_true',
                        default=False)


    args = parser.parse_args()
    return args

def renderVideo(ageFile, landmarkFile, inputVideo, outputVideo, numberOfFrames, renderBox, renderLandmarks):

    print("Creating video...")

    cap = cv2.VideoCapture(inputVideo)
    resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)


    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(outputVideo + '.avi', fourcc, fps, resolution, 1)

    with open(ageFile) as f:
        ageList = f.read().splitlines()

    for index in range(0, len(ageList)):
        ageList[index] = ageList[index].split()

    if renderBox or renderLandmarks:
        with open(landmarkFile) as l:
            landmarks = l.read().splitlines()
        counterOfLandmarksFrames = 0

    frameId = 0


    while(True):

        ret, frame = cap.read()
        if(ret==False):
            break

        frameId += 1

        ageLabel = countLabel(numberOfFrames, frameId, ageList)
        frame = renderAgeToFrame(frame, ageLabel)

        if (renderBox or renderLandmarks):
            landmarksLabels = landmarks[counterOfLandmarksFrames].split()

            if (landmarksLabels[0] == str(frameId)):
                if renderBox:
                    boxLand = landmarksLabels[2].split(':')
                    boxLand = [int(point) for point in boxLand]
                    boxPoints = []
                    for index in range(0,2):
                        boxPoints.append((boxLand[index*2], boxLand[index*2 + 1]))
                    frame = renderBoxToFrame(frame, boxPoints)

                if renderLandmarks:
                    landmarksPoints = landmarksLabels[3:]
                    landmarksPoints = [(int(point.split(':')[0]), int(point.split(':')[1])) for point in landmarksPoints]
                    frame = renderLandmarksToFrame(frame, landmarksPoints)

                counterOfLandmarksFrames += 1

                try:
                    while (landmarks[counterOfLandmarksFrames].split()[0] == str(frameId)):
                        counterOfLandmarksFrames += 1
                except Exception:
                        counterOfLandmarksFrames -= 1

        writer.write(frame)

        if frameId % 100 == 0:
            print('{} images processed'.format(frameId))

    cap.release()
    writer.release()

def countLabel(numberOfFrames, frameId, ageList):

    meanAgeLabel = np.zeros(len(ageList[0][2:]))
    counterOfAgeFrames = 0

    for index in range(-numberOfFrames, numberOfFrames + 1):
        for ageLabel in ageList:
            if (frameId + index) == int(ageLabel[0]):
                meanAgeLabel += np.asarray([float(age) for age in ageLabel[2:]])
                counterOfAgeFrames += 1
                break

    if counterOfAgeFrames != 0:
        meanAgeLabel = meanAgeLabel/float(counterOfAgeFrames)

    return meanAgeLabel


def renderAgeToFrame(frame, ageLabel):

    xShiftBar = frame.shape[1] - frame.shape[1]/15
    yShiftBar = frame.shape[1]/25
    heightShiftBar = frame.shape[1]/35
    widhtShiftBar = frame.shape[1]/6
    shiftBar = frame.shape[1]/30

    valuesAgeCatogories = [np.sum(ageLabel[:4]),
                           np.sum(ageLabel[4:8]),
                           np.sum(ageLabel[8:14]),
                           np.sum(ageLabel[14:23]),
                           np.sum(ageLabel[23:34]),
                           np.sum(ageLabel[34:45]),
                           np.sum(ageLabel[45:57]),
                           np.sum(ageLabel[57:])]
    ageCategories = ["(0, 2)", "(4, 6)", "(8, 12)", "(15, 20)", "(25, 32)", "(35, 43)", "(46, 53)", "(60, 100)"]
    for i, age in enumerate(ageCategories):
        cv2.putText(frame, age, (frame.shape[1] - frame.shape[1]/6, 2 * yShiftBar + shiftBar * (i + 2)), cv2.FONT_HERSHEY_SIMPLEX, frame.shape[0]/520.0, (0, 255, 0), thickness=2)
        cv2.rectangle(frame, (frame.shape[1] - frame.shape[1]/6, 2 * yShiftBar - heightShiftBar + shiftBar * (i + 2)), (frame.shape[1] - frame.shape[1]/6 - int(widhtShiftBar * float(valuesAgeCatogories[i])), 2 * yShiftBar + shiftBar * (i + 2)), (0, 255, 0), cv2.FILLED)

    meanAge = 0
    for i, value in enumerate(ageLabel):
        meanAge += value*i
    cv2.putText(frame, 'Mean Age: ' + str(meanAge), (frame.shape[1] - frame.shape[1]/3, 2 * yShiftBar + shiftBar * 10), cv2.FONT_HERSHEY_SIMPLEX, frame.shape[0]/520.0, (0, 255, 0),  thickness=2)

    return frame

def renderBoxToFrame(frame, boxPoints):

    cv2.rectangle(frame, boxPoints[0], boxPoints[1], (0, 0, 255), frame.shape[1]/600)

    return frame

def renderLandmarksToFrame(frame, landmarksPoints):

    rangeChin = 16
    rangeBrow = 4
    rangeNose = 8
    rangeEye = 5
    rangeOuterMouth = 11
    rangeInnerMouth = 7

    landCounter = 0

    thicknessOfLine = frame.shape[1]/600
    colorOfLine = (255, 0, 0)

    landmarksRange = [rangeChin, rangeBrow, rangeBrow, rangeNose, rangeEye, rangeEye, rangeOuterMouth, rangeInnerMouth]

    for landRange in landmarksRange:
        for index in range(0, landRange):
            cv2.line(frame, landmarksPoints[index + landCounter], landmarksPoints[index + 1 + landCounter], colorOfLine, thicknessOfLine)
        if landRange == rangeNose:
            cv2.line(frame, landmarksPoints[landCounter + 3], landmarksPoints[index + 1 + landCounter], colorOfLine, thicknessOfLine)
        if (landRange == rangeEye) or (landRange == rangeOuterMouth) or (landRange == rangeInnerMouth):
            cv2.line(frame, landmarksPoints[landCounter], landmarksPoints[index + 1 + landCounter], colorOfLine, thicknessOfLine)

        landCounter += landRange + 1


    return frame


def main():
    args = parse_args()

    renderVideo(args.file_Class, args.file_Land, args.input_video, args.output_video, args.number_Frames, args.render_Box, args.render_Landmarks)

if __name__ == "__main__":
    main()
