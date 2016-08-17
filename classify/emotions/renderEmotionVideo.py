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
    parser.add_argument('-ec', '--emoticons-dir',
                        required=True,
                        help='Directory with images of emoticons for each emotion (angry.png, disgust.png, fear.png, happy.png, sad.png, surprise.png, neutral.png)')
    parser.add_argument('-fc', '--file-Class',
                        required=True,
                        help='File with emotions predictions.')
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

def renderVideo(emotionsFile, landmarkFile, emoticonsDirectory, inputVideo, outputVideo, numberOfFrames, renderBox, renderLandmarks):

    print("Creating video...")

    cap = cv2.VideoCapture(inputVideo)
    resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)


    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(outputVideo + '.avi', fourcc, fps, resolution, 1)

    with open(emotionsFile) as f:
        emotionsList = f.read().splitlines()

    for index in range(0, len(emotionsList)):
        emotionsList[index] = emotionsList[index].split()

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

        emotionsLabel = countLabel(numberOfFrames, frameId, emotionsList)
        frame = renderEmotionsToFrame(frame, emotionsLabel, emoticonsDirectory)

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

def countLabel(numberOfFrames, frameId, emotionsLabels):

    meanEmotionLabel = np.zeros(7)
    counterOfEmotionsFrames = 0

    for index in range(-numberOfFrames, numberOfFrames + 1):
        for emotionsLabel in emotionsLabels:
            if (frameId + index) == int(emotionsLabel[0]):
                meanEmotionLabel += np.asarray([float(emotion) for emotion in emotionsLabel[2:]])
                counterOfEmotionsFrames += 1
                break

    if counterOfEmotionsFrames != 0:
        meanEmotionLabel = meanEmotionLabel/float(counterOfEmotionsFrames)

    return meanEmotionLabel


def renderEmotionsToFrame(frame, emotionsLabel, emoticonsDirectory):

    xShiftBar = frame.shape[1]/15
    yShiftBar = frame.shape[1]/25
    heightShiftBar = frame.shape[1]/35
    widhtShiftBar = frame.shape[1]/6
    shiftBar = frame.shape[1]/30

    emotionImageFiles = ["angry.png", "disgust.png", "fear.png", "happy.png", "sad.png", "surprise.png", "neutral.png"]
    emotionIcons = [cv2.imread(os.path.join(emoticonsDirectory, name), 1) for name in emotionImageFiles]
    emotionIcons = [cv2.resize(im, (heightShiftBar, heightShiftBar), interpolation=cv2.INTER_AREA) for im in emotionIcons]

    for i, icon in enumerate(emotionIcons):
        frame[yShiftBar + shiftBar *i : yShiftBar + shiftBar*i + icon.shape[0], 0:icon.shape[1]] = icon
        cv2.rectangle(frame, (xShiftBar, yShiftBar + shiftBar*i), (xShiftBar + int(widhtShiftBar*float(emotionsLabel[i])), yShiftBar + heightShiftBar + shiftBar*i), (0, 255, 0), cv2.FILLED)

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

    renderVideo(args.file_Class, args.file_Land, args.emoticons_dir, args.input_video, args.output_video, args.number_Frames, args.render_Box, args.render_Landmarks)

if __name__ == "__main__":
    main()
