import cv2
import numpy as np
import utils
import time
from pyzbar.pyzbar import decode
#import serial

curveList = []
avgVal = 10


def sendToArduino(message):
    serialcmd = str(message) + "\n"
    ser.write(serialcmd.encode())
    # line = ser.readline().decode("utf-8").rstrip()
    # print(line)


def qrReader(cap):
    with open("/home/beesh/Desktop/pyPro/dmt2f/myDatafile.txt") as f:
        myDatalist = f.read().splitlines()

    while True:
        myOutput = 0
        print("working....")
        success, img = cap.read()
        for barcode in decode(img):
            myData = barcode.data.decode("utf-8")
            if myData in myDatalist:
                myOutput = 100

            else:
                myOutput = 200

        if myOutput != 0:
            return myOutput


def getLaneCurve(img, imag, imgContour, areaminimum=5000, display=2):

    global imgLanecolor
    imgCopy = img.copy()
    imgResult = img.copy()
    ###### STEP1 #######

    imgThres = utils.thresholding(img)

    ###### STEP2 #######

    hT, wT, c = img.shape
    points = utils.valTrackbars()
    imgWarp = utils.warpImg(imgThres, points, wT, hT)
    imgWarpPoints = utils.drawPoints(imgCopy, points)

    ###### STEP3 #######

    midPoint, imgHist = utils.getHistogram(imgWarp, display=True, minPer=0.5, region=4)
    curveAveragePoint, imgHist = utils.getHistogram(imgWarp, display=True, minPer=0.9)
    curveRaw = curveAveragePoint - midPoint

    ###### STEP4 #######

    curveList.append(curveRaw)
    if len(curveList) > avgVal:
        curveList.pop(0)
    curve = int(sum(curveList) / len(curveList))

    ###### STEP5 #######

    if display != 0:
        imgInvwarp = utils.warpImg(imgWarp, points, wT, hT, inv=True)
        imgInvwarp = cv2.cvtColor(imgInvwarp, cv2.COLOR_GRAY2BGR)
        imgInvwarp[0 : hT // 3, 0:wT] = 0, 0, 0
        imgLanecolor = np.zeros_like(img)
        imgLanecolor[:] = 0, 255, 0
        imgLanecolor = cv2.bitwise_and(imgInvwarp, imgLanecolor)
        imgResult = cv2.addWeighted(imgResult, 1, imgLanecolor, 1, 0)
        midY = 450
        cv2.putText(
            imgResult,
            str(curve),
            (wT // 2 - 80, 85),
            cv2.FONT_HERSHEY_COMPLEX,
            2,
            (255, 0, 0),
            5,
        )
        cv2.line(
            imgResult, (wT // 2, midY), (wT // 2 + (curve * 3), midY), (255, 0, 255), 5
        )
        cv2.line(
            imgResult,
            ((wT // 2 + (curve * 3)), midY - 25),
            (wT // 2 + (curve * 3), midY),
            (255, 0, 255),
            5,
        )
        for x in range(-30, 30):
            w = wT // 20
            cv2.line(
                imgResult,
                (w * x + int(curve // 50), midY - 10),
                (w * x + int(curve // 50), midY + 10),
                (0, 0, 255),
                2,
            )

    if display == 2:
        stack = utils.stackImages(
            0.8, ([img, imgLanecolor, imgWarp], [imgWarpPoints, imgHist, imgResult])
        )

        cv2.imshow("Result", stack)
    elif display == 1:
        cv2.imshow("Result", imgResult)

    ##### NORMALIZATION #####
    curve = curve / 100
    if curve > 1:
        curve == 1
    if curve < -1:
        curve == -1
    ##### GET CONTOURS #####
    im2, contours, hierarchy = cv2.findContours(
        imag, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = areaminimum
        if area > areaMin:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)

            if len(approx) == 4:
                cv2.putText(
                    imgContour,
                    "square",
                    (x, y),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                )
                return midPoint, 50

            if len(approx) == 3:
                cv2.putText(
                    imgContour,
                    "Triangle",
                    (x, y),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                )
                return midPoint, 60

    return midPoint, 0


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    initialTrackbarVals = [158, 147, 160, 220]
    utils.initializeTrackbars(initialTrackbarVals)
    frameCounter = 0
    #ser = serial.Serial("/dev/ttyACM0", baudrate=57600, timeout=1)
    #ser.flush
    while True:
        frameCounter += 1
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0
        success, img = cap.read()
        if success:
            img = cv2.resize(img, (480, 240))
            imgContour = img.copy()
            imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            threshold1 = 150
            threshold2 = 255
            imgCanny = cv2.Canny(imgGrey, threshold1, threshold2)
            kernel = np.ones((5, 5))
            imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
            curve, contours = getLaneCurve(img, imgDil, imgContour, areaminimum=5000, display=2)
            curve = curve / 10
            sendToArduino(curve)
            print(curve)
            #time.sleep(0.5)
            if contours > 0:
                sendToArduino(contours)
                print(contours)
            if contours == 60:
                qr = qrReader(cap)
                print(qr)
                sendToArduino(qr)

        cv2.waitKey(1)
