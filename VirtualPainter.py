import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

#######################

brushThickness = 15
eraserThickness = 50

#######################

folderPath ='Painting'
myList = os.listdir(folderPath)
overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
header = overlayList[0]
drawColor = (255,0,255)

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector = htm.handDetector(detectionCon=0.85)
xp,yp =0,0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    #1.import image
    success, img = cap.read()
    img = cv2.flip(img,1)
    #2. Find Hans landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img=img,draw=False)

    if len(lmList) != 0:
        #print(lmList)

        # tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]



        #3. Check which fingers are up
        fingers = detector.fingersUp()
        print(x1,y1)

        #4. If selection mode - Two Finger are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0

            #print("selection mode")

            #checking for the click
            if y1 < 125 :
                if 0<x1<140:
                    header = overlayList[0]
                    drawColor = (0,0,0)
                if 140<x1<280:
                    header = overlayList[1]
                    drawColor = (0,255,0)
                if 280<x1<440:
                    header = overlayList[2]
                    drawColor = (255,255,255)
                if 440<x1<580:
                    header = overlayList[3]
                    drawColor = (255, 0, 0)
                elif 580<x1<740:
                    header = overlayList[4]
                    drawColor = (0, 0, 255)
                elif 740<x1<920:
                    header = overlayList[5]
                    drawColor = (0,0,0)

            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)


        #5. If Drawing mode - Index finger is up
        if fingers[1] and fingers[2]==False:
            cv2.circle(img,(x1,y1),10,(255,0,255),cv2.FILLED)
            #print("Drawing mode")
            if xp==0 and yp == 0:
                xp,yp = x1,y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            cv2.line(img, (xp,yp), (x1,y1), drawColor, brushThickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _,imgInv = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    #setting the header image
    img[0:120,0:920]=header
    img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("camera",img)
    cv2.imshow("Canvas", imgCanvas)
    cv2.waitKey(1)