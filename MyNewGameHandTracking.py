import cv2
import  mediapipe as mp
import time
import HandTrackingModule as htm


pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector = htm.handDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
        print(lmList[4])

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    img = cv2.flip(img, 1)
    cv2.putText(img, str(round(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)