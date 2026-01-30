import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0

folder = "C:\\Users\\ASUS\\OneDrive\\Desktop\\AI SIGN\\Data\\No"

while True:
    success, img = cap.read()
    if not success:
        break

    hands, img = detector.findHands(img) 
    if hands:
        hand1 = hands[0]
    
        x, y, w, h = hand1["bbox"]   

        
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

     
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(img.shape[1], x + w + offset)
        y2 = min(img.shape[0], y + h + offset)

        imgCrop = img[y1:y2, x1:x2]

        
        if imgCrop.size == 0:
            
            continue

       
        aspectRatio = h / w

        if aspectRatio > 1:
           
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
           
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wGap + wCal] = imgResize

        else:
           
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hGap + hCal, :] = imgResize

       
        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
       
        counter += 1
        filename = f"{folder}/Image_{int(time.time())}_{counter}.jpg"
        cv2.imwrite(filename, imgWhite)
        print("Saved:", filename, "Count:", counter)

    
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
