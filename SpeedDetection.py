from Tracking import *
import cv2
import numpy as np
import time
import os

f = 25
w = int(1000/(f-1))
print(w)
end = 0

limit = 45 #km/hr
dist = 28.87  #meters(area width between starting point of  blue and green area)

file = open("SpeedRecord.txt","w")
file.write("ID \t SPEED\n------\t-------\n")
file.close()

#Creater Tracker Object
tracker = EuclideanDistTracker()

if not os.path.exists('captures'):
    os.makedirs('captures')
if not os.path.exists('captures/ exceeded'):
    os.makedirs('captures/ exceeded')

#cap = cv2.VideoCapture("Resources/traffic3.mp4")
cap = cv2.VideoCapture("traffic4.mp4")

#Object Detection
object_detector = cv2.createBackgroundSubtractorMOG2(history=None, varThreshold=None)
#100,5

car_in = {}
car_et = {}
img_cut= {}

#KERNALS
kernalOp = np.ones((3, 3), np.uint8)
kernalOp2 = np.ones((5, 5), np.uint8)
kernalCl = np.ones((11, 11), np.uint8)
fgbg=cv2.createBackgroundSubtractorMOG2(detectShadows=True)
kernal_e = np.ones((5,5),np.uint8)


while True:
    ret,frame = cap.read()
    roi = cv2.resize(frame, None, fx=0.5, fy=0.5)

    #area1 = [(294, 203), (645, 203), (679, 230), (277, 230)]
    area2 = [(277, 230), (679, 230),(802,352),(143,355)]
    area3 = [(106,466), (792,463), (820,516),(94,513)]
    area_capture = [(165,309), (760,309), (760,184), (165,184)]

    #cv2.polylines(roi, [np.array(area_capture, np.int32)], True, (225, 220, 10), 3)
    #cv2.polylines(roi, [np.array(area2, np.int32)], True, (22, 220, 10), 3)
    #cv2.polylines(roi, [np.array(area3, np.int32)], True, (225, 22, 10), 3)


    #height,width, = frame.shape
    #print(height,width)

    #Extract ROI
    #roi = frame[50:540,200:960]

    #MASKING METHOD 1
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)

    #DIFFERENT MASKING METHOD 2 -> This is used
    fgmask = fgbg.apply(roi)
    ret, imBin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    mask1 = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernalOp)
    mask2 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernalCl)
    e_img = cv2.erode(mask2, kernal_e)


    contours,_ = cv2.findContours(e_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        #THRESHOLD
        if area > 1000:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),3)
            detections.append([x,y,w,h])

    #Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x,y,w,h,id = box_id
        cx = (x + x + w) // 2
        cy = (y + y +h+h) // 2

        detect3 = cv2.pointPolygonTest(np.array(area3, np.int32),(int(cx),int(cy)), False)
        if detect3 >= 0:
            car_in[id] = time.time()
            cv2.circle(roi, (cx, cy), 5, (255, 200, 220), -1)

        if id in car_in:
            cv2.circle(roi, (cx, cy), 5, (255, 200, 220), -1)
            detect2= cv2.pointPolygonTest(np.array(area2, np.int32), (int(cx), int(cy)), False)
            if detect2 >= 0:
                elapsed = time.time() - car_in[id]
                if id not in car_et:
                    car_et[id] =elapsed
                if id in car_et:
                    elapsed = car_et[id]

                    detect_capture = cv2.pointPolygonTest(np.array(area_capture, np.int32), (int(cx), int(cy)), False)

                    #SPEED CALCULATION
                    v_ms = dist/ elapsed
                    v_khr = v_ms*3.6
                    filet = open("SpeedRecord.txt", "a")

                    if limit>= v_khr:
                        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        cv2.rectangle(roi, (x, y), (x+165, y-30), (0, 255, 0), -1)
                        cv2.putText(roi, "car "+str(id)+",  "+f'{v_khr:.2f}'+" km/hr", (x, y - 10), 0, 0.5, (255, 255, 255), 2)

                        if detect_capture >= 0:
                            if id not in img_cut:
                                filet.write("car " + str(id) + " \t " + f'{v_khr:.2f}' + " km/hr" + "\n")
                                img_cut[id] = roi[y - 5:y + h + 5, x - 5:x + w + 5]
                                cv2.imwrite('./captures/car_id ' + str(id) + " speed " + f'{v_khr:.2f}' + '.jpg',
                                            img_cut[id])
                            if id in img_cut:
                                filet = filet
                                img_cut[id] = img_cut[id]

                    else:
                        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 3)
                        cv2.rectangle(roi, (x, y), (x + 165, y - 30), (0, 0, 255), -1)
                        cv2.putText(roi, "car " + str(id) + ",  " + f'{v_khr:.2f}' + " km/hr", (x, y - 10), 0, 0.5,(255, 255, 255), 2)
                        if detect_capture >= 0:
                            if id not in img_cut:
                                filet.write("car " + str(id) + " \t " + f'{v_khr:.2f}' + " km/hr" + "<---exceeded" + "\n")
                                img_cut[id] = roi[y - 5:y + h + 5, x - 5:x + w + 5]
                                cv2.imwrite('./captures/ exceeded/car_id '+str(id)+" speed "+ f'{v_khr:.2f}'+'.jpg', img_cut[id])
                            if id in img_cut:
                                filet =filet
                                img_cut[id]= img_cut[id]





    cv2.imshow("ROI", roi)

    key = cv2.waitKey(w-10)
    if key==27:
        tracker.end()
        end=1
        break

if(end!=1):
    tracker.end()

cap.release()
cv2.destroyAllWindows()