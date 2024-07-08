import numpy as np
import math
import random
import cv2

class KalmanFilter:
    kf = cv2.KalmanFilter(4,2)
    kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
    kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)

    def predict(self, cordX, cordY):
        measured = np.array([[np.float32(cordX)],[np.float32(cordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        x,y = int(predicted[0]), int(predicted[1])
        return x,y


kf = KalmanFilter()

cap = cv2.VideoCapture('archery.mp4')

frameCounter = 0
vel = 0 
centers = []   # Board Centers
predicted_center = []  # Axis
angularVelocities = []

while True:
    _, img = cap.read()
    frameCounter += 1
    #If the last frame is reached, reset the capture and the frame_counter
    if frameCounter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        frameCounter = 0
        centers = []
        predicted_center = []
        angularVelocities = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Hough Transform
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.blur(gray, (7, 7)) 
    canny = cv2.Canny(gray_blurred, 100,200)
    detected_circles = cv2.HoughCircles(canny,  
                    cv2.HOUGH_GRADIENT, 1, 20, param1 = 40, 
                param2 = 25, minRadius = 70, maxRadius = 85)

    # Draw circles that are detected. 
    if detected_circles is not None: 

        # Convert the circle parameters a, b and r to integers. 
        detected_circles = np.uint16(np.around(detected_circles)) 

        for pt in detected_circles[0, :]: 
            a, b, r = pt[0], pt[1], pt[2] 
            centers.append((a,b))

            cv2.circle(img, (a, b), r, (0, 255, 0), 2) # Circumference
            cv2.circle(img, (a, b), 1, (0, 0, 255), 3) # Center
            break
    
    for pt in centers:
        cv2.circle(img, (pt[0], pt[1]), 2, (255, 0, 0), 1)
    if len(centers) > 100: del centers[0]   # remove old points

    if len(centers) == 100:
        # Calculate Axis
        pt1, pt2, pt3 = random.sample(centers, 3)
        den = np.array([[pt1[0],pt1[1],1],[pt2[0],pt2[1],1],[pt3[0],pt3[1],1]])
        x_num = np.array([[pt1[0]**2+pt1[1]**2, pt1[1], 1], [pt2[0]**2+pt2[1]**2, pt2[1], 1], [pt3[0]**2+pt3[1]**2, pt3[1], 1]])
        y_num = np.array([[pt1[0]**2+pt1[1]**2, pt1[0], 1], [pt2[0]**2+pt2[1]**2, pt2[0], 1], [pt3[0]**2+pt3[1]**2, pt3[0], 1]])

        if round(np.linalg.det(den)) != 0:
            x0 = int(0.5*round(np.linalg.det(x_num))/round(np.linalg.det(den)))
            y0 = int(-0.5*round(np.linalg.det(y_num))/round(np.linalg.det(den)))
            predicted_center.append((x0, y0))

            c = np.average(np.array(predicted_center), axis=0).astype(int)

            cv2.circle(img, (c[0], c[1]), 5, (0, 0, 255), -1)

            # Radius
            cv2.line(img, c, centers[-1], (0,0,255), 2)
            rad = np.linalg.norm(c - np.array(centers[-1]))
            # r = []
            # for pt in centers:
            #     r.append(np.linalg.norm(np.array(pt) - c))
            cv2.putText(img, f"R - {rad:.2f}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

            # Slope
            if (c[0] - centers[-1][0]) != 0:
                slope = (c[1] - centers[-1][1])/(c[0] - centers[-1][0])
            # print(slope)
            cv2.putText(img, f"m - {slope:.2f}", (50,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

            # Using Kalman Filter
            predicted = centers[-20][0], centers[-10][1]
            for i in range(28):
                predicted = kf.predict(predicted[0], predicted[1])
                if -20+i < 0: predicted = centers[-20+i][0], centers[-20+i][1]
            cv2.circle(img, (int(predicted[0]), int(predicted[1])), 10, (0, 0, 255), 2)

            
            # ANgular Velocity
            # if (c[0] - predicted[0]) != 0:
            #     slope_predicted = (c[1] - predicted[1])/(c[0] - predicted[0])
            #     angle = math.atan(abs((slope_predicted - slope)/(1-slope*slope_predicted)))
            #     w = angle * 60
            #     angularVelocities.append(w)
            #     cv2.putText(img, f"w - {np.average(angularVelocities):.2f}", (50,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)


            


    cv2.imshow("Detected Circle", img)
    # cv2.imshow('canny', canny)
    # cv2.imshow('blur', gray_blurred)
            # cv2.waitKey(0) 

# cv2.imshow('video',img)

    if cv2.waitKey(25) & 0xFF == ord('q'): 
        
        break
    
cap.release()
cv2.destroyAllWindows()