import dees
import time
import cv2
import numpy as np
detect = True
red = 30
yellow = 3
green = 25
detect =False
#colors = np.random.uniform(0, 255, size=(len(dees.CLASSES), 3))

# Load yolo model
net = cv2.dnn.readNetFromDarknet(dees.YOLOV3_CFG, dees.YOLOV3_WEIGHT)

# Read first frame
cap = cv2.VideoCapture(r'E:\NAMHOC\python\hif.mp4')
ret_val, frame = cap.read()
width = frame.shape[1]
height = frame.shape[0]

# Define format of output
video_format = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(r'E:\NAMHOC\python\Pytorch\vehicles.avi', video_format, 25, (width, height))
# Define tracking object
list_object = []
number_frame = 0
number_vehicle = 0
while cap.isOpened():
    number_frame += 1
    ret_val, frame = cap.read()
    if frame is None:
        break
    #Tracking old object
    frame = dees.countign_vehicle(detect, number_frame, ret_val, frame, width, height, net,list_object)
    out.write(frame)
cap.release()
out.release()
cv2.destroyAllWindows()
     