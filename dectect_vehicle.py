import cv2
import numpy as np
import math
import msvcrt
import plate
START_POINT = 450
END_POINT = 0
CLASSES = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
           "boat", "traffic light"]
# Define vehicle class
VEHICLE_CLASSES = [1, 2, 3, 5, 6, 7]

# get it at https://pjreddie.com/darknet/yolo/
YOLOV3_CFG = r'E:\NAMHOC\python\DO_AN\yolov3-tiny.cfg'
YOLOV3_WEIGHT = r'E:\NAMHOC\python\DO_AN\yolov3-tiny.weights'

CONFIDENCE_SETTING = 0.4
YOLOV3_WIDTH = 416
YOLOV3_HEIGHT = 416

MAX_DISTANCE = 40


def get_output_layers(net):
    """
    Get output layers of darknet
    :param net: Model
    :return: output_layers
    """
    try:
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers
    except:
        print("Can't get output layers")
        return None


def detections_yolo3(net, image, confidence_setting, yolo_w, yolo_h, frame_w, frame_h, classes=None):
    """
    Detect object use yolo3 model
    :param net: model
    :param image: image
    :param confidence_setting: confidence setting
    :param yolo_w: dimension of yolo input
    :param yolo_h: dimension of yolo input
    :param frame_w: actual dimension of frame
    :param frame_h: actual dimension of frame
    :param classes: name of object
    :return:
    """
    img = cv2.resize(image, (yolo_w, yolo_h))
    blob = cv2.dnn.blobFromImage(img, 0.00392, (yolo_w, yolo_h), swapRB=True, crop=False)
    net.setInput(blob)
    layer_output = net.forward(get_output_layers(net))

    boxes = []
    class_ids = []
    confidences = []

    for out in layer_output:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_setting and class_id in VEHICLE_CLASSES:
                #print("Object name: " + classes[class_id] + " - Confidence: {:0.2f}".format(confidence * 100))
                center_x = int(detection[0] * frame_w)
                center_y = int(detection[1] * frame_h)
                w = int(detection[2] * frame_w)
                h = int(detection[3] * frame_h)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    return boxes, class_ids, confidences


def draw_prediction(classes, colors, img, class_id, confidence, x, y, width, height):
    """
    Draw bounding box and put classe text and confidence
    :param classes: name of object
    :param colors: color for object
    :param img: immage
    :param class_id: class_id of this object
    :param confidence: confidence
    :param x: top, left
    :param y: top, left
    :param width: width of bounding box
    :param height: height of bounding box
    :return: None
    """
    try:
        label = str(classes[class_id])
        color = colors[class_id]
        center_x = int(x + width / 2.0)
        center_y = int(y + height / 2.0)
        x = int(x)
        y = int(y)
        width = int(width)
        height = int(height)

        cv2.rectangle(img, (x, y), (x + width, y + height), color, 1)
        #cv2.circle(img, (center_x, center_y), 2, (0, 255, 0), -1)
        cv2.putText(img, label + ": {:0.2f}%".format(confidence * 100), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    except (Exception, cv2.error) as e:
        print("Can't draw prediction for class_id {}: {}".format(class_id, e))


def check_location(box_y, box_height, height):
    """
    Check center point of object that passing end line or not
    :param box_y: y value of bounding box
    :param box_height: height of bounding box
    :param height: height of image
    :return: Boolean
    """
    center_y = int(box_y + box_height / 2.0)
    if center_y > height - END_POINT:
        return True
    else:
        return False


def check_start_line(box_y, box_height):
    """
    Check center point of object that passing start line or not
    :param box_y: y value of bounding box
    :param box_height: height of bounding box
    :return: Boolean
    """
    center_y = int(box_y + box_height / 2.0)
    if center_y < START_POINT:
        return True
    else:
        return False

def readch():

    ch = msvcrt.getch()
    if ch in b'\x00\xe0':  # Arrow or function key prefix?
        ch = msvcrt.getch()  # Second call returns the actual key code.
    return ch 

def put_data_to_csv(classes,class_id,data,number_frame):
    file_open = open(r'E:\NAMHOC\python\Pytorch\data\data.csv',mode = "r+", encoding ="utf-8-sig")
    time = number_frame/60
    print(data)
    time = str(time)
    label = str(classes[class_id])
    data =str(data)
    if data != "":
        lines = file_open.readlines()
        header =data+","+label+","+time+"\n"
        file_open.write(header)

def counting_vehicle(video_input, video_output, skip_frame=10):
    colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # Load yolo model
    net = cv2.dnn.readNetFromDarknet(YOLOV3_CFG, YOLOV3_WEIGHT)

    # Read first frame
    cap = cv2.VideoCapture(video_input)
    ret_val, frame = cap.read()
    width = frame.shape[1]
    height = frame.shape[0]

    # Define format of output
    video_format = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_output, video_format, 25, (width, height))
    Stoptracker_box_current = False
    data = 0
    new_tracker = 0
    number_reset_traker_change = 0
    temp = 0
    timeout =0
    number_reset_traker = 0

    # Define tracking object
    list_object = []
    number_frame = 0
    number_vehicle = 0
    red_light = False
    #while cap.isOpened():
        number_frame += 1
        # Read frame
        ret_val, frame = cap.read()
        if frame is None:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def dectected():        

        if red_light is True: #check redlight 
            # Tracking old object
            tmp_list_object = list_object
            list_object = []
            #print(tmp_list_object)
            for obj in tmp_list_object:

                tracker = obj['tracker']
                class_id = obj['id']
                confidence = obj['confidence']
                #print(obj['box'])
                #kiem tra box tren fame moi (kiem tra co xi dich ve toa do)
                check, box = tracker.update(frame)
                #khao tao 1 list trong va 1 list bo nho de nho cai box truoc do
                # check neu box do vuot qua thi k luu lai con chua thi lai luu
                if check:
                    box_x, box_y, box_width, box_height = box
                    draw_prediction(CLASSES, colors, frame, class_id, confidence,
                                    box_x, box_y, box_width, box_height)
                    obj['tracker'] = tracker
                    obj['box'] = box
                    if Stoptracker_box_current: # neu ma truy xuat dc bien so thi k tracker 
                        # This object passed the end line
                        put_data_to_csv(CLASSES,class_id,data,number_frame)
                        Stoptracker_box_current =False
                    else:
                        list_object.append(obj)
            #print(list_object)
            if number_frame % skip_frame == 0:
                # Detect object and check new object
                boxes, class_ids, confidences = detections_yolo3(net, frame, CONFIDENCE_SETTING, YOLOV3_WIDTH,
                                                                YOLOV3_HEIGHT, width, height, classes=CLASSES)
                #print(boxes)
                # quet tat ca id box va box trong boxes
                for idx, box in enumerate(boxes):
                    box_x, box_y, box_width, box_height = box
                    #print(box)
                    if not Stoptracker_box_current:
                        #print(Stoptracker_box_current)
                        # This object doesnt pass the end line
                        box_center_x = int(box_x + box_width / 2.0)
                        box_center_y = int(box_y + box_height / 2.0)
                        check_new_object = True
                        for tracker in list_object: # check cac box cu so voi box neu khoang cach be hon 80 thi break
                            # box trong nay la box hien tai, khi update fame
                            # Check exist object
                            current_box_x, current_box_y, current_box_width, current_box_height = tracker['box']
                            current_box_center_x = int(current_box_x + current_box_width / 2.0)
                            current_box_center_y = int(current_box_y + current_box_height / 2.0)
                            # Calculate distance between 2 object
                            distance = math.sqrt((box_center_x - current_box_center_x) ** 2 +
                                                (box_center_y - current_box_center_y) ** 2)
                            if distance < MAX_DISTANCE:
                                # Object is existed
                                check_new_object = False
                                break
                        #print("ca")
                        #print(new_tracker)
                        temp_tracker = new_tracker
                        if check_new_object and check_start_line(box_y, box_height):
                            # Append new object to list
                            new_tracker = cv2.TrackerKCF_create()
                            new_tracker.init(frame, tuple(box))
                            new_object = {
                                'id': class_ids[idx],
                                'tracker': new_tracker,
                                'confidence': confidences[idx],
                                'box': box
                            }
                            list_object.append(new_object)
                            # Draw new object
                            draw_prediction(CLASSES, colors, frame, new_object['id'], new_object['confidence'],
                                            box_x, box_y, box_width, box_height)
                            #print(temp_tracker)
                            #print(new_tracker)
                        #print(box)
                        if temp_tracker is new_tracker and box_center_x > width/4 and box_center_x <3*width/4 :
                            #img = frame[int(box_y-box_y/4):int(box_y+box_height),int(box_x-box_x/4):int(box_x+box_width)]
                            img = frame[int(abs(box_y)):int(abs(box_y)+abs(box_height)),int(abs(box_x)):int(abs(box_x)+abs(box_width))]
                            # record bien so
                            #temp_data = data
                            data =plate.recognize(img)
                            # kiem tra 2 frame co cung bien so 
                            #print(data)
                            #if data == data :
                                #print(data)
                            Stoptracker_box_current = True
                            temp_data = 0
                            #data =0
                            timeout = 0
                            number_reset_traker +=1
                            if number_reset_traker ==2:
                                new_tracker = 0
                                number_reset_traker = 0
        number_reset_traker_change +=1
        if number_reset_traker_change ==5:
            number_reset_traker_change ==0
            new_tracker = 0   
        # Draw start line
        cv2.line(frame, (0, START_POINT), (width, START_POINT), (255, 255, 255), 5)
        cv2.imshow("video", frame)
        if msvcrt.kbhit():
            key = ord(readch())
            print(key)
            if key == 114:  # ord('r')
                red_light = True
                print("la")
            elif key == 98:  # b key?
                red_light = False
            elif key == 113:
                break
        key = cv2.waitKey(1)
        out.write(frame)


if __name__ == '__main__':
    counting_vehicle(r'E:\NAMHOC\python\Pytorch\data\video\video2.mp4', 'plate.avi')