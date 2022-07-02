import numpy as np
import cv2
import pytesseract
import math
import time

detect = 0
n = 1
start_time = time.time()
Min_char = 0.01
Max_char = 0.09
######
GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 19 
ADAPTIVE_THRESH_WEIGHT = 9  
data =""
link = r'E:\NAMHOC\python\Pytorch\data\image\41.jpg'
img=cv2.imread(link)
#cv2.imshow('img', img)

imgGrayscale = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
#cv2.imshow('gray', imgGrayscale)
height, width = imgGrayscale.shape

imgBlurred = np.zeros((height, width, 1), np.uint8)
imgBlurred = cv2.GaussianBlur(imgGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
#cv2.imshow('blur',imgBlurred)
imgThreshplate = cv2.adaptiveThreshold(imgGrayscale, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)
#cv2.imshow('thresh', imgThreshplate)
################ Tiền xử lý ảnh #################
#imgThreshplate = cv2.adaptiveThreshold(imgGrayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,2)
canny_image = cv2.Canny(imgThreshplate,250,255) #Tách biên bằng canny
#cv2.imshow('canny', canny_image)
kernel = np.ones((2,2), np.uint8)
dilated_image =cv2.dilate(canny_image,kernel,iterations=1) #tăng sharp cho egde (Phép nở)
#cv2.imshow('delate', dilated_image)
###### vẽ contour và lọc biển số  #############
_,contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours= sorted(contours, key = cv2.contourArea, reverse = True)[:10] #Lấy 10 contours có diện tích lớn nhất
cv2.drawContours(img, contours, -1, (0, 255,255 ), 3) # Vẽ tất cả các contour trong hình lớn
cv2.imshow('contour',img)
largest = []
area_max = 0
largest_rectangle =None
for cnt in contours:
    peri = cv2.arcLength(cnt, True) #Tính chu vi
    approx = cv2.approxPolyDP(cnt, 0.06 * peri, True) 
    [x, y, w, h] = cv2.boundingRect(approx.copy())
    ratio = w/h
    #print(ratio)
    #print(len(approx))
    if len(approx)==4 and ratio <= 5 and ratio >1:
        #print(x,y,w,h)
        area = w*h
        if area > area_max:
            area_max =area
            largest_rectangle = approx
largest.append(largest_rectangle)
if  largest and largest_rectangle is None:
    print("No detect")
else:
    detect = True
if detect is True:
    cv2.drawContours(img, [largest_rectangle], 0, (255, 255, 0), 3) #Khoanh vùng biển số xe
    cv2.imshow('draw', img)
    ############## Tìm góc xoay ảnh #####################
    print(largest_rectangle)
    (ax,ay) = largest_rectangle[3,0]
    (bx,by) = largest_rectangle[2,0]
    (cx,cy) = largest_rectangle[1,0]
    (dx,dy) = largest_rectangle[0,0]
    point = [(ax,ay), (bx,by), (cx,cy), (dx,dy)]
    point.sort(reverse=True, key=lambda x:x[1])
    (cx,cy) = point[0]
    (dx,dy) = point[1]
    cv2.circle(img, (cx, cy), 2, (0, 255, 255), -1)
    cv2.circle(img, (dx, dy), 2, (255, 255, 255), -1)
    cv2.imshow('img', img)
    doi = abs(dy-cy)
    ke = abs(dx-cx)
    # tim diem c va d de chon tam quay
    if cx < dx:
        (cx,cy) = point[0]
        (dx,dy) = point[1]
    else:
        (cx,cy) = point[1]
        (dx,dy) = point[0]   
    angle = math.atan(doi/ke)*(180.0/math.pi)
    ########## Cắt biển số ra khỏi ảnh và xoay ảnh ################
    x,y,w,h = cv2.boundingRect(largest_rectangle)
    roi =img[y:y+h,x:x+w]
    cv2.imshow('cat', roi)
    height, width = roi.shape[:2]
    C_rot = (0, height)
    D_rot = (width,height)
    if cy >= dy :
        rotate_matrix = cv2.getRotationMatrix2D(center=C_rot, angle=-angle, scale=1)
        print(angle)
    else:
        rotate_matrix = cv2.getRotationMatrix2D(center=D_rot, angle=angle, scale=1)
        print(angle)
    roi = cv2.warpAffine(src=roi, M=rotate_matrix,dsize=(w,h))
    roi = cv2.resize(roi,(0,0),fx = 2, fy = 2)
    cv2.imshow('roi', roi)
    ####################################
    #################### Tiền xử lý ảnh đề phân đoạn kí tự ####################
    ###############
    gray_crop = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_crop, (5,5),0)
    _, thresh = cv2.threshold (blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    #cv2.imshow('m', opening)
    invert = 255 - opening
    #cv2.imshow('in', invert)
    #=======
    _,cont,hier = cv2.findContours(opening,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 

    cv2.drawContours(roi, cont, -1, (100, 100, 255), 2) #Vẽ contour các kí tự trong biển số
    cv2.imshow('draw_con', roi)
    ##################### Lọc vùng kí tự #################
    char_x_ind = {}
    char_x = []
    height, width,_ = roi.shape
    roiarea = height * width

    for ind,cnt in enumerate(cont) :
        (x,y,w,h) = cv2.boundingRect(cont[ind])
        ratiochar = w/h
        char_area = w*h
        #cv2.putText(roi, str(char_area), (x, y+20),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 2)
        #cv2.putText(roi, str(ratiochar), (x, y+20),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 2)

        if (Min_char*roiarea < char_area < Max_char*roiarea) and ( 0.25 < ratiochar < 0.7):
            if x in char_x: #Sử dụng để dù cho trùng x vẫn vẽ được
                x = x + 1
            char_x.append(x)    
            char_x_ind[x] = ind

            #cv2.putText(roi, str(char_area), (x, y+20),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 2)          
    ############ Cắt và nhận diện kí tự ##########################

    char_x = sorted(char_x)
    row1 =0
    row2 =0
    data =""
    data_row1 =""
    data_row2 ="" 
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    for i in char_x:
        (x,y,w,h) = cv2.boundingRect(cont[char_x_ind[i]])
        cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),2)
        imgROI = invert[y:y+h,x:x+w]  # cắt kí tự ra khỏi hình
        #kernel = np.ones((10,10), np.uint8)
        #imgROI = cv2.dilate(imgROI,kernel)
        #cv2.imshow(str(i),imgROI)
        if y>height/2.5: # tim ki tu hang 2
            row1 = invert[0:y-10,0:width]
            row2 = invert[y-10:height,0:width]
            data_row1 = pytesseract.image_to_string(row1, lang='eng', config='--psm 6')  
            data_row2 = pytesseract.image_to_string(row2, lang='eng', config='--psm 6')
            data =data_row1 + data_row2
            break        
        else:

            data2 = pytesseract.image_to_string(invert, lang='eng', config='--psm 6')
            data = data2
            #print(data_row2)      
    #cv2.imshow('roi', roi)

    # xu ly ki tu
    data = ''.join(filter(str.isalnum, data)) # loc ki tu dac biet
    print(data)
    if len(data) >2:
          
        if data[2] is "4": # loc ki tu thuong nhan sai A thanh 4
            data =list(data)
            data[2] = 'A'
            data =''.join(data)
        if data[2] is "8": # loc ki tu thuong nhan sai A thanh 4
            data =list(data)
            data[2] = 'B'
            data =''.join(data)
    print(data)
print("--- %s seconds ---" % (time.time() - start_time))
cv2.waitKey()
cv2.destroyAllWindows()