import cv2
import numpy as np
import Preprocess
import math
import time

ADAPTIVE_THRESH_BLOCK_SIZE = 19 
ADAPTIVE_THRESH_WEIGHT = 9  

n = 1

Min_char = 0.01
Max_char = 0.09
start_time = time.time()
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30
link = r'E:\NAMHOC\python\img\.jpg'
img = cv2.imread(link)
img = cv2.resize(img,dsize = (1920,1080))


###################### So sánh việc sử dụng độ tương phản#############
#img2 = cv2.imread("1.jpg")
#imgGrayscaleplate2, _ = Preprocess.preprocess(img)
#imgThreshplate2 = cv2.adaptiveThreshold(imgGrayscaleplate2, 250, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE ,ADAPTIVE_THRESH_WEIGHT )
#cv2.imshow("imgThreshplate2",imgThreshplate2)
###############################################################

######## Tải lên mô hình KNN ######################
npaClassifications = np.loadtxt(r'E:\NAMHOC\python\DO_AN\tinhgon\classifications.txt', np.float32)
npaFlattenedImages = np.loadtxt(r'E:\NAMHOC\python\DO_AN\tinhgon\flattened_images.txt', np.float32) 
npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))# reshape numpy array to 1d, necessary to pass to call to train
kNearest = cv2.ml.KNearest_create()                   # instantiate KNN object
kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
#########################

################ Tiền xử lý ảnh #################
imgGrayscaleplate, imgThreshplate = Preprocess.preprocess(img)
canny_image = cv2.Canny(imgThreshplate,250,255) #Tách biên bằng canny
kernel = np.ones((3,3), np.uint8)
dilated_image = cv2.dilate(canny_image,kernel,iterations=1) #tăng sharp cho egde (Phép nở)
#cv2.imshow("dilated_image",dilated_image)

###########################################

###### vẽ contour và lọc biển số  #############
contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours= sorted(contours, key = cv2.contourArea, reverse = True)[:10] #Lấy 10 contours có diện tích lớn nhất
#cv2.drawContours(img, contours, -1, (0, 0,255 ), 3) # Vẽ tất cả các ctour trong hình lớn
screenCnt = []
for c in contours:
    peri = cv2.arcLength(c, True) #Tính chu vi
    approx = cv2.approxPolyDP(c, 0.06 * peri, True) # làm xấp xỉ đa giác, chỉ giữ contour có 4 cạnh
    [x, y, w, h] = cv2.boundingRect(approx.copy())
    ratio = w/h
    #cv2.putText(img, str(len(approx.copy())), (x,y),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)
    #cv2.putText(img, str(ratio), (x,y),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)
    if (len(approx) == 4) :
        screenCnt.append(approx)  
        
        cv2.putText(img, str(len(approx.copy())), (x,y),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)

if screenCnt is None:
    detected = 0
    print ("No plate detected")
else:
    detected = 1

if detected == 1:

    for screenCnt in screenCnt:
        cv2.drawContours(img, [screenCnt], -1, (255, 255, 0), 3) #Khoanh vùng biển số xe
        
        ############## Tìm góc xoay ảnh #####################
        (x1,y1) = screenCnt[0,0]
        (x2,y2) = screenCnt[1,0]
        (x3,y3) = screenCnt[2,0]
        (x4,y4) = screenCnt[3,0]
        print(screenCnt)
        array = [[x1, y1], [x2,y2], [x3,y3], [x4,y4]]
        sorted_array = array.sort(reverse=True, key=lambda x:x[1])
        print(array)
        (x1,y1) = array[0]
        (x2,y2) = array[1]
        doi = abs(y1 - y2)
        ke = abs (x1 - x2)
        angle = math.atan(doi/ke) * (180.0 / math.pi)
        print(angle)
        ####################################

        ########## Cắt biển số ra khỏi ảnh và xoay ảnh ################
        
        mask = np.zeros(imgGrayscaleplate.shape, np.uint8)
        cv2.imshow('mask', mask)
        new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
        #cv2.imshow("new_image",new_image) 
            # Now crop
        (x, y) = np.where(mask == 255)       
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        print(topx,topy,bottomx,bottomy)

        roi = img[topx:bottomx, topy:bottomy]

        imgThresh = imgThreshplate[topx:bottomx, topy:bottomy]
        ptPlateCenter = (bottomx - topx)/2, (bottomy - topy)/2

        if x1 < x2:
            rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
        else:
            rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, angle, 1.0)

        roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx ))
        cv2.imshow('roi', roi)
        imgThresh = cv2.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx ))
        roi = cv2.resize(roi,(0,0),fx = 3, fy = 3)
        cv2.imshow('roi', roi)
        imgThresh = cv2.resize(imgThresh,(0,0),fx = 3, fy = 3)

        ####################################
        #################### Tiền xử lý ảnh đề phân đoạn kí tự ####################
        ###############
        gray_crop = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        thresh_crop = cv2.adaptiveThreshold(gray_crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101,2)
        #cv2.imshow('thresh_cop', thresh_crop)
        # Otsu sau khi lọc Gaussian

        blur = cv2.GaussianBlur(gray_crop, (5,5),0)
        #cv2.imshow('blur', blur)
        #cv2.imshow('gray_crop', gray_crop)
        _, thresh = cv2.threshold (blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        #cv2. imshow ('CROP', thresh)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        cv2.imshow('m', opening)
        invert = 255 - opening
        #=======
        kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        thre_mor = cv2.morphologyEx(imgThresh,cv2.MORPH_OPEN,kerel3,)
        cont,hier = cv2.findContours(opening,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 

        cv2.imshow(str(n+20),thre_mor)
        cv2.drawContours(roi, cont, -1, (100, 255, 255), 2) #Vẽ contour các kí tự trong biển số

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
        strFinalString = ""
        first_line = ""
        second_line = ""

        for i in char_x:
            (x,y,w,h) = cv2.boundingRect(cont[char_x_ind[i]])
            cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),2)
        
            imgROI = thre_mor[y:y+h,x:x+w]     # cắt kí tự ra khỏi hình

            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))     # resize lại hình ảnh
            npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))      # đưa hình ảnh về mảng 1 chiều
            #cHUYỂN ảnh thành ma trận có 1 hàng và số cột là tổng số điểm ảnh trong đó
            npaROIResized = np.float32(npaROIResized)       # chuyển mảng về dạng float
            _, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 3)     # call KNN function find_nearest; neigh_resp là hàng xóm 
            strCurrentChar = str(chr(int(npaResults[0][0])))  # Lấy mã ASCII của kí tự đang xét
            #cv2.putText(roi, strCurrentChar, (x, y+50),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255,0), 3)

            if (y < height/3): # Biển số 1 hay 2 hàng
                first_line = first_line + strCurrentChar
            else:
                second_line = second_line + strCurrentChar

        print ("\n License Plate " +str(n)+ " is: " + first_line + " - " + second_line + "\n")
        roi = cv2.resize(roi, None, fx=0.75, fy=0.75) 
        cv2.imshow(str(n),cv2.cvtColor(roi,cv2.COLOR_BGR2RGB))
        
        #cv2.putText(img, first_line + "-" + second_line ,(topy ,topx),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)
        n = n + 1
        


img = cv2.resize(img, None, fx=0.5, fy=0.5) 
cv2.imshow('License plate', img)
print("--- %s seconds ---" % (time.time() - start_time))
cv2.waitKey(0)


