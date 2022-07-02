import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
im=cv.imread(r'E:\NAMHOC\python\img\3.jpg')
im_gray=cv.cvtColor(im,cv.COLOR_BGR2GRAY)
cv.imshow('gray', im_gray)
noise_removal=cv.bilateralFilter(im_gray,9,75,75)
cv.imshow('loc', noise_removal)
equal_histogram=noise_removal
#cv.imshow('equ', equal_histogram)
kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
morph_image = cv.morphologyEx(equal_histogram, cv.MORPH_OPEN, kernel, iterations=20)
sub_morp_image=cv.subtract(equal_histogram,morph_image)
ret,thresh_image=cv.threshold(sub_morp_image,0,255,cv.THRESH_OTSU)
canny_image=cv.Canny(thresh_image,250,255)
cv.imshow('cann',canny_image)
kernel=np.ones((3,3),np.uint8)
dilated_image=cv.dilate(canny_image,kernel,iterations=1)
contours,hierarchy=cv.findContours(dilated_image,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
contours=sorted(contours,key=cv.contourArea,reverse=True)[:10]
dst = cv.Canny(thresh_image, 50, 200, None, 3)
    
    # Copy edges to the images that will display the results in BGR
cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
cdstP = np.copy(cdst)
c = cv.findContours(image, mode, method)
lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)


linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)

cv.imshow("Source", im_gray)
cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

cv.waitKey()

cv.destroyAllWindows()