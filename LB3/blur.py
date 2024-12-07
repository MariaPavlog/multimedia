import cv2
import numpy as np
import math

def gauss (x,y,a,b,sigma):
    return math.exp(-((x-a)**2+(y-b)**2)/(2*sigma**2))/(2*math.pi*sigma**2)


def create_ker(size_ker, sigma):

#size_ker = 7
#sigma=0.5
    ker = np.zeros((size_ker, size_ker))
    a=b=size_ker//2
#print(a)
    for i in range (size_ker):
        for j in range (size_ker):
            ker[i][j]=gauss(i,j,a,b,sigma)

    sum_ker = np.sum(ker)  # Сумма всех элементов матрицы
    if sum_ker != 0:  # Проверяем, чтобы избежать деления на ноль
        ker /= sum_ker  # Делим каждый элемент на сумму


    for i in range(size_ker):
        for j in range(size_ker):
            print(ker[i][j])
        print()

    return ker


def filter(img, ker):
    B = img.copy()
    x0 = ker.shape[0] // 2
    y0 = ker.shape[1] // 2
    for i in range(x0, B.shape[0] - x0):
        for j in range(y0, B.shape[1] - y0):
           val = 0

           for k in range(-(size_ker // 2), size_ker // 2 + 1):
                for l in range(-(size_ker // 2), size_ker // 2 + 1):
                    val += img[i + k, j + l] * ker[k + (size_ker // 2), l + (size_ker // 2)]
                B[i, j] = val

    return B

size_ker=7
sigma=10
img = cv2.imread(r'C:/Users/HP/PycharmProjects/atsom3/img.jpg', cv2.IMREAD_GRAYSCALE)

ker=create_ker(size_ker,sigma)
B=filter(img,ker)
blurOpencv=cv2.GaussianBlur(img,(size_ker, size_ker), sigma)
cv2.imshow('myBlur', B)
cv2.imshow('opencv', blurOpencv)
cv2.waitKey(0)
