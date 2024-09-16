import cv2


img = cv2.imread('1.jpg', cv2.IMREAD_REDUCED_COLOR_2 )
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imshow('BGR', img)
cv2.imshow('HSV', img_hsv)

cv2.waitKey(0)
cv2.destroyAllWindows()