import cv2

img2=cv2.imread(r"1.jpg", cv2.IMREAD_GRAYSCALE)
img3=cv2.imread(r"1.jpg", cv2.IMREAD_UNCHANGED)
img4=cv2.imread(r"1.jpg", cv2.IMREAD_REDUCED_COLOR_4 )

cv2.namedWindow('normal', cv2.WINDOW_NORMAL)
cv2.namedWindow('autosize', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('fullscreen', cv2.WINDOW_FULLSCREEN)

cv2.imshow('normal', img2)
cv2.imshow('autosize', img3)
cv2.imshow('fullscreen', img4)
cv2.waitKey(0)
cv2.destroyAllWindows()

