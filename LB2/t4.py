import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1) # фильтрация (изображение, левая и права границы), на выходе набор пикселей со значениями 0/1
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2) # побитовая операция ИЛИ

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # erosion + dilation
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # dilation + erosion

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        moments = cv2.moments(contour, True)

        dM01 = moments['m01']  # Y
        dM10 = moments['m10']  # X
        dArea = moments['m00']


        if dArea > 1000:
            posX = int(dM10 / dArea)
            posY = int(dM01 / dArea)
            cv2.circle(frame, (posX, posY), 5, (0, 0, 0), -1)

            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 3)
            cv2.putText(frame, f'Area: {int(dArea)}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Rectangle_frame', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()