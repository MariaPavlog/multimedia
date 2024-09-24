import cv2
import numpy as np

cap = cv2.VideoCapture(0)

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
center_x = frame_w // 2
center_y = frame_h // 2

rectangles = np.array([  # значения [x1, y1], [x2, y2]
    [[  0, 140], [260, 180]],
    [[110,   0], [150, 140]],
    [[110, 180], [150, 320]]
])

offset_x = frame_w // 2 - rectangles[:, :, 0].max() // 2
offset_y = frame_h // 2 - rectangles[:, :, 1].max() // 2


ret, frame = cap.read()


center_pixel = frame[center_y][center_x]

max_color_index = np.argmax(center_pixel)
print(f'R = {center_pixel[2]}, G = {center_pixel[1]}, B = {center_pixel[0]}')
color = [0, 0, 0]
color[max_color_index] = 255

for rect in rectangles:
    x1, y1 = rect[0]
    x2, y2 = rect[1]
    cv2.rectangle(frame, (x1 + offset_x, y1 + offset_y), (x2 + offset_x, y2 + offset_y), color, -1)

cv2.imshow("Color cross", frame)
cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()