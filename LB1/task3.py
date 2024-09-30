import cv2

cap = cv2.VideoCapture('v1.MOV', cv2.CAP_ANY)
cap1 = cv2.VideoCapture('v1.MOV', cv2.CAP_ANY)
cap2 = cv2.VideoCapture('v1.MOV', cv2.CAP_ANY)

#fps = cap.get(cv2.CAP_PROP_FPS)
#print(fps)


while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('normal', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
#cv2.destroyWindow('normal')

while True:
    ret, frame = cap1.read()
    if not ret:
        break
    frame = cv2.resize(frame, (480, 640))
    cv2.imshow('resize', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Escape key
        break
cv2.destroyWindow('resize')
while True:
    ret, frame = cap2.read()
    if not ret:
        break
    frame = cv2.resize(frame, (480, 640))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow('color', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Escape key
        break

cv2.destroyWindow('color')