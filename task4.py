import cv2


video = cv2.VideoCapture('v1.MOV')
w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  #кодек для сжатия видео
video_writer = cv2.VideoWriter("output4.mov", fourcc, 25, (w, h))
while True:
    ret, img = video.read()
    if not ret or cv2.waitKey(1) & 0xFF == 27:
        break
   # cv2.imshow('video', img)
    video_writer.write(img)
video.release()
cv2.destroyAllWindows()