import cv2


ip_address = '10.217.22.127'
port = '8080'
video_url = f"http://{ip_address}:{port}/video"

video = cv2.VideoCapture(video_url)

while True:
    ok, img = video.read()
    img=cv2.resize(img,(640,380))

    if ok:
        cv2.imshow(f'Video stream from {ip_address}:{port}', img)

    # Выход из цикла при нажатии клавиши 'Esc'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Освобождение ресурсов
video.release()
cv2.destroyAllWindows()
