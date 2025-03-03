import os
import cv2
import numpy as np
from keras_facenet import FaceNet
from mtcnn import MTCNN
import query_db as db


offsetPerW = 10
offsetPerH = 10
count = 0

# Khởi tạo MTCNN và FaceNet
detector = MTCNN()
embedder = FaceNet()

# Thiết lập camera
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Chiều rộng video
cam.set(4, 480)  # Chiều cao video

# Nhận thông tin người dùng
face_id = input("\n Enter your user ID and press Enter: ")
name = input("\n Enter your name and press Enter: ")
print("\n Start collecting face photos. Please look at the camera....")


while True:
    ret, img = cam.read()
    if not ret:
        break

    # Phát hiện khuôn mặt
    faces = detector.detect_faces(img)

    for face in faces:
        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)  # Đảm bảo không có giá trị âm
        face_img = img[y:y + h, x:x + w]

        offsetW = (offsetPerW / 100) * w
        x = int(x - offsetW)
        w = int(w + offsetW * 2.5)
        offsetH = (offsetPerH / 100) * h
        y = int(y - offsetH * 3)
        h = int(h + offsetH * 4)

        cv2.imshow('face img', face_img)

        # Trích xuất vector đặc trưng khuôn mặt
        embedding = embedder.embeddings([face_img])[0]

        # Tạo thư mục lưu trữ nếu chưa có
        if not os.path.exists("dataset"):
            os.makedirs("dataset")

        # Lưu vector đặc trưng vào tệp .npy
        np.save(f"dataset/User_{face_id}_{count}.npy", embedding)
        count += 1

        # Vẽ hình chữ nhật quanh khuôn mặt
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow('image', img)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif count >= 100:  # Lưu 100 embedding và dừng
        break

db.insertOrUpdate(face_id,name)
print(f"User {name} (ID: {face_id}) has been registered successfully.")

print("End program")
cam.release()
cv2.destroyAllWindows()
