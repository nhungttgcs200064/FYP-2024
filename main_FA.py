import os
import time
import query_db as db
import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from ultralytics import YOLO
import joblib

# Tải mô hình đã huấn luyện và LabelEncoder
model = joblib.load('models/face_recognition_model.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

# Khởi tạo MTCNN, FaceNet và YOLO
detector = MTCNN()
embedder = FaceNet()
spoof_model = YOLO("models/train_ver_2_100.pt")

last_time_checked = time.time()
modeType = 6
# timeOut = None

# Tải ảnh nền và các khung chế độ
imgBackGround = cv2.imread('img/background.png')
folderModePath = 'img'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    img = cv2.imread(os.path.join(folderModePath, path))
    imgModeList.append(img)

# Mở camera
cam = cv2.VideoCapture(0)
cam.set(3, 740)  # Chiều rộng video
cam.set(4, 720)  # Chiều cao video

print("Start taking attendance...")

while True:
    ret, img = cam.read()
    if not ret:
        break

    # if timeOut and (time.time() - timeOut >= 5):
    #     if check_status == 'Check-in':
    #         print("Check-in successfully")
    #     elif check_status == 'Check-out':
    #         print("Check-out successfully")
    #     break

    frame_resized = cv2.resize(img, (732, 720))
    imgBackGround[0:720, 0:732] = frame_resized
    imgBackGround[44:44+634, 800:800+414] = imgModeList[modeType]

    # Phát hiện giả mạo bằng YOLO
    results = spoof_model(frame_resized, stream=True, verbose=False)
    is_fake = False
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1 = int(box.xyxy[0][0])
            y1 = int(box.xyxy[0][1])
            x2 = int(box.xyxy[0][2])
            y2 = int(box.xyxy[0][3])
            cls = int(box.cls[0])
            confidence = box.conf[0]

            if cls == 0 and confidence > 0.8:
                is_fake = True
                cv2.rectangle(imgBackGround, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(imgBackGround, "Fake", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Nếu phát hiện giả mạo, bỏ qua khung hình hiện tại
    if is_fake:
        modeType = 2  # Chế độ giả mạo
        cv2.imshow('Face Attendance', imgBackGround)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Phát hiện khuôn mặt bằng MTCNN
    faces = detector.detect_faces(frame_resized)
    if len(faces) == 0:
        modeType = 6  # Không có khuôn mặt
    else:
        for face in faces:
            x, y, w, h = face['box']
            x, y = max(0, x), max(0, y)
            face_img = frame_resized[y:y + h, x:x + w]
            cv2.rectangle(imgBackGround, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Trích xuất embedding của khuôn mặt
            embedding = embedder.embeddings([face_img])[0]
            prediction = model.predict([embedding])
            proba = model.predict_proba([embedding])
            confidence = np.max(proba) * 100

            if confidence > 80:
                name = label_encoder.inverse_transform(prediction)[0]
                profile = db.getProfile(int(name))

                if profile is not None:
                    cv2.putText(imgBackGround, profile[1], (x + 10, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(imgBackGround, f"{confidence:.2f}%", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

                    current_time = time.time()
                    if (current_time - last_time_checked) >= 15:
                        check = db.checkInAndCheckOut(profile[0])
                        # check_status = None
                        if check:
                            modeType = 0
                            # check_status = 'Check-in'
                            # timeOut = time.time()
                        else:
                            modeType = 4
                            # check_status = 'Check-out'
                            # timeOut = time.time()
                        last_time_checked = current_time
            else:
                cv2.putText(imgBackGround, "Unknown", (x + 10, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                modeType = 3  # Chế độ không nhận diện được

    # Hiển thị ảnh nền với chế độ
    cv2.imshow('Face Attendance', imgBackGround)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("End program")
cam.release()
cv2.destroyAllWindows()
