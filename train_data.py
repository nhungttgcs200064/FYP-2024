import numpy as np
import os
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Thư mục chứa các tệp .npy đã lưu embeddings
path = 'dataset'

def load_embeddings(path):
    embeddings = []
    labels = []

    for file in os.listdir(path):
        if file.endswith(".npy"):
            embedding = np.load(os.path.join(path, file))
            embeddings.append(embedding)

            # Lấy ID từ tên tệp (tên tệp có định dạng "User_<id>_<count>.npy")
            label = file.split("_")[1]
            labels.append(label)

    return np.array(embeddings), np.array(labels)

# Load embeddings và labels
print("Loading embeddings and labels...")
embeddings, labels = load_embeddings(path)

# Mã hóa nhãn bằng LabelEncoder
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels_encoded, test_size=0.2, random_state=42)

# Sử dụng SVM để huấn luyện mô hình
print("Training face recognition model...")
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Kiểm tra độ chính xác trên tập kiểm tra
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy * 100:.2f}%")

# Tạo thư mục 'models' nếu chưa tồn tại
if not os.path.exists("models"):
    os.makedirs("models")

# Lưu mô hình và LabelEncoder
joblib.dump(model, 'models/face_recognition_model.pkl')
joblib.dump(label_encoder, 'models/label_encoder.pkl')

print("Training completed and model saved!")
