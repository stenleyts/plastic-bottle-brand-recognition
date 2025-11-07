import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time

# ===== Load model yang sudah dilatih =====
print("Memuat model...")
model = load_model("bottle_classifier.h5")
print("Model berhasil dimuat")

# Daftar label sesuai urutan folder dataset
labels = ["aqua", "leminerale", "nestle"]

# ===== Buka kamera =====
print("Mengaktifkan kamera...")
cap = cv2.VideoCapture(0)

# Tambahkan sedikit jeda agar kamera sempat inisialisasi
time.sleep(2)

if not cap.isOpened():
    print("Kamera gagal dibuka. Pastikan izin kamera diberikan.")
    exit()

print("Kamera aktif. Tekan 'q' untuk keluar.")

# ===== Atur ambang keyakinan minimum =====
confidence_threshold = 0.7  # jika kurang dari ini -> dianggap tidak ada botol

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera.")
        break

    # Resize frame agar sesuai input model
    img = cv2.resize(frame, (150, 150))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    # Prediksi merek botol
    prediction = model.predict(img, verbose=0)
    confidence = np.max(prediction)
    label_index = np.argmax(prediction)

    # Tentukan teks yang akan ditampilkan
    if confidence < confidence_threshold:
        text = "No bottle detected"
        color = (0, 0, 255)  # merah untuk tidak ada botol
    else:
        label = labels[label_index]
        text = f"{label} ({confidence*100:.2f}%)"
        color = (0, 255, 0)  # hijau untuk deteksi valid

    # Tampilkan hasil di layar
    cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Bottle Detection", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Kamera dimatikan. Program selesai")