import cv2
import easyocr
import numpy as np

# Inisialisasi EasyOCR reader dengan bahasa Indonesia ('id') dan Inggris ('en')
reader = easyocr.Reader(['en', 'id'])  # menambahkan 'id' untuk Bahasa Indonesia

def process_frame(frame):
    """
    Proses frame dari webcam untuk ekstraksi teks dengan langkah praproses: grayscale, thresholding, dan blurring.
    """
    # Langkah 1: Konversi ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Langkah 2: Terapkan GaussianBlur untuk mengurangi noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Langkah 3: Terapkan thresholding adaptif untuk meningkatkan kontras
    thresholded = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Langkah 4: Gunakan EasyOCR untuk mengenali teks pada gambar yang telah dipraproses
    results = reader.readtext(thresholded)
    
    return results, frame  # Mengembalikan hasil OCR dan gambar asli

def main():
    """
    Fungsi utama untuk membaca teks dari webcam secara real-time.
    """
    # Buka webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Tidak dapat membuka webcam.")
        return
    
    print("Tekan 'q' untuk keluar.")
    
    while True:
        # Baca frame dari webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Tidak dapat membaca frame dari webcam.")
            break
        
        # Ubah ukuran tampilan webcam (misalnya menjadi 640x480)
        frame_resized = cv2.resize(frame, (640, 480))  # Menyesuaikan ukuran frame menjadi lebih kecil
        
        # Proses frame untuk membaca teks
        results, output_frame = process_frame(frame_resized)
        
        # Menampilkan hasil OCR di terminal (cetak teks yang dikenali)
        print("\nHasil OCR:")
        for (bbox, text, prob) in results:
            print(f"Teks: {text}, Probabilitas: {prob:.2f}")

        # Gambar kotak pembatas untuk setiap teks yang terdeteksi
        for (bbox, text, prob) in results:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            
            # Convert the points to integers
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))
            
            # Gambar kotak hijau di sekitar teks yang terdeteksi
            cv2.rectangle(output_frame, top_left, bottom_right, (0, 255, 0), 2)
            
            # Tampilkan teks di atas kotak
            cv2.putText(output_frame, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Tampilkan hasil video dengan ukuran yang lebih kecil
        cv2.imshow("Webcam OCR dengan Easy OCR", output_frame)
        
        # Keluar dengan menekan tombol 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release webcam dan tutup jendela
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
