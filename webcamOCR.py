import cv2
import pytesseract
from pytesseract import Output
from PIL import Image
import numpy as np

# Konfigurasi path Tesseract (sesuaikan dengan sistem Anda)
# Untuk macOS/Linux
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

# Untuk Windows
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(frame):
    """
    Preprocessing frame untuk meningkatkan kualitas deteksi teks.
    """
    # Konversi frame ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Histogram Equalization untuk meningkatkan kontras
    gray = cv2.equalizeHist(gray)
    
    # Gaussian Blurring untuk mengurangi noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu's thresholding untuk menghasilkan citra biner
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary

def process_frame(frame):
    """
    Proses frame dari webcam untuk ekstraksi teks dan mendapatkan kotak pembatas.
    """
    # Preprocessing frame
    processed_frame = preprocess_image(frame)
    
    # Konversi ke format PIL untuk digunakan oleh pytesseract
    pil_image = Image.fromarray(processed_frame)
    
    # Ekstraksi teks dan informasi kotak pembatas
    data = pytesseract.image_to_data(pil_image, output_type=Output.DICT, lang='eng')
    return data

def main():
    """
    Fungsi utama untuk membaca teks dari webcam secara real-time dan menandai teks yang terbaca.
    """
    # Buka webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Tidak dapat membuka webcam")
        return
    
    print("Tekan 'q' untuk keluar.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Tidak dapat membaca frame dari webcam")
            break
        
        # Proses frame untuk membaca teks dan mendapatkan informasi kotak pembatas
        data = process_frame(frame)
        
        # Gambar kotak pembatas untuk setiap kata yang terdeteksi
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            if int(data['conf'][i]) > 0:  # Hanya menggambar kotak untuk kata yang terdeteksi dengan kepercayaan > 0
                (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Gambar kotak hijau di sekitar kata
                cv2.putText(frame, data['text'][i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Menampilkan teks di atas kotak

        # Tampilkan hasil video
        cv2.imshow("Webcam OCR", frame)
        
        # Keluar dengan menekan 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release webcam dan tutup jendela
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
