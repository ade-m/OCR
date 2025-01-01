import cv2
import easyocr
import tkinter as tk
from tkinter import filedialog
import os
import numpy as np

# Inisialisasi EasyOCR reader dengan bahasa Indonesia ('id') dan Inggris ('en')
reader = easyocr.Reader(['en', 'id'])  # menambahkan 'id' untuk Bahasa Indonesia

def process_image(image_path):
    """
    Proses gambar untuk ekstraksi teks dengan langkah praproses: grayscale, thresholding, dan blurring.
    """
    # Baca gambar dari file
    image = cv2.imread(image_path)

    # Langkah 1: Konversi ke grayscale (untuk mempermudah pengolahan gambar)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Langkah 2: Terapkan GaussianBlur untuk mengurangi noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Langkah 3: Terapkan thresholding adaptif untuk meningkatkan kontras
    thresholded = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Langkah 4: Coba meningkatkan kualitas gambar dengan rescaling jika teks terlalu kecil
    # Perbesar gambar jika diperlukan
    height, width = thresholded.shape
    scale_factor = 1  # Atur pengaturan skala untuk memperbesar
    resized = cv2.resize(thresholded, (width * scale_factor, height * scale_factor))

    # Langkah 5: Gunakan EasyOCR untuk mengenali teks pada gambar yang telah dipraproses
    results = reader.readtext(resized)
    
    return results, image  # Mengembalikan hasil OCR dan gambar asli



def main():
    """
    Fungsi utama untuk memilih gambar dan menampilkan hasil OCR dengan kotak pembatas.
    """
    # Pilih gambar menggunakan file dialog
    image_path = "/Users/ademaulana/Documents/OCRImg/struk2.png"
    
    if not image_path:
        print("Tidak ada gambar yang dipilih.")
        return
    
    print(f"Membaca gambar dari: {image_path}")

    # Proses gambar dan dapatkan hasil OCR serta gambar asli
    results, image = process_image(image_path)
    
    # Menampilkan hasil OCR di terminal
    print("\nHasil OCR:")
    for (bbox, text, prob) in results:
        print(f"Teks: {text}, Probabilitas: {prob:.2f}")

    # Gambar kotak pembatas untuk setiap teks yang terdeteksi
    for (bbox, text, prob) in results:
        # Extracting the four points of the bounding box
        (top_left, top_right, bottom_right, bottom_left) = bbox
        
        # Convert the points to integers
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))
        
        # Gambar kotak biru di sekitar teks yang terdeteksi
        cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)
        
        # Tampilkan teks di atas kotak
        cv2.putText(image, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Tampilkan hasil gambar yang telah diberi kotak pembatas (gambar asli)
    cv2.imshow("EasyOCR - Hasil Gambar", image)
    
    # Tunggu hingga tombol 'q' ditekan untuk menutup
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
