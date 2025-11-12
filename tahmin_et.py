import tensorflow as tf
import numpy as np
import cv2 
import os

# GEREKLİ FONKSİYONU İMPORT ET
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# YENİ Kaydedilmiş modeli yükle
print("Model 'tas_kagit_makas_modeli_v4.h5' yükleniyor...")
model = tf.keras.models.load_model('tas_kagit_makas_modeli_v4.h5', compile=False)
print("Model başarıyla yüklendi.")

sinif_isimleri = ['Tas', 'Kagit', 'Makas']
IMG_SIZE = 160

# Resim hazırlama fonksiyonu (Aynı)
def resmi_hazirla(resim_yolu):
    img = cv2.imread(resim_yolu)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(img, axis=0)
    return preprocess_input(img_array)

# Tahmin yapılacak resmin yolu
resim_yolu = 'indir3.jfif' # Makas resminiz

if os.path.exists(resim_yolu):
    hazirlanmis_resim = resmi_hazirla(resim_yolu)
    tahmin_sonuclari = model.predict(hazirlanmis_resim)
    
    tahmin_indeksi = np.argmax(tahmin_sonuclari[0])
    tahmin_edilen_sinif = sinif_isimleri[tahmin_indeksi]
    tahmin_skoru = tahmin_sonuclari[0][tahmin_indeksi]

    print("\n========== TAHMİN SONUCU ==========")
    print(f"Resim: {resim_yolu}")
    print(f"Tahminim: {tahmin_edilen_sinif} (Skor: {tahmin_skoru:.2f})")
    print("===================================")
    print(f"(Ham skorlar: Tas={tahmin_sonuclari[0][0]:.2f}, Kagit={tahmin_sonuclari[0][1]:.2f}, Makas={tahmin_sonuclari[0][2]:.2f})")
else:
    print(f"HATA: '{resim_yolu}' adında bir dosya bulunamadı.")


OUTPUT:
Model başarıyla yüklendi.
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 1s/step

========== TAHMİN SONUCU ==========
Resim: indir3.jfif
Tahminim: Kagit (Skor: 0.53)
===================================
(Ham skorlar: Tas=-0.75, Kagit=0.53, Makas=-0.69)



