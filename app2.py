import pathlib
from torch.nn import Conv2d
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Windows uyumluluğu için Path ayarı
pathlib.PosixPath = pathlib.WindowsPath

# Sabit değerler
CONFIDENCE_THRESHOLD = 0.05
INPUT_SIZE = (416, 416)
OVERLAY_ALPHA = 0.6
BOX_COLOR = (0, 255, 0)  # RGB
TEXT_COLOR = (255, 255, 255)  # RGB
BOX_THICKNESS = 2
TEXT_SCALE = 0.5
TEXT_THICKNESS = 2

# YOLOv5 modeli yükleme
model = torch.hub.load('./yolov5', 'custom', path='best.pt', source='local')
model.eval()

# Modeldeki son Conv2D katmanını bul
def find_last_conv_layer(model):
    for layer in reversed(list(model.modules())):
        if isinstance(layer, Conv2d):
            return layer
    raise ValueError("Modelde Conv2D katmanı bulunamadı!")

target_layer = find_last_conv_layer(model.model)

def process_image_with_eigencam(img_path, model, eigen_cam):
    """EigenCAM ile görüntü işleme ve görselleştirme"""
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, INPUT_SIZE)
    
    # Tensor dönüşümü
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    
    try:
        # EigenCAM hesaplama
        with torch.no_grad():
            grayscale_cam = eigen_cam(input_tensor=img_tensor, aug_smooth=True, eigen_smooth=True)[0]
        
        # Isı haritasını görüntü üzerine bindir
        grayscale_cam_resized = cv2.resize(grayscale_cam, (img_rgb.shape[1], img_rgb.shape[0]))
        visualization = show_cam_on_image(img_rgb / 255.0, grayscale_cam_resized, use_rgb=True)

        # Sonuçları kaydet
        output_path = f"eigencam_result_{img_path.split('/')[-1]}"
        cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        print(f"EigenCAM sonucu {output_path} dosyasına kaydedildi.")
        
        # Matplotlib ile görselleştirme
        plt.figure(figsize=(12, 8))
        plt.imshow(visualization)
        plt.title("EigenCAM Isı Haritası", pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    except ValueError as e:
        print("Hata:", e)

# Ana kod
if __name__ == "__main__":
    # EigenCAM hazırlığı
    eigen_cam = EigenCAM(model.model, [target_layer])
    
    # Görüntüleri sırayla işle
    image_paths = [
        'dota/5.png', 'dota/6.png', 'dota/7.png',
        'dota/8.png', 'dota/9.png', 'dota/10.png',
        'dota/11.png', 'dota/12.png', 'dota/123.png'
    ]

    for img_path in image_paths:
        process_image_with_eigencam(img_path, model, eigen_cam)
