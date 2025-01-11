import pathlib
from torch.nn import Conv2d
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Sabit değerleri tanımlayalım
CONFIDENCE_THRESHOLD = 0.01
INPUT_SIZE = (416, 416)
OVERLAY_ALPHA = 0.6
BOX_COLOR = (0, 255, 0)  # RGB
TEXT_COLOR = (255, 255, 255)  # RGB
BOX_THICKNESS = 2
TEXT_SCALE = 0.5
TEXT_THICKNESS = 2

# YOLOv5 modelini yükle
model = torch.hub.load('/content/drive/MyDrive/deneme/yolov5', 'custom', path='/content/drive/MyDrive/deneme/best.pt', source='local')
model.eval()

# Isı haritasını oluşturup görüntüye bindirme
def overlay_heatmap(img_rgb, cam):
    """Isı haritası oluştur ve görüntü üzerine bindir"""
    # Isı haritasını normalize et
    heatmap = cv2.resize(cam, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    # TURBO renk haritasını kullan
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Isı haritasını normalize ederek görüntüye daha şık bindirme
    heatmap_overlay = cv2.addWeighted(img_rgb, 1 - OVERLAY_ALPHA, heatmap_colored, OVERLAY_ALPHA, 0)
    return heatmap_overlay

# Belirtilen sıradaki Conv2D katmanını bul
def find_nth_conv_layer(model, n):
    conv_layers = [layer for layer in model.modules() if isinstance(layer, Conv2d)]
    if len(conv_layers) >= n:
        return conv_layers[n - 1]  # n. Conv2D katmanı
    raise ValueError(f"Modelde {n}. bir Conv2D katmanı bulunamadı!")

target_layer = find_nth_conv_layer(model.model, 8)  # 8. Conv2D katmanı

# Score-CAM sınıfı
class ScoreCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.hooks()

    def hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        self.target_layer.register_forward_hook(forward_hook)

    def generate(self, input_tensor, box_idx=0):
        # Modelin ileri geçişi
        output = self.model(input_tensor)
        # Tahminleri işleme
        predictions = self.decode_predictions(output)

        # Tahminlerin varlığını kontrol et
        if len(predictions) == 0:
            raise ValueError("Model tarafından tahmin yapılmadı. Girdi görüntüsünü ve modeli kontrol edin.")

        # En yüksek güvene sahip kutuyu seç
        predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        target_box = predictions[box_idx]
        print(f"Kullanılan kutu: {target_box}")

        # Aktivasyon haritaları ile Score-CAM hesaplama
        activations = self.activations.squeeze(0).detach().cpu().numpy()
        scores = []
        for i in range(activations.shape[0]):
            mask = activations[i]
            mask = cv2.resize(mask, (input_tensor.shape[3], input_tensor.shape[2]))
            mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-9)
            masked_input = input_tensor * torch.tensor(mask).to(input_tensor.device).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                score = self.model(masked_input)[0, box_idx, 4].item()
            scores.append(score)

        scores = np.array(scores)
        weighted_map = (scores[:, None, None] * activations).sum(axis=0)
        cam = np.maximum(weighted_map, 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
        return cam, predictions

    @staticmethod
    def decode_predictions(output):
        """Ham tensor çıktısını okunabilir tahminlere dönüştür."""
        predictions = []
        for i in range(output.shape[1]):
            box = output[0, i, :4].detach().cpu().numpy()
            confidence_tensor = output[0, i, 4]
            confidence = float(confidence_tensor.detach().cpu().numpy())
            if confidence > CONFIDENCE_THRESHOLD:
                predictions.append({
                    'box': box,
                    'confidence': confidence,
                    'confidence_tensor': confidence_tensor
                })
        return predictions

def process_image(img_path, model, score_cam):
    """Görüntü işleme ve görselleştirme"""
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, INPUT_SIZE)

    # Tensor dönüşümü
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    img_tensor = img_tensor.to(model.model.device)
    model.model = model.model.to(model.model.device)

    try:
        # Score-CAM hesaplama
        cam, predictions = score_cam.generate(img_tensor)

        # Isı haritası oluştur ve bindir
        final_image = overlay_heatmap(img_rgb, cam)

        # Sonucu kaydet
        output_path = "scorecam_result.jpg"
        final_image_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, final_image_bgr)
        print(f"Score-CAM sonucu {output_path} dosyasına kaydedildi.")

        # Matplotlib ile görselleştirme
        plt.figure(figsize=(12, 8))
        plt.imshow(final_image)
        plt.title("Score-CAM Isı Haritası ve Tespitler", pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    except ValueError as e:
        print("Hata:", e)

# Ana kod
if __name__ == "__main__":
    # Score-CAM hazırlığı
    target_layer = find_nth_conv_layer(model.model, 8)  # 8. Conv2D katmanını seç
    score_cam = ScoreCAM(model.model, target_layer)

    # Görüntü işleme
    process_image('/content/drive/MyDrive/deneme/dota/P1421.png', model, score_cam)