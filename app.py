import pathlib
from torch.nn import Conv2d
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Windows uyumluluğu için Path ayarı
#pathlib.PosixPath = pathlib.WindowsPath

# Sabit değerleri en üstte tanımlayalım
CONFIDENCE_THRESHOLD = 0.25
INPUT_SIZE = (416, 416)
OVERLAY_ALPHA = 0.6
BOX_COLOR = (0, 255, 0)  # RGB
TEXT_COLOR = (255, 255, 255)  # RGB
BOX_THICKNESS = 2
TEXT_SCALE = 0.5
TEXT_THICKNESS = 2

# YOLOv5 modelini yükle
model = torch.hub.load('./yolov5', 'custom', path='best.pt', source='local')
model.eval()

# Modeldeki son Conv2D katmanını bul
def find_last_conv_layer(model):
    for layer in reversed(list(model.modules())):
        if isinstance(layer, Conv2d):
            return layer
    raise ValueError("Modelde Conv2D katmanı bulunamadı!")

target_layer = find_last_conv_layer(model.model)

# Grad-CAM sınıfı
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks()

    def hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

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
        confidence_score = target_box['confidence_tensor']

        print(f"Kullanılan kutu: {target_box}")

        self.model.zero_grad()
        confidence_score.backward(retain_graph=True)

        # Grad-CAM hesaplama
        gradients = self.gradients
        activations = self.activations

        if gradients.shape[2:] != activations.shape[2:]:
            gradients = torch.nn.functional.interpolate(
                gradients, size=activations.shape[2:], mode='bilinear', align_corners=False
            )

        cam = (gradients.mean(dim=(2, 3), keepdim=False) * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().detach().cpu().numpy()
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
            if confidence > 0.25:
                predictions.append({
                    'box': box,
                    'confidence': confidence,
                    'confidence_tensor': confidence_tensor
                })
        return predictions

    def generate_heatmap(self, img_rgb, cam):
        """Isı haritası oluştur ve görüntü üzerine bindir"""
        # Isı haritasını normalize et
        heatmap = cv2.resize(cam, (img_rgb.shape[1], img_rgb.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        
        # Farklı bir renk haritası kullanalım (TURBO daha modern ve görsel olarak çekici)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        return heatmap_colored

    def draw_boxes(self, img, predictions):
        """Tespit kutularını ve etiketleri çiz"""
        img_with_boxes = img.copy()
        
        for pred in predictions:
            box = pred['box']
            confidence = pred['confidence']
            x_min, y_min, x_max, y_max = map(int, box)
            
            # Kutu çizimi
            cv2.rectangle(img_with_boxes, 
                         (x_min, y_min), 
                         (x_max, y_max), 
                         BOX_COLOR, 
                         BOX_THICKNESS)
            
            # Etiket arka planı için dikdörtgen
            label = f'Conf: {confidence:.2f}'
            (label_w, label_h), _ = cv2.getTextSize(label, 
                                                   cv2.FONT_HERSHEY_SIMPLEX,
                                                   TEXT_SCALE, 
                                                   TEXT_THICKNESS)
            cv2.rectangle(img_with_boxes,
                         (x_min, y_min - label_h - 10),
                         (x_min + label_w, y_min),
                         BOX_COLOR,
                         -1)  # -1 ile içi dolu dikdörtgen
            
            # Etiket metni
            cv2.putText(img_with_boxes,
                       label,
                       (x_min, y_min - 5),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       TEXT_SCALE,
                       TEXT_COLOR,
                       TEXT_THICKNESS)
            
        return img_with_boxes

def process_image(img_path, model, grad_cam):
    """Görüntü işleme ve görselleştirme"""
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, INPUT_SIZE)
    
    # Tensor dönüşümü
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    img_tensor.requires_grad = True
    
    try:
        # Grad-CAM hesaplama
        cam, predictions = grad_cam.generate(img_tensor)
        
        # Isı haritası oluştur
        heatmap_colored = grad_cam.generate_heatmap(img_rgb, cam)
        
        # Isı haritası bindirme
        overlay = cv2.addWeighted(img_rgb, OVERLAY_ALPHA, 
                                heatmap_colored, 1 - OVERLAY_ALPHA, 0)
        
        # Kutuları çiz
        final_image = grad_cam.draw_boxes(overlay, predictions)
        
        # Sonucu kaydet
        output_path = "gradcam_result.jpg"
        final_image_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, final_image_bgr)
        print(f"Grad-CAM sonucu {output_path} dosyasına kaydedildi.")
        
        # Matplotlib ile görselleştirme
        plt.figure(figsize=(12, 8))
        plt.imshow(final_image)
        plt.title("Grad-CAM Isı Haritası ve Tespitler", pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    except ValueError as e:
        print("Hata:", e)

# Ana kod
if __name__ == "__main__":
    # Model yükleme
    model = torch.hub.load('./yolov5', 'custom', path='best.pt', source='local')
    model.eval()
    
    # Grad-CAM hazırlığı
    target_layer = find_last_conv_layer(model.model)
    grad_cam = GradCAM(model.model, target_layer)
    
    # Görüntü işleme
    process_image('img.jpeg', model, grad_cam)
