import torch
import os
import random
import numpy as np

class Config:
    # ==============================================================================
    # 1. ORTAM VE DOSYA YOLLARI (Environment & Paths)
    # ==============================================================================
    # Projenin çalıştığı ana dizini bulur.
    BASE_DIR = os.getcwd()
    
    # Veri ve çıktı klasörleri
    DATA_DIR = os.path.join(BASE_DIR, "data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    
    # Modellerin ve XAI görsellerinin kaydedileceği yerler
    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "saved_models")
    XAI_PLOT_PATH = os.path.join(OUTPUT_DIR, "xai_results")
    LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

    # Klasörler yoksa otomatik oluşturur (Hata almayı engeller)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(XAI_PLOT_PATH, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # ==============================================================================
    # 2. MODEL SEÇİMİ VE BATCH SIZE STRATEJİSİ (En Kritik Bölüm)
    # ==============================================================================
    
    # ⚠️ DİKKAT: Çalışmaya başlamadan önce sadece burayı değiştirin!
    # Seçenekler: "resnet50", "densenet121", "vgg16"
    # Örnek: Ayşe çalışırken burayı "vgg16" veya "densenet121" yapacak.
    CURRENT_MODEL = "resnet50"

    # Her modelin donanım ihtiyacına göre optimize edilmiş Batch Size'lar
    # VGG16 çok VRAM harcadığı için düşük, ResNet daha verimli olduğu için yüksek.
    BATCH_SIZES_MAP = {
        "resnet50": 32,      # Sudenaz'ın modeli (Daha hızlı eğitim)
        "densenet121": 32,   # Ayşe'nin 2. modeli (Orta seviye yük)
        "vgg16": 16          # Ayşe'nin 1. modeli (Çok ağır, düşük batch şart!)
    }

    # ==============================================================================
    # 3. VERİ STANDARTLARI (Data Config)
    # ==============================================================================
    IMAGE_SIZE = (224, 224) 
    NUM_CLASSES = 2 
    CLASS_NAMES = ['NORMAL', 'PNEUMONIA']
    
    # Transfer Learning için ImageNet İstatistikleri (Standarttır, değiştirmeyin)
    # Görüntüleri 0-1 arasına değil, bu dağılıma göre normalize edeceğiz.
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

    # ==============================================================================
    # 4. EĞİTİM HİPERPARAMETRELERİ (Training Hyperparams)
    # ==============================================================================
    EPOCHS = 25
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5  # Overfitting engellemek için L2 Regularization
    
    # Donanım kontrolü: GPU varsa kullanır, yoksa CPU'ya geçer.
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==============================================================================
    # 5. XAI VE HOOK AYARLARI (Explainability)
    # ==============================================================================
    # Grad-CAM ve diğer XAI araçlarının "kanca" atacağı (dinleyeceği) katmanlar.
    # Bu katmanlar, modelin özellik çıkardığı son evrelerdir.
    TARGET_LAYERS = {
        "resnet50": "layer4",           # ResNet son blok
        "vgg16": "features.29",         # VGG son konvolüsyon (MaxPooling öncesi)
        "densenet121": "features.norm5" # DenseNet son normalizasyon katmanı
    }

    # ==============================================================================
    # 6. DAYANIKLILIK TESTLERİ (Robustness & Stress Tests)
    # ==============================================================================
    # Sudenaz'ın test aşamasında görüntüye ekleyeceği gürültü oranları
    STRESS_TEST_NOISE_LEVELS = [0.01, 0.05, 0.1, 0.2] 

    # Raporlanacak Metrikler
    METRICS = ["Accuracy", "F1-Score", "Precision", "Recall", "AUC-ROC"]

    # ==============================================================================
    # 7. YARDIMCI METOTLAR (Helper Methods)
    # ==============================================================================
    
    @staticmethod
    def get_batch_size():
        """Seçili modele (CURRENT_MODEL) uygun batch size'ı döndürür."""
        return Config.BATCH_SIZES_MAP[Config.CURRENT_MODEL]

    @staticmethod
    def get_target_layer_name():
        """Seçili modelin XAI için hedef katman ismini döndürür."""
        return Config.TARGET_LAYERS[Config.CURRENT_MODEL]

    @staticmethod
    def seed_everything(seed=42):
        """
        Reproducibility (Tekrarlanabilirlik) ayarı.
        Ayşe ve Sudenaz'ın bilgisayarlarında verinin aynı şekilde karışmasını sağlar.
        """
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"[INFO] Random Seed sabitlendi: {seed} | Model: {Config.CURRENT_MODEL}")

# ==============================================================================
# KULLANIM ÖRNEĞİ (Bunu kodunuzun en başına yazın)
# ==============================================================================
# Config.seed_everything(42)
# print(f"Aktif Model: {Config.CURRENT_MODEL}")
# print(f"Kullanılacak Batch Size: {Config.get_batch_size()}")
# print(f"Hedef XAI Katmanı: {Config.get_target_layer_name()}")