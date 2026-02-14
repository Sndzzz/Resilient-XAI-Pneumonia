import torch
import os
import random
import numpy as np

class Config:
    # --- 1. ORTAM VE DOSYA YOLLARI ---
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, "data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    
    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "saved_models")
    XAI_PLOT_PATH = os.path.join(OUTPUT_DIR, "xai_results")
    LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

    # CSV Dosya İsimleri (Ayşe ile ortak olması şart!)
    TRAIN_CSV = "train_list.csv"
    VAL_CSV = "val_list.csv"
    TEST_CSV = "test_list.csv"

    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(XAI_PLOT_PATH, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # --- 2. MODEL SEÇİMİ ---
    CURRENT_MODEL = "resnet50" # Ayşe bunu "vgg16" yapacak

    BATCH_SIZES_MAP = {
        "resnet50": 32,
        "densenet121": 32,
        "vgg16": 16
    }

    # --- 3. VERİ STANDARTLARI ---
    IMAGE_SIZE = (224, 224) 
    NUM_CLASSES = 2 
    CLASS_NAMES = ['NORMAL', 'PNEUMONIA']
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

    # --- 4. EĞİTİM PARAMETRELERİ ---
    EPOCHS = 25
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 5. XAI AYARLARI ---
    TARGET_LAYERS = {
        "resnet50": "layer4",
        "vgg16": "features.29",
        "densenet121": "features.norm5"
    }
    # Eğitim sırasında kaç epoch'ta bir XAI görseli üretilsin?
    XAI_EVERY_N_EPOCH = 5 

    # --- 6. DAYANIKLILIK ---
    STRESS_TEST_NOISE_LEVELS = [0.01, 0.05, 0.1, 0.2] 
    METRICS = ["Accuracy", "F1-Score", "Precision", "Recall", "AUC-ROC"]

    # --- 7. YARDIMCI METOTLAR ---
    @staticmethod
    def get_batch_size():
        return Config.BATCH_SIZES_MAP[Config.CURRENT_MODEL]

    @staticmethod
    def get_target_layer_name():
        return Config.TARGET_LAYERS[Config.CURRENT_MODEL]

    @staticmethod
    def seed_everything(seed=42):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False