import os
import pandas as pd
from sklearn.model_selection import train_test_split

# --- AYARLAR ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Yolun sonuna 'chest_xray' klasörünü ekledik
DATA_PATH = os.path.join(BASE_DIR, "data", "chest_xray")
OUTPUT_PATH = os.path.join(BASE_DIR, "data") # CSV'ler ana data klasörüne gitsin

def get_patient_id(file_name):
    """Dosya isminden hasta veya çekim bazlı benzersiz ID ayıklar."""
    # Küçük harfe çevirerek kontrolü garantileyelim
    fn = file_name.lower()
    
    if "person" in fn:
        return fn.split('_')[0] # person1672
    
    if "normal2-im" in fn:
        # normal2-im-0383-0001.jpeg -> normal2-im-0383 kısmını alır
        parts = file_name.split('-')
        return "-".join(parts[:3])
        
    if "im-" in fn:
        # im-0761-0001.jpeg -> im-0761 kısmını alır
        parts = file_name.split('-')
        return "-".join(parts[:2])
    
    # Eğer yukarıdakilere uymuyorsa, ilk alt çizgiye kadar olan kısmı al
    return file_name.split('_')[0]

def run_split():
    all_data = []
    # Senin yapında val yerine 'val' mi 'value' mi var kontrol et. 
    # Genelde 'val' olur, eğer klasör adın 'value' ise aşağıdakini değiştir.
    sub_folders = ['train', 'test', 'val'] 
    categories = ['NORMAL', 'PNEUMONIA']

    print(f"📂 Aranan Ana Dizin: {DATA_PATH}")

    for sub in sub_folders:
        for cat in categories:
            folder_path = os.path.join(DATA_PATH, sub, cat)
            
            if not os.path.exists(folder_path):
                print(f"❌ Klasör bulunamadı: {folder_path}")
                continue
            
            files = os.listdir(folder_path)
            print(f"✅ Klasör bulundu: {sub}/{cat} | Dosya: {len(files)}")
            
            for img in files:
                if img.lower().endswith(('.jpeg', '.jpg', '.png')):
                    patient_id = get_patient_id(img)
                    # Dosya yolunu kaydederken 'chest_xray' kısmını da ekliyoruz
                    all_data.append({
                        'patient_id': patient_id,
                        'file_path': os.path.join('chest_xray', sub, cat, img),
                        'label': 1 if cat == 'PNEUMONIA' else 0
                    })

    df = pd.DataFrame(all_data)
    if df.empty:
        print("\n❌ HATA: Hala resim bulunamadı! Lütfen klasör adının 'val' mı yoksa 'value' mı olduğunu kontrol et.")
        return

    unique_patients = df['patient_id'].unique()
    train_ids, temp_ids = train_test_split(unique_patients, test_size=0.3, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)
    
    train_df = df[df['patient_id'].isin(train_ids)]
    val_df = df[df['patient_id'].isin(val_ids)]
    test_df = df[df['patient_id'].isin(test_ids)]
    
    train_df.to_csv(os.path.join(OUTPUT_PATH, 'train_list.csv'), index=False)
    val_df.to_csv(os.path.join(OUTPUT_PATH, 'val_list.csv'), index=False)
    test_df.to_csv(os.path.join(OUTPUT_PATH, 'test_list.csv'), index=False)
    
    print("\n" + "="*45)
    print(f"🚀 BAŞARIYLA TAMAMLANDI!")
    print(f"Toplam Görüntü: {len(df)}")
    print(f"CSV Dosyaları {OUTPUT_PATH} konumuna kaydedildi.")
    print("="*45)

if __name__ == "__main__":
    run_split()