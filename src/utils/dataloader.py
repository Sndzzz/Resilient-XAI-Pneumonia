import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

class PneumoniaDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_info = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        # Ayşe'nin Tavsiyesi: Sütun adıyla (file_path) çağırmak daha güvenlidir
        img_relative_path = self.data_info.iloc[idx]['file_path'].replace('\\', '/')
        img_path = os.path.join(self.root_dir, img_relative_path)
        
        image = Image.open(img_path).convert('RGB')
        label = int(self.data_info.iloc[idx]['label']) 

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data_loaders(data_dir, batch_size=32):
    # 1. Eğitim için Veri Artırma (EDA'daki parlaklık ve açı farklarını çözmek için)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), 
        transforms.RandomRotation(15),     # Röntgenlerdeki küçük kaymalar için
        transforms.ColorJitter(brightness=0.15, contrast=0.15), # Parlaklık dengesizliği için
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Test/Val için sadece boyutlandırma ve normalizasyon
    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = PneumoniaDataset(os.path.join(data_dir, 'train_list.csv'), data_dir, transform=train_transforms)
    
    # 2. Weighted Sampler: %73-%26 dengesizliğini matematiksel olarak çözen kısım
    labels = train_ds.data_info['label'].values
    class_sample_count = pd.Series(labels).value_counts().sort_index().values
    
    # Ağırlık hesabı: Az olan sınıfa (Normal) daha yüksek ağırlık verilir
    weights = 1. / torch.tensor(class_sample_count, dtype=torch.float)
    samples_weights = weights[labels]
    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

    # Eğitimde sampler kullanıldığı için shuffle=False olmalıdır
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    
    val_ds = PneumoniaDataset(os.path.join(data_dir, 'val_list.csv'), data_dir, transform=val_test_transforms)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    test_ds = PneumoniaDataset(os.path.join(data_dir, 'test_list.csv'), data_dir, transform=val_test_transforms)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader