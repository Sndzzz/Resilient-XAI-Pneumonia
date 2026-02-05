import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class PneumoniaDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        csv_file: 'data/train_list.csv' gibi CSV dosyasının yolu.
        root_dir: Resimlerin bulunduğu ana dizin (data/).
        transform: Resimler üzerinde yapılacak işlemler (boyutlandırma vb.).
        """
        self.data_info = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        # CSV'den dosya yolunu al
        img_relative_path = self.data_info.iloc[idx, 1]
        img_path = os.path.join(self.root_dir, img_relative_path)
        
        # Resmi aç ve RGB'ye çevir (bazı röntgenler gri tonlamalı olabilir)
        image = Image.open(img_path).convert('RGB')
        label = int(self.data_info.iloc[idx, 2]) # 0 veya 1

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data_loaders(data_dir, batch_size=32):
    """DataLoader nesnelerini oluşturur."""
    
    # Resimleri 224x224 yap ve normalize et (Plana göre)
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset nesnelerini oluştur
    train_ds = PneumoniaDataset(os.path.join(data_dir, 'train_list.csv'), data_dir, transform=data_transforms)
    val_ds = PneumoniaDataset(os.path.join(data_dir, 'val_list.csv'), data_dir, transform=data_transforms)
    test_ds = PneumoniaDataset(os.path.join(data_dir, 'test_list.csv'), data_dir, transform=data_transforms)

    # DataLoader nesnelerini oluştur
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader