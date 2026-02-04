import torch
import torch.nn as nn
import torchvision.models as models

class PneumoniaEnsemble(nn.Module):
    def __init__(self):
        super(PneumoniaEnsemble, self).__init__()
        
        # 1. ResNet50
        self.resnet = models.resnet50(pretrained=True)
        in_features_res = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features_res, 2) # Normal ve Pneumonia için 2 çıkış
        
        # 2. DenseNet121
        self.densenet = models.densenet121(pretrained=True)
        in_features_dense = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(in_features_dense, 2)
        
        # 3. VGG16
        self.vgg = models.vgg16(pretrained=True)
        in_features_vgg = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Linear(in_features_vgg, 2)
        
        # Ensemble Karar Katmanı (3 modelden gelen toplam 6 çıktıyı birleştirir)
        # Her model 2 ihtimal (Normal/Pneumonia) döndürür. Toplam 6 girdi.
        self.ensemble_classifier = nn.Linear(6, 2)

    def forward(self, x):
        out_res = self.resnet(x)
        out_dense = self.densenet(x)
        out_vgg = self.vgg(x)
        
        # Çıktıları yan yana birleştir (Concatenate)
        combined = torch.cat((out_res, out_dense, out_vgg), dim=1)
        
        # Son kararı ver
        final_output = self.ensemble_classifier(combined)
        return final_output