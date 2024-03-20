from dataset import NumberOfKeypoints

import torch.nn as nn
import torch
import torchvision
import timm

class DenseNetModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.timm_model = timm.create_model('densenet201', pretrained=pretrained, in_chans=1)
        self.fc = nn.Linear(1000, 30)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.timm_model(x)
        x = self.relu(x)
        x = self.fc(x)
        return x
    
class EfficientNetModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.timm_model = timm.create_model('tf_efficientnetv2_s', pretrained=pretrained, in_chans=1)
        self.fc = nn.Linear(1000, 30)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.timm_model(x)
        x = self.relu(x)
        x = self.fc(x)
        return x
    
class MobileNetModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.timm_model = timm.create_model('mobilenetv3_large_100', pretrained=pretrained, in_chans=1)
        self.fc = nn.Linear(1000, 30)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.timm_model(x)
        x = self.relu(x)
        x = self.fc(x)
        return x
    
class InceptionNextModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.timm_model = timm.create_model('inception_next_small', pretrained=pretrained, in_chans=1)
        self.fc = nn.Linear(1000, 30)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.timm_model(x)
        x = self.relu(x)
        x = self.fc(x)
        return x
    
    
def available_models():
    return [MobileNetModel(), InceptionNextModel(), DenseNetModel(), EfficientNetModel()]
    

def load_model_checkpoint(checkpoint):
    data = torch.load(checkpoint)

    if data['model_kind'] == "DenseNetModel":
        model = DenseNetModel(False)
    elif data['model_kind'] == "EfficientNetModel":
        model = EfficientNetModel(False)
    elif data['model_kind'] == "MobileNetModel":
        model = MobileNetModel(False)
    elif data['model_kind'] == "InceptionNextModel":
        model = InceptionNextModel(False)

    model.load_state_dict(data['model_state_dict'])

    return model