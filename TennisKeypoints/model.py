from dataset import NumberOfKeypoints, ImageChannels

import torch.nn as nn
import torch
import torchvision
import timm

class BaseModel_OLD(nn.Module):
    def __init__(self, kind, pretrained=True):
        super().__init__()
        self.timm_model = timm.create_model(kind, pretrained=pretrained, in_chans=ImageChannels)
        self.fc = nn.Linear(1000, NumberOfKeypoints * 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.timm_model(x)
        x = self.relu(x)
        x = self.fc(x)
        return x
    
class BaseModel(nn.Module):
    def __init__(self, kind, pretrained=True):
        super().__init__()
        self.timm_model = timm.create_model(kind, pretrained=pretrained, in_chans=ImageChannels, num_classes=NumberOfKeypoints * 2)

    def forward(self, x):
        x = self.timm_model(x)
        return x

class MobileNetModel(BaseModel):
    def __init__(self, pretrained=True):
        super().__init__('mobilenetv3_large_100', pretrained)

class ResNetModel(BaseModel):
    def __init__(self, pretrained=True):
        super().__init__('resnet50', pretrained)
    
class DenseNetModel(BaseModel):
    def __init__(self, pretrained=True):
        super().__init__('densenet201', pretrained)

class EfficientNetModel(BaseModel):
    def __init__(self, pretrained=True):
        super().__init__('tf_efficientnetv2_s', pretrained)

class InceptionNextModel(BaseModel):
    def __init__(self, pretrained=True):
        super().__init__('inception_next_small', pretrained)

class TinyNetModel(BaseModel):
    def __init__(self, pretrained=True):
        super().__init__('tinynet_e.in1k', pretrained)
    
    
    
    
def available_models():
    return [MobileNetModel(), TinyNetModel()]
    

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
    elif data['model_kind'] == "ResNetModel":
        model = ResNetModel(False)
    elif data['model_kind'] == "TinyNetModel":
        model = TinyNetModel(False)

    model.load_state_dict(data['model_state_dict'])

    return model