from dataset import NumberOfKeypoints, ImageChannels

import torch.nn as nn
import torch
import timm

class BaseModel(nn.Module):
    def __init__(self, kind, pretrained=True):
        super().__init__()
        self.kind = kind
        self.timm_model = timm.create_model(kind, pretrained=pretrained, in_chans=ImageChannels)
        self.fc = nn.Linear(1000, NumberOfKeypoints * 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.timm_model(x)
        x = self.relu(x)
        x = self.fc(x)
        return x
     
def available_models():
    # [BaseModel('resnet50', True), BaseModel('densenet201', True), BaseModel('tf_efficientnetv2_s', True), BaseModel('inception_next_small', True)]
    return [BaseModel('mobilenetv3_large_100', True), BaseModel('tinynet_e.in1k', True), BaseModel('densenet201', True)]


def load_model_checkpoint(checkpoint):
    data = torch.load(checkpoint)
    model = BaseModel(data['model_kind'], False)
    model.load_state_dict(data['model_state_dict'])
    return model