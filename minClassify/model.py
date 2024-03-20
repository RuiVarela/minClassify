import logging
from math import ceil
from torch import nn
import torch
import torchvision
from torch.nn import init

#
# Very Light Network inspired on AlexNet
# https://github.com/wzyjsha-00/CNN-for-Fashion-MNIST
#
class AlexNetLight(nn.Module):
    def __init__(self, input_size, factor, classes):
        super().__init__()

        cs = int(96 * factor)
        cm = int(256 * factor)
        cb = int(384 * factor)

        lb = int(1024 * factor)
        lm = int(256 * factor)
        assert(input_size[2] == input_size[3])
        
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=input_size[1], out_channels=cs, kernel_size=(3, 3), padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        c1_size = input_size[2] 
        c1_size = int(c1_size / 2)

        self.c2 = nn.Sequential(
            nn.Conv2d(in_channels=cs, out_channels=cm, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2) 
        )
        c2_size = c1_size
        c2_size = int((c2_size - 3 + 1) / 2)

        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=cm, out_channels=cb, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )
        c3_size = c2_size

        self.c4 = nn.Sequential(
            nn.Conv2d(in_channels=cb, out_channels=cb, kernel_size=(3, 3), padding=1), 
            nn.ReLU()
        )
        c4_size = c3_size

        self.c5 = nn.Sequential(
            nn.Conv2d(in_channels=cb, out_channels=cm, kernel_size=(2, 2), padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2) 
        )
        c5_size = c4_size
        c5_size = int(ceil(c5_size / 2))

        # print(f"c1_size {c1_size}")
        # print(f"c2_size {c2_size}")
        # print(f"c3_size {c3_size}")
        # print(f"c4_size {c4_size}")
        # print(f"c5_size {c5_size}")

        self.c6 = nn.Sequential(
            nn.Linear(in_features=cm * c5_size * c5_size, out_features=lb),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.c7 = nn.Sequential(
            nn.Linear(in_features=lb, out_features=lm),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.c8 = nn.Sequential(
            nn.Linear(in_features=lm, out_features=classes)
        )

    def forward(self, x):
        # print(f"x {x.shape}")
        x = self.c1(x)
        # print(f"c1 {x.shape}")
        x = self.c2(x)
        # print(f"c2 {x.shape}")
        x = self.c3(x)
        # print(f"c3 {x.shape}")
        x = self.c4(x)
        # print(f"c4 {x.shape}")
        x = self.c5(x)
        # print(f"c5 {x.shape}")
        x = torch.flatten(x, 1)
        x = self.c6(x)
        x = self.c7(x)
        logits = self.c8(x)
        return logits
    
def createAlexNetLightModel(input_size, classes, device):
    logging.info(f"Creating AlexNetLight model with input_size={input_size} classes={classes}")
    model = AlexNetLight(input_size, 1.0 / 8, classes).to(device)
    return model


#
# Classifier Wrapper
#
class WrapClassifier(nn.Module):
    FINE_TUNE_ALL = 0
    FINE_TUNE_CLASSIFIER = 1
    FINE_TUNE_NEW_LAYERTS = 2

    def __init__(self):
        super().__init__()
        self.model = None
        self.classifier = []
        self.new = []

    def forward(self, x):
        return self.model(x)

    def fine_tune(self, kind):
        # The requires_grad parameter controls whether this parameter is
        # trainable during model training.
        for p in self.model.parameters():
            p.requires_grad = False

        if kind == SqueezeNetClassifier.FINE_TUNE_NEW_LAYERTS:
            for l in self.new:
                for p in l.parameters():
                    p.requires_grad = True
        elif kind == SqueezeNetClassifier.FINE_TUNE_CLASSIFIER:
            for l in self.classifier:
                for p in l.parameters():
                    p.requires_grad = True
        else:
            for p in self.model.parameters():
                p.requires_grad = True

#
# SqueezeNet1_1
#
class SqueezeNetClassifier(WrapClassifier):

    def __init__(self, classes, load_pretrained):
        super().__init__()

        weights = None
        if load_pretrained:
            weights = torchvision.models.SqueezeNet1_1_Weights.IMAGENET1K_V1

        self.model = torchvision.models.squeezenet1_1(weights=weights)

        final_conv = nn.Conv2d(512, classes, kernel_size=1)
        self.model.classifier[1] = final_conv

        init.normal_(final_conv.weight, mean=0.0, std=0.01)
        if final_conv.bias is not None:
            init.constant_(final_conv.bias, 0)

        self.classifier = [self.model.classifier]
        self.new = [final_conv]

def createSqueezeNetModel(input_size, classes, device):
    logging.info(f"Creating SqueezeNetClassifier model with input_size={input_size} classes={classes}")
    model = SqueezeNetClassifier(classes, True).to(device)
    #model.fine_tune(WrapClassifier.FINE_TUNE_CLASSIFIER)
    return model


#
# MobileNet
#
class MobileNet(WrapClassifier):
    def __init__(self, small, classes, load_pretrained):
        super().__init__()

        if small:
            weights = None
            if load_pretrained:
                weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            self.model = torchvision.models.mobilenet_v3_small(weights=weights)
        else:
            weights = None
            if load_pretrained:
                weights = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
            self.model = torchvision.models.mobilenet_v3_large(weights=weights)            

        last_channel= self.model.classifier[3].in_features
        final_linear = nn.Linear(last_channel, classes)
        self.model.classifier[3] = final_linear

        self.classifier = [self.model.classifier]
        self.new = [final_linear]

def createMobileNetSmallModel(input_size, classes, device):
    logging.info(f"Creating MobileNetSmall model with input_size={input_size} classes={classes}")
    model = MobileNet(True, classes, True).to(device)
    #model.fine_tune(WrapClassifier.FINE_TUNE_CLASSIFIER)
    return model

def createMobileNetLargeModel(input_size, classes, device):
    logging.info(f"Creating MobileNetLarge model with input_size={input_size} classes={classes}")
    model = MobileNet(False, classes, True).to(device)
    #model.fine_tune(WrapClassifier.FINE_TUNE_CLASSIFIER)
    return model

#
# Resnet
#
class ResnetNet(WrapClassifier):
    def __init__(self, size, classes, load_pretrained):
        super().__init__()

        if size == 18:
            weights = None
            if load_pretrained:
                weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            self.model = torchvision.models.resnet18(weights=weights)
        elif size == 34:
            weights = None
            if load_pretrained:
                weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1
            self.model = torchvision.models.resnet34(weights=weights)
        elif size == 50:
            weights = None
            if load_pretrained:
                weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
            self.model = torchvision.models.resnet50(weights=weights)
        elif size == 101:
            weights = None
            if load_pretrained:
                weights = torchvision.models.ResNet101_Weights.IMAGENET1K_V2
            self.model = torchvision.models.resnet101(weights=weights)
        elif size == 152:
            weights = None
            if load_pretrained:
                weights = torchvision.models.ResNet152_Weights.IMAGENET1K_V2
            self.model = torchvision.models.resnet152(weights=weights)

        last_channel= self.model.fc.in_features
        final_linear = nn.Linear(last_channel, classes)
        self.model.fc = final_linear

        self.classifier = [final_linear]
        self.new = [final_linear]

def _createResnet(input_size, size, classes, device):
    logging.info(f"Creating ResNet{size} model with input_size={input_size} classes={classes}")
    model = ResnetNet(size, classes, True)
    # model.fine_tune(WrapClassifier.FINE_TUNE_NEW_LAYERTS)
    return model.to(device)

def createResNet18Model(input_size, classes, device):
    return _createResnet(input_size, 18, classes, device)

def createResNet34Model(input_size, classes, device):
    return _createResnet(input_size, 34, classes, device)

def createResNet50Model(input_size, classes, device):
    return _createResnet(input_size, 50, classes, device)

def createResNet101Model(input_size, classes, device):
    return _createResnet(input_size, 101, classes, device)

def createResNet152Model(input_size, classes, device):
    return _createResnet(input_size, 152, classes, device)