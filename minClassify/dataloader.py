import logging
import torch
from torch.utils.data import DataLoader, TensorDataset

import torchvision
from torchvision.datasets import FashionMNIST, CIFAR10, Flowers102
from torchvision.transforms import RandomHorizontalFlip, ColorJitter, RandomAffine
from torchvision.transforms import Compose, Resize, Normalize, CenterCrop, ToTensor, Lambda
from torchvision.transforms import InterpolationMode

def createDataLoader(type, name, classes, split, to_train, batch_size, load_gpu):
    i_transform = ToTensor()
    o_transform = Lambda(lambda y: torch.nn.functional.one_hot(torch.tensor(y), classes).to(torch.float))

    if split is not None:
        data = type(root=".data", split=split, download=True, transform=i_transform, target_transform=o_transform)
    else:
        data = type(root=".data", train=to_train, download=True,  transform=i_transform, target_transform=o_transform)

    if to_train is not None:
        split = "train" if to_train else "test"

    if load_gpu:
        logging.info(f"Loading {split} {name} into gpu")
        features = [x for x, y in data]
        labels = [y for x, y in data]
        features = torch.stack(features).to("cuda")
        labels = torch.stack(labels).to("cuda")
        data = TensorDataset(features, labels)

    loader = DataLoader(data, batch_size=batch_size, shuffle=to_train)
    return loader

def createFashionLoaders(batch_size, load_gpu=False):
    logging.info(f"Creating FashionMNIST loaders with batch_size={batch_size}")
    test_dl = createDataLoader(FashionMNIST, "FashionMNIST", 10, None, False, batch_size, load_gpu)
    train_dl = createDataLoader(FashionMNIST, "FashionMNIST", 10, None, True, batch_size, load_gpu)
    return train_dl, test_dl, 10

def createCIFAR10Loaders(batch_size, load_gpu=False):
    logging.info(f"Creating CIFAR10 loaders with batch_size={batch_size}")
    test_dl = createDataLoader(CIFAR10, "CIFAR10", 10, None, False, batch_size, load_gpu)
    train_dl = createDataLoader(CIFAR10, "CIFAR10", 10, None, True, batch_size, load_gpu)
    return train_dl, test_dl, 10

def createFlowers102Loaders(batch_size, load_gpu=False):
    logging.info(f"Creating Flowers102 loaders with batch_size={batch_size}")
    classes = 102
    num_workers = pow(2, 4)

    divisor = 1
    resize_size = int(256 / divisor)
    center_crop_size = int(224 / divisor)

    preprocess = Compose([
        Resize(resize_size, interpolation=InterpolationMode.BILINEAR, antialias=True),
        CenterCrop(center_crop_size),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    i_transform = Compose([
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.3, contrast=0.3),
        RandomAffine(degrees=30, shear=20),

        preprocess
    ])
    o_transform = Lambda(lambda y: torch.nn.functional.one_hot(torch.tensor(y), classes).to(torch.float))
    data = Flowers102(root=".data", split="train", download=True, transform=i_transform, target_transform=o_transform)
    train_dl = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    i_transform = preprocess
    o_transform = Lambda(lambda y: torch.nn.functional.one_hot(torch.tensor(y), classes).to(torch.float))
    data = Flowers102(root=".data", split="val", download=True, transform=i_transform, target_transform=o_transform)
    test_dl = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dl, test_dl, classes 