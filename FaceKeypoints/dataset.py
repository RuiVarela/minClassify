import os
import random
import math

import numpy as np

import pandas as pd 

import torch 
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2

NumberOfKeypoints = 15

class FaceDataset(data.Dataset):
    def __init__(self, mode, transform=True):
        super().__init__()
        self.do_transforms = transform and (mode == "train")
        self.mode = mode
        self.images = None
        self.targets = None

        raw = pd.read_csv("ds/training.zip" , compression='zip')
        raw = raw.sample(frac=1)
        features_name = list(raw)   
        features_name.remove('Image')
        images = raw['Image']
        targets = raw.drop(columns=['Image'])
        targets = targets.fillna(-1).to_numpy()

        samples = raw.shape[0]
        train_factor = 0.8

        if mode == "train":
            self.images = images[0: int(samples * train_factor)]
            self.targets = targets[0: int(samples * train_factor)]
        else:
            self.images = images[int(samples * (1.0 - train_factor)): ]
            self.targets = targets[int(samples * (1.0 - train_factor)): ]

        self.images = self.reshape_images(self.images)

    def reshape_images(self, images):
        images_reshaped = np.zeros((images.shape[0], 96, 96, 1))
        for i, img in enumerate(images):
            img = img.split(' ')
            img = np.array([int(num) for num in img])
            img = img.reshape((96, 96, 1))
            images_reshaped[i] = img
        
        return images_reshaped
    
    def transform(self, image, target):
        image = transforms.ToTensor()(image)

        if self.do_transforms:
            p = random.random()
            if p > 0.5:
                image = transforms.GaussianBlur(5)(image)
                           
            p = random.random()
            if p > 0.5:
                mask = torch.rand(size=image.shape)
                mask[mask > 0.95] = 0
                mask[mask <= 0.95] = 1
                image = mask * image

            p = random.random()
            if p > 0.5:  
                angle = random.randint(-20, 20)
                image = transforms.functional.rotate(image, angle=angle)
                angle = -angle
                idx = target == -1
                angle_radians = math.radians(angle)
                
                x = (target[::2] - 48) * math.cos(angle_radians) - (target[1::2] - 48) * math.sin(angle_radians)
                y = (target[::2] - 48) * math.sin(angle_radians) + (target[1::2] - 48) * math.cos(angle_radians)

                target[::2] = x + 48
                target[1::2] = y + 48
                target[idx] = -1

        return image, torch.tensor(target).float()
        
    def __getitem__(self, index):
        return self.transform(self.images[index], target=self.targets[index])

    def __len__(self):
        return self.images.shape[0]





def get_dataset(split, transforms):
    if split == "test":
        return None
    
    return FaceDataset(split, transform=transforms)

    