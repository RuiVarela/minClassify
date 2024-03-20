import logging
import os
import json

from PIL import Image, ImageOps

import torch 
import torch.utils.data as data

from torchvision.transforms import v2

NumberOfKeypoints = 14
ImageChannels = 3

class CourtDataset(data.Dataset):
    def __init__(self, mode, scale = 3):
        self.mode = mode
        assert mode in ['train', 'val'], 'incorrect mode'

        if scale <= 0:
            self.output_height = 224
            self.output_width = 224
        else:
            self.output_height = int(720 / scale)
            self.output_width = int(1280 / scale)

        self.path_dataset = os.path.join('ds', 'data')
        self.path_images = os.path.join(self.path_dataset, 'images')
        with open(os.path.join(self.path_dataset, 'data_{}.json'.format(mode)), 'r') as f:
            self.data = json.load(f)

        logging.info(f"mode={mode} len={len(self.data)} output_width={self.output_width} output_height={self.output_height}")

        self.transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((self.output_height, self.output_width)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        img_name = self.data[index]['id'] + '.png'
        image = None
        with open(os.path.join(self.path_images, img_name), "rb") as f:
            image = Image.open(f)
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")

        h, w = image.height, image.width
            
        image = self.transform(image)

        kps = self.data[index]['kps']
        keypoints = []
        for i in range(len(kps)):
            x = int((kps[i][0] / w) * self.output_width)
            y = int((kps[i][1] / h) * self.output_height)
            keypoints.append(x)
            keypoints.append(y)

        return image.float(), torch.tensor(keypoints).float()
        
        
    def __len__(self):
        return len(self.data)


def get_dataset(split, transforms):
    if split == "test":
        return None
    
    return CourtDataset(split)

    