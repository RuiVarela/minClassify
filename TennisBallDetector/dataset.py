import logging
import os
import random
import math
import pickle


from PIL import Image, ImageOps
import numpy as np

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

NumberOfKeypoints = 1
#ImageChannels = 3 * 3
ImageChannels = 3 * 1

class BallDataset(Dataset):
    def __init__(self, mode, scale = -3, cache_folder=None):
        self.mode = mode
        assert mode in ['train', 'val'], 'incorrect mode'

        if scale <= 0:
            self.output_height = 224
            self.output_width = 224
        else:
            self.output_height = int(720 / scale)
            self.output_width = int(1280 / scale)

        self.path_dataset = os.path.join('source', 'Dataset')
        self.cache_folder = cache_folder

        if self.cache_folder is not None and not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)

        all_data = []
        self.data = []

        for game_id in range(1,11):
            game = f"game{game_id}"
            clips = os.listdir(os.path.join(self.path_dataset, game))
            for clip in clips:
                path_labels = os.path.join(os.path.join(self.path_dataset, game, clip), 'Label.csv')
                labels = pd.read_csv(path_labels)    
                for idx in range(labels.shape[0]):
                    file_name, vis, x, y, _ = labels.loc[idx, :]

                    if idx < 2:
                        continue

                    if math.isnan(x) or math.isnan(y):
                        #logging.info(f"Discarding {game} {clip} {idx} - math.isnan(x) or math.isnan(y)")
                        continue

                    if vis == 0:
                        #logging.info(f"Discarding {game} {clip} {idx} - vis")
                        continue

                    if x < 0:
                        logging.info(f"Discarding {game} {clip} {idx} - x < 0")
                        continue

                    if x > 1280:
                        logging.info(f"Discarding {game} {clip} {idx} - x > 1280")
                        continue

                    if y < 0:
                        logging.info(f"Discarding {game} {clip} {idx} - y < 0")
                        continue

                    if y > 720:
                        logging.info(f"Discarding {game} {clip} {idx} - y > 720")
                        continue

                    current = {
                        "p-0": os.path.join(self.path_dataset, game, clip, f"{idx - 0:04}.jpg"),
                        "p-1": os.path.join(self.path_dataset, game, clip, f"{idx - 1:04}.jpg"),
                        "p-2": os.path.join(self.path_dataset, game, clip, f"{idx - 2:04}.jpg"),
                        "x": x,
                        "y": y,
                        "v": vis
                    }

                    all_data.append(current)

                    #print(f"filename: {file_name} | {idx} | {current}")

        random.shuffle(all_data)

        train_rate = 0.7
        train_size = int(len(all_data) * train_rate)
        if mode == "train":
            self.data = all_data[:train_size]
        else:
            self.data = all_data[train_size:]

        logging.info(f"mode={mode} len={len(self.data)} output_width={self.output_width} output_height={self.output_height}")


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.cache_folder is not None:
            cache_file = os.path.join(self.cache_folder, f"{self.mode}_{idx}.pik")
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    unserialized_data = pickle.load(f)
                    return unserialized_data
        
        current = self.data[idx]

        x = current["x"]
        y = current["y"]
        
        p0 = current["p-0"] # path
        p1 = current["p-1"] # prev
        p2 = current["p-2"] # prev prev

        channels, h, w = self.get_input(p0, p1, p2)

        x = int((x / w) * self.output_width)
        y = int((y / h) * self.output_height)

        if x > self.output_width:
            logging.info(f"Issue {idx} x={x} ! {w}x{h}")

        if y > self.output_height:
            logging.info(f"Issue {idx} y={y} ! {w}x{h}")

        data = (channels, torch.tensor([x, y]).float())

        if self.cache_folder is not None:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return data
    
    def load_image(self, path):
        use_gray = ImageChannels == 3 * 1

        image = None
        with open(path, "rb") as f:
            image = Image.open(f)
            image = ImageOps.exif_transpose(image)
            if use_gray:
                image = ImageOps.grayscale(image)
            else:
                image = image.convert("RGB")

        h, w = image.height, image.width

        transforms = [
            v2.ToImage(),
            v2.Resize((self.output_height, self.output_width)),
            v2.ToDtype(torch.float32, scale=True)
        ]

        if not use_gray:
            transforms.append(v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        image = v2.Compose(transforms)(image)
        return image, h, w

    def get_input(self, path, path_prev, path_preprev):
        img0, img0_h, img0_w = self.load_image(path)
        img1, img1_h, img1_w = self.load_image(path_prev)
        img2, img2_h, img2_w = self.load_image(path_preprev)

        imgs = torch.cat((img0, img1, img2), axis=0)
        return imgs, img0_h, img0_w
    
def get_dataset(split, transforms):
    if split == "test":
        return None
    
    return BallDataset(split)