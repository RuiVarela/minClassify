import logging
import os
import random
import math
import json

from PIL import Image, ImageOps
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

import xml.etree.ElementTree as ET

NumberOfKeypoints = 4
ImageChannels = 3

#
# Creates the list of images from the source videos
#
def generate_source_images():
    dir = os.path.join("source", "testDataset")
    output = os.path.join("source", "images")
    if (not os.path.isdir(output)):
        os.mkdir(output)

    for folder in os.listdir(dir):
        if os.path.isdir(os.path.join(dir, folder)):
            dir_temp = os.path.join(dir, folder)
            for file in os.listdir(dir_temp):
                print(file)
                from subprocess import call

                if (file.endswith(".avi")):
                    current_output = os.path.join(output, folder)
                    if (not os.path.isdir(current_output)):
                        os.mkdir(current_output)

                    if (os.path.isdir(os.path.join(current_output, file))):
                        print("Folder already exist")
                    else:
                        call("cd " + current_output + " && mkdir " + file, shell=True)
                        call("ls", shell=True)

                        location = os.path.join(dir, folder, file)
                        gt_address = "cp " + location[0:-4] + ".gt.xml " + current_output + "/" + file + "/" + file + ".gt"
                        call(gt_address, shell=True)
                        command = "ffmpeg -i " + location + " " + current_output + "/" + file + "/%3d.jpg"
                        print(command)
                        call(command, shell=True)

def sort_gt(gt):
    '''
    Sort the ground truth labels so that TL corresponds to the label with smallest distance from O
    :param gt: 
    :return: sorted gt
    '''

    gt = np.asarray(gt)

    tmp = gt * gt
    sum_array = tmp.sum(axis=1)
    tl_index = np.argmin(sum_array)
    tl = gt[tl_index]
    tr = gt[(tl_index + 1) % 4]
    br = gt[(tl_index + 2) % 4]
    bl = gt[(tl_index + 3) % 4]

    output = np.asarray((tl, tr, br, bl)).reshape((1, 8))

    return output[0]

def generate_dataset():

    source = os.path.join("source", "images")
    output = "ds"

    data = []
    labels = []

    for folder in os.listdir(source):
        if os.path.isdir(os.path.join(source, folder)):
            for file in os.listdir(os.path.join(source, folder)):
                images_dir = os.path.join(source, folder, file)
                if os.path.isdir(images_dir):
                    list_gt = []
                    tree = ET.parse(images_dir + "/" + file + ".gt")
                    root = tree.getroot()
                    for a in root.iter("frame"):
                        list_gt.append(a)
                        im_no = 0

                    for image in os.listdir(images_dir):
                        if image.endswith(".jpg"):
                            im_no += 1

                            # Now we have opened the file and GT. Write code to create multiple files and scale gt
                            list_of_points = {}

                            data.append(os.path.join(images_dir, image))

                            for point in list_gt[int(float(image[0:-4])) - 1].iter("point"):
                                myDict = point.attrib
                                list_of_points[myDict["name"]] = (int(float(myDict['x'])), int(float(myDict['y'])))

                            ground_truth = (list_of_points["tl"], list_of_points["tr"], list_of_points["br"], list_of_points["bl"])
                            ground_truth = sort_gt(ground_truth)
                            labels.append(ground_truth)

    logging.info("Ground Truth Shape: %s", str(len(labels)))
    logging.info("Data shape %s", str(len(data)))

    zipped = list(zip(data, labels))
    random.shuffle(zipped)

    dataset_size = len(zipped)
    val_size = int(math.floor(dataset_size * 0.3))
    train_size = dataset_size - val_size

    splits = [("train", zipped[:train_size]), ("val", zipped[-val_size:])]
    print(f"dataset_size={dataset_size}")

    counter = 1
    for name, split in splits:
        print(f"copy {name} {len(split)}")

        folder = os.path.join(output, name)
        if not os.path.exists(folder):
            os.makedirs(folder)

        records = []
        for source_file, target in split:
            base_filename = f"{counter:05}.jpg"
            target_filename = os.path.join(folder, base_filename) 
            counter = counter + 1

            factor = 1.0
            with open(source_file, "rb") as f:
                image = Image.open(f)
                image = ImageOps.exif_transpose(image)
                image = image.convert("RGB")

                factor = 1024 / max(image.size) 
                img_w = int(image.size[0] * factor)
                img_h = int(image.size[1] * factor)

                image = image.resize((img_w, img_h))

                image.save(target_filename) 

            record = {
                "filename": base_filename,
                "keypoints": [
                    target[0] * factor, target[1] * factor,
                    target[2] * factor, target[3] * factor,
                    target[4] * factor, target[5] * factor,
                    target[6] * factor, target[7] * factor
                ]
            }
            records.append(record)

        data_file = os.path.join(folder, "data.json")
        with open(data_file, 'w') as f:
            json.dump(records, f, indent=4)


class KeypointsDataset(Dataset):
    def __init__(self, split="train"):
        self.split = split

        self.output_height = 224
        self.output_width = 224

        root_dir="ds"
        self.folder = os.path.join(root_dir, split)
        self.data = None

        data_file = os.path.join(self.folder, "data.json")
        with open(data_file) as f:
            self.data = json.load(f)

        self.transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((self.output_height, self.output_width)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        element = self.data[idx]
        filename = os.path.join(self.folder, element['filename'])

        image = None
        with open(filename, "rb") as f:
            image = Image.open(f)
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
        h, w = image.height, image.width

        keypoints = element["keypoints"]
        keypoints = [
            (keypoints[0] / w) * self.output_width, (keypoints[1] / h) * self.output_height,
            (keypoints[2] / w) * self.output_width, (keypoints[3] / h) * self.output_height,
            (keypoints[4] / w) * self.output_width, (keypoints[5] / h) * self.output_height,
            (keypoints[6] / w) * self.output_width, (keypoints[7] / h) * self.output_height,
        ]
        
        keypoints = torch.tensor(keypoints).float()
        image = self.transform(image)

        return image, keypoints

def get_dataset(split, transforms):
    if split == "test":
        return None
    
    return KeypointsDataset(split)