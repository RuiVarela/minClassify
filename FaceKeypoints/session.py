
import logging
import os

import random
import csv

import numpy as np

import torch.nn as nn
import torch
import torch.utils.data as data
import torchvision
from torchvision.transforms import v2

import matplotlib
import matplotlib.pyplot as plt

from dataset import NumberOfKeypoints, get_dataset
from model import load_model_checkpoint

class Session:
    def __init__(self):
        self.device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.seed = None
        self.save_period = 5

        self.session_dir = None
        self.best_loss = None
        self.best_epoch = None
        self.best_weights = None
        self.best_weights_saved = None

        self.cvs_fieldnames = ['epoch', 'train loss', 'test loss']
        self.cvs_filename = "report.cvs"
    
    def ensure_session_folder(self):
        if not os.path.exists(self.session_dir):
            os.makedirs(self.session_dir)

    def configure_log(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        logger.handlers.clear()
        
        if not logger.handlers:
            self.ensure_session_folder()
            file_handler = logging.FileHandler(os.path.join(self.session_dir, 'log.txt'))
            file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter('%(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)


    def setup(self):
        self.best_loss = 1000000.0
        self.best_epoch = 0
        self.best_weights = None
        self.best_weights_saved = False

        session_root = os.path.join(os.getcwd(), 'session')
        self.session_dir = os.path.join(session_root, f'{0:04d}')

        if os.path.exists(session_root):
            folders = [f for f in os.listdir(session_root) if os.path.isdir(os.path.join(session_root, f))]
            folders.sort()
            if len(folders) > 0:
                last = int(folders[-1])
                last = last + 1
                self.session_dir = os.path.join(session_root, f'{last:04d}')

        self.configure_log()

        if self.seed:
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
            torch.manual_seed(self.seed)
            random.seed(self.seed)
            logging.info(f"Using seed {self.seed}")

    def end_epoch(self, epoch, model, losses, lr, is_last):    
        train_loss = losses[0]
        test_loss = losses[1]

        message = f"epoch {epoch:04} lr: {lr:7.5} trn: {train_loss:7.4f} evl: {test_loss:7.4f}"
        logging.info(message)

        self.ensure_session_folder()
    
        if test_loss < self.best_loss:
            self.best_loss = test_loss
            self.best_epoch = epoch
            self.best_weights = model.state_dict().copy()
            self.best_weights_saved = False
        

        if (epoch % self.save_period) == 0 or is_last:
            if not self.best_weights_saved:
                name = f"{self.best_epoch:05}.pt"
                filepath = os.path.join(self.session_dir, name)

                state = {
                    'model_kind': model.__class__.__name__,
                    'model_state_dict': self.best_weights,
                }
                torch.save(state, filepath)
                self.best_weights_saved = True


        #
        # Save csv
        #
        filepath = os.path.join(self.session_dir, self.cvs_filename)
        add_header = not os.path.exists(filepath)
        with open(filepath,'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.cvs_fieldnames)
            if add_header:
                writer.writeheader()
            writer.writerow({'epoch': epoch, 'train loss': train_loss, 'test loss': test_loss})

        #
        # Generate plot
        #
        data_epoch = []
        data_train_loss = []
        data_test_loss = []
        with open(filepath, newline='') as f:
            reader = csv.DictReader(f, fieldnames=self.cvs_fieldnames)
            next(reader, None)  # skip the headers
            for row in reader:
                data_epoch.append(int(row['epoch']))
                data_train_loss.append(float(row['train loss']))
                data_test_loss.append(float(row['test loss']))

        matplotlib.use('Agg')

        filepath = os.path.join(self.session_dir, "report.png")
        plt.clf()
        plt.plot(data_epoch, data_train_loss, color='red', label='Train Loss')
        plt.plot(data_epoch, data_test_loss, color='blue', label='Test Loss')
        plt.xticks(data_epoch, ['{}'.format(i+1) for i in data_epoch])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper left')
        plt.grid(alpha=0.4)
        plt.savefig(filepath)


    def plot_dataset_tagging(self):
        counter = 0
        for split in ["train", "val", "test"]:
            dataset = get_dataset(split, True)
            if dataset is None:
                continue

            plot_folder = os.path.join(self.session_dir, "plot_dataset_tagging", split)
            if not os.path.exists(plot_folder):
                os.makedirs(plot_folder)

            for index, (image, target) in enumerate(dataset):
                filename = os.path.join(plot_folder, f"{counter:05}.jpg")

                if image.shape[0] == 1:
                    image = torch.cat((image, image, image), 0)
                    
                image = v2.functional.to_dtype(image, torch.uint8, scale=False)

                keypoints = torch.reshape(target, (1, NumberOfKeypoints, 2))
                keypoints = v2.functional.to_dtype(keypoints, torch.int, scale=False)

                image = torchvision.utils.draw_keypoints(image, keypoints, colors="red", radius=2)

                image = image / 255.0
                torchvision.utils.save_image(image, filename)
                
                counter = counter + 1        

    def test(self, checkpoint):
        batch_size = 8

        model = load_model_checkpoint(checkpoint)
        for split in ["train", "val", "test"]:
            dataset = get_dataset(split, False)
            if dataset is None:
                continue

            loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=2)
            model = model.to(self.device)
            
            loss_func = nn.MSELoss()
            
            # eval
            model.eval() 
            with torch.no_grad():
                epoch_loss, num_samples = 0, 0
                for samples, targets in loader:
                    samples = samples.to(self.device).float()
                    targets = targets.to(self.device).float()

                    preds = model(samples)

                    idx = targets == -1
                    preds[idx] = -1
                    loss = loss_func(preds, targets)

                    epoch_loss += loss.item() * targets.shape[0] 
                    num_samples += targets.shape[0]

                epoch_loss = np.sqrt(epoch_loss / num_samples)
                logging.info(f"{split} Loss: {epoch_loss:7.4f}")


    def plot_inference_results(self, checkpoint):
        # load checkpoint
        model = load_model_checkpoint(checkpoint)
        model = model.to(self.device)

        # switch to eval model
        model.eval()
        with torch.no_grad():
            for split in ["train", "val", "test"]:
                dataset = get_dataset(split, False)
                if dataset is None:
                    continue

                plot_folder = os.path.join(self.session_dir, "plot_inference_results", split)
                if not os.path.exists(plot_folder):
                    os.makedirs(plot_folder)

                for index, (image, Y) in enumerate(dataset):
                    filename = os.path.join(plot_folder, f"{index:05}.jpg")

                    image = image.to(self.device).float()
                    X = image.view(1, image.shape[0], image.shape[1], image.shape[2])
                    pred = model(X)


                    image = torch.cat((image, image, image), 0)
                    image = v2.functional.to_dtype(image, torch.uint8, scale=False)

                    keypoints = torch.reshape(pred, (1, NumberOfKeypoints, 2))
                    keypoints = v2.functional.to_dtype(keypoints, torch.int, scale=False)

                    image = torchvision.utils.draw_keypoints(image, keypoints, colors="red", radius=2)

                    image = image / 255.0
                    torchvision.utils.save_image(image, filename)

    def export_weights(self, filename):
        dataset = get_dataset("val", False)
        input_size = next(iter(dataset))[0].shape
        input_size = (1, input_size[0], input_size[1], input_size[2])

        # filename without extension
        target = os.path.splitext(filename)[0] + ".onnx"
        logging.info(f"Exporting model to ONNX format. {filename} -> {target}")
        
        model = load_model_checkpoint(filename)

        x = torch.randn(input_size, requires_grad=True)
        logging.info(f"{x.shape}")

        # Export the model
        torch.onnx.export(model,             # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  target,                    # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=17,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
