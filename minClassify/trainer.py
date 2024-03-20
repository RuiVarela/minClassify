
import copy
import logging
import os
import torch
import csv
import torchinfo
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt

from torch import nn
from torch.optim import lr_scheduler

class Trainer:
    def __init__(self):
        self.dataloader_creator = None
        self.model_creator = None
        self.seed = None

        self.batch_size = 8
        self.epochs = 30

        self.learning_rate = 1e-3
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.step_size = 7
        self.gamma = 0.1

        self.use_adam = False

        self.can_upload_dataset_to_gpu = True

        self.device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = None
        self.test_dl = None
        self.train_dl = None
        self.loss_fn = None
        self.optimizer = None
        self.scheduler = None

        self.session_dir = os.path.join(os.getcwd(), 'session', 'latest')
        self.best_accuracy = 0.0
        self.best_epoch = 0
        self.train_loss = 0.0

        self.cvs_fieldnames = ['epoch', 'name', 'train loss', 'test accuracy', 'test loss', 'learning rate']
        self.cvs_filename = "report.cvs"

    def ensure_session_folder(self):
        if not os.path.exists(self.session_dir):
            os.makedirs(self.session_dir)

    def configure_log(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

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

    def save_step(self, epoch, test_accuracy, test_loss):
        logging.info(f"Epoch={epoch:03d} Accuracy={test_accuracy:>0.3f} Avgloss={test_loss:>8f} lr={self.get_lr()}")

        #
        # Save checkpoints
        #
        state = {
            'epoch': epoch,
            'best_accuracy': self.best_accuracy, 
            'best_epoch': self.best_epoch,

            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }

        name = f"{epoch}_{test_accuracy:>0.3f}_{test_loss:>8f}.pt"
        filepath = os.path.join(self.session_dir, name)
        self.ensure_session_folder()
        torch.save(state, filepath)
    
        if test_accuracy > self.best_accuracy:
            self.best_accuracy = test_accuracy
            self.best_epoch = epoch
            filepath = os.path.join(self.session_dir, "best.pt")
            torch.save(state, filepath)

        #
        # Save csv
        #
        filepath = os.path.join(self.session_dir, self.cvs_filename)
        add_header = not os.path.exists(filepath)
        with open(filepath,'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.cvs_fieldnames)
            if add_header:
                writer.writeheader()
            writer.writerow({'epoch': epoch, 'name': name,
                            'train loss': self.train_loss,
                            'test accuracy': test_accuracy, 'test loss': test_loss,
                            'learning rate': self.get_lr() })
        #
        # Generate plot
        #
        data_epoch = []
        data_train_loss = []
        data_test_accuracy = []
        data_test_loss = []
        with open(filepath, newline='') as f:
            reader = csv.DictReader(f, fieldnames=self.cvs_fieldnames)
            next(reader, None)  # skip the headers
            for row in reader:
                data_epoch.append(int(row['epoch']))
                data_train_loss.append(float(row['train loss']))
                data_test_accuracy.append(float(row['test accuracy']))
                data_test_loss.append(float(row['test loss']))

        matplotlib.use('Agg')

        filepath = os.path.join(self.session_dir, "report.png")
        plt.clf()
        plt.plot(data_epoch, data_train_loss, color='red', label='Train Loss')
        plt.plot(data_epoch, data_test_loss, color='blue', label='Test Loss')
        plt.plot(data_epoch, data_test_accuracy, color='green', label='Test Accuracy')
        plt.xticks(data_epoch, ['{}'.format(i+1) for i in data_epoch])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper left')
        plt.grid(alpha=0.4)
        plt.savefig(filepath)

    def resume(self):
        filepath = os.path.join(self.session_dir, self.cvs_filename)
        if not os.path.exists(filepath):
            logging.error(f"Unable to resume session. csv not found")
            return -1

        epoch = -1
        name = ""
        with open(filepath, newline='') as f:
            reader = csv.DictReader(f, fieldnames=self.cvs_fieldnames)
            next(reader, None)  # skip the headers
            for row in reader:
                epoch = int(row['epoch'])
                name = row['name']

        if epoch < 0:
            logging.error(f"Unable to resume session. invalid epoch {epoch}")
            return -1

        filepath = os.path.join(self.session_dir, name)
        if not os.path.exists(filepath):
            logging.error(f"Unable to resume session. checkpoint not found")
            return -1

        logging.info(f"Resuming {name} from session {self.session_dir}")

        checkpoint = torch.load(filepath)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        self.best_accuracy = checkpoint['best_accuracy']
        self.best_epoch = checkpoint['best_epoch']
    
        logging.info(f"Restarting epoch={epoch} best_epoch={self.best_epoch} best_accuracy={self.best_accuracy}")
        return epoch + 1
    
    def load_weights(self, filename):
        logging.info(f"Loading weights from {filename}")
        
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def export_weights(self, filename, input_size):
        # filename without extension
        target = os.path.splitext(filename)[0] + ".onnx"
        logging.info(f"Exporting model to ONNX format. {filename} -> {target}")
        
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        x = torch.randn(input_size, requires_grad=True)
        logging.info(f"{x.shape}")

        # Export the model
        torch.onnx.export(self.model,        # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  target,                    # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})


    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def train_step(self, epoch):
        dataloader = self.train_dl

        num_batches = len(dataloader)
        
        # Set the model to training mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        self.model.train()
        self.train_loss = 0.0

        with tqdm(dataloader, unit="batch") as tepoch:
            for batch, (X, y) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")

                X = X.to(self.device)
                y = y.to(self.device)

                # Zero your gradients for every batch!
                self.optimizer.zero_grad()
                
                # Make predictions for this batch
                pred = self.model(X)

                # Compute the loss and its gradients
                loss = self.loss_fn(pred, y)
                loss.backward()

                # Adjust learning weights
                self.optimizer.step()

                # Gather data and report
                self.train_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

            self.scheduler.step()
            self.train_loss /= num_batches

    def test_step(self, epoch):
        dataloader = self.test_dl

        # Set the model to evaluation mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        self.model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
        with torch.no_grad():
            for X, y in dataloader:

                X = X.to(self.device)
                y = y.to(self.device)
                
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

        test_loss /= num_batches
        test_accuracy = correct / size

        self.save_step(epoch, test_accuracy, test_loss)
            

    def train(self, resume, load, export):
        session_root = os.path.join(os.getcwd(), 'session')
        self.session_dir = os.path.join(session_root, f'{0:04d}')
        if os.path.exists(session_root):
            folders = [f for f in os.listdir(session_root) if os.path.isdir(os.path.join(session_root, f))]
            folders.sort()
            if len(folders) > 0:
                last = int(folders[-1])
                if not resume:
                    last = last + 1
                self.session_dir = os.path.join(session_root, f'{last:04d}')

        self.configure_log()

        if self.seed:
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
            torch.manual_seed(self.seed)
            logging.info(f"Using seed {self.seed}")

        upload_to_gpu_ram = self.device == "cuda" and self.can_upload_dataset_to_gpu
        self.train_dl, self.test_dl, classes = self.dataloader_creator(self.batch_size, upload_to_gpu_ram)
        logging.info(f"Batch Size: {self.batch_size} Train samples: {len(self.train_dl.dataset)} / test samples: {len(self.test_dl.dataset)}")

        x, y = next(iter(self.test_dl))
        input_size = x.shape
        self.model = self.model_creator(input_size, classes, self.device)
        logging.info(f"Using {self.device} device")
        logging.info(f"Using epochs={self.epochs} learning_rate={self.learning_rate} momentum={self.momentum} weight_decay={self.weight_decay} step_size={self.step_size} gamma={self.gamma} use_adam={self.use_adam}")

        logging.info("\n" + str(torchinfo.summary(self.model, input_size=input_size, verbose=0)))

        self.loss_fn = nn.CrossEntropyLoss()
        if self.use_adam:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        # Decay LR by a factor of gamma every step_size epochs
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)

        epoch = 0
        if resume:
            epoch = self.resume()
            if epoch < 0:
                return
                        
        if load and os.path.exists(load):
            self.load_weights(load)

        if export:
            if os.path.exists(export):
                self.export_weights(export, input_size)
            return

        for t in range(epoch, self.epochs):
            self.train_step(t)
            self.test_step(t)

        logging.info("Completed!")
        logging.info(f"Best Epoch={self.best_epoch:03d} Accuracy={self.best_accuracy:>0.3f} ")
