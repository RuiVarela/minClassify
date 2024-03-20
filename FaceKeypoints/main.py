import argparse
import logging

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from dataset import FaceDataset

from session import Session
from model import available_models


session = Session()

def train():
    loss_func = nn.MSELoss()
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 210
    decay_every = 30
    decay = 0.25
    warmup_epochs = 6

    train_dataset = FaceDataset('train')
    val_dataset = FaceDataset('val')

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    models = available_models()

    for model in models:
        session.setup()
    
        model = model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
        scheduler = StepLR(optimizer, decay_every, gamma=decay)

        logging.info(f"Starting model {model.__class__.__name__}")

        for epoch in range(1, num_epochs + 1):

            # train
            model.train()
            for samples, targets in train_loader:
                samples = samples.to(device).float()
                targets = targets.to(device).float()
                preds = model(samples)

                idx = targets == -1
                preds[idx] = -1
                loss = loss_func(preds, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step() 

            scheduler.step()

            # eval
            if epoch < warmup_epochs:
                logging.info(f"epoch {epoch:04} warm up")
            else:
                model.eval() 
                with torch.no_grad():
                    losses = []
                    for mode, loader in zip(['train', 'val'], [train_loader, val_loader]):
                        epoch_loss, num_samples = 0, 0
                        for samples, targets in loader:
                            samples = samples.to(device).float()
                            targets = targets.to(device).float()
                            preds = model(samples)

                            idx = targets == -1
                            preds[idx] = -1
                            loss = loss_func(preds, targets)

                            epoch_loss += loss.item() * targets.shape[0] 
                            num_samples += targets.shape[0]

                        epoch_loss = np.sqrt(epoch_loss / num_samples)
                        losses.append(epoch_loss)


                    lr = 0.0
                    for param_group in optimizer.param_groups:
                        lr = param_group['lr']
                    session.end_epoch(epoch, model, losses, lr, epoch == num_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='detector', description='detector')
    parser.add_argument('--plot_dataset_tagging', action='store_true', help="plot the original dataset tagging")
    parser.add_argument('--plot_inference_results', help="plot the results of a trained model")
    parser.add_argument('--test_checkpoint', help="test a specific checkpoint")
    parser.add_argument('--export', help="export a trained model")

    args = parser.parse_args()

    session.setup()

    if args.plot_dataset_tagging:
        session.plot_dataset_tagging()
    elif args.plot_inference_results:
        session.plot_inference_results(args.plot_inference_results)
    elif args.test_checkpoint:
        session.test(args.test_checkpoint)
    elif args.export:
        session.export_weights(args.export)
    else:
        train()