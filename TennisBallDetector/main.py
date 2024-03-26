import argparse
import logging
import os

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from dataset import BallDataset

from session import Session
from model import available_models


session = Session()

def train():
    loss_func = nn.MSELoss()
    batch_size = 32
    learning_rate = 0.0001 # 0.001
    num_epochs = 200
    weight_decay = 0.1
    decay_every = 30
    decay = 0.25
    warmup_epochs = 3
    workers = 6

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    models = available_models()

    for model in models:
        session.setup()

        cache_folder = os.path.join(session.session_dir, "cache")

        train_dataset = BallDataset(mode='train', cache_folder=cache_folder)
        val_dataset = BallDataset(mode='val', cache_folder=cache_folder)

        train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
        val_loader = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    
        model = model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, decay_every, gamma=decay)

        logging.info(f"Starting model {model.kind}")
        logging.info(f"batch_size={batch_size} Optim[lr={learning_rate} weight_decay={weight_decay}] Step[decay_every={decay_every} decay={decay}]")

        for epoch in range(1, num_epochs + 1):
            # train
            train_loss, train_samples = 0, 0
            model.train()
            with tqdm(train_loader, unit="batch") as tepoch:
                for batch, (samples, targets) in enumerate(tepoch):
                    tepoch.set_description(f"epoch {epoch:04} trn")

                    samples = samples.to(device).float()
                    targets = targets.to(device).float()
                    preds = model(samples)

                    loss = loss_func(preds, targets)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step() 

                    loss_item = loss.item()
                    train_loss += loss_item * targets.shape[0] 
                    train_samples += targets.shape[0]

                    tepoch.set_postfix(loss=loss_item)

            scheduler.step()

            # eval
            if epoch < warmup_epochs:
                logging.info(f"epoch {epoch:04} warm up")
            else:
                model.eval() 
                test_loss, test_samples = 0, 0
                with torch.no_grad():
                    with tqdm(val_loader, unit="batch") as tepoch:
                        for batch, (samples, targets) in enumerate(tepoch):
                            tepoch.set_description(f"epoch {epoch:04} val")

                            samples = samples.to(device).float()
                            targets = targets.to(device).float()
                            preds = model(samples)

                            loss = loss_func(preds, targets)

                            loss_item = loss.item()
                            test_loss += loss_item * targets.shape[0] 
                            test_samples += targets.shape[0]

                            session.eval_result(samples, targets, preds)

                            tepoch.set_postfix(loss=loss_item)

                        lr = 0.0
                        for param_group in optimizer.param_groups:
                            lr = param_group['lr']

                        train_loss = np.sqrt(train_loss / train_samples)
                        test_loss = np.sqrt(test_loss / test_samples)
                        session.end_epoch(epoch, model, train_loss, test_loss, lr, epoch == num_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='detector', description='detector')
    parser.add_argument('--plot_inference_results', help="plot the results of a trained model")
    parser.add_argument('--test_checkpoint', help="test a specific checkpoint")
    parser.add_argument('--export', help="export a trained model")

    args = parser.parse_args()

    session.setup()

    if args.plot_inference_results:
        session.plot_inference_results(args.plot_inference_results)
    elif args.test_checkpoint:
        session.test(args.test_checkpoint)
    elif args.export:
        session.export_weights(args.export)
    else:
        train()