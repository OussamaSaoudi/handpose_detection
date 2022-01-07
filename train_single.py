#!/usr/bin/env python3
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
#preputils
import torch.distributed as dist
from FreiHAND import FreiHAND
from StackedHourGlass import StackedHourGlass
from Trainer import Trainer
def train():
    torch.cuda.empty_cache()
    device = torch.device("cuda")

    model = StackedHourGlass(num_stacks=3,num_residual=3).to(device)
    config = {
        "data_dir": "/data/",
        "epochs": 100,
        "batch_size": 1,
        "learning_rate": 0.01,
        "device": device
    }

    #Initialize training dataset
    train_dataset = FreiHAND(config=config, set_type="train")
    train_dataloader = DataLoader(
        train_dataset,
        config["batch_size"],
        num_workers=4,
        shuffle=True
    )

    #Initialize validation dataset
    val_dataset = FreiHAND(config=config, set_type="val")
    val_dataloader = DataLoader(
        val_dataset,
        config["batch_size"],
        num_workers=4,
        shuffle=True
    )

    #Initialize loss, optimizer, and scheduler
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, factor = 0.2, patience=10, verbose=True
    ) 
    torch.cuda.empty_cache()
    trainer = Trainer(model, optimizer, loss_fn, device, lambda x: (x["image"], x["heatmaps"]), scheduler=scheduler, epochs=config["epochs"],epoch_load_bar=False, early_stop=True,early_stop_patience=10)
    trainer.train(train_dataloader, val_dataloader)
train()
