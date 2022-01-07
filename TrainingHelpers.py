import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
#preputils
import torch.distributed as dist
import os
from FreiHAND import FreiHAND
from Trainer import Trainer
from torch.utils.data import DistributedSampler
from StackedHourGlass import StackedHourGlass
from collections import OrderedDict

def train_model(rank, world_size, config):
    #Configure process, model, and hyperparameters
    torch.cuda.empty_cache()
    device = torch.device("cuda:{}".format(rank))
    model = StackedHourGlass(num_stacks=8,num_residual=1)

    #If there is a checpoint to start from:
    #state_dict = torch.load("/data/model_03012022094415_070.pt")
    #model.load_state_dict(state_dict)

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    print("training ", sum(p.numel() for p in model.parameters()), " parameters")
    #Initialize training dataset
    train_dataset = FreiHAND(config=config, set_type="train")
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    train_dataloader = DataLoader(
        train_dataset,
        config["batch_size"],
        num_workers=8,
        sampler=sampler,
        pin_memory=True
    )

    #Initialize validation dataset
    val_dataset = FreiHAND(config=config, set_type="val")
    sampler2 = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        config["batch_size"],
        num_workers=8,
        pin_memory=True,
        sampler=sampler2
    )

    #Initialize optimizer and scheduler
    loss_fn = nn.MSELoss()
    optimizer = optim.RMSprop(ddp_model.parameters(), lr=config["learning_rate"], alpha=0.99,eps=1e-8)

    warm_up = lambda epoch: epoch / config["warmup_epochs"] if epoch <= config["warmup_epochs"] else 1   # (L6)
    scheduler_wu = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_up)   # (L7)
    scheduler_re = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.2, patience=10, verbose=True)   # (L8)

    #Aoex
    trainer = Trainer(ddp_model,
                    optimizer,
                    loss_fn,
                    device,
                    lambda x: (x["image"], x["heatmaps"]),
                    warmup_epochs=config["warmup_epochs"],
                    epochs=config["epochs"],
                    scheduler=[scheduler_wu,scheduler_re],
                    rank=rank,
                    early_stop=False,
                    checkpoint_frequency=20,
                    train_sampler=[sampler,sampler2]
                    )
    trainer.train(train_dataloader, val_dataloader)
    dist.destroy_process_group()

def init_process(rank, size, fn, config, backend="nccl"):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank,size,config)


def train_hogwild(model, rank):
    #Configure process, model, and hyperparameters
    device = torch.device("cuda:{}".format(rank))
    config = {
        "data_dir": "./",
        "epochs": 100,
        "batch_size": 1,
        "learning_rate": 0.1,
        "batches_per_epoch": None,
        "batches_per_epoch_val": None,
        "device": device
    }

    #Initialize training dataset
    train_dataset = FreiHAND(config=config, set_type="train")
    train_dataloader = DataLoader(
        train_dataset,
        config["batch_size"],
        num_workers=8,
        shuffle=True
    )
    #Initialize validation dataset
    val_dataset = FreiHAND(config=config, set_type="val")

    val_dataloader = DataLoader(
        val_dataset,
        config["batch_size"],
        num_workers=8,
        shuffle=True
    )

    #Initialize optimizer and scheduler
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0025 )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, factor = 0.2, patience=10, verbose=True
    )
    trainer = Trainer(model, optimizer, loss_fn, device, lambda x: (x["image"], x["heatmaps"]), epochs = config["epochs"],scheduler=scheduler, rank=rank)
    trainer.train(train_dataloader, val_dataloader)
    dist.destroy_process_group()