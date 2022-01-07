import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from TrainingHelpers import init_process, train_model 

if __name__ == "__main__":
    world_size = 4
    processes = []
    config = {
        "data_dir": "/data/",
        "epochs": 200,
        "batch_size": 16,
        "learning_rate": 0.00025,
        "warmup_epochs": 8
    }

    try: 
        mp.set_start_method("spawn")
    except:
        pass
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, train_model, config, "nccl"))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()