import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from TrainingHelpers import init_process, train_model, train_hogwild
from StackedHourGlass import StackedHourGlass
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    processes = []
    model = StackedHourGlass(num_stacks=1,num_residual=1)
    mp.set_start_method('spawn')
    model.share_memory()
    for rank in range(world_size):
        p = mp.Process(target=train_hogwild, args=(model, rank))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()