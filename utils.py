import torch
import os
import random
import numpy as np

from torch.optim.lr_scheduler import StepLR


def normalize(x, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1).expand(x.shape[0], 3, x.shape[2], x.shape[2]).to(x.device)
    std = torch.tensor(std).view(3, 1, 1).expand(x.shape[0], 3, x.shape[2], x.shape[2]).to(x.device)
    return (x - mean)/std


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class my_step_lr_Scheduler(StepLR):

    def __init__(self, initial_lr, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = last_epoch
        for group in self.optimizer.param_groups:
            group["initial_lr"] = initial_lr
        if self.last_epoch != -1:
            self.resume_the_last_epoch()
        super(my_step_lr_Scheduler, self).__init__(self.optimizer, step_size, gamma, last_epoch)

    def resume_the_last_epoch(self):
        for group in self.optimizer.param_groups:
            group['lr'] = group['initial_lr'] * self.gamma ** (self.last_epoch // self.step_size)