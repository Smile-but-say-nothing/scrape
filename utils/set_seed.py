import torch
import numpy as np
import random

def set_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        # torch.set_deterministic(True)
        np.random.seed(seed)
        random.seed(seed)
