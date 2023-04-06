import torch
import random
import numpy as np

def set_random_seed(seed=13):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)