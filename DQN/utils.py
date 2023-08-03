import random
import numpy as np
import torch

def set_random_seed(seed: int = 42):
    """
    乱数シードを設定する関数
    Args:
        seed: 乱数シード
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True