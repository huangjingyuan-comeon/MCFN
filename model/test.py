import torch, copy
import torch.nn as nn

lin = nn.Linear(100, 100)

lin_co = copy.deepcopy(lin)

