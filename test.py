# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from data_process import *
from model import *
import optuna


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda')
torch.manual_seed(1)
torch.cuda.manual_seed(2)

li = nn.Linear(18, 1)
x = torch.randn(40, 256, 18)
out = li(x)
print(out.shape)