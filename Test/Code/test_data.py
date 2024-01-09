import argparse
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
import os
import time
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys
CODE_DIR_1  ='/home/halin/Master/Transformer/'
sys.path.append(CODE_DIR_1)

from dataHandler.datahandler import get_data, prepare_data, find_hyperparameters, get_data_binary_class

train_data, val_data, test_data = get_data_binary_class(test=True)
