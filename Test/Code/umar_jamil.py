import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
path = os.getcwd()
print(path)


import os
myhost = os.uname()[1]

print("Host name: ", myhost)
if myhost == 'LAPTOP-9FBI5S57':
    PATH = '/home/hansalin/Code/Transformer'
    import matplotlib.pyplot as plt
    sys.path.append('/home/hansalin/Code/Transformer/Test/')
else:
  PATH = '/home/halin/Master/Transformer'

import dataHandler.handler as dh
import models.models_1 as md
import training.train as tr

# device = None
device = (
    "mps"
    if getattr(torch, "has_mps", False)
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")


x_train, x_test, y_train, y_test = dh.get_test_data()
train_loader, test_loader = dh.prepare_data(x_train, x_test, y_train, y_test, 32)

model = md.build_encoder_transformer(embed_size=64,
                                      seq_len=100,
                                      d_model=512,
                                      N=6,
                                      h=8,
                                      dropout=0.1)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

trained_model = tr.training(model, 
                            train_loader, 
                            test_loader, 
                            device, 
                            optimizer, 
                            criterion,
                            epochs=2)
tr.save_data()
