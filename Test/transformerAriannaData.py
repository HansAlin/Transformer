import numpy as np
from sklearn.preprocessing import StandardScaler
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Costum libries
import dataHandler.handler as dh
import models
import training as tr

import os
myhost = os.uname()[1]

print("Host name: ", myhost)
if myhost == 'LAPTOP-9FBI5S57':
    PATH = '/home/hansalin/Code/Transformer'
    import matplotlib.pyplot as plt
else:
  PATH = '/home/halin/Master/Transformer'

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

model = models.TransformerModel(d_model=128,nhead=8)  
model = model.to(device)
# Train model
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# TODO try to understand ReduceLROnPlateau
# scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)

trained_model = tr.training()


if myhost == 'LAPTOP-9FBI5S57':
  plt.plot(train_length, train_losses)  
  loss_path = PATH + 'Test/Models/test_loss_plot.png'
  plt.savefig(loss_path)
  plt.plot(train_length, val_accs)
  acc_path = PATH + 'Test/Models/test_acc_plot.png'
  plt.savefig(acc_path)
  



  
