import numpy as np
from sklearn.preprocessing import StandardScaler
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Costum libries
import Test.dataHandler.datahandler as dh
import models.models as md
import training.train as tr

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

model = md.EncoderTransformerModel(d_model=64,nhead=8)  
model = model.to(device)

# Print model's state_dict
# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())


# Train model
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# TODO try to understand ReduceLROnPlateau
# scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)

trained_model = tr.training(model, 
                            train_loader, 
                            test_loader, 
                            device, 
                            optimizer, 
                            criterion,
                            epochs=2)
tr.save_data()

