import numpy as np
from sklearn.preprocessing import StandardScaler
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
myhost = os.uname()[1]

print("Host name: ", myhost)
if myhost == 'LAPTOP-9FBI5S57':
    PATH = '/home/hansalin/Code/Transformer/'
else:
  PATH = '/home/halin/Master/Transformer/Test/test_data/test_data.npy'

device = None
# device = (
#     "mps"
#     if getattr(torch, "has_mps", False)
#     else "cuda"
#     if torch.cuda.is_available()
#     else "cpu"
# )
# print(f"Using device: {device}")

def prepare_data(**kwarg):
  batch_size = 32
  for arg in kwarg:
    if arg == 'batch_size':
      batch_size = kwarg.get('batch_size')
    if arg == 'PATH':
      PATH = kwarg.get('PATH') + 'Test/test_data/test_data.npy'  

    
  with open(PATH, 'rb') as f:

    x_train = np.load(f)
    x_test = np.load(f)
    y_train = np.load(f)
    y_test = np.load(f)

  scaler = StandardScaler()
  x_train = scaler.fit_transform(x_train)
  x_test = scaler.fit_transform(x_test)
  # to torch
  x_train = torch.tensor(x_train, dtype=torch.float32).view(-1, len(x_train[0]), 1)
  x_test = torch.tensor(x_test, dtype=torch.float32).view(-1, len(x_test[0]), 1)
  y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, len(y_train[0]), 1)
  y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, len(y_test[0]), 1)

  train_dataset = TensorDataset(x_train, y_train)
  train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

  test_dataset = TensorDataset(x_test, y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

  return train_loader, test_loader

train_loader, test_loader = prepare_data(batch_size=64, PATH=PATH)
print()

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, drop_out=0.2, max_len=500):
    super(PositionalEncoding, self).__init__()

    self.dropout = nn.Dropout(p=drop_out)
    
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position*div_term)
    pe[:, 1::2] = torch.cos(position*div_term)
    # add dimensions and transpose TODO understand the dimensions
    pe = pe.unsqueeze(0).transpose(0,1)
    # This I don't understand
    self.register_buffer('pe', pe)

  def forward(self, x):
    x = x + self.pe[:x.size(0), :]
    return self.dropout(x)   

# Define model
class TransformerModel(nn.Module):
  def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=2, dropout=0.2):
    super(TransformerModel, self).__init__()

    self.encoder = nn.Linear(input_dim, d_model)
    self.pos_encoder = PositionalEncoding(d_model=d_model, drop_out=dropout, max_len=5000)
    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
    self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
    self.decoder = nn.Linear(d_model, 1)

  def forward(self, x):
    x = self.encoder(x)
    x = self.pos_encoder(x)
    x = self.transformer(x)
    x = self.decoder(x[:, -1, :])
    y_pred = torch.sigmoid(x)
    return y_pred

model = TransformerModel()  

# Train model
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# TODO try to understand ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

epochs = 2
early_stop_count = 0
min_val_loss = float('inf')  # TODO ????
val_losses = []
train_losses = []
for epoch in range(epochs):
  # set the model in training mode
  model.train()
  train_loss = []
  for batch in train_loader:
    x_batch, y_batch = batch
    if device != None:
      x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    optimizer.zero_grad()
    outputs = model(x_batch)
    loss = criterion(outputs, y_batch)
    train_loss.append(loss.item()) 
    loss.backward()
    optimizer.step()

  train_loss = np.mean(train_loss)
  # validation
  # set model in evaluation mode
  val_loss = []
  
  # This part should not be part of the model thats the reason for .no_grad()
  with torch.no_grad():
    for batch in test_loader:
      x_batch, y_batch = batch
      if device != None:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
      outputs = model(x_batch)
      loss = criterion(outputs, y_batch)
      val_loss.append(loss.item())   
          
    val_loss = np.mean(val_loss)
    scheduler.step(val_loss)

  if val_loss < min_val_loss:
      min_val_loss = val_loss
      early_stop_count = 0
  else:
      early_stop_count += 1

  if early_stop_count >= 5:
      print("Early stopping!")
      break
  
  print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss}, Validation Loss: {val_loss:.4f}")
  train_losses.append(train_loss)
  val_losses.append(val_loss)

train_length = range(len(train_losses))
plt.plot(train_length, train_losses)  
plt.savefig(PATH + 'Test/Models/test_loss_plot.png')
plt.show() 

with open(PATH + 'Test/Models/test_train_data.npy', 'wb') as f:
  np.save(f, np.array(train_length))
  np.save(f, np.array(train_losses))
  np.save(f, np.array(val_losses))
  
