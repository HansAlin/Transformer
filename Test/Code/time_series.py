import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, MultiplicativeLR, StepLR, MultiStepLR

device = (
    "mps"
    if getattr(torch, "has_mps", False)
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")

names = ['year', 'month', 'day', 'dec_year', 'sn_value', 'sn_error', 'obs_num', 'unuesed1']

df = pd.read_csv(
    "https://data.heatonresearch.com/data/t81-558/SN_d_tot_V2.0.csv",
    sep=';', header=None, names=names,
    na_values=['-1'], index_col=False)
# starts with first none zero value
l = df[df['obs_num'] == 0].index.tolist()
start_id = max(l) + 40000
sorted_data = df[start_id:].copy()

sorted_data['sn_value'] = sorted_data['sn_value'].astype(float)
df_train = sorted_data[sorted_data['year'] < 2000]
df_test = sorted_data[sorted_data['year'] >= 2000]

spots_train = df_train['sn_value'].to_numpy().reshape(-1, 1).flatten().tolist()
spots_test = df_test['sn_value'].to_numpy().reshape(-1, 1).flatten().tolist()

def to_sequence(seq_size, obs):
  x = []
  y = []
  for i in range(len(obs) - seq_size):
    window = obs[i:(i + seq_size)]
    after_window = obs[i + seq_size]
    x.append(window)
    y.append(after_window)

  return torch.tensor(x, dtype=torch.float32).view(-1, seq_size, 1), torch.tensor(y, dtype=torch.float32).view(-1,1,1) 

# Scaler
def scaler(x):
  m = x.mean()
  std = x.std()
  x = (x - m) / std
  return x

SEQUENCE_SIZE = 10
x_train, y_train = to_sequence(seq_size=SEQUENCE_SIZE, obs=spots_train)
x_test, y_test = to_sequence(seq_size=SEQUENCE_SIZE, obs=spots_test)
x_train = scaler(x_train)
x_test = scaler(x_test)
y_train = scaler(y_train)
y_test = scaler(y_test)
print(x_train.shape)

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)

# Positional encoding
class PositionalEncoding(nn.Module):
  def __init__(self, d_model, dropout=0.1, max_len=5000):
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout)

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
  def __init__(self, input_dim=1, seq_len=10, d_model=64, nhead=4, num_layers=2, dropout=0.2):
    super(TransformerModel, self).__init__()

    self.encoder = nn.Linear(input_dim, d_model)
    self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=5000)
    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
    self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
    self.decoder = nn.Linear(d_model, input_dim)
   

  def forward(self, x):
    # (batch_size (32), seq_len (10) , 1)
    x = self.encoder(x)# --> (batch_size, seq_len, d_model)
    x = self.pos_encoder(x) # --> (batch_size, seq_len, d_model)
    x = self.transformer(x) # --> (batch_size, seq_len, d_model)
    x = x.mean(axis=-2) # --> (batch_size, d_model)
    
    x = self.decoder(x).unsqueeze(-1) # --> (batch_size, 1)
    return x  
    

model = TransformerModel().to(device)

# Train model
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# TODO try to understand ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=3, verbose=True)
# lambda1 = lambda epoch: epoch / 10
# scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lambda1)
# lambda1 = lambda epoch: 0.95
# scheduler = MultiplicativeLR(optimizer=optimizer, lr_lambda=lambda1)


epochs = 50
early_stop_count = 0
min_val_loss = float('inf')  
val_losses = []
train_losses = []

for epoch in range(epochs):

  # set the model in training mode
  model.train()
  train_loss = []
  for batch in train_loader:
    x_batch, y_batch = batch
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    
    outputs = model(x_batch)
    loss = criterion(outputs, y_batch)
    train_loss.append(loss.item()) 
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

  train_loss = np.mean(train_loss)

  # validation
  # set model in evaluation mode
  val_loss = []
  # This part should not be part of the model thats the reason for .no_grad()
  model.eval()
  with torch.no_grad():
    for batch in test_loader:
      x_batch, y_batch = batch
      x_batch, y_batch = x_batch.to(device), y_batch.to(device)
      outputs = model(x_batch)
      loss = criterion(outputs, y_batch)
      val_loss.append(loss.item())   
          
    val_loss = np.mean(val_loss)


  scheduler.step(epoch=epoch, metrics=val_loss)
  print(f"Learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}")
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

# Evaluation
model.eval()
predictions = []
with torch.no_grad():
  for batch in test_loader:
    x_batch, y_batch = batch
    x_batch = x_batch.to(device)
    outputs = model(x_batch)
    predictions.extend(outputs.squeeze().tolist())

print() 
pred = np.array(predictions).reshape(-1,1)
print(pred.shape)
y = np.array(y_test.reshape(-1,1))
print(y.shape)

print(np.array(predictions).reshape(-1, 1).shape)
print(scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).shape)

print(y_test.numpy().reshape(-1, 1).shape)
print(scaler.inverse_transform(y_test.numpy().reshape(-1, 1)).shape)


rmse = np.sqrt(np.mean((scaler.inverse_transform(np.array(predictions).reshape(-1, 1)) - scaler.inverse_transform(y_test.numpy().reshape(-1, 1)))**2))
print(f"Score (RMSE): {rmse:.4f}")

dif = (pred-y)
print(dif.shape)
square = dif**2
print(square.shape)
mse = np.sqrt(np.mean(square))
print(rmse)