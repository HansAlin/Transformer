import numpy as np
from sklearn.preprocessing import StandardScaler
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau


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
  
class TransformerModel(nn.Module):
  def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=2, dropout=0.2):
    super(TransformerModel, self).__init__()

    self.encoder = nn.Linear(input_dim, d_model)
    self.pos_encoder = PositionalEncoding(d_model=d_model, drop_out=dropout, max_len=5000)
    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
    # TODO should it be a activation function here?
    self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
    self.fcc = nn.Linear(d_model, 1)
    print()


  def forward(self, x):
    print(f"Input shape: {x.shape}")
    x = self.encoder(x)
    x = self.pos_encoder(x)
    x = self.transformer(x)
    print(f"Shape: {x.shape}")
    print(f"Shape: {x[:, -1, :].shape}")
    x = self.fcc(x[:, -1, :])
    y_pred = torch.sigmoid(x)
    return y_pred
