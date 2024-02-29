import torch 
import torch.nn as nn
import math
from typing import List
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer
from dataHandler.datahandler import save_data, get_model_path
import sys
import torch.nn.functional as F

# 
class LayerNormalization(nn.Module):
  """From https://github.com/hkproj/pytorch-transformer/blob/main/model.py
      This layer nomalize the features over the d_model dimension according 
      to the paper "Attention is all you need".
  """
  def __init__(self, features: int, eps: float = 10**-6) -> None:
    super().__init__()
    self.eps = eps
    self.alpha = nn.Parameter(torch.ones(features))
    self.bias = nn.Parameter(torch.zeros(features))
  

  def forward(self, x):
    """Normalize over the d_model dimension.
    
    """
    # (batch_size, seq_len, d_model)
    mean = x.mean(dim = -1, keepdim=True) # (b, seq_len, 1)
    std = x.std(dim = -1, keepdim=True) # (b, seq_len, 1)
    x = self.alpha * (x - mean) / (std + self.eps) + self.bias

    # (batch_size, seq_len, d_model)
    return x
  
class BatchNormalization(nn.Module): 
  #TODO Check whether this is correct implemented 
  """Instead of normalizing the features over the layers as in "Attention is all you need"
  this class normalizes the features over the batch dimension.
  
  Args:
      eps (float, optional): [description]. Defaults to 10**-6.
      d_model (int, optional): [description]. Defaults to 512.
  """
  def __init__(self, eps: float = 10**-6, features: int = 512) -> None:
    super().__init__()
    self.eps = eps
    self.features = features
    self.batch_norm = nn.BatchNorm1d(num_features=self.features, eps=eps)

  def forward(self, x):
    # (batch_size, seq_len, d_model)
    # TODO check whether this is correct implemented reshape? 
    x = x.permute(1,2,0) # (batch_size, d_model, seq_len)
    x = self.batch_norm(x)
    x = x.permute(2,0,1)
    return x
  

class FeedForwardBlock(nn.Module):

  def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation='relu'):
    # d_ff is the hidden layer size
    super().__init__()
    self.linear_1 = nn.Linear(d_model, d_ff) # W1 and bias
    self.dropout = nn.Dropout(dropout)
    self.linear_2 = nn.Linear(d_ff, d_model) # W2 and bias
    if activation == None:
      self.activation = nn.Identity()
    elif activation == 'relu':
      self.activation = nn.ReLU() 
    elif activation == 'gelu':
      self.activation = nn.GELU()

  def forward(self, x):
    # (batch_size, seq_len, d_model) 
    x = self.linear_1(x)
    # (batch_size, seq_len, d_ff) 
    x = self.activation(x)
    x = self.dropout(x)
    x = self.linear_2(x)


    # (batch_size, seq_len, d_model)
    return x

class CnnInputEmbeddings(nn.Module):
  """This layer maps the input to the d_model space using a CNN.

  Args:
      channels (int): The number of channels in the input.
      d_model (int): The dimension of the model.
      padding (int, optional): The padding. Defaults to 1.
      stride (int, optional): The stride. Defaults to 1.
      kernel_size (int, optional): The kernel size. Defaults to 3.
  """
  def __init__(self, channels, d_model, padding: int = 1, stride: int = 1, kernel_size: int = 3):
    super().__init__()

    self.kernel_size = kernel_size

    rows = channels // kernel_size + 1
    vertical_padding = (rows * kernel_size - channels) // 2

    out_put_size = d_model//channels
    self.vertical_conv = nn.Conv2d(1, out_put_size, kernel_size=(kernel_size, stride), padding=(vertical_padding, 0), stride=(1, stride))
    self.horizontal_conv = nn.Conv2d(1, out_put_size, kernel_size=(1, kernel_size), padding=(0, 0), stride=(1, stride))

  def forward(self, x):
    # x: (batch_size, seq_len, channels)
    x = x.transpose(1, 2).unsqueeze(1)  # now x: (batch_size, channels, seq_len, 1)
    out1 = self.vertical_conv(x).squeeze(-1)  # out1: (batch_size, d_model, seq_len)
    out1 = out1.transpose(1, 2)  # swap d_model with seq_len, now out1: (batch_size, seq_len, d_model)
     # Calculate padding dynamically
    seq_len = x.size(2)
    padding = 0
    if seq_len % self.kernel_size != 0:
        padding = (self.kernel_size - seq_len % self.kernel_size) // 2

    x = F.pad(x, (0, padding))  # Add padding to the sequence length dimension
    out2 = self.horizontal_conv(x).squeeze(-1)  # out2: (batch_size, d_model, seq_len)
    out2 = out2.transpose(1, 2)  # swap d_model with seq_len, now out2: (batch_size, seq_len, d_model)
    out = out1 + out2  # add the outputs
    out = out.transpose(2,3).transpose(1,2)  # swap d_model with seq_len, now out: (batch_size, seq_len, d_model)
    out = out.flatten(start_dim=2, end_dim=3)  # Remove the height dimension  
    return out


class ViTEmbeddings(nn.Module):

  def __init__(self,
               channels: int,
               out_put_size: int,
               kernel_size: int,
               stride: int,
               padding: int,
               ) -> None:
    super().__init__()
    self.channels = channels
    self.out_put_size = out_put_size
    self.height = kernel_size


    
    rows = channels // kernel_size + 1

    vertical_padding = (rows * kernel_size - channels) // 2

    vertical_stride = kernel_size
    horizontal_stride = kernel_size
    self.conv = nn.Conv2d(1, self.out_put_size, kernel_size=(self.height, self.height), stride=(horizontal_stride, vertical_stride), padding=(0, vertical_padding))

  def forward(self, x):
    x = x.unsqueeze(1)  # Add an extra dimension for channels

    # Calculate padding dynamically
    seq_len = x.size(2)
    padding = 0
    if seq_len % self.height != 0:
        padding = (self.height - seq_len % self.height) // 2

    x = F.pad(x, (padding, padding))  # Add padding to the sequence length dimension
    x = self.conv(x)
    x = x.transpose(1,2)
    x = x.flatten(start_dim=2, end_dim=3)  # Remove the height dimension
    # x = x.transpose(1, 2)  # Swap the "seq_len" and "d_model" dimensions
    return x  



class InputEmbeddings(nn.Module):
  """This layer maps feature space from the input to the d_model space.

  
  Args:
      d_model (int): The dimension of the model.
      dropout (float, optional): The dropout rate. Defaults to 0.1.
      channels (int, optional): The number of channels in the input. Defaults to 1.
      embed_type (str, optional): The type of the embedding. Defaults to 'lin_relu_drop'.
      kwargs: Additional arguments for the CnnInputEmbeddings layer, padding, stride, kernel_size.
  
  """
  def __init__(self, d_model: int, dropout: float = 0.1, channels: int = 4, embed_type='lin_relu_drop', **kwargs) -> None:
    super().__init__()
    self.d_model = d_model
    self.channels = channels

    if embed_type == 'lin_relu_drop':
      self.embedding = nn.Linear(channels, d_model)
      self.activation = nn.ReLU()
      self.dropout = nn.Dropout(dropout)
    elif embed_type == 'lin_gelu_drop':
      self.embedding = nn.Linear(channels, d_model)
      self.activation = nn.GELU()
      self.dropout = nn.Dropout(dropout)
    elif embed_type == 'linear':
      self.embedding = nn.Linear(channels, d_model)
      self.activation = nn.Identity()
      self.dropout = nn.Dropout(0)  
    elif embed_type == 'cnn':
      self.embedding = CnnInputEmbeddings(channels, d_model, **kwargs)
      self.activation = nn.ReLU()
      self.dropout = nn.Dropout(dropout)
    elif embed_type == 'ViT':
      self.embedding = ViTEmbeddings(channels, d_model, **kwargs)
      self.activation = nn.ReLU()
      self.dropout = nn.Dropout(dropout)  

    else:
      raise ValueError(f"Unsupported embed type: {embed_type}")

  def forward(self, x):
    # (batch_size, seq_len, channels)

    x = self.embedding(x)
    x = self.activation(x)
    x = self.dropout(x)
    # --> (batch_size, seq_len, d_model)
    return x 

class PositionalEncoding(nn.Module):
   
  def __init__(self, d_model: int, dropout: float, max_seq_len: int, omega: float) -> None:
    super().__init__()
    self.d_model = d_model
    self.dropout = nn.Dropout(p=dropout)
    self.max_seq_len = max_seq_len
    self.omega = omega
    
    # Create a matrix of shape (seq_len, d_model) 
    # In this case the sequence length is 100 and the model dimension is 512
    # and the seq_len is originaly the max length of the sentence that can
    # be considered in the model. In this case it is eseentially the max length
    # of the data that we have, in this case 100.
    pe = torch.zeros(max_seq_len, d_model)
    # Create a vector of shape (seq_len, 1)
    position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
    # The exponential form is used here for stabilty purposes
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(omega) / d_model))  # 1/10000^(2i/d_model)
    # Apply the sin to even indices in the array; 2i
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    # Add a batch dimension to the positional encoding in 
    # front of the sequence length: (1, seq_len, d_model)
    pe = pe.unsqueeze(0)

    # Register the buffer to the model object
    # The buffer won't be considered a model parameter
    # and won't be trained but saved and moved together with the model
    self.register_buffer('pe', pe)

  def forward(self, x):
    # Add the positional encoding to the input tensor
    # the input is from the previous embedding layer
    # for sentences previous word embedding layer is used.
    # x.shape[1] is the sequence length
    x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
    # (b, seq_len, d_model)
    return self.dropout(x)

class RelativePositionalEncoding(nn.Module):
  """
  This part is taken from:
  https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py
  """
  def __init__(self, d_h, max_relative_position=20):
    super().__init__()
    # d_h is the dimension of head
    self.d_h = d_h
    self.max_relative_position = max_relative_position
    self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, d_h))
 

  def forward(self, length_q, length_k):
    #length_q = length_k = seq_len
    range_vec_q = torch.arange(length_q)
    range_vec_k = torch.arange(length_k)
    distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
    # distance_mat = (seq_len, seq_len)
    # [ 0,  1,  2,  3,  4]
    # [-1,  0,  1,  2,  3]
    # [-2, -1,  0,  1,  2]
    # [-3, -2, -1,  0,  1]
    # [-4, -3, -2, -1,  0]
   
    distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
    # distance_mat_clipped = (seq_len, seq_len)
    # if max_relative_position = 3
    # [ 0,  1,  2,  3,  3]
    # [-1,  0,  1,  2,  3]
    # [-2, -1,  0,  1,  2]
    # [-3, -2, -1,  0,  1]
    # [-3, -3, -2, -1,  0]
    final_mat = distance_mat_clipped + self.max_relative_position
    # final_mat = (seq_len, seq_len)
    # [3, 4, 5, 6, 6]
    # [2, 3, 4, 5, 6]
    # [1, 2, 3, 4, 5]
    # [0, 1, 2, 3, 4]
    # [0, 0, 1, 2, 3]
    final_mat = torch.LongTensor(final_mat)
    embeddings = self.embeddings_table[final_mat] # (seq_len, seq_len, d_h)
    # Each value in the final_mat is used as an index in the embeddings_table
    # E.g. final_mat[0,0] = 3, embeddings_table[3] => third row in the embeddings_table
    # and final_mat[0,1] = 4, embeddings_table[4] => fourth row in the embeddings_table
    # so the first row in the embeddings matrix will be:
    # [[embeddings_table[3]], [embeddings_table[4]], ...]
     

    return embeddings

class LearnablePositionalEncoding(nn.Module):
  """ This is from   https://github.com/gzerveas/mvts_transformer/blob/master/src/models/ts_transformer.py
  """
  def __init__(self, d_model, max_len=2048, dropout=0.1):
    super().__init__()
    self.d_model = d_model
    self.positional_encoding = nn.Parameter(torch.empty(1, max_len, d_model))
    nn.init.uniform_(self.positional_encoding, -0.02, 0.02)
    self.dropout = nn.Dropout(p=dropout)

  def forward(self, x):
    b = self.positional_encoding[:, :x.size(1), :]
    x = x + b
    x = self.dropout(x)
    return x  
   
class FinalBlock(nn.Module):
  """ This layer maps the d_model space to the output space.
      Args:
          d_model (int): The dimension of the model.
          seq_len (int): The length of the sequence.
          dropout (float, optional): The dropout rate. Defaults to 0.1.
          out_put_size (int, optional): The size of the output. Defaults to 1.
          forward_type (str, optional): The type of the forward pass. Defaults to 'basic'.
  """
  def __init__(self, d_model: int, seq_len: int, dropout: float = 0.1,  out_put_size: int = 1, forward_type='double_linear'):
    super().__init__()
    if forward_type == 'double_linear':
      self.linear_1 = nn.Linear(d_model, out_put_size)
      self.dropout = nn.Dropout(dropout)
      self.linear_2 = nn.Linear(seq_len, out_put_size)
      self.forward_type = self.double_linear_forward

    elif forward_type == 'single_linear':
      self.linear_1 = nn.Linear(d_model*seq_len, out_put_size)  
      self.forward_type = self.single_linear_forward

    elif forward_type == 'seq_average_linear':  
      self.linear = nn.Linear(seq_len, out_put_size) #TODO check whether this is correct seq_len
      self.forward_type = self.average_forward
      self.dim = 2

    elif forward_type == 'd_model_average_linear':
      self.linear = nn.Linear(d_model, out_put_size)
      self.forward_type = self.average_forward
      self.dim = 1
    else:
      raise ValueError(f"Unsupported forward type: {forward_type}")  
 
  def double_linear_forward(self, x):
    #(batch_size, seq_len, d_model)
    x = self.linear_1(x)
    x = x.squeeze()
    x = self.dropout(x)
    x = self.linear_2(x) 
    x = x.squeeze()
    x = self.dropout(x)
    # (batch_size)

    return x

  def single_linear_forward(self, x):
    # (batch_size, seq_len, d_model)
    x = x.view(x.shape[0], -1)
    # (batch_size, seq_len*d_model)
    x = self.linear_1(x)
    x = x.squeeze()
    # (batch_size, out_put_size)
    return x
  
  def average_forward(self, x):
    # (batch_size, seq_len, d_model)
    x =  x.mean(dim=self.dim) # --> (batch_size, seq_len)#TODO check whether this is correct 1 or 2
    x = self.linear(x) # --> (batch_size, 1)

    x = x.squeeze()

    # (batch_size)
    return x

  def forward(self, x):
    x = self.forward_type(x)
    return x

class MultiHeadAttentionBlock(nn.Module):
  def __init__(self, 
               d_model: int, 
               h: int, 
               max_seq_len: int = 128, 
               dropout: float = 0.1, 
               max_relative_position = 10, 
               relative_positional_encoding: str = 'Sinusoidal', 
               GSA: bool =False,
               projection_type: str = 'linear',
               ):
    super().__init__()
    self.d_model = d_model
    self.h = h
    self.scale = math.sqrt(self.d_model)
    self.relative_positional_encoding = relative_positional_encoding
    self.projection_type = projection_type

    assert d_model % h == 0, "d_model must be divisible by h"
    self.d_h = d_model // h

    if self.projection_type == 'linear':
      # TODO its an option to have biases or not in the linear layers
      self.W_q = nn.Linear(d_model, d_model) # W_q and bias
      self.W_k = nn.Linear(d_model, d_model) # W_k and bias
      self.W_v = nn.Linear(d_model, d_model) # W_v and bias

    elif self.projection_type == 'cnn':
      self.W_q = nn.Conv2d(self.d_h, self.d_h, kernel_size=(1,1), stride=(1,1))
      self.W_k = nn.Conv2d(self.d_h, self.d_h, kernel_size=(1,1), stride=(1,1))
      self.W_v = nn.Conv2d(self.d_h, self.d_h, kernel_size=(1,1), stride=(1,1))

      self.W_0 = nn.Conv2d(d_model, d_model, kernel_size=(5,5), stride=(1,1), padding=(2,2))


    self.W_0 = nn.Linear(self.h*self.d_h, d_model) # W_0 and bias wher self.h*self.d_h = d_model
    self.dropout = nn.Dropout(dropout)

    # For relative positional encoding
    if self.relative_positional_encoding == 'Relative':
      self.max_relative_position = max_relative_position
      self.relative_positional_k = RelativePositionalEncoding(self.d_h, max_relative_position)
      self.relative_positional_v = RelativePositionalEncoding(self.d_h, max_relative_position)

    


    self.GSA = GSA
    if GSA:
      row = torch.arange(max_seq_len).reshape(-1, 1)
      col = torch.arange(max_seq_len)
      self.sigma = nn.Parameter(torch.ones(1))
      self.G = torch.exp(-((row - col).float()/ self.sigma)**2)
    else:
      self.G = None  

      


  @staticmethod # Be able to call this method without instantiating the class
  def attention(query, key, value, mask, dropout : nn.Dropout, G=None, GSA=False):
    # mask is used for sentences and to ignore the padding
    # that is used to fill out a sentence to the max length
    d_h = query.shape[-1] # Get the last dimension of the query matrix

    # Compute the scaled dot product attention
    # key.transpose(-2, -1) swaps the last two dimensions of the key matrix
    # (batch_size, h, seq_len, d_h)  @ (batch_size, h, d_h, seq_len) --> (batch_size, h, seq_len, seq_len)
    attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_h) 

    # Apply the mask
    if mask is not None:
      attention_scores.masked_fill(mask == 0, -1e9)

    attention_scores = attention_scores.softmax(dim=-1) # (batch_size, h, seq_len, seq_len)
    if GSA:
      seq_len = attention_scores.shape[-1]
      attention_scores = attention_scores * G[:seq_len, :seq_len]

    # Apply the dropout
    if dropout is not None:
      attention_scores = dropout(attention_scores)

    # (batch_size, h, seq_len, seq_len) @ (batch_size, h, seq_len, d_h) --> 
    # (batch_size, h, seq_len, d_h)
    x = (attention_scores @ value)
    return x, attention_scores

  @staticmethod
  def attention_with_relative_position(query, key, value, h, d_h, relative_position_k, relative_position_v, scale, mask, dropout : nn.Dropout):
    # query = (batch size, seq_len, d_model)
    # key = (batch size, seq_len, d_model)
    # value = (batch size, seq_len, d_model)
    len_k = key.shape[1] # seq_len
    len_q = query.shape[1] # seq_len
    len_v = value.shape[1] # seq_len
    batch_size = query.shape[0] # batch_size
    

    q = query.view(query.shape[0], query.shape[1], h, d_h).transpose(1,2) # (batch_size, n_head, seq_len, d_h)
    k = key.view(key.shape[0], key.shape[1], h, d_h).transpose(1,2) # (batch_size, n_head, seq_len, d_h)
    normal_attention_scores = (q @ k.transpose(-2, -1)) # (batch_size, n_head, seq_len, seq_len)
    # TODO I might have to divide by math.sqrt(d_h) here as well
    # and use a softmax function here as well
    
    relative_q = query.permute(1, 0, 2).contiguous().view(len_q, batch_size*h, -1) # (seq_len, batch_size*n_head, d_h)
    relative_k = relative_position_k(len_q, len_k) # (seq_len, seq_len, d_h)
    # realative_k = (seq_len, seq_len, d_h) --> (seq_len, d_h, seq_len)
    # (seq_len, batch_size*n_head, d_h) @ (seq_len, d_h, seq_len) --> (seq_len, batch_size*n_head, seq_len)
    # (seq_len, batch_size*n_head, seq_len) --> (batch_size*n_head, seq_len, seq_len)
    #
    relative_attention_scores = torch.matmul(relative_q, relative_k.transpose(1, 2)).transpose(0, 1)
    relative_attention_scores = relative_attention_scores.contiguous().view(batch_size, h, len_q, len_k)
    # (batch_size, n_head, seq_len, seq_len)
    attention_scores = (normal_attention_scores + relative_attention_scores) / scale

    # Apply the mask
    if mask is not None:
      attention_scores.masked_fill(mask == 0, -1e9)

    # Apply the dropout
    if dropout is not None:
      attention_scores = dropout(attention_scores)

    relative_v_1 = value.view(batch_size, -1, h, d_h).permute(0, 2, 1, 3) # (batch_size, n_head, seq_len, d_h)
    weight1 = (attention_scores @ relative_v_1) # (batch_size, n_head, seq_len, seq_len) @ (batch_size, n_head, seq_len, d_h) --> 
    # (batch_size, n_head, seq_len, d_h)
    relative_v_2 = relative_position_v(len_q, len_v) # (seq_len, seq_len, d_h) 
    weight2 = attention_scores.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size*h, len_k)  # (seq_len, batch_size*n_head, seq_len)
    weight2 = (weight2 @ relative_v_2).transpose(0, 1).contiguous().view(batch_size, h, len_q, d_h)

    x = weight1 + weight2

    return x, attention_scores

  def forward(self, q, k, v, mask):
    # secretly q,k,v are all the same
    if self.projection_type == 'linear':
      query =self.W_q(q) # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
      key = self.W_k(k) # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
      value = self.W_v(v) # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
    elif self.projection_type == 'cnn':
      batch_size, seq_len, _ = q.shape
      q = q.view(batch_size, self.d_h, self.h, seq_len)
      k = k.view(batch_size, self.d_h, self.h, seq_len)
      v = v.view(batch_size, self.d_h, self.h, seq_len)
      query = self.W_q(q)
      key = self.W_k(k)
      value = self.W_v(v)
      query = query.view(batch_size, seq_len, self.d_h * self.h)
      key = key.view(batch_size, seq_len, self.d_h * self.h)
      value = value.view(batch_size, seq_len, self.d_h * self.h)

          
    # Split the query, into h heads
    # query.shape[0] = batch_size
    # query.shape[1] = seq_len
    # (batch_size, seq_len, d_model) --> (batch_size, seq_len, h, d_h) -->
    # transpose(1,2) --> (batch_size, h, seq_len, d_h)
    # transpose(1,2) swaps the seq_len and h dimensions dimeinstion 1 and 2


    if self.relative_positional_encoding == 'Realtive':
      x, self.attention_scores = MultiHeadAttentionBlock.attention_with_relative_position(query, 
                                                                                          key, 
                                                                                          value, 
                                                                                          self.h,
                                                                                          self.d_h, 
                                                                                          self.relative_positional_k, 
                                                                                          self.relative_positional_v, 
                                                                                          self.scale, 
                                                                                          mask, 
                                                                                          self.dropout)
    else:

      query = query.view(query.shape[0], query.shape[1], self.h, self.d_h).transpose(1,2) # (batch_size, seq_len, d_model) --> (batch_size, h, seq_len, d_h)
      key = key.view(key.shape[0], key.shape[1], self.h, self.d_h).transpose(1,2) # (batch_size, seq_len, d_model) --> (batch_size, h, seq_len, d_h)
      value = value.view(value.shape[0], value.shape[1], self.h, self.d_h).transpose(1,2) # (batch_size, seq_len, d_model) --> (batch_size, h, seq_len, d_h)

      x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout, self.G, self.GSA)  
    # Concatenate the heads
    # (batch_size, h, seq_len, d_h) --> (batch_size, seq_len, h, d_h) -->
    #  (batch_size, seq_len, h*d_h) = (batch_size, seq_len, d_model)
    x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h*self.d_h)

    # Apply the last linear layer
    # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model) 
    if self.projection_type == 'linear':
      x = self.W_0(x)
    elif self.projection_type == 'cnn':
      batch_size, seq_len, d_model = x.size()
      x = x.view(batch_size, 1, seq_len, d_model)  # reshape
      x = self.W_0(x)  # apply Conv2d
      x = x.view(batch_size, seq_len, d_model)  # reshape back
    return x
  
class ResidualConnection(nn.Module):
  """ This layer adds the residual connection from sublayer to the input.
      It also appies a 
  """
  def __init__(self, features: int, dropout: float = 0.1, residual_type='post_ln', normalization='layer'):
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.residual_type = residual_type
    if normalization == 'layer':
      self.norm = LayerNormalization(features=features)
    elif normalization == 'batch':
      self.norm = BatchNormalization(features=features)
    
  def forward(self, x, sublayer):
    if self.residual_type == 'post_ln':
      return self.forward_post_ln(x, sublayer)
    elif self.residual_type == 'pre_ln':
      return self.forward_pre_ln(x, sublayer)

  def forward_post_ln(self, x, sublayer):
    out = sublayer(x)
    x = x + self.dropout(out)
    x = self.norm(x)
    return x
    
  def forward_pre_ln(self, x, sublayer):
    out = self.norm(x)
    out = sublayer(out)
    x = x + self.dropout(out)

    # (batch_size , seq_len, d_model)
    return x
  
class EncoderBlock(nn.Module):
  def __init__(self, 
               features: int,
               self_attention_block: MultiHeadAttentionBlock,
               feed_forward_block: FeedForwardBlock, 
               dropout: float = 0.1,
               residual_type='post_ln',
               normalization: str = 'layer'):
    super().__init__()
    self.self_attention_block = self_attention_block
    self.feed_forward_block =  feed_forward_block
    self.residual_connection_1 = ResidualConnection(features=features, 
                                                    dropout=dropout, 
                                                    residual_type=residual_type,
                                                    normalization=normalization)
    self.residual_connection_2 = ResidualConnection(features=features,
                                                    dropout=dropout, 
                                                    residual_type=residual_type,
                                                    normalization=normalization)

  

  def forward(self, x, src_mask):
    # x = self.residual_1(x, lambda x: self.self_attention_block(x, x, x, src_mask))
    # Is this the same as:
    # y = self.attention_block(x, x, x, src_mask)
    # x = self.residual_1(x, y) ?
    # x = self.residual_2(x, self.feed_forward_block)

    #  TODO I don't understand the differnece between the two residual connections
    x = self.residual_connection_1(x, lambda y: self.self_attention_block(y, y, y, src_mask))
    x = self.residual_connection_2(x, self.feed_forward_block)
    return x  
  

  
class VanillaEncoderBlock(nn.Module):
  def __init__(self, d_model: int, h: int, d_ff: int, dropout: float = 0.1, normalization='layer', activation='relu', residual_type='post_ln'):
    super().__init__()
    self.self_attention_block = MultiheadAttention(d_model, h, dropout)
    self.dropout_1 = nn.Dropout(dropout)
    self.dropout_2 = nn.Dropout(dropout)
    if normalization == 'layer':
      self.norm_1 = LayerNormalization(d_model)
      self.norm_2 = LayerNormalization(d_model)
    elif normalization == 'batch': 
      self.norm_1 = BatchNormalization(d_model)
      self.norm_2 = BatchNormalization(d_model)  
 
    self.feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout, activation)
    self.residual_type = residual_type

  def forward(self, x, src_mask, src_key_padding_mask=None):
    # (batch_size, seq_len, d_model)
    # Multi head attention block
    

    # Residual connection and normalization
    if self.residual_type == 'pre_ln':
      x2 = self.norm_1(x) # --> (seq_len, batch_size, d_model)
      x2 = x2.permute(1, 0, 2) # --> (seq_len, batch_size, d_model)
      x2  = self.self_attention_block(x2, x2, x2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]# --> (seq_len, batch_size, d_model)
      x = x.permute(1, 0, 2) # --> (seq_len, batch_size, d_model)
      x = x + self.dropout_1(x2) # --> (seq_len, batch_size, d_model)
      x = x.permute(1, 0, 2) # --> (batch_size, seq_len, d_model)

      # Feed forward block
      x2 = self.norm_2(x) # --> (seq_len, batch_size, d_model)
      x2 = self.feed_forward_block(x2) # --> (seq_len, batch_size, d_model)
      x = x + self.dropout_2(x2)

    elif self.residual_type == 'post_ln':
      x2  = self.self_attention_block(x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
      x = x + self.dropout_1(x2) # --> (seq_len, batch_size, d_model)
      x = x.permute(1, 0, 2) # --> (batch_size, seq_len, d_model)
      x = self.norm_1(x) # --> (batch_size, seq_len, d_model)
      
      # Feed forward block
      x2 = self.feed_forward_block(x) # --> (batch_size, seq_len, d_model)
      # Residual connection and normalization
      x = x + self.dropout_2(x2) # --> (batch_size, seq_len, d_model)
      x = self.norm_2(x) # --> (batch_size, seq_len, d_model)



    return x
               
class VanillaEncoderTransformer(nn.Module):
  def __init__(self, encoders: nn.Module, 
              src_embed: List[InputEmbeddings], 
               src_pos: List[PositionalEncoding],   
               final_block: FinalBlock,
               residual_type: str = 'post_ln',
               normalization: str = 'layer',
               features: int = 512
               ) -> None:
    super().__init__()

    self.encoder = encoders
    self.src_embed = src_embed[0]
    self.src_pos = src_pos[0]
    self.final_block = final_block
    self.residual_type = residual_type
    if residual_type == 'post_ln':
      self.norm = nn.Identity()
    elif residual_type == 'pre_ln':
      if normalization == 'layer':
        self.norm = LayerNormalization(features=features)
      elif normalization == 'batch':
        self.norm = BatchNormalization(features=features)


  def encode(self, src, src_mask=None, src_key_padding_mask=None): 
    # (batch_size, seq_len, d_model)
    src = self.src_embed(src)

    src = self.src_pos(src)

    src = self.encoder(src, src_mask, src_key_padding_mask)

    src = self.final_block(src)

    return src 
    


class Encoder(nn.Module):
  def __init__(self, layers: nn.ModuleList) -> None:
    super().__init__()
    self.layers = layers
    
    

  def forward(self, x, mask=None):
    for layer in self.layers:
      x = layer(x, mask)

    return x



# class ProjectionLayer(nn.Module):
#   def __init__(self, d_model: int, vocab_size: int) -> None:
#     super().__init__()
#     self.proj = nn.Linear(d_model, vocab_size)     
        
#   def forward(self, x):
#     # (Batch, seq_len, d_model) --> (Batch, seq_len, vocab_size)
#     return torch.log_softmax(self.proj(x), dim=-1)
  

# TODO this is a test
# class EncoderTransformer(nn.Module):
#   def __init__(self, 
#                features:int,
#                encoders: List[nn.Module], 
#                src_embed: List[InputEmbeddings], 
#                src_pos: List[PositionalEncoding],
#                final_block: FinalBlock,
#                encoding_type: str = 'normal',
#                residual_type: str = 'post_ln',
#                normalization: str = 'layer'

#                ) -> None:
#     super().__init__()

#     if residual_type == 'post_ln':
#       self.norm = nn.Identity()
#     elif residual_type == 'pre_ln':
#       if normalization == 'layer':
#         self.norm = LayerNormalization(features=features)
#       elif normalization == 'batch':
#         self.norm = BatchNormalization(features=features)
#       else:
#         raise ValueError(f"Unsupported normalization type: {normalization}")  
#     else:
#       raise ValueError(f"Unsupported residual type: {residual_type}")
      



#     if encoding_type == 'bypass':
#       for i, encoder in enumerate(encoders):
#         setattr(self, f'encoder_{i+1}', encoder)
#         setattr(self, f'src_embed_{i+1}', src_embed[i])
#         setattr(self, f'src_pos_{i+1}', src_pos[i])
#       self.final_block = final_block
#       self.encode_type = self.bypass_encode

#     elif encoding_type == 'none':
#       self.src_embed = src_embed[0]
#       self.src_pos = src_pos[0]
#       self.final_block = final_block
#       self.encode_type = self.non_encode

#     elif encoding_type == 'normal':
#       self.encoder = encoders[0]
#       self.src_embed = src_embed[0]
#       self.src_pos = src_pos[0]
#       self.final_block = final_block
#       self.encode_type = self.normal_encode

    
#     else:
#       raise ValueError(f"Unsupported encoding type: {encoding_type}")


#   def normal_encode(self, src, src_mask=None):
#     # (batch_size, seq_len, d_model)
#     src = self.src_embed(src)

#     src = self.src_pos(src)
    
#     src = self.encoder(src, src_mask)
#     # print(f"Encoder output standard deviation: {src.std().item()} ", end=' ')

#     src = self.norm(src)

#     src = self.final_block(src)
#     # print(f"Final block output standard deviation: {src.std().item()} ", end='\r')
#     return src
  
#   def non_encode(self, src, src_mask=None):
#     # (batch_size, seq_len, d_model)
#     src = self.src_embed(src)

#     src = self.src_pos(src)

#     src = self.final_block(src)
#     return src

#   def bypass_encode(self, src, src_mask=None):
#     # (batch_size, seq_len, d_model)
#     src_slices = src.split(1, dim=-1)
#     # (batch_size, seq_len, 1) x n_ant
#     src_embeds = [getattr(self, f'src_embed_{i+1}')(src_slice) for i, src_slice in enumerate(src_slices)]
#     src_poss = [getattr(self, f'src_pos_{i+1}')(src_embed) for i, src_embed in enumerate(src_embeds)]
#     src_encoders = [getattr(self, f'encoder_{i+1}')(src_pos, src_mask) for i, src_pos in enumerate(src_poss)]

#     src = torch.cat(src_encoders, dim=-1)

#     src = self.norm(src)

#     src = self.final_block(src)

#     return src

#   def encode(self, src, src_mask=None):
#     return self.encode_type(src, src_mask)
  

# TODO this is a test
class EncoderTransformer(nn.Module):
    def __init__(self, 
                 features:int,
                 encoders: List[nn.Module], 
                 src_embed: List[InputEmbeddings], 
                 src_pos: List[PositionalEncoding],
                 final_block: FinalBlock,
                 encoding_type: str = 'normal',
                 residual_type: str = 'post_ln',
                 normalization: str = 'layer',
                 data_type: str = 'trigger'
                 ) -> None:
        super().__init__()

        self.data_type = data_type
        norm = self.get_norm(residual_type, normalization, features)
        self.network_blocks = nn.ModuleList()



    
        if encoding_type == 'normal':
            self.encode_type = self.normal_encode
            block1 = nn.Sequential(src_embed[0], src_pos[0], encoders[0], norm)
            block2 = nn.Sequential(final_block)
            self.network_blocks.extend([block1, block2])


        # if encoding_type == 'normal':

        #     self.encode_type = self.normal_encode
        #     self.network_blocks.extend([nn.Sequential(src_embed[0], src_pos[0], encoders[0], norm), nn.Sequential(final_block)])


        elif encoding_type == 'bypass':
            self.encode_type = self.bypass_encode
            self.src_embed = src_embed
            self.src_pos = src_pos
            self.encoders = encoders
            self.network_blocks.extend([nn.Sequential(src_embed[i], src_pos[i], encoders[i]) for i in range(len(encoders))])
            self.network_blocks.append(final_block)

        # TODO test this
        elif encoding_type == 'none':
            self.src_embed = src_embed[0]
            self.src_pos = src_pos[0]
            self.final_block = final_block
            self.encode_type = self.non_encode
            self.network_blocks.extend([nn.Sequential(self.src_embed, self.src_pos), nn.Sequential(self.final_block)])
        
        else:
            raise ValueError(f"Unsupported encoding type: {encoding_type}")

    def get_norm(self, residual_type, normalization, features):
        norm_types = {'post_ln': nn.Identity, 'pre_ln': {'layer': LayerNormalization, 'batch': BatchNormalization}}
        try:
            if residual_type == 'pre_ln':
                return norm_types[residual_type][normalization](features=features)
            return norm_types[residual_type]()
        except KeyError:
            raise ValueError(f"Unsupported residual or normalization type: {residual_type}, {normalization}")

    def normal_encode(self, src, src_mask=None):
        for module in self.network_blocks:
            src = module(src)
        return src
    
     # TODO test this
    def bypass_encode(self, src, src_mask=None):
        src_slices = src.split(1, dim=-1)
        src_embeds = [self.src_embed[i](src_slice) for i, src_slice in enumerate(src_slices)]
        src_poss = [self.src_pos[i](src_embed) for i, src_embed in enumerate(src_embeds)]
        src_encoders = [self.encoders[i](src_pos, src_mask) for i, src_pos in enumerate(src_poss)]
        src = torch.cat(src_encoders, dim=-1)
        src = self.norm(src)
        src = self.network_blocks[-1](src)  # Access the final_block through network_blocks
        return src
    
    def non_encode(self, src, src_mask=None):
        for module in self.network_blocks:
            src = module(src)
        return src

    def forward(self, src, src_mask=None):
        if self.data_type == 'chunked':
            src = src.permute(0, 2, 1)
            src = self.encode_type(src, src_mask)
            return src.unsqueeze(-1)
        return self.encode_type(src, src_mask)
    
    def obtain_pre_activation(self, x):

        x = self.forward(x)
        return x


# class build_encoder_transformer(nn.Module):
#   def __init__(self, config):
#     super(build_encoder_transformer, self).__init__()
#     if 'transformer' in config:
#       config = config['transformer']
#     self.encoder = build_encoder_transformer(config)
  

#     if config['architecture']['data_type'] == 'trigger':
#       self.first = 0
#       self.second = 1
#       self.third = 2
#     elif config['architecture']['data_type'] == 'chunked':
#       self.first = 0
#       self.second = 1
#       self.third = 2
#     else:
#       raise ValueError(f"Unsupported data type: {config['architecture']['data_type']}")  

#   def forward(self, x):
#     x = x.permute(self.first,self.second,self.third)
#     return self.encoder(x, src_mask=None)

def build_encoder_transformer(config): 
  
  seq_len=        config['architecture']['seq_len'] 
  d_model=        config['architecture']['d_model'] 
  N=              config['architecture']['N'] 
  h=              config['architecture']['h'] 
  omega=          config['architecture']['omega'] 
  d_ff=           config['architecture']['d_ff'] 
  output_size =   config['architecture'].get('output_size', 1) 
  config['architecture']['output_size'] = output_size 
  encoder_type =  config['architecture'].get('encoder_type', 'normal') 
  config['architecture']['encoder_type'] = encoder_type 
  normalization = config['architecture'].get('normalization', 'layer')
  # config['architecture']['normalization'] = normalization 
  residual_type =      config['architecture'].get('residual_type', 'post_ln')
  max_seq_len =    1024 #config['architecture']['seq_len'] # TODO make some other implementation
  activation =    config['architecture']['activation']
  n_ant =         config['architecture']['n_ant'] 
  max_relative_position = config['architecture'].get('max_relative_position', 100)
  config['architecture']['max_relative_position'] = max_relative_position
  old_residual =  config['architecture'].get('old_residual', False)
  if old_residual:
    features = 1
  else:
    features = d_model  

  relative_positional_encoding = config['architecture']['pos_enc_type'] 
 
  dropout=config['training']['dropout'] 
  data_type = config['architecture']['data_type']
  if data_type == 'chunked':
    data_order = 'bcs'
  elif data_type == 'trigger':
    data_order = 'bsc'  

  if encoder_type == 'bypass':
    by_pass = True
    num_embeddings = n_ant 
    channels = 1
  else:
    by_pass = False 
    channels = n_ant
    num_embeddings = 1  

 

  #########################################################
  # Create the input embeddings                           #
  #########################################################    

  src_embed = [InputEmbeddings(d_model=d_model, channels=channels, dropout=dropout, embed_type=config['architecture']['embed_type']) for _ in range(num_embeddings)]
  
  #########################################################
  # Create the positional encodings                       #
  #########################################################
  pos_enc_config = {
      'Sinusoidal': lambda: PositionalEncoding(d_model=d_model, dropout=dropout, max_seq_len=max_seq_len, omega=omega),
      'None': lambda: nn.Identity(),
      'Relative': lambda: nn.Identity(),
      'Learnable': lambda: LearnablePositionalEncoding(d_model=d_model, max_len=max_seq_len, dropout=dropout)
  }

  pos_enc_type = config['architecture']['pos_enc_type'] 

  if pos_enc_type in pos_enc_config:
      src_pos_func = pos_enc_config[pos_enc_type]
  else:
      raise ValueError(f"Unsupported positional encoding type: {pos_enc_type}")

  num_embeddings = n_ant if by_pass else 1 

  src_pos = [src_pos_func() for _ in range(num_embeddings)]

  #########################################################
  # Create the encoder layers                             #
  #########################################################
  if encoder_type == 'normal' or encoder_type == 'bypass':

    num_encoders = n_ant if by_pass else 1

    encoders = []
    
    for _ in range(num_encoders):
      encoder_blocks = []
      for _ in range(N):
        if config['architecture']['pos_enc_type'] == 'Relative':
          encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, max_seq_len=max_seq_len, dropout=dropout, max_relative_position=max_relative_position, relative_positional_encoding=relative_positional_encoding)
        else:
          encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)

        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout, activation=activation)
        encoder_block = EncoderBlock(features=features,
                                    self_attention_block=encoder_self_attention_block, 
                                    feed_forward_block=feed_forward_block, 
                                    dropout=dropout,
                                    residual_type=residual_type,
                                    normalization=normalization)
        encoder_blocks.append(encoder_block)
      encoders.append(Encoder(layers=nn.ModuleList(encoder_blocks)))
  elif encoder_type == 'none':
    encoders = [None]
  elif encoder_type == 'vanilla':
    vanilla_layer = VanillaEncoderBlock(d_model=d_model, 
                                        h=h, 
                                        d_ff=d_ff, 
                                        dropout=dropout,
                                        normalization=normalization,
                                        activation=activation,
                                        residual_type=residual_type)
    encoders = nn.TransformerEncoder(vanilla_layer, num_layers=N)
  else:
    raise ValueError(f"Unsupported encoding type: {encoder_type}")        
    

  #########################################################
  # Create the final block                                #
  #########################################################
  final_type = config['architecture'].get('final_type', 'double_linear') 
  if by_pass:
    factor = 4
  else:
    factor = 1  
  final_block = FinalBlock(d_model=d_model*factor,
                            seq_len=seq_len, 
                            dropout=dropout, 
                            out_put_size=output_size, 
                            forward_type=final_type)

  # Create the transformer
  if encoder_type == 'vanilla':
    encoder_transformer = VanillaEncoderTransformer(encoders=encoders,
                                                    src_embed=src_embed,
                                                    src_pos= src_pos, 
                                                    final_block=final_block,
                                                    residual_type=residual_type,
                                                    normalization=normalization,
                                                    features=features)
  else:
    encoder_transformer = EncoderTransformer(features=features,
                                            encoders=encoders,
                                            src_embed=src_embed,
                                            src_pos= src_pos, 
                                            final_block=final_block,
                                            encoding_type=encoder_type,
                                            residual_type=residual_type,
                                            normalization=normalization,
                                            data_type=data_type)

  # Initialize the parameters
  for p in encoder_transformer.parameters():
    if p.dim() > 1:
      nn.init.xavier_uniform_(p)  
 
  return encoder_transformer    

 
def get_n_params(model):
  pp = 0
  for p in list(model.parameters()):
    nn = 1
    for s in list(p.size()):
      nn = nn*s
    pp += nn
  return pp    



class ModelWrapper(nn.Module):
  """
    This code was provided by Copilot AI.
    This wrapper makes it possible to use get_model_complexity_info() from 
    ptflops. It is not possible to use the function directly on the model
    because the get_model_complexity_info() adds a dimention.
    Args:
        model (torch.nn.Module): The PyTorch model.
        batch_size (int): The batch size.
        seq_len (int): The length of the sequence.
        channels (int): The number of channels in the input.
    Returns:
        The model with the wrapper.    
  """
  def __init__(self, model, batch_size=1,  seq_len=256, channels=4):
    super().__init__()
    self.model = model
    self.batch_size = batch_size
    self.seq_len = seq_len
    self.channels = channels


  def forward(self, x):
    # Reshape the input to the correct shape
    x = x.squeeze(0)
    # x = x.view(self.batch_size, self.seq_len, self.channels)
    src_mask = None #torch.zeros(self.batch_size, self.seq_len, self.seq_len)
    return self.model(x)
    
def load_model(config, text='early_stop'):
  """
    Load model from config file
    
    Args:
        config (dict): The configuration dictionary. Should be loaded from a yaml file and should be the
                        same as the one used to train the model and also be the model config and not the 
                        training config.
        text (str, optional): The text to add to the model name. Defaults to 'early_stop'. Other options are
                              'final'.
    Returns:
            model

  """
  
  model = build_encoder_transformer(config)
  model_dict = model.state_dict()

  model_path = get_model_path(config, text=f'{text}')

  print(f'Preloading model {model_path}')
  state = torch.load(model_path)

  state_dict = state['model_state_dict']

  def count_matching_chars_from_right(s1, s2):
      # Reverse both strings
      s1 = s1[::-1]
      s2 = s2[::-1]

      # Count the number of matching characters from the start
      count = 0
      for c1, c2 in zip(s1, s2):
          if c1 == c2:
              count += 1
          else:
              break

      return count/len(s1)

  print()
  new_state_dict = {}
  print(f"{'Model: keys':<90} {'Model: value shape':<20} {'Loaded: keys':<90} {'Loaded: value shape':<20}")
  nr_of_model_items = len(model_dict.items())
  nr_of_state_items = len(state_dict.items())
  for (k1, v1) in model_dict.items():
      
      best_match = 0
      count_states = 0
      for (k2, v2) in state_dict.items():
          count_states += 1
          if k1 == 'network_blocks.0.2.layers.0.self_attention_block.relative_positional_k.embeddings_table':
            matching = count_matching_chars_from_right(k1, k2)
            print(f"{k1:<90} {str(v1.shape):<25} {k2:<90} {str(v2.shape):<25} {matching:<5}")
          matching_count = count_matching_chars_from_right(k1, k2)
          if v1.shape != v2.shape:
              matching_count = 0
          if matching_count > best_match:
              best_match = matching_count
              best_k2 = k2
              best_v2 = v2

          # if best_match == 0 and count_states == nr_of_state_items:
          #     for (k2, v2) in state_dict.items():
          #       matching_count = count_matching_chars_from_right(k1, k2)
          #       if matching_count > best_match:
          #         best_match = matching_count
          #         best_k2 = k2
          #         length_v2 = v2.shape[1]
          #         length_v1 = v1.shape[1]
          #         zeros = torch.zeros(v2.shape[0], length_v1 - length_v2, v2.shape[2]).to(v2.device)
          #         v2 = torch.cat((v2, zeros), dim=1)

          #         best_v2 = v2


      print(f"{k1:<90} {str(v1.shape):<25} {best_k2:<90} {str(best_v2.shape):<25} {best_match:<5}")
      new_state_dict[k1] = best_v2

  # Now load the new state dict
  model.load_state_dict(new_state_dict, strict=False)



  return model