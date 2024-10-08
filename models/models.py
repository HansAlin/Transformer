import torch 
import torch.nn as nn
import math
from typing import List
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer
import torch.nn.functional as F
import dataHandler.datahandler as dd
import sys
import os
import torch.nn.functional as F
from itertools import zip_longest
from ptflops import get_model_complexity_info

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
    x = x.permute(0, 2, 1) # (batch_size, d_model, seq_len)
    x = self.batch_norm(x)
    x = x.permute(0, 2, 1)
    return x
  

class FeedForwardBlock(nn.Module):

  def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation='relu'):
    # d_ff is the hidden layer size
    super().__init__()
    self.linear_1 = nn.Linear(d_model, d_ff, bias=False) # W1
    self.dropout = nn.Dropout(dropout)
    self.linear_2 = nn.Linear(d_ff, d_model, bias=False) # W2
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
  def __init__(self, channels, d_model, stride: int = 1, kernel_size: int = 3, max_pool: bool = False):
    super().__init__()

    self.kernel_size = kernel_size
    self.channels = channels
    vertical_padding = (kernel_size - 1 ) // 2
    self.stride = stride
    self.out_put_size = d_model//channels
    self.horizontal_padding = (kernel_size - 1) // 2

    if max_pool:
      self.maxpool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
      self.out_put_size = self.out_put_size * 2
    else:
      self.maxpool = nn.Identity()

    self.horizontal_conv = nn.Conv2d(1, self.out_put_size, kernel_size=(kernel_size, 1), padding=(self.horizontal_padding, 0), stride=(stride, 1))
    self.vertical_conv = nn.Conv2d(1, self.out_put_size, kernel_size=(stride, kernel_size), padding=(0,vertical_padding), stride=(stride, 1))
    
    

  def forward(self, x):
    # x: (batch_size, seq_len, channels)
    x = x.unsqueeze(1)  # now x: (batch_size, 1, seq_len, channels)
    out1 = self.horizontal_conv(x).squeeze(-1)  # out1: (batch_size, out_put_size, seq_len, channels)


    out2 = self.vertical_conv(x).squeeze(-1)  # out2: (batch_size, out_put_size, seq_len, channels)
    out = out1 + out2  # add the outputs
    out = out.transpose(1,2) # swap (batch_size, seq_len, out_put_size, channels)
    out = out.flatten(start_dim=2, end_dim=3)  # (batch_size, seq_len, out_put_size*channels) --> (batch_size, seq_len, d_model)
    out = self.maxpool(out)

    return out


class ViTEmbeddings(nn.Module):

  def __init__(self,
               channels: int,
               out_put_size: int,
               kernel_size: int,
               stride: int = 1,
               max_pool: bool = False,
               ) -> None:
    super().__init__()
    self.channels = channels
    self.out_put_size = out_put_size
    self.kernel_size = kernel_size
    self.height = kernel_size

    if max_pool:
      self.max_pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
      self.out_put_size = self.out_put_size * 2
    else:
      self.max_pool = nn.Identity()
    
    if kernel_size == channels:
      self.vertical_stride = 1
    else:
      self.vertical_stride = kernel_size
    assert channels % kernel_size == 0, "The number of channels must be divisible by the kernel size"
    self.horizontal_stride = kernel_size
    self.conv = nn.Conv2d(1, self.out_put_size, kernel_size=(self.height, self.height), stride=(self.vertical_stride, self.horizontal_stride), padding=(0, 0))

  def forward(self, x):
    x = x.transpose(1, 2) # (batch_size, seq_len, channels) --> (batch_size, channels, seq_len)
    x = x.unsqueeze(1)  # Add an extra dimension for "channels" (batch_size, 1, channels, seq_len)
     
    # Calculate padding dynamically
    seq_len = x.size(-1)
   
    if seq_len % self.height != 0:
      total_padding = (self.kernel_size - seq_len % self.kernel_size) % self.kernel_size
      padding_left = total_padding // 2
      padding_right = total_padding - padding_left
    else:
      padding_left = 0
      padding_right = 0  

    x = F.pad(x, (padding_left, padding_right))  # Add padding to the sequence length dimension
    x = self.conv(x) # (batch_size, out_put_size (d_model), channels//kernel_size, seq_len//kernel_size)
    x = x.permute(0, 2, 3, 1)  # (batch_size, channels//kernel_size, seq_len//kernel_size, out_put_size (d_model)
    x = x.flatten(start_dim=1, end_dim=2)  # (batch_size, d_model, new_seq_len) Remove the height dimension

    x = self.max_pool(x)
    
    # x = x.transpose(1, 2)  # Swap the "seq_len" and "d_model" dimensions
    return x  

class InputEmbeddings(nn.Module):
  """This layer maps feature space from the input to the d_model space.

  
  Args:
      d_model (int): The dimension of the model.
      dropout (float, optional): The dropout rate. Defaults to 0.1.
      channels (int, optional): The number of channels in the input. Defaults to 1.
      embed_type (str, optional): The type of the embedding. Defaults to 'linear'.
      kwargs: Additional arguments for the CnnInputEmbeddings and ViTEmbeddings layer, padding, stride, kernel_size.
  
  """
  def __init__(self, d_model: int, dropout: float = 0.1, activation: str = 'relu',n_ant: int = 4, embed_type='linear', **kwargs) -> None:
    super().__init__()
    self.d_model = d_model
    self.channels = n_ant
    if activation == 'relu':
      self.activation = nn.ReLU()
    elif activation == 'gelu':
      self.activation = nn.GELU()
    else:
      self.activation = nn.Identity

    if embed_type == 'linear':
      self.embedding = nn.Linear(n_ant, d_model)
    elif embed_type == 'cnn':
      self.embedding = CnnInputEmbeddings(n_ant, d_model, **kwargs)
    elif embed_type == 'ViT':
      self.embedding = ViTEmbeddings(n_ant, d_model, **kwargs)
    else:
      raise ValueError(f"Unsupported embed type: {embed_type}")
    
    self.dropout = nn.Dropout(dropout)  
  def forward(self, x):
    # (batch_size, seq_len, channels)

    x = self.embedding(x)
    x = self.activation(x)
    x = self.dropout(x)
    # --> (batch_size, seq_len, d_model)
    return x 

class PositionalEncoding(nn.Module):
   
  def __init__(self, d_model: int, dropout: float, max_seq_len: int, omega: float = 10000) -> None:
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

class MultiHeadAttentionBlock(nn.Module):
  def __init__(self, 
               d_model: int, 
               h: int, 
               max_seq_len: int = 128, 
               dropout: float = 0.1, 
               max_relative_position = 10, 
               positional_encoding: str = 'Sinusoidal', 
               GSA: bool =False,
               projection_type: str = 'linear',
               **kwargs
               ):
    super().__init__()
    self.d_model = d_model
    self.h = h
    self.scale = math.sqrt(self.d_model)
    self.positional_encoding = positional_encoding
    self.projection_type = projection_type

    if kwargs:
      if 'pre_def_dot_product' in kwargs:
        self.pre_def_dot_product = kwargs['pre_def_dot_product']
    else:
      self.pre_def_dot_product = False    

    assert d_model % h == 0, "d_model must be divisible by h"
    self.d_h = d_model // h

    if self.projection_type == 'linear':
      # Note! No biases
      self.W_q = nn.Linear(d_model, d_model, bias=False) # W_q 
      self.W_k = nn.Linear(d_model, d_model, bias=False) # W_k 
      self.W_v = nn.Linear(d_model, d_model, bias=False) # W_v 
      self.W_0 = nn.Linear(self.h*self.d_h, d_model, bias=False) # W_0 wher self.h*self.d_h = d_model

    elif self.projection_type == 'cnn':
      self.W_q = nn.Conv2d(self.h, self.h, kernel_size=(1,1), stride=(1,1), bias=False)
      self.W_k = nn.Conv2d(self.h, self.h, kernel_size=(1,1), stride=(1,1), bias=False)
      self.W_v = nn.Conv2d(self.h, self.h, kernel_size=(1,1), stride=(1,1), bias=False)

      self.W_0 = nn.Conv2d(d_model, d_model, kernel_size=(5,5), stride=(1,1), padding=(2,2), bias=False)


    self.dropout_value = dropout
    self.dropout = nn.Dropout(dropout)

    # For relative positional encoding
    if self.positional_encoding == 'Relative':
      self.max_relative_position = max_relative_position
      self.relative_positional_k = RelativePositionalEncoding(self.d_h, max_relative_position)
      self.relative_positional_v = RelativePositionalEncoding(self.d_h, max_relative_position)

    


    self.GSA = GSA
    if GSA:
      row = torch.arange(max_seq_len).reshape(-1, 1)
      col = torch.arange(max_seq_len)
      self.sigma = nn.Parameter(torch.ones(1))
      G = torch.exp(-((row - col).float()/ self.sigma)**2)
      self.register_buffer('G', G)
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
    # Note! No scaling here
    
    relative_q = query.permute(1, 0, 2).contiguous().view(len_q, batch_size*h, -1) # (seq_len, batch_size*n_head, d_h)
    relative_k = relative_position_k(len_q, len_k) # (seq_len, seq_len, d_h)
    # realative_k = (seq_len, seq_len, d_h) --> (seq_len, d_h, seq_len)
    # (seq_len, batch_size*n_head, d_h) @ (seq_len, seq_len, d_h)--> (seq_len, batch_size*n_head, seq_len)
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
      q = q.reshape(batch_size, self.h, self.d_h, seq_len)
      k = k.reshape(batch_size, self.h, self.d_h, seq_len)
      v = v.reshape(batch_size, self.h, self.d_h, seq_len)
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


    if self.positional_encoding == 'Relative': 
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

      if self.pre_def_dot_product:
        x = F.scaled_dot_product_attention(query=query,
                                           key=key, 
                                           value=value, 
                                           dropout_p=self.dropout_value)


      else:
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
      x = x.view(batch_size, d_model, 1, seq_len)  # reshape
      x = self.W_0(x)  # apply Conv2d
      x = x.view(batch_size, seq_len, d_model)  # reshape back
    return x
  
class ResidualConnection(nn.Module):
  """ This layer adds the residual connection from sublayer to the input.
      It also appies a 
  """
  def __init__(self, d_model: int, batch_size: int, dropout: float = 0.1, residual_type='post_ln', normalization='layer'):
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.residual_type = residual_type
    if normalization == 'layer':
      self.norm = LayerNormalization(features=d_model)
    elif normalization == 'batch':
      self.norm = BatchNormalization(features=d_model)
    
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
               d_model: int,
               batch_size: int,
               self_attention_block: MultiHeadAttentionBlock,
               feed_forward_block: FeedForwardBlock, 
               dropout: float = 0.1,
               residual_type='post_ln',
               normalization: str = 'layer'):
    super().__init__()
    self.self_attention_block = self_attention_block
    self.feed_forward_block =  feed_forward_block
    self.residual_connection_1 = ResidualConnection(d_model=d_model, 
                                                    batch_size=batch_size,
                                                    dropout=dropout, 
                                                    residual_type=residual_type,
                                                    normalization=normalization)
    self.residual_connection_2 = ResidualConnection(d_model=d_model,
                                                    batch_size=batch_size,
                                                    dropout=dropout, 
                                                    residual_type=residual_type,
                                                    normalization=normalization)

  

  def forward(self, x, src_mask):
    x = self.residual_connection_1(x, lambda y: self.self_attention_block(y, y, y, src_mask))
    x = self.residual_connection_2(x, self.feed_forward_block)
    return x  
  

  
class VanillaEncoderBlock(nn.Module):
  def __init__(self, d_model: int, h: int, d_ff: int, dropout: float = 0.1, normalization='layer', activation='relu', residual_type='post_ln'):
    super().__init__()
    self.self_attn = MultiheadAttention(d_model, h, dropout, bias=False)
    self.dropout_1 = nn.Dropout(dropout)
    self.dropout_2 = nn.Dropout(dropout)
    if normalization == 'layer':
      self.norm_1 = LayerNormalization(d_model)
      self.norm_2 = LayerNormalization(d_model)
    elif normalization == 'batch': 
      self.norm_1 = BatchNormalization(features=d_model)
      self.norm_2 = BatchNormalization(features=d_model)  
 
    self.feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout, activation)
    self.residual_type = residual_type

  def get_attention_scores(self):
    return self.attention_scores

  def forward(self, x, src_mask, src_key_padding_mask=None, is_causal=None):
    # (batch_size, seq_len, d_model)
    # Multi head attention block
    

    # Residual connection and normalization
    if self.residual_type == 'pre_ln':
      x2 = self.norm_1(x) # --> (seq_len, batch_size, d_model)
      x2 = x2.permute(1, 0, 2) # --> (seq_len, batch_size, d_model)
      x2, self.attention_scores  = self.self_attn(x2, x2, x2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)# --> (seq_len, batch_size, d_model)
      x = x.permute(1, 0, 2) # --> (seq_len, batch_size, d_model)
      x = x + self.dropout_1(x2) # --> (seq_len, batch_size, d_model)
      x = x.permute(1, 0, 2) # --> (batch_size, seq_len, d_model)

      # Feed forward block
      x2 = self.norm_2(x) # --> (seq_len, batch_size, d_model)
      x2 = self.feed_forward_block(x2) # --> (seq_len, batch_size, d_model)
      x = x + self.dropout_2(x2)

    elif self.residual_type == 'post_ln':
      x2, self.attention_scores  = self.self_attn(x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
      x = x + self.dropout_1(x2) # --> (seq_len, batch_size, d_model)
      x = x.permute(1, 0, 2) # --> (batch_size, seq_len, d_model)
      x = self.norm_1(x) # --> (batch_size, seq_len, d_model)
      
      # Feed forward block
      x2 = self.feed_forward_block(x) # --> (batch_size, seq_len, d_model)
      # Residual connection and normalization
      x = x + self.dropout_2(x2) # --> (batch_size, seq_len, d_model)
      x = self.norm_2(x) # --> (batch_size, seq_len, d_model)



    return x

class FinalBlock(nn.Module):
  """ This layer maps the d_model space to the output space.
      Args:
          d_model (int): The dimension of the model.
          seq_len (int): The length of the sequence.
          dropout (float, optional): The dropout rate. Defaults to 0.1.
          out_put_size (int, optional): The size of the output. Defaults to 1.
          forward_type (str, optional): The type of the forward pass. Defaults to 'd_model_average_linear'.
  """
  def __init__(self, d_model: int, seq_len: int, dropout: float = 0.1,  out_put_size: int = 1, forward_type='double_linear'):
    super().__init__()

    # Two linear layers after each other
    if forward_type == 'double_linear':
      self.linear_1 = nn.Linear(d_model, out_put_size)
      self.dropout = nn.Dropout(dropout)
      self.linear_2 = nn.Linear(seq_len, out_put_size)
      self.forward_type = self.double_linear_forward

    # One linear layer
    elif forward_type == 'single_linear':
      self.linear_1 = nn.Linear(d_model*seq_len, out_put_size)  
      self.forward_type = self.single_linear_forward

    # Average the sequence and apply a linear layer
    elif forward_type == 'seq_average_linear':  
      self.linear = nn.Linear(seq_len, out_put_size) 
      self.forward_type = self.average_forward
      self.dim = 2

    # Average the d_model and apply a linear layer
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
    x =  x.mean(dim=self.dim) # --> (batch_size, seq_len)
    x = self.linear(x) # --> (batch_size, 1)

    x = x.squeeze()

    # (batch_size)
    return x

  def forward(self, x):
    x = self.forward_type(x)
    return x  


class VanillaEncoderTransformer(nn.Module):
  def __init__(self, encoders: nn.Module, 
              src_embed: List[InputEmbeddings], 
               src_pos: List[PositionalEncoding],   
               final_block: FinalBlock,
               residual_type: str = 'post_ln',
               normalization: str = 'layer',
               d_model: int = 512,
               batch_size: int = 128,
               data_type: str = 'trigger'
               ) -> None:
    super().__init__()

    # self.encoder = encoders
    # self.src_embed = src_embed
    # self.src_pos = src_pos
    # self.final_block = final_block
    self.data_type = data_type
    self.residual_type = residual_type
    if residual_type == 'post_ln':
      self.norm = nn.Identity()
    elif residual_type == 'pre_ln':
      if normalization == 'layer':
        self.norm = LayerNormalization(features=d_model)
      elif normalization == 'batch':
        self.norm = BatchNormalization(features=d_model)

    self.network_blocks = nn.ModuleList()
    block1 = nn.Sequential(src_embed, src_pos, encoders, self.norm)
    block2 = nn.Sequential(final_block)
    self.network_blocks.extend([block1, block2])

  def forward(self, src, src_mask=None, src_key_padding_mask=None): 
    if self.data_type == 'chunked':
      src = src.permute(0, 2, 1)
      # (batch_size, seq_len, d_model)
      src = self.network_blocks[0][0](src)
      src = self.network_blocks[0][1](src)
      src = self.network_blocks[0][2](src)
      src = self.network_blocks[0][3](src)
      src = self.network_blocks[1](src)
      return src.unsqueeze(-1) # (batch_size, 1)
    else:
      # (batch_size, seq_len, d_model)
      src = self.network_blocks[0][0](src)
      src = self.network_blocks[0][1](src)
      src = self.network_blocks[0][2](src)
      src = self.network_blocks[0][3](src)
      src = self.network_blocks[1](src)
      return src # (batch_size)

  def obtain_pre_activation(self, x):

        x = self.forward(x)
        return x


class Encoder(nn.Module):
  def __init__(self, layers: nn.ModuleList) -> None:
    super().__init__()
    self.layers = layers
    
    

  def forward(self, x, mask=None):
    for layer in self.layers:
      x = layer(x, mask)

    return x

class EncoderTransformer(nn.Module):
    def __init__(self, 
                 d_model:int,
                 batch_size:int,
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
        self.d_model = d_model
        self.batch_size = batch_size
        self.data_type = data_type
        norm = self.get_norm(residual_type, normalization)
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

        elif encoding_type == 'none':
            self.src_embed = src_embed[0]
            self.src_pos = src_pos[0]
            self.final_block = final_block
            self.encode_type = self.non_encode
            self.network_blocks.extend([nn.Sequential(self.src_embed, self.src_pos), nn.Sequential(self.final_block)])
        
        else:
            raise ValueError(f"Unsupported encoding type: {encoding_type}")

    def get_norm(self, residual_type, normalization):
        norm_types = {'post_ln': nn.Identity, 'pre_ln': {'layer': LayerNormalization, 'batch': BatchNormalization}}
        try:
            if residual_type == 'pre_ln':
                if normalization == 'layer':
                  features = self.d_model
                elif normalization == 'batch':
                  features = self.d_model
                return norm_types[residual_type][normalization](features=features)
            return norm_types[residual_type]()
        except KeyError:
            raise ValueError(f"Unsupported residual or normalization type: {residual_type}, {normalization}")

    def normal_encode(self, src, src_mask=None):
        for module in self.network_blocks:
            src = module(src)
        return src
    
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
            # chunked comes as (batch_size, n_ant, seq_len)
            src = src.permute(0, 2, 1)
            src = self.encode_type(src, src_mask)
            return src.unsqueeze(-1)
        return self.encode_type(src, src_mask)
    
    def obtain_pre_activation(self, x):

        x = self.forward(x)
        return x


def build_encoder_transformer(config): 
  

  max_seq_len =    1024 # For the positional encoding

  # if config['transformer']['architecture'].get('old_residual', False):
  #   features = 1
  # else:
  #   features = config['transformer']['architecture']['d_model']   
  # # TODO remove this not used
  # if config['transformer']['architecture']['data_type'] == 'chunked':
  #   data_order = 'bcs'
  # elif config['transformer']['architecture']['data_type'] == 'trigger':
  #   data_order = 'bsc'  

  if config['transformer']['architecture'].get('encoder_type', 'normal')   == 'bypass':
    by_pass = True
    num_embeddings = config['n_ant']  
    channels = 1
  else:
    by_pass = False 
    channels = config['n_ant'] 
    num_embeddings = 1  

 

  #########################################################
  # Create the input embeddings                           #
  #########################################################  
  max_pool = False   

  if config['transformer']['architecture']['embed_type'] == 'cnn' or config['transformer']['architecture']['embed_type'] == 'ViT':  
    kernel_size = config['transformer']['architecture']['input_embeddings']['kernel_size']
    stride = config['transformer']['architecture']['input_embeddings']['stride']
    if 'max_pool' in config['transformer']['architecture']:
      max_pool = config['transformer']['architecture']['max_pool']

  else:
    kernel_size = None
    stride = None
  
  src_embed = [InputEmbeddings(d_model=config['transformer']['architecture']['d_model'] , 
                               n_ant=channels, dropout=config['training']['dropout'], 
                               embed_type=config['transformer']['architecture']['embed_type'], 
                               kernel_size=kernel_size,
                               stride=stride,
                                max_pool=max_pool,
                               ) for _ in range(num_embeddings)]
  
  #########################################################
  # Create the positional encodings                       #
  #########################################################
  pos_enc_config = {
      'Sinusoidal': lambda: PositionalEncoding(d_model=config['transformer']['architecture']['d_model'] , dropout=config['training']['dropout'], max_seq_len=max_seq_len, omega=config['transformer']['architecture']['omega'] ),
      'None': lambda: nn.Identity(),
      'Relative': lambda: nn.Identity(),
      'Learnable': lambda: LearnablePositionalEncoding(d_model=config['transformer']['architecture']['d_model'] , max_len=max_seq_len, dropout=config['training']['dropout'])
  }

  pos_enc_type = config['transformer']['architecture']['pos_enc_type'] 

  if pos_enc_type in pos_enc_config:
      src_pos_func = pos_enc_config[pos_enc_type]
  else:
      raise ValueError(f"Unsupported positional encoding type: {pos_enc_type}")


  src_pos = [src_pos_func() for _ in range(num_embeddings)]

  #########################################################
  # Create the encoder layers                             #
  #########################################################
  if config['transformer']['architecture'].get('encoder_type', 'normal')   == 'normal' or config['transformer']['architecture'].get('encoder_type', 'normal')   == 'bypass':

    num_encoders = config['n_ant']  if by_pass else 1

    encoders = []
    
    for _ in range(num_encoders):
      encoder_blocks = []
      for _ in range(config['transformer']['architecture']['N'] ):
        if config['transformer']['architecture']['pos_enc_type'] == 'Relative':
          encoder_self_attention_block = MultiHeadAttentionBlock(d_model=config['transformer']['architecture']['d_model'] , 
                                                                 h=config['transformer']['architecture']['h'] ,
                                                                 max_seq_len=max_seq_len, 
                                                                 dropout=config['training']['dropout'], 
                                                                 max_relative_position=config['transformer']['architecture']['max_relative_position'], 
                                                                 positional_encoding=config['transformer']['architecture']['pos_enc_type'],
                                                                 GSA=config['transformer']['architecture'].get('GSA', False),
                                                                 projection_type=config['transformer']['architecture'].get('projection_type', 'linear'),
                                                                 pre_def_dot_product=config['transformer']['architecture'].get('pre_def_dot_product', False)

                                                                 )
        else:
          encoder_self_attention_block = MultiHeadAttentionBlock(d_model=config['transformer']['architecture']['d_model'] , 
                                                                 h=config['transformer']['architecture']['h'] , 
                                                                 max_seq_len=max_seq_len,
                                                                 dropout=config['training']['dropout'], 
                                                                 max_relative_position=None,
                                                                 positional_encoding=config['transformer']['architecture']['pos_enc_type'], 
                                                                 GSA=config['transformer']['architecture'].get('GSA', False),
                                                                 projection_type=config['transformer']['architecture'].get('projection_type', 'linear'),
                                                                 pre_def_dot_product=config['transformer']['architecture'].get('pre_def_dot_product', False)
                                                                 )

        feed_forward_block = FeedForwardBlock(d_model=config['transformer']['architecture']['d_model'] , 
                                              d_ff=config['transformer']['architecture']['d_ff'] ,
                                               dropout=config['training']['dropout'], 
                                               activation=config['transformer']['architecture']['activation'])
        encoder_block = EncoderBlock(d_model=config['transformer']['architecture']['d_model'],
                                     batch_size=config['training']['batch_size'],
                                    self_attention_block=encoder_self_attention_block, 
                                    feed_forward_block=feed_forward_block, 
                                    dropout=config['training']['dropout'],
                                    residual_type=config['transformer']['architecture'].get('residual_type', 'post_ln'),
                                    normalization=config['transformer']['architecture'].get('normalization', 'layer')  )
        encoder_blocks.append(encoder_block)
      encoders.append(Encoder(layers=nn.ModuleList(encoder_blocks)))
  elif config['transformer']['architecture'].get('encoder_type', 'normal')   == 'none':
    encoders = [None]
  elif config['transformer']['architecture'].get('encoder_type', 'normal')   == 'vanilla':
    vanilla_layer = VanillaEncoderBlock(d_model=config['transformer']['architecture']['d_model'] , 
                                        h=config['transformer']['architecture']['h'] , 
                                        d_ff=config['transformer']['architecture']['d_ff'] , 
                                        dropout=config['training']['dropout'],
                                        normalization=config['transformer']['architecture'].get('normalization', 'layer')  ,
                                        activation=config['transformer']['architecture']['activation'],
                                        residual_type=config['transformer']['architecture'].get('residual_type', 'post_ln'))
    encoders = nn.TransformerEncoder(vanilla_layer, num_layers=config['transformer']['architecture']['N'], enable_nested_tensor=False )
  else:
    raise ValueError(f"Unsupported encoding type: {config['transformer']['architecture'].get('encoder_type', 'normal')  }")        
    

  #########################################################
  # Create the final block                                #
  #########################################################
  final_type = config['transformer']['architecture'].get('final_type', 'double_linear') 
  if by_pass:
    factor = 4
  else:
    factor = 1  
  final_block = FinalBlock(d_model=config['transformer']['architecture']['d_model'] *factor,
                            seq_len=config['input_length'] , 
                            dropout=config['training']['dropout'], 
                            out_put_size=config['transformer']['architecture'].get('output_size', 1) , 
                            forward_type=final_type)

  # Create the transformer
  if config['transformer']['architecture'].get('encoder_type', 'normal')   == 'vanilla':
    encoder_transformer = VanillaEncoderTransformer(encoders=encoders,
                                                    src_embed=src_embed[0],
                                                    src_pos= src_pos[0], 
                                                    final_block=final_block,
                                                    residual_type=config['transformer']['architecture'].get('residual_type', 'post_ln'),
                                                    normalization=config['transformer']['architecture'].get('normalization', 'layer')  ,
                                                    d_model=config['transformer']['architecture']['d_model'],
                                                    batch_size=config['training']['batch_size'],
                                                    data_type=config['transformer']['architecture']['data_type']
                                                    )
  else:
    encoder_transformer = EncoderTransformer(d_model=config['transformer']['architecture']['d_model'],
                                            batch_size=config['training']['batch_size'],
                                            encoders=encoders,
                                            src_embed=src_embed,
                                            src_pos= src_pos, 
                                            final_block=final_block,
                                            encoding_type=config['transformer']['architecture'].get('encoder_type', 'normal')  ,
                                            residual_type=config['transformer']['architecture'].get('residual_type', 'post_ln'),
                                            normalization=config['transformer']['architecture'].get('normalization', 'layer')  ,
                                            data_type=config['transformer']['architecture']['data_type'])

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
    
def get_last_model(model_path):
  files = os.listdir(model_path)
  epochs = [file_name.split('_')[-1].split('.')[0] for file_name in files]
  new_epochs = []
  for epoch in epochs:
      try:
          new_epochs.append(int(epoch))
      except:
          pass  
  epochs = new_epochs    
  last_epoch = max(epochs)

  return last_epoch

def get_last_model_chunked(model_path):
  files = os.listdir(model_path)
  epochs = [file_name.split('_')[-1].split('.')[0] for file_name in files]
  new_epochs = []
  for epoch in epochs:
      try:
          new_epochs.append(int(epoch))
      except:
          pass  
  epochs = new_epochs    
  last_epoch = max(epochs)

  return last_epoch

def get_state(config, text):
  model_path = dd.get_model_path(config, text=f'{text}')
  state = torch.load(model_path)
  return state

def load_model(config, text='last', verbose=False):
  """
    Load model from config file
    
    Args:
        config (dict): The configuration dictionary. Should be loaded from a yaml file and should be the
                        same as the one used to train the model and also be the model config and not the 
                        training config.
        text (str, optional): The text to add to the model name. Defaults to 'last'.
                              Normaly ones specifies the specific epoch, as a sting,
                                one wants to load.
        verbose (bool, optional): If True, print several informations during evaluation. Defaults to False.
    Returns:
            model

  """


  model = build_encoder_transformer(config)
  model_dict = model.state_dict()
  if text == 'last':
    try:
      transformer_model_path = dd.get_value(config['transformer'], 'model_path') + 'saved_model'
      last_epoch = get_last_model(transformer_model_path)
    except:
      transformer_model_path = dd.get_value(config['transformer'], 'model_path') 
      
      if dd.get_value(config['transformer'], 'data_type') == 'chunked':
        last_epoch = 'last'
      else:
        last_epoch = get_last_model(transformer_model_path)

    
    
    text = last_epoch

  if type(text) == str:
    try:
      text = int(text)
    except ValueError:
      pass

  if type(text) == int:
    text = "{:03d}".format(text)

  model_path = dd.get_model_path(config['transformer'], text=f'_{text}.pth')

  print(f'Preloading model {model_path}') #if verbose else None
  state = torch.load(model_path)

  if '/mnt/md0/halin/Models/' not in config['transformer']['basic']['model_path']  :
    try:
      state = torch.load(model_path, map_location=torch.device("cpu"))
      model.load_state_dict(state, strict=True)
      model.multiplys = get_FLOPs(model, config, verbose=verbose)
      model.adds =  0
      return model
    except:
      pass
  try:
    state_dict = state['model_state_dict']
  except:
    state_dict = state  

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

  
  new_state_dict = {}
  nr_ofs = len(model_dict.items())
  nr_of_state_items = len(state_dict.items())

  if verbose:
    print(f"{'Model: keys':<90} {'Model: value shape':<20} {'Loaded: keys':<90} {'Loaded: value shape':<20}")
    


    for (k1, v1), (k2, v2) in zip_longest(model_dict.items(), state_dict.items(), fillvalue=('', None)):
        v1_shape = str(v1.shape) if v1 is not None else ''
        v2_shape = str(v2.shape) if v2 is not None else ''
        print(f"{k1:<90} {v1_shape:<25} {k2:<90} {v2_shape:<25}")
    print()
  
  for (k1, v1) in model_dict.items():
      nr_of_model_items = nr_ofs
      # if 'embeddings_table' in k1:
      #   continue

      best_match = 0
      count_states = 0
      for (k2, v2) in state_dict.items():
          count_states += 1
          if k1 == k2:
              best_match = 1
              best_k2 = k2
              best_v2 = v2
              nr_of_model_items = count_states
              break
             
          matching_count = count_matching_chars_from_right(k1, k2)
          if v1.shape != v2.shape:
              matching_count = 0
          if matching_count > best_match:
              best_match = matching_count
              best_k2 = k2
              best_v2 = v2

          


          if best_match == 0 and k1.split('.')[-1] == 'pe':
              if verbose:
                print(f"Warning: {k1} did not match any state dict keys")
              for (k2, v2) in state_dict.items():
                matching_count = count_matching_chars_from_right(k1, k2)
                if matching_count > best_match:
                  best_match = matching_count
                  best_k2 = k2
                  length_v2 = v2.shape[1]
                  length_v1 = v1.shape[1]
                  zeros = torch.zeros(v2.shape[0], length_v1 - length_v2, v2.shape[2]).to(v2.device)
                  v2 = torch.cat((v2, zeros), dim=1)

                  best_v2 = v2
      if count_states != nr_of_model_items:
        if verbose:
          print(f"Warning! The number of state items is not equal to the number of model items")            

      if verbose:
        print(f"{k1:<90} {str(v1.shape):<25} {best_k2:<90} {str(best_v2.shape):<25} {best_match:<5}")
      new_state_dict[k1] = best_v2

  # Now load the new state dict
      
  model.load_state_dict(new_state_dict, strict=False)
  model.multiplys = get_FLOPs(model, config, verbose=verbose)
  model.adds =  0
 
  return model

def get_FLOPs(model, config, verbose=False):
  """
    Get the number of FLOPs for the model. Note the model needs to have the modules named
    as: InputEmbeddings, PositionalEncoding, ResidualConnection, MultiHeadAttentionBlock,
    FeedForwardBlock, FinalBlock. The model also needs to have the configuration dictionary
    as an argument.
    Args:
        model (nn.Module): The model.
        config (dict): The configuration dictionary.
    Returns:
        int: The number of FLOPs.
  """

  batch_size = 1
  seq_len = config['input_length']
  d_model = config['transformer']['architecture']['d_model']
  d_ff = config['transformer']['architecture']['d_ff']
  n_ant = config['n_ant']
  max_relative_position = config['transformer']['architecture']['max_relative_position']
  max_pool = config['transformer']['architecture'].get('max_pool', False)
  n_heads = config['transformer']['architecture']['h']
  embed_type = config['transformer']['architecture']['embed_type']
  d_h = d_model // n_heads
  out_put_size = 1
  flops = 0

  seq_len_updated = False

  for name, module in model.named_modules():
    ## Input embeddings
    if isinstance(module, InputEmbeddings):
       input_flops = 0
       print(f"Is a input embedding: {name}") if verbose else None

       if isinstance(module.embedding, torch.nn.Linear):

        input_flops +=  module.embedding.in_features * module.embedding.out_features  * seq_len # for the multiplication
        input_flops += module.embedding.out_features *seq_len # for the bias addition

       elif isinstance(module.embedding, CnnInputEmbeddings): 

          cnn1 = module.embedding.kernel_size * 1 * module.embedding.channels * module.embedding.out_put_size * seq_len / module.embedding.stride
          cnn2 = module.embedding.kernel_size * module.embedding.stride * module.embedding.out_put_size * module.embedding.channels * seq_len / module.embedding.stride
          input_flops += 2 * (int(cnn1) + int(cnn2))

       elif isinstance(module.embedding, ViTEmbeddings):
            
          if seq_len % module.embedding.height != 0:
            total_padding = (module.embedding.kernel_size - seq_len % module.embedding.kernel_size) % module.embedding.kernel_size
          else:
            total_padding = 0

          new_seq_len = (seq_len + total_padding) // module.embedding.kernel_size

          if not seq_len_updated:
            seq_len = new_seq_len
            seq_len_updated = True

          ViT = ( 1 * new_seq_len ) * (module.embedding.kernel_size * module.embedding.kernel_size) * module.embedding.out_put_size
          input_flops += 2 * ViT 

       if module.dropout.p > 0:
        input_flops += 2 * batch_size * seq_len * d_model

       if isinstance(module.activation, torch.nn.ReLU) or isinstance(module.activation, torch.nn.GELU):
          input_flops += batch_size * seq_len * d_model

       print(f"Input embedding flops: {input_flops}") if verbose else None
       flops += input_flops   



    ## Positional encoding      
    elif isinstance(module, PositionalEncoding):
       print(f"Is a positional encoding: {name}") if verbose else None
       pos_flops = batch_size * seq_len * d_model
       print(f"Positional encoding flops: {pos_flops}") if verbose else None

       flops +=  pos_flops

    ## Residual connection
    elif isinstance(module, ResidualConnection):
      if max_pool and embed_type != 'linear':
         seq = seq_len // 2
      else:
         seq = seq_len  

      print(f"Is a residual connection: {name}") if verbose else None

      flops += batch_size * seq * d_model

      if module.dropout.p > 0:
        flops += 2 * batch_size * seq * d_model

      if isinstance(module.norm, BatchNormalization) or isinstance(module.norm, LayerNormalization):
        flops += 4 * batch_size * seq * d_model

    ## MultiHeadAttentionBlock
    elif isinstance(module, MultiHeadAttentionBlock):
      print(f"Is a multi head attention block: {name}") if verbose else None
      if max_pool and embed_type != 'linear':
         seq = seq_len // 2
      else:
         seq = seq_len   

      MHA_flops = 0

      if module.projection_type == 'linear':

        MHA_flops = 4 * seq*d_model**2 + 2*seq**2*d_model



      elif module.projection_type == 'cnn':

        W_q_k_v = seq * d_model * module.W_q.kernel_size[0] * module.W_q.kernel_size[1] * module.W_q.out_channels
        W_0 = seq * d_model * module.W_0.kernel_size[0] * module.W_0.kernel_size[1] * module.W_0.out_channels
        MHA_flops = W_0 + W_q_k_v


      if module.positional_encoding == True:
         
         relative_attention_1 = batch_size * seq * seq * n_heads * d_h

         additative_1 = batch_size * n_heads * seq * seq
         
         relative_attention_2 = 2 * batch_size  * n_heads * d_h * seq

         additative_2 = batch_size * n_heads * d_h * seq
         
         MHA_flops += relative_attention_1 + relative_attention_2 + additative_1 + additative_2

      if module.dropout.p > 0:
        MHA_flops += 2 * batch_size * seq * d_model

      print(f"Multi head attention block flops: {MHA_flops}") if verbose else None
      flops += MHA_flops

    elif isinstance(module, MultiheadAttention):
      if max_pool and embed_type != 'linear':
         seq = seq_len // 2
      else:
         seq = seq_len

      MHA_flops = 4 * seq*d_model**2 + 2*seq**2*d_model

      if module.dropout > 0:
        MHA_flops += 2 * batch_size * seq * d_model

      print(f"Multi head attention block flops: {MHA_flops}") if verbose else None
      flops += MHA_flops

    ## FeedForwardBlock
    elif isinstance(module, FeedForwardBlock):

      if max_pool and embed_type != 'linear':
         seq = seq_len // 2
      else:
          seq = seq_len   

      print(f"Is a feed forward block: {name}") if verbose else None
      ffb_flops = 0
      # FLOPs for wieghts a
      ffb_flops += batch_size * seq * d_model * d_ff +  batch_size * seq * d_ff * d_model


      if isinstance(module.activation, torch.nn.ReLU) or isinstance(module.activation, torch.nn.GELU):
        ffb_flops += batch_size * seq * d_ff

      print(f"Feed forward block flops: {ffb_flops}") if verbose else None
      flops += ffb_flops  

    ## FinalBlock
    elif isinstance(module, FinalBlock):
        if max_pool and embed_type != 'linear':
         seq = seq_len // 2
        else:
          seq = seq_len
        print(f"Is a final block: {name}") if verbose else None

        if module.forward_type == module.double_linear_forward:
            flops += 2 * batch_size * d_model * out_put_size + out_put_size  # for the first linear transformation
            flops += 2 * batch_size * seq * out_put_size + out_put_size  # for the second linear transformation

        elif module.forward_type == module.single_linear_forward:
            flops += 2 * batch_size * (d_model * seq) * out_put_size + out_put_size

        elif module.forward_type == module.average_forward:
            if module.dim == 2:  # seq_average_linear
                flops += batch_size * seq  # for the average operation
                flops += 2 * batch_size * seq * out_put_size + out_put_size  # for the linear transformation
            else:  # d_model_average_linear
                flops += batch_size * d_model  # for the average operation
                flops += 2 * batch_size * d_model * out_put_size + out_put_size  # for the linear transformation


    elif isinstance(module, LayerNormalization):
      print(f"Is a layer normalization: {name}") if verbose else None
      # Perform calculations for LayerNormalization
      flops += 4 * batch_size * seq_len * d_model

    else:
      print(f"Is a unknown block: {name} the type is: {type(module)}") if verbose else None
      # Perform calculations for unknown block

  print(f"FLOPs: {flops}") if verbose else None
  return flops

def count_parameters(model, verbose=False):
  """ Originaly from Copilot AI
  Counts the number of parameters in a model and prints the result.
  Args:
    model: The model to count the parameters of.
    verbose: Whether to print the number of parameters or not.
    
    Returns:
        dict: A dictionary containing the following keys:
            - 'total_param': The total number of parameters in the model.
            - 'total_trainable_param': The total number of trainable parameters in the model.
            - 'encoder_param': The number of parameters in the encoder.
            - 'src_embed_param': The number of parameters in the source embedding.
            - 'final_param': The number of parameters in the final layer.
            - 'buf_param': The number of parameters in the buffer.

    Example:
      results = count_parameters(model)\n
      param_1 = results['total_param']\n
      param_2 = results['total_trainable_param']\n
      param_3 = results['encoder_param']\n
      param_4 = results['src_embed_param']\n
      param_5 = results['final_param']\n
      param_6 = results['buf_param']\n

    """
  
  total_param = 0
  total_trainable_param = 0  
  encoder_param = 0
  src_embed_param = 0
  final_param = 0
  buf_param = 0 
  for name, param in model.named_parameters():
    if param.requires_grad:
      if verbose:
        print(f"Trainable layer: {name} | Size: {param.size()} | Number of Parameters: {param.numel()}")
      total_trainable_param += param.numel()
           
    else:
      if verbose:
        print(f"Non-trainable layer: {name} | Size: {param.size()} | Number of Parameters: {param.numel()}")

    total_param += param.numel() 
    split_names = name.split('.') 
    if 'encoder' in name or (split_names[1] == '0' and split_names[2] in ['2', '3']):
      encoder_param += param.numel()
    elif 'src_embed' in name  or (split_names[1] == '0' and split_names[2] in ['0', '1']):
       src_embed_param += param.numel()
    elif 'final' in name or split_names[1] == '1':
      final_param += param.numel() 


  for name, buf in model.named_buffers():
    if verbose:
      print(f"Buffer: {name} | Size: {buf.size()} | Number of Parameters: {buf.numel()}")
    total_param += buf.numel()
    buf_param += buf.numel()
  if verbose:
    print(f'\nTotal Encoder Parameters: {encoder_param}')
    print(f'\nTotal src_embed Parameters: {src_embed_param}')
    print(f'\nTotal final Parameters: {final_param}')
    print(f'\nTotal Buffer Parameters: {buf_param}')  
    print(f'\nTotal Trainable Number of Parameters: {total_trainable_param}')
    print(f'\nTotal Number of Parameters: {total_param}')

  return {
        'total_param': total_param,
        'total_trainable_param': total_trainable_param,
        'encoder_param': encoder_param,
        'src_embed_param': src_embed_param,
        'final_param': final_param,
        'buf_param': buf_param
    }

     