import torch 
import torch.nn as nn
import math
from typing import List


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
  def _init__(self, eps: float = 10**-6, d_model: int = 512) -> None:
    super().__init__()
    self.eps = eps
    self.batch_norm = nn.BatchNorm1d(d_model, eps=eps)

  def forward(self, x):
    #   
    return self.batch_norm(x)
  

class FeedForwardBlock(nn.Module):

  def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation='relu'):
    # d_ff is the hidden layer size
    super().__init__()
    self.linear_1 = nn.Linear(d_model, d_ff) # W1 and bias
    self.dropout = nn.Dropout(dropout)
    self.linear_2 = nn.Linear(d_ff, d_model) # W2 and bias
    if activation == None:
      self.activation_1 = nn .ReLU()
      self.activation_2 = nn.ReLU()
    elif activation == 'relu':
      self.activation_1 = nn.ReLU() 
      self.activation_2 = nn.ReLU() 
    elif activation == 'gelu':
      self.activation_1 = nn.GELU()
      self.activation_2 = nn.GELU()  

  def forward(self, x):
    # (batch_size, seq_len, d_model) 
    x = self.linear_1(x)
    # (batch_size, seq_len, d_ff) 
    x = self.activation_1(x)
    x = self.dropout(x)
    x = self.linear_2(x)

    # TODO mvts_transformer does not have this activation function
    # and it does not seem so many others have it either
    # x = self.activation_2(x)
    
    # (batch_size, seq_len, d_model)
    return x


class InputEmbeddings(nn.Module):
  """This layer maps feature space from the input to the d_model space.

  
  Args:
      d_model (int): The dimension of the model.
      channels (int, optional): The number of channels in the input. Defaults to 1.
      dropout (float, optional): The dropout rate. Defaults to 0.1.
      activation (str, optional): The activation function. Defaults to 'relu'.
  
  """
  def __init__(self, d_model: int, dropout: float = 0.1, channels: int = 4, activation='relu'):
    super().__init__()
    self.d_model = d_model
    self.embedding = nn.Linear(channels, d_model)
    self.dropout = nn.Dropout(dropout)
    
    activations = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'none': nn.Identity(),  
        }
    try:
      self.activation = activations[activation]
    except KeyError:
      raise KeyError(f"{activation} is not a valid activation function. Choose between {activations.keys()}")  


  def forward(self, x):
    # (batch_size, seq_len, channels)
    x = self.embedding(x)
    x = self.activation(x)
    x = self.dropout(x)
    # --> (batch_size, seq_len, d_model)
    return x 

class PositionalEncoding(nn.Module):
   
  def __init__(self, d_model: int, dropout: float, seq_len: int, omega: float) -> None:
    super().__init__()
    self.d_model = d_model
    self.dropout = nn.Dropout(p=dropout)
    self.seq_len = seq_len
    self.omega = omega
    
    # Create a matrix of shape (seq_len, d_model) 
    # In this case the sequence length is 100 and the model dimension is 512
    # and the seq_len is originaly the max length of the sentence that can
    # be considered in the model. In this case it is eseentially the max length
    # of the data that we have, in this case 100.
    pe = torch.zeros(seq_len, d_model)
    # Create a vector of shape (seq_len, 1)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
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
  def __init__(self, d_model):
    super().__init__()
    self.d_model = d_model
    self.positional_encoding = nn.Parameter(torch.zeros(1, d_model))

  def forward(self, x):
    x += self.positional_encoding
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
  def __init__(self, d_model: int, seq_len: int, dropout: float = 0.1,  out_put_size: int = 1, forward_type='basic'):
    super().__init__()
    if forward_type == 'basic':
      self.linear_1 = nn.Linear(d_model, out_put_size)
      self.dropout = nn.Dropout(dropout)
      self.linear_2 = nn.Linear(seq_len, out_put_size)
      self.activation = nn.Sigmoid()
      self.forward_type = self.basic_forward
    elif forward_type == 'slim':
      self.linear_1 = nn.Linear(d_model*seq_len, out_put_size)  
      self.forward_type = self.slim_forward
    elif forward_type == 'maxpool':  
      self.maxpool = nn.AdaptiveAvgPool2d((1, 1))
      self.forward_type = self.maxpool_forward
    else:
      raise ValueError(f"Unsupported forward type: {forward_type}")  
 
  def basic_forward(self, x):
    #(batch_size, seq_len, d_model)
    x = self.linear_1(x)
    x = x.squeeze()
    x = self.dropout(x)
    x = self.linear_2(x) 
    x = x.squeeze()
    x = self.activation(x)
    x = self.dropout(x)
    # (batch_size)

    return x

  def slim_forward(self, x):
    # (batch_size, seq_len, d_model)
    x = x.view(x.shape[0], -1)
    # (batch_size, seq_len*d_model)
    x = self.linear_1(x)
    x = x.squeeze()
    # (batch_size, out_put_size)
    return x
  
  def maxpool_forward(self, x):
    # (batch_size, seq_len, d_model)
    x = self.maxpool(x)
    # (batch_size, 1,)
    x = x.squeeze()
    # (batch_size)
    return

  def forward(self, x):
    x = self.forward_type(x)
    return x

class MultiHeadAttentionBlock(nn.Module):
  def __init__(self, d_model: int, h: int, dropout: float = 0.1, max_relative_position = 10, relative_positional_encoding: bool = False):
    super().__init__()
    self.d_model = d_model
    self.h = h
    
    assert d_model % h == 0, "d_model must be divisible by h"
    self.d_h = d_model // h

    # TODO its an option to have biases or not in the linear layers
    self.W_q = nn.Linear(d_model, d_model) # W_q and bias
    self.W_k = nn.Linear(d_model, d_model) # W_k and bias
    self.W_v = nn.Linear(d_model, d_model) # W_v and bias

    self.W_0 = nn.Linear(self.h*self.d_h, d_model) # W_0 and bias wher self.h*self.d_h = d_model
    self.dropout = nn.Dropout(dropout)

    # For relative positional encoding
    self.max_relative_position = max_relative_position
    self.relative_positional_k = RelativePositionalEncoding(self.d_h, max_relative_position)
    self.relative_positional_v = RelativePositionalEncoding(self.d_h, max_relative_position)
    self.scale = math.sqrt(self.d_model)
    self.relative_positional_encoding = relative_positional_encoding


  @staticmethod # Be able to call this method without instantiating the class
  def attention(query, key, value, mask, dropout : nn.Dropout):
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
    query =self.W_q(q) # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
    key = self.W_k(k) # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
    value = self.W_v(v) # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)

    # Split the query, into h heads
    # query.shape[0] = batch_size
    # query.shape[1] = seq_len
    # (batch_size, seq_len, d_model) --> (batch_size, seq_len, h, d_h) -->
    # transpose(1,2) --> (batch_size, h, seq_len, d_h)
    # transpose(1,2) swaps the seq_len and h dimensions dimeinstion 1 and 2


    if self.relative_positional_encoding:
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

      x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)  
    # Concatenate the heads
    # (batch_size, h, seq_len, d_h) --> (batch_size, seq_len, h, d_h) -->
    #  (batch_size, seq_len, h*d_h) = (batch_size, seq_len, d_model)
    x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h*self.d_h)

    # Apply the last linear layer
    # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model) 
    x = self.W_0(x)
    return x
  
class ResidualConnection(nn.Module):
  """ This layer adds the residual connection from sublayer to the input.
      It also appies a 
  """
  def __init__(self, features: int, dropout: float = 0.1, normalization='layer'):
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    if normalization == 'layer':
      self.norm = LayerNormalization(features=features)
    else:
      self.norm = BatchNormalization()  
    

  def forward(self, x, sublayer):
    # There are other ways to add the residual connection
    # For example, you can add it sublayer(x) and then apply the normalization.

    # (batch_size, seq_len, d_model)

    # This is how mvts_transformer does it
    out = sublayer(x)
    x = x + self.dropout(out)
    x = self.norm(x)

    # This is how Jamil does it
    # out = self.norm(x)
    # out = sublayer(out)
    # out = self.dropout(out)
    # x = x + out

    # (batch_size , seq_len, d_model)
    return x
  
class EncoderBlock(nn.Module):
  def __init__(self, 
               features: int,
               self_attention_block: MultiHeadAttentionBlock,
               feed_forward_block: FeedForwardBlock, 
               dropout: float = 0.1,
               normalization='layer'):
    super().__init__()
    self.self_attention_block = self_attention_block
    self.feed_forward_block =  feed_forward_block
    self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout, normalization) for _ in range(2)])

    # self.residual_1 = ResidualConnection(dropout)
    # self.residual_2 = ResidualConnection(dropout)

  def forward(self, x, src_mask):
    # x = self.residual_1(x, lambda x: self.self_attention_block(x, x, x, src_mask))
    # Is this the same as:
    # y = self.attention_block(x, x, x, src_mask)
    # x = self.residual_1(x, y) ?
    # x = self.residual_2(x, self.feed_forward_block)

    #  TODO I don't understand the differnece between the two residual connections
    x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
    x = self.residual_connections[1](x, self.feed_forward_block)
    return x  
  
class Encoder(nn.Module):
  def __init__(self, features: int, layers: nn.ModuleList, normalization='layer') -> None:
    super().__init__()
    self.layers = layers
    if normalization == 'layer':
      self.norm = LayerNormalization(features=features)
    else:
      self.norm = BatchNormalization()  
    

  def forward(self, x, mask=None):
    for layer in self.layers:
      x = layer(x, mask)

    return self.norm(x)  



# class ProjectionLayer(nn.Module):
#   def __init__(self, d_model: int, vocab_size: int) -> None:
#     super().__init__()
#     self.proj = nn.Linear(d_model, vocab_size)     
        
#   def forward(self, x):
#     # (Batch, seq_len, d_model) --> (Batch, seq_len, vocab_size)
#     return torch.log_softmax(self.proj(x), dim=-1)
  

  
class EncoderTransformer(nn.Module):
  def __init__(self, encoders: List[Encoder], 
               src_embed: List[InputEmbeddings], 
               src_pos: List[PositionalEncoding],
               final_block: FinalBlock
               ) -> None:
    super().__init__()

    if len(encoders) > 1:
      for i, encoder in enumerate(encoders):
        setattr(self, f'encoder_{i+1}', encoder)
        setattr(self, f'src_embed_{i+1}', src_embed[i])
        setattr(self, f'src_pos_{i+1}', src_pos[i])
      self.final_block = final_block
      self.encode_type = self.bypass_encode
    else:
      self.encoder = encoders[0]
      self.src_embed = src_embed[0]
      self.src_pos = src_pos[0]
      self.final_block = final_block
      self.encode_type = self.non_bypass_encode


  def non_bypass_encode(self, src, src_mask=None):
    # (batch_size, seq_len, d_model)
    src = self.src_embed(src)

    src = self.src_pos(src)
    
    src = self.encoder(src, src_mask)

    src = self.final_block(src)
    return src
  
  def bypass_encode(self, src, src_mask=None):
    # (batch_size, seq_len, d_model)
    src_slices = src.split(1, dim=-1)
    # (batch_size, seq_len, 1) x n_ant
    src_embeds = [getattr(self, f'src_embed_{i+1}')(src_slice) for i, src_slice in enumerate(src_slices)]
    src_poss = [getattr(self, f'src_pos_{i+1}')(src_embed) for i, src_embed in enumerate(src_embeds)]
    src_encoders = [getattr(self, f'encoder_{i+1}')(src_pos, src_mask) for i, src_pos in enumerate(src_poss)]


    src = torch.cat(src_encoders, dim=-1)

    src = self.final_block(src)

    return src

  def encode(self, src, src_mask=None):
    return self.encode_type(src, src_mask)
  

 

def build_encoder_transformer(config) -> EncoderTransformer:

  seq_len=config['seq_len']
  d_model=config['d_model']
  N=config['N']
  h=config['h']
  dropout=config['dropout']
  omega=config['omega']
  d_ff=config['d_ff']
  output_size = config.get('output_size', 1)
  by_pass = config.get('by_pass', False)
  
  #########################################################
  # Create the input embeddings                           #
  #########################################################    
  embed_config = {
      'relu_drop': ('relu', dropout),
      'gelu_drop': ('gelu', dropout),
      'basic': ('none', 0)
  }

  embed_type = config.get('embed_type', 'relu_drop')

  if embed_type in embed_config:
      activation, emb_drop = embed_config[embed_type]
  else:
      raise ValueError(f"Unsupported Embedding type: {embed_type}")

  
  if by_pass:
    num_embeddings = config['n_ant']
    channels = 1
  else:
    channels = config['n_ant']
    num_embeddings = 1  
 
  src_embed = [InputEmbeddings(d_model=d_model, channels=channels, dropout=emb_drop, activation=activation) for _ in range(num_embeddings)]
  
  #########################################################
  # Create the positional encodings                       #
  pos_enc_config = {
      'Sinusoidal': lambda: PositionalEncoding(d_model=d_model, dropout=dropout, seq_len=seq_len, omega=omega),
      'None': lambda: nn.Identity(),
      'Relative': lambda: nn.Identity(),
      'Learnable': lambda: LearnablePositionalEncoding(d_model=d_model)
  }

  pos_enc_type = config['pos_enc_type']

  if pos_enc_type in pos_enc_config:
      src_pos_func = pos_enc_config[pos_enc_type]
  else:
      raise ValueError(f"Unsupported positional encoding type: {pos_enc_type}")

  num_embeddings = config['n_ant'] if by_pass else 1

  src_pos = [src_pos_func() for _ in range(num_embeddings)]

  #########################################################
  # Create the encoder layers                             #
  #########################################################
  num_encoders = config['n_ant'] if by_pass else 1

  encoders = []
  
  for _ in range(num_encoders):
    encoder_blocks = []
    for _ in range(N):
      if config['pos_enc_type'] == 'Relative':
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout, max_relative_position=100, relative_positional_encoding=True)
      else:
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)

      feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
      encoder_block = EncoderBlock(features=d_model, 
                                   self_attention_block=encoder_self_attention_block, 
                                   feed_forward_block=feed_forward_block, 
                                   dropout=dropout)
      encoder_blocks.append(encoder_block)
    encoders.append(Encoder(features=d_model,
                            layers=nn.ModuleList(encoder_blocks), 
                            normalization='layer'))
    

  #########################################################
  # Create the final block                                #
  #########################################################
  final_type = config.get('final_type', 'basic')
  if by_pass:
    factor = 4
  else:
    factor = 1  
  final_block = FinalBlock(d_model=d_model*factor, seq_len=seq_len, dropout=dropout, out_put_size=output_size, forward_type=final_type)

  # Create the transformer
  encoder_transformer = EncoderTransformer(encoders=encoders,
                                           src_embed=src_embed,
                                           src_pos= src_pos, 
                                           final_block=final_block)

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

# def set_max_split_size_mb(model, max_split_size_mb):
#     """
#     Set the max_split_size_mb parameter in PyTorch to avoid fragmentation.

#     Args:
#         model (torch.nn.Module): The PyTorch model.
#         max_split_size_mb (int): The desired value for max_split_size_mb in megabytes.
#     """
#     for param in model.parameters():
#         param.requires_grad = False  # Disable gradient calculation to prevent unnecessary memory allocations

#     # Dummy forward pass to initialize the memory allocator
#     dummy_input = torch.randn(32, 100,1)
#     model.encoder(dummy_input, mask=None)

#     # Get the current memory allocator state
#     allocator = torch.cuda.memory._get_memory_allocator()

#     # Update max_split_size_mb in the memory allocator
#     allocator.set_max_split_size(max_split_size_mb * 1024 * 1024)

#     for param in model.parameters():
#         param.requires_grad = True  # Re-enable gradient calculation for training

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
        x = x.view(self.batch_size, self.seq_len, self.channels)
        return self.model.encode(x, src_mask=None)