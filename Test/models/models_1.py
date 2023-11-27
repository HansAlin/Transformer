import torch 
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

  def __init__(self, d_model: int, vocab_size: int) -> None:
    super().__init__()
    self.d_model = d_model
    self.vocab_size = vocab_size
    self.embedding = nn.Embedding(vocab_size,d_model)

  def forward(self, x):
      x = self.embedding(x) * math.sqrt(self.d_model)
      return x

class PositionalEncoding(nn.Module):
   
  def __init__(self, d_model: int, dropout: float, seq_len: int) -> None:
    super().__init__()
    self.d_model = d_model
    self.dropout = nn.Dropout(p=dropout)
    self.seq_len = seq_len
    
    # Create a matrix of shape (seq_len, d_model) 
    # In this case the sequence length is 100 and the model dimension is 512
    # and the seq_len is originaly the max length of the sentence that can
    # be considered in the model. In this case it is eseentially the max length
    # of the data that we have, in this case 100.
    pe = torch.zeros(seq_len, d_model)
    # Create a vector of shape (seq_len, 1)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    # The exponential form is used here for stabilty purposes
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 1/10000^(2i/d_model)
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
    return self.dropout(x)


class LayerNormalization(nn.Module):
  def __init__(self, d_model: int, eps: float = 1e-6) -> None:
    super().__init__()
    self.d_model = d_model
    self.eps = eps
    self.alpha = nn.Parameter(torch.ones(1))
    self.bias = nn.Parameter(torch.zeros(0))

  def forward(self, x):
    # keepdim=True will keep the mean and std dimension
    # same as the input tensor.
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True)
    return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):

  def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
    # d_ff is the hidden layer size
    super().__init__()
    self.linear_1 = nn.Linear(d_model, d_ff) # W1 and bias
    self.dropout = nn.Dropout(dropout)
    self.linear_2 = nn.Linear(d_ff, d_model) # W2 and bias

  def forward(self, x):
    # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_ff) --> (batch_size, seq_len, d_model)
    x = self.linear_1(x)
    x = torch.relu(x)
    x = self.dropout(x)
    x = self.linear_2(x)
    return x
  
class MultiHeadAttentionBlock(nn.Module):
  def __init__(self, d_model: int, h: int, dropout: float = 0.1):
    super().__init__()
    self.d_model = d_model
    self.h = h
    assert d_model % h == 0, "d_model must be divisible by h"
    self.d_k = d_model // h

    self.W_q = nn.Linear(d_model, d_model) # W_q and bias
    self.W_k = nn.Linear(d_model, d_model) # W_k and bias
    self.W_v = nn.Linear(d_model, d_model) # W_v and bias

    self.W_0 = nn.Linear(self.h*self.d_k, d_model) # W_0 and bias wher self.h*self.d_k = d_model
    self.dropout = nn.Dropout(dropout)



    self.dropout = nn.Dropout(dropout)

    # Create the query, key, value matrices
    self.query = nn.Linear(d_model, d_model)
    self.key = nn.Linear(d_model, d_model)
    self.value = nn.Linear(d_model, d_model)

    # Create the output layer
    self.output = nn.Linear(d_model, d_model)  

  @staticmethod # Be able to call this method without instantiating the class
  def attention(query, key, value, mask, dropout : nn.Dropout):
    d_k = query.shape[-1] # Get the last dimension of the query matrix

    # Compute the scaled dot product attention
    # key.transpose(-2, -1) swaps the last two dimensions of the key matrix
    # (batch_size, h, seq_len, d_k)  @ (batch_size, h, d_k, seq_len) --> (batch_size, h, seq_len, seq_len)
    attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) 

    # Apply the mask
    if mask is not None:
      attention_scores.masked_fill(mask == 0, -1e9)

    attention_scores = attention_scores.softmax(dim=-1) # (batch_size, h, seq_len, seq_len)

    # Apply the dropout
    if dropout is not None:
      attention_scores = dropout(attention_scores)

    # (batch_size, h, seq_len, seq_len) @ (batch_size, h, seq_len, d_k) --> 
    # (batch_size, h, seq_len, d_k)
    return (attention_scores @ value), attention_scores


  def forward(self, q, k, v, mask):
    query =self.W_q(q) # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
    key = self.W_k(k) # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
    value = self.W_v(v) # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)

    # Split the query, into h heads
    # query.shape[0] = batch_size
    # query.shape[1] = seq_len
    # (batch_size, seq_len, d_model) --> (batch_size, seq_len, h, d_k) -->
    # transpose(1,2) --> (batch_size, h, seq_len, d_k)
    # transpose(1,2) swaps the seq_len and h dimensions dimeinstion 1 and 2
    query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2) # (batch_size, seq_len, d_model) --> (batch_size, h, seq_len, d_k)
    key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2) # (batch_size, seq_len, d_model) --> (batch_size, h, seq_len, d_k)
    value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2) # (batch_size, seq_len, d_model) --> (batch_size, h, seq_len, d_k)

    x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

    # Concatenate the heads
    # (batch_size, h, seq_len, d_k) --> (batch_size, seq_len, h, d_k) -->
    #  (batch_size, seq_len, h*d_k) = (batch_size, seq_len, d_model)
    x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h*self.d_k)

    # Apply the last linear layer
    # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model) 
    x = self.W_0(x)
    return x
  
class ResidualConnection(nn.Module):
  def __init__(self, dropout: float = 0.1):
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.norm = LayerNormalization()

  def forward(self, x, sublayer):
    # There are other ways to add the residual connection
    # For example, you can add it sublayer(x) and then apply the normalization
    out = self.norm(x)
    out = sublayer(out)
    out = self.dropout(out)
    return x + out
  
class EncoderBlock(nn.Module):
  def __init__(self, 
               self_attention_block: MultiHeadAttentionBlock,
               feed_forward_block: FeedForwardBlock, 
               dropout: float = 0.1):
    super().__init__()
    self.attention = MultiHeadAttentionBlock()
    self.residual_1 = ResidualConnection(dropout)
    self.feed_forward = FeedForwardBlock()
    self.residual_2 = ResidualConnection(dropout)

  def forward(self, x, src_mask):
    x = self.residual_1(x, lambda x: self.attention(x, x, x, src_mask))
    x = self.residual_2(x, self.feed_forward)
    return x  