import torch 
import torch.nn as nn
import math



class LayerNormalization(nn.Module):
  def __init__(self, eps: float = 10**-6) -> None:
    super().__init__()
    self.eps = eps
    self.alpha = nn.Parameter(torch.ones(1))
    self.bias = nn.Parameter(torch.zeros(1))
    # TODO might send in features instaed of just set size
    # to 1
  

  def forward(self, x):
    # keepdim=True will keep the mean and std dimension
    # same as the input tensor.
    # TODO dont understand why my costum norm does not work
    # x: (b, seq_len, d_model)
    mean = x.mean(dim = -1, keepdim=True) # (b, seq_len, 1)
    std = x.std(dim = -1, keepdim=True) # (b, seq_len, 1)
    x = self.alpha * (x - mean) / (std + self.eps) + self.bias

    # (B, seq_len, d_model)
    return x

class FeedForwardBlock(nn.Module):

  def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
    # d_ff is the hidden layer size
    super().__init__()
    self.linear_1 = nn.Linear(d_model, d_ff) # W1 and bias
    self.dropout = nn.Dropout(dropout)
    self.linear_2 = nn.Linear(d_ff, d_model) # W2 and bias

  def forward(self, x):
    # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_ff) --> (batch_size, seq_len, d_model)
    return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class InputEmbeddings(nn.Module):

  def __init__(self, d_model: int, vocab_size: int) -> None:
      super().__init__()
      self.d_model = d_model
      self.vocab_size = vocab_size
      self.embedding = nn.Embedding(vocab_size, d_model)

  def forward(self, x):
      # (batch, seq_len) --> (batch, seq_len, d_model)
      # Multiply by sqrt(d_model) to scale the embeddings according to the paper
      return self.embedding(x) * math.sqrt(self.d_model) 
  
class TimeInputEmbeddings(nn.Module):
  def __init__(self, d_model: int, dropout: float = 0.1):
    super().__init__()
    self.d_model = d_model
    self.embedding = nn.Linear(1, d_model)
    self.dropout = nn.Dropout(dropout)
    self.activation = nn.ReLU()

  def forward(self, x):
    # (B, seq_len, 1)
    x = self.embedding(x)
    x = self.activation(x)
    x = self.dropout(x)
    # --> (B, seq_len, d_model)
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
    # Initialize the weights
    nn.init.xavier_uniform_(self.embeddings_table)

  def forward(self, length_q, length_k):
    range_vec_q = torch.arange(length_q)
    range_vec_k = torch.arange(length_k)
    distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
    distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
    final_mat = distance_mat_clipped + self.max_relative_position
    final_mat = torch.LongTensor(final_mat)
    embeddings = self.embeddings_table[final_mat]

    return embeddings

   
class FinalBinaryBlock(nn.Module):
  def __init__(self, d_model: int, seq_len: int, dropout: float = 0.1):
    super().__init__()
    self.linear_1 = nn.Linear(d_model, 1)
    self.sigmoid = nn.Sigmoid()
    self.dropout = nn.Dropout(dropout)
    self.linear_2 = nn.Linear(seq_len, 1)
    self.activation = nn.ReLU()


  def forward(self, x):
    #(Batch, seq_len, d_model) --> ()

    x = self.linear_1(x)
    x = x.squeeze()
    x = self.dropout(x)
    x = self.linear_2(x) 
    x = self.sigmoid(x)

    return x

class FinalMultiBlock(nn.Module):
  def __init__(self, d_model: int, seq_len: int, dropout: float = 0.1):
    super().__init__()
    self.linear_1 = nn.Linear(d_model, 1)
    self.activation = nn.ReLU()

  def forward(self, x):
    #(Batch, seq_len, d_model) --> ()
    x = self.linear_1(x)
    x = self.activation(x)
    x = x.squeeze()
    return x


class MultiHeadAttentionBlock(nn.Module):
  def __init__(self, d_model: int, h: int, dropout: float = 0.1, max_relative_position = 10, relative_positional_encoding: bool = False):
    super().__init__()
    self.d_model = d_model
    self.h = h
    
    assert d_model % h == 0, "d_model must be divisible by h"
    self.d_h = d_model // h

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
  def attention_with_relative_position(query, key, value, h, relative_position_k, relative_position_v, scale, mask, dropout : nn.Dropout):
    # Should be
    #query = [batch size, query len, hid dim]
    #key = [batch size, key len, hid dim]
    #value = [batch size, value len, hid dim]
    len_k = key.shape[1]
    len_q = query.shape[1]
    len_v = value.shape[1]
    batch_size = query.shape[0]
    d_h = query.shape[-1]

    normal_attention_scores = (query @ key.transpose(-2, -1)) # (batch_size, seq_len, seq_len)

    relative_q = query.permute(1, 0, 2).contiguous().view(len_q, batch_size*h, -1) # (seq_len, batch_size*h, d_h)
    relative_k = relative_position_k(len_q, len_k) # (seq_len, seq_len, d_h)
    relative_attention_scores = torch.matmul(relative_q, relative_k.transpose(1, 2)).transpose(0, 1)
    relative_attention_scores = relative_attention_scores.contiguous().view(batch_size, h, len_q, len_k)

    attention_scores = (normal_attention_scores + relative_attention_scores) / scale

    # Apply the mask
    if mask is not None:
      attention_scores.masked_fill(mask == 0, -1e9)

    # Apply the dropout
    if dropout is not None:
      attention_scores = dropout(attention_scores)

    relative_v_1 = value.view(batch_size, -1, h, d_h).permute(0, 2, 1, 3)
    weight1 = (attention_scores @ relative_v_1)
    relative_v_2 = relative_position_v(len_q, len_v)
    weight2 = attention_scores.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size*h, len_k)
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

    query = query.view(query.shape[0], query.shape[1], self.h, self.d_h).transpose(1,2) # (batch_size, seq_len, d_model) --> (batch_size, h, seq_len, d_h)
    key = key.view(key.shape[0], key.shape[1], self.h, self.d_h).transpose(1,2) # (batch_size, seq_len, d_model) --> (batch_size, h, seq_len, d_h)
    value = value.view(value.shape[0], value.shape[1], self.h, self.d_h).transpose(1,2) # (batch_size, seq_len, d_model) --> (batch_size, h, seq_len, d_h)

    if self.relative_positional_encoding:
      x, self.attention_scores = MultiHeadAttentionBlock.attention_with_relative_position(query, 
                                                                                          key, 
                                                                                          value, 
                                                                                          self.h, 
                                                                                          self.relative_positional_k, 
                                                                                          self.relative_positional_v, 
                                                                                          self.scale, 
                                                                                          mask, 
                                                                                          self.dropout)
    else:


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
  def __init__(self, dropout: float = 0.1):
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.norm = LayerNormalization()

  def forward(self, x, sublayer):
    # There are other ways to add the residual connection
    # For example, you can add it sublayer(x) and then apply the normalization
    # but this is the way Jamil did it.
    # out = self.norm(x)
    # out = sublayer(out)
    # out = self.dropout(out)
    # # (b , seq_len, d_model)
    return x + self.dropout(sublayer(self.norm(x)))
  
class EncoderBlock(nn.Module):
  def __init__(self, 
               self_attention_block: MultiHeadAttentionBlock,
               feed_forward_block: FeedForwardBlock, 
               dropout: float = 0.1):
    super().__init__()
    self.self_attention_block = self_attention_block
    self.feed_forward_block =  feed_forward_block
    self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

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
  def __init__(self, layers: nn.ModuleList ) -> None:
    super().__init__()
    self.layers = layers
    self.norm = LayerNormalization()

  def forward(self, x, mask=None):
    for layer in self.layers:
      x = layer(x, mask)

    return self.norm(x)  



class DecoderBlock(nn.Module):
  def __init__(self, 
               self_attention_block: MultiHeadAttentionBlock,
               cross_attention_block: MultiHeadAttentionBlock,
               feed_forward_block: FeedForwardBlock,
               dropout: float = 0.1):  
    super().__init__()
    self.self_attention_block = self_attention_block
    self.cross_attention_block = cross_attention_block
    self.feed_forward_block = feed_forward_block
    self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

  def forward(self, x, encoder_output, src_mask, tgt_mask):
    x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
    x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
    x = self.residual_connections[2](x, self.feed_forward_block)

    return x
  
class Decoder(nn.Module):
  def __init__(self, layers: nn.ModuleList) -> None:
    super().__init__()
    self.layers = layers
    self.norm = LayerNormalization()

  def forward(self, x, encoder_output, src_mask, tgt_mask):
    for layer in self.layers:
      x = layer(x, encoder_output, src_mask, tgt_mask)

    return self.norm(x)

class ProjectionLayer(nn.Module):
  def __init__(self, d_model: int, vocab_size: int) -> None:
    super().__init__()
    self.proj = nn.Linear(d_model, vocab_size)     
        
  def forward(self, x):
    # (Batch, seq_len, d_model) --> (Batch, seq_len, vocab_size)
    return torch.log_softmax(self.proj(x), dim=-1)
  
class Transformer(nn.Module):
  def __init__(self, encoder: Encoder, 
               decoder: Decoder, 
               src_embed: InputEmbeddings, 
               tgt_embed: InputEmbeddings,
               src_pos: PositionalEncoding,
               tgt_pos: PositionalEncoding,
               projeciton_layer: ProjectionLayer) -> None:
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.src_embed = src_embed
    self.tgt_embed = tgt_embed
    self.src_pos = src_pos
    self.tgt_pos = tgt_pos
    self.projection_layer = projeciton_layer

  def encode(self, src, src_mask=None):
    src = self.src_embed(src)
    src = self.src_pos(src)
    return self.encoder(src, src_mask)

  def decode(self, encoder_output, src_mask, tgt, tgt_mask):
    tgt = self.tgt_embed(tgt)
    tgt = self.tgt_pos(tgt)
    return self.decoder(tgt, encoder_output, src_mask, tgt_mask)  
  
  def project(self, x):
    return self.projection_layer(x)
  
class EncoderTransformer(nn.Module):
  def __init__(self, encoder: Encoder, 
               src_embed: TimeInputEmbeddings, 
               src_pos: PositionalEncoding,
               final_block: FinalBinaryBlock) -> None:
    super().__init__()
    self.encoder = encoder
    self.src_embed = src_embed
    self.src_pos = src_pos
    self.final_block = final_block


  def encode(self, src, src_mask=None):
    # (b, seq_len, d_model)
    src = self.src_embed(src)

    if self.src_pos != None:
      # (b,seq_len,1)
      src = self.src_pos(src)
    
   
    # (b,1) !!!!! TODO does not seem right
    src = self.encoder(src, src_mask)

    src = self.final_block(src)
    return src
  

def build_transformer(src_vocab_size: int, 
                      tgt_vocab_size: int, 
                      src_seq_len: int,
                      tgt_seq_len: int,
                      d_model: int = 512,
                      N: int = 6,
                      h: int = 8, 
                      dropout: float = 0.1,
                      d_ff: int = 2048) -> Transformer:
  # Create the input embeddings
  src_embed = InputEmbeddings(d_model, src_vocab_size)
  tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

  # Create the positional encodings
  src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
  tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

  # Create the encoder layers
  encoder_blocks = []
  for _ in range(N):
    encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
    feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
    encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
    encoder_blocks.append(encoder_block)

  # Create the decoder blocks
  decoder_blocks = []
  for _ in range(N):
    decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
    decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
    feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
    decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
    decoder_blocks.append(decoder_block)  

  # Create the encoder and decoder
  encoder = Encoder(nn.ModuleList(encoder_blocks))
  decoder= Decoder(nn.ModuleList(decoder_blocks))

  # Create the projection layer
  projection_layer = ProjectionLayer(d_model, tgt_vocab_size)   

  # Create the transformer
  transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

  # Initialize the parameters
  for p in transformer.parameters():
    if p.dim() > 1:
      nn.init.xavier_uniform_(p)  

  return transformer    

def build_encoder_transformer(config,
                              embed_size: int, 
                              seq_len: int,
                              d_model: int = 512,
                              N: int = 6,
                              h: int = 8, 
                              dropout: float = 0.1,
                              d_ff: int = 2048,
                              omega: float = 10000) -> EncoderTransformer:
  # Create the input embeddings
  src_embed = TimeInputEmbeddings(d_model=d_model)
 
  # Create the positional encodings
  if config['pos_enc_type'] == 'normal':
    src_pos = PositionalEncoding(d_model=d_model, dropout=dropout, seq_len=seq_len, omega=omega)
  elif config['pos_enc_type'] == 'none' or config['pos_enc_type'] == 'relative':
    src_pos = None  

  
    


  # Create the encoder layers
  encoder_blocks = []
  for _ in range(N):
    if config['pos_enc_type'] == 'relative':
      encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout, max_relative_position=100, relative_positional_encoding=True)
    encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
    feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
    encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
    encoder_blocks.append(encoder_block)

  # Create the encoder and decoder
  encoder = Encoder(nn.ModuleList(encoder_blocks))

  final_block = FinalBinaryBlock(d_model=d_model, seq_len=seq_len, dropout=dropout)

  # Create the transformer
  encoder_transformer = EncoderTransformer(encoder=encoder,
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

def set_max_split_size_mb(model, max_split_size_mb):
    """
    Set the max_split_size_mb parameter in PyTorch to avoid fragmentation.

    Args:
        model (torch.nn.Module): The PyTorch model.
        max_split_size_mb (int): The desired value for max_split_size_mb in megabytes.
    """
    for param in model.parameters():
        param.requires_grad = False  # Disable gradient calculation to prevent unnecessary memory allocations

    # Dummy forward pass to initialize the memory allocator
    dummy_input = torch.randn(32, 100,1)
    model.encoder(dummy_input, mask=None)

    # Get the current memory allocator state
    allocator = torch.cuda.memory._get_memory_allocator()

    # Update max_split_size_mb in the memory allocator
    allocator.set_max_split_size(max_split_size_mb * 1024 * 1024)

    for param in model.parameters():
        param.requires_grad = True  # Re-enable gradient calculation for training
