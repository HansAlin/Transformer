
# Other code

class EncoderTransformer(nn.Module):
  def __init__(self, encoders: List[nn.Module], 
               src_embed: List[InputEmbeddings], 
               src_pos: List[PositionalEncoding],
               final_block: FinalBlock,
               encoding_type: str = 'vanilla'
               ) -> None:
    super().__init__()

    if encoding_type == 'vanilla':
      self.encoder = encoders[0]
      self.src_embed = src_embed
      self.src_pos = src_pos[0]
      self.final_block = final_block
      self.encode_type = self.vanilla_encode
    
    else:
      raise ValueError(f"Unsupported encoding type: {encoding_type}")

  def vanilla_encode(self, src, src_mask=None):
    # (batch_size, seq_len, channels)
    src = self.src_embed(src)  # --> (batch_size, seq_len, d_model)
    src = self.src_pos(src) # --> (batch_size, seq_len, d_model)

    src = self.encoder(src)[0] # --> (batch_size, seq_len, d_model)

    src = self.final_block(src)

    return src

  def encode(self, src, src_mask=None):
    return self.encode_type(src, src_mask)
  
class VanillaEncoderBlock(nn.Module):
  def __init__(self, d_model: int, h: int, d_ff: int, dropout: float = 0.1, activation='relu'):
    super().__init__()
    self.self_attention_block = MultiheadAttention(d_model, h, dropout)
    self.feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout, activation)
    self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)])
    self.dropout_1 = nn.Dropout(dropout)
    self.dropout_2 = nn.Dropout(dropout)
    self.norm_1 = LayerNormalization(d_model)
    self.norm_2 = LayerNormalization(d_model)


  def forward(self, x, src_mask, src_key_padding_mask=None):
    # (batch_size, seq_len, d_model)
    # Multi head attention block
    x2 = self.self_attention_block(x, x, x, src_mask, src_key_padding_mask)[0] # --> (batch_size, seq_len, d_model)
    # Residual connection and normalization
    x = x + self.dropout_1(x2) # --> (batch_size, seq_len, d_model)
    x = self.norm_1(x) # --> (batch_size, seq_len, d_model)
    # Feed forward block
    x2 = self.feed_forward_block(x) # --> (batch_size, seq_len, d_model)
    # Residual connection and normalization
    x = x + self.dropout_2(x2) # --> (batch_size, seq_len, d_model)
    x = self.norm_2(x) # --> (batch_size, seq_len, d_model)

    return x  
  

# other code
  
vanilla_layer = VanillaEncoderBlock(d_model, h, d_ff, dropout, activation='relu')
encoders = [nn.TransformerEncoder(vanilla_layer, num_layers=N)]    

encoder_transformer = EncoderTransformer(encoders=encoders,
                                           src_embed=src_embed,
                                           src_pos= src_pos, 
                                           final_block=final_block,
                                           encoding_type=encoding_type)

# other code

