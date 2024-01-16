import os
from os.path import dirname, abspath, join
import sys
import unittest
import numpy as np
# Find code directory relative to our directory
CODE_DIR_1  ='/home/halin/Master/Transformer'
sys.path.append(CODE_DIR_1)

# from models.models import InputEmbeddings, LayerNormalization, FinalBlock, MultiHeadAttentionBlock, LearnablePositionalEncoding
from models.models import InputEmbeddings, LayerNormalization, FinalBlock, MultiHeadAttentionBlock, LearnablePositionalEncoding, build_encoder_transformer
from dataHandler.datahandler import get_data, prepare_data
from model_configs.config import get_config
import torch


# PATH = '/home/halin/Master/Transformer/Test/data/test_100_data.npy'

# x_train, x_val, x_test, y_train, y_val, y_test = get_data(path=PATH)

# train_loader, val_loader, test_loader = prepare_data(x_train, x_val, x_test, y_train, y_val, y_test, batch_size=32, multi_channel=True)    

# train_loader, val_loader, test_loader = get_data(batch_size=32, seq_len=256, subset=True)

# input_data = None
# for i in train_loader:
#     input_data = i[0]
#     break
config = get_config(num=1)

model = build_encoder_transformer(config)
input_data = torch.ones((32, 256, 4))
output = model.encode(input_data, src_mask=None)

# time_input_model = InputEmbeddings(d_model=512, channels=4, dropout=0, activation='none' )
# print(f"Shape input time embedding: {input_data.shape}")
# output = time_input_model(input_data)
# print(f"Shape output time embedding: {output.shape}")

# learn_pos_layer = LearnablePositionalEncoding(d_model=512)
# print(f"Shape input learnable position: {output.shape}")
# output = learn_pos_layer(output)
# print(f"Shape output learnable position: {output.shape}")

# multi_head = MultiHeadAttentionBlock(d_model=512, h=8, dropout=0.1, max_relative_position=10, relative_positional_encoding=True)
# print(f"Shape input to multi head: {output.shape}")
# output = multi_head(output, output, output, mask=None)
# print(f"Shape output from multi head: {output.shape}")

# final_layer = FinalBlock(d_model=512, seq_len=256, out_put_size=1, forward_type='new')
# final_output = final_layer(output)
# print(f"Shape input: {final_output.shape}")
