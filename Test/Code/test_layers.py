import os
from os.path import dirname, abspath, join
import sys
import unittest
import numpy as np
# Find code directory relative to our directory
CODE_DIR_1  ='/home/halin/Master/Transformer'
sys.path.append(CODE_DIR_1)
type(sys.path)
for path in sys.path:
   print(path)
from models.models import TimeInputEmbeddings, LayerNormalization, FinalBinaryBlock, MultiHeadAttentionBlock, LearnablePositionalEncoding
from dataHandler.datahandler import get_data, prepare_data
import torch

# PATH = '/home/halin/Master/Transformer/Test/data/test_100_data.npy'

# x_train, x_val, x_test, y_train, y_val, y_test = get_data(path=PATH)

# train_loader, val_loader, test_loader = prepare_data(x_train, x_val, x_test, y_train, y_val, y_test, batch_size=32, multi_channel=True)    

train_loader, val_loader, test_loader = get_data(batch_size=32, seq_len=256, subset=True)

input_data = None
for i in train_loader:
    input_data = i[0]
    break

time_input_model = TimeInputEmbeddings(d_model=512, channels=4 )
print(f"Shape input time embedding: {input_data.shape}")
output = time_input_model(input_data)
print(f"Shape output time embedding: {output.shape}")

learn_pos_layer = LearnablePositionalEncoding(d_model=512)
print(f"Shape input learnable position: {output.shape}")
output = learn_pos_layer(output)
print(f"Shape output learnable position: {output.shape}")

multi_head = MultiHeadAttentionBlock(d_model=512, h=8, dropout=0.1, max_relative_position=10, relative_positional_encoding=True)
print(f"Shape input to multi head: {output.shape}")
output = multi_head(output, output, output, mask=None)
print(f"Shape output from multi head: {output.shape}")

final_layer = FinalBinaryBlock(d_model=512, seq_len=100)
final_output = final_layer(output)
print(f"Shape input: {final_output.shape}")
