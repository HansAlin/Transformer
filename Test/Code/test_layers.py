import os
from os.path import dirname, abspath, join
import sys
import unittest

# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, '..', ''))
sys.path.append(CODE_DIR)

from models.models_1 import TimeInputEmbeddings, LayerNormalization, FinalBinaryBlock, MultiHeadAttentionBlock, LearnablePositionalEncoding
from dataHandler.datahandler import get_data, prepare_data
import torch



path = os.getcwd()

data_path = path + '/Test/data/test_100_data.npy'
x_train, x_test, x_val, y_train, y_val, y_test = get_data(path=data_path)

train_loader, val_loader, test_loader, number_of_noise, number_of_signals = prepare_data(x_train, x_val, x_test, y_train, y_val, y_test, 32)

input_data = None
for i in train_loader:
    input_data = i[0]
    break

time_input_model = TimeInputEmbeddings(d_model=512)
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
