import os
from os.path import dirname, abspath, join
import sys
import unittest

# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, '..', ''))
sys.path.append(CODE_DIR)

from models.models_1 import TimeInputEmbeddings, LayerNormalization, FinalBinaryBlock
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
print(f"Shape input: {input_data.shape}")
output = time_input_model(input_data)
print(f"Shape output: {output.shape}")

final_layer = FinalBinaryBlock(d_model=512, seq_len=100)
final_output = final_layer(output)
print(f"Shape input: {final_output.shape}")
