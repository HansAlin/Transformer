import torch
import math

batch_size = 1
seq_lens = [32,64,128,256,512]
d_models = [32,64,128,256,512,1024,2048]


for seq_len in seq_lens:
    for d_model in d_models:
        tensor1 = torch.rand(batch_size, seq_len, d_model)
        tensor2 = tensor1.transpose(2, 1)
        result = torch.matmul(tensor1, tensor2)
        max_value = torch.max(result)
        divided_value = max_value/d_model
        print(f"Seq_len: {seq_len}, D_model: {d_model}, Max_value: {max_value}, Divided_value: {divided_value}")


