import torch
import math



values = [[1,2,1,2],
          [2,4,2,4],
          [2,3,1,2],
          [1,2,1,2]]
tensor1 = torch.tensor(values, dtype=torch.float)
tensor2 = tensor1.transpose(0, 1)
result = torch.matmul(tensor1, tensor2)
softed = torch.softmax(result, dim=-1)

print(result)
print(softed)
print(torch.matmul(softed, tensor1))
