import unittest
from models.models_1 import TimeInputEmbeddings, LayerNormalization
from dataHandler.datahandler import get_data, prepare_data
import torch


data_path = '/home/halin/Master/Transformer/Test/data/mini_test_data.npy'
x_train, x_test, y_train, y_test = get_data(path=data_path, test=True)

train_loader, test_loader = prepare_data(x_train, x_test, y_train, y_test, 32)

input_data = None
for i in train_loader:
    input_data = i[0]
    break

output = None


class TestReturnDimenstions(unittest.TestCase):
    def tesTimeEmbedding(self):
        timeinput = TimeInputEmbeddings(d_model=512, embed_dim=1)
        output = timeinput(input_data)
        print(f"Shape input: {input_data.shape}")
        batch_size = output.size(dim=0)
        seq_len = output.size(dim=1)
        d_model = output.size(dim=2)
        print(f"Shape output: {output.shape}")
        self.assertEqual([batch_size, seq_len, d_model], [32 , 100, 512], "Incorrect size!")

    def testLayerNormalization(self):
        normal_layer = LayerNormalization()
        input_data = torch.tensor([32,100,512], dtype=torch.float64)
        print(f"Shape input: {input_data.shape}")
        output = normal_layer(input_data)
        batch_size = output.size(dim=0)
        seq_len = output.size(dim=1)
        d_model = output.size(dim=2)
        print(f"Shape output: {output.shape}")
        self.assertEqual([batch_size, seq_len, d_model], [32 , 100, 512], "Incorrect size!")

     

unittest.main()    