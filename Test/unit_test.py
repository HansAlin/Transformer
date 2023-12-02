import unittest
from models.models_1 import TimeInputEmbeddings
from dataHandler.datahandler import get_data, prepare_data


data_path = '/home/halin/Master/Transformer/Test/data/mini_test_data.npy'
x_train, x_test, y_train, y_test = get_data(path=data_path, test=True)

train_loader, test_loader = prepare_data(x_train, x_test, y_train, y_test, 32)

input_data = None
for i in train_loader:
    input_data = i[0]
    break



class TestReturnDimenstions(unittest.TestCase):
    def runBatchTest(self):
        timeinput = TimeInputEmbeddings(d_model=512, embed_dim=1)
        output = timeinput(input_data)
        batch_size = output.size(dim=0)
        seq_len = output.size(dim=1)
        d_model = output.size(dim=2)
        print(batch_size)
        self.assertEqual(batch_size, 200, "Incorrect size!")

     

unittest.main()    