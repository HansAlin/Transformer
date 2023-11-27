import sys
import os
path = os.getcwd()

sys.path.append('/home/hansalin/Code/Transformer/Test/dataHandler')
sys.path.append('/home/hansalin/Code/Transformer/Test/models')

import handler as dh
import models

x_train, x_test, y_train, y_test = dh.get_test_data()
train_loader, test_loader = dh.prepare_data(x_train, x_test, y_train, y_test, 32)

model = models.TransformerModel(d_model=128,nhead=8) 
print()