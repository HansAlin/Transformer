import sys
import torch
import pickle
import os
from torchinfo import summary



from os.path import dirname, abspath, join
myhost = os.uname()[1]
print("Host name: ", myhost)

model_num = 13
if myhost == "phy-RD-DL1":
   CODE_DIR_1  = '/home/halin/Master/Transformer'
   MODEL_PATH = f'/mnt/md0/halin/Models/model_13/saved_model/model_{model_num}.pth'
   CONFIG_PATH = f'/mnt/md0/halin/Models/model_{model_num}/config.txt'

else:
  CODE_DIR_1  = '/home/hansalin/Code/Transformer'
  MODEL_PATH = f'/home/hansalin/Code/Transformer/Test/ModelsResults/model_{model_num}/saved_model/model_{model_num}.pth'
  CONFIG_PATH = f'/home/hansalin/Code/Transformer/Test/ModelsResults/model_{model_num}/config.txt'   

THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, '..', ''))
sys.path.append(CODE_DIR)

sys.path.append(CODE_DIR_1)
type(sys.path)
import torch.nn as nn
from models.models import build_encoder_transformer


with open(CONFIG_PATH, 'rb') as f:
    config = pickle.load(f)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")           
print(f"Using device: {device}")
  
model = build_encoder_transformer(config).to(device)
if device.type == 'cpu':
  state = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
else:  
  state = torch.load(MODEL_PATH)
out = model.load_state_dict(state['model_state_dict'])
print(out)
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # Replace with the actual parameters your model needs
        mask = None
        return self.model.encoder(x, src_mask=None)

wrapped_model = model.encoder
summary(wrapped_model, input_size=(config['batch_size'], config['seq_len'], config['n_ant']), device=device.type)




