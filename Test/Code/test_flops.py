import sys
import torch
import pickle



CODE_DIR_1  = '/home/hansalin/Code/Transformer'
sys.path.append(CODE_DIR_1)

from models.models import build_encoder_transformer

MODEL_PATH = '/home/hansalin/Code/Transformer/Test/ModelsResults/model_13/saved_model/model_13.pth'
CONFIG_PATH = '/home/hansalin/Code/Transformer/Test/ModelsResults/model_13/config.txt'
with open(CONFIG_PATH, 'rb') as f:
    config = pickle.load(f)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")           
print(f"Using device: {device}")
  
model = build_encoder_transformer(config).to(device)
if device.type == 'cpu':
  state = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
else:  
  state = torch.load(MODEL_PATH).to(device)
out = model.load_state_dict(state['model_state_dict'])
print(out)



