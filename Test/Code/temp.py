import torch
import sys

CODE_DIR_1  ='/home/halin/Master/Transformer/'
sys.path.append(CODE_DIR_1)

import dataHandler.datahandler as dd
import models.models as mm


import os
import re
import torch

if torch.cuda.is_available(): 
    device = torch.device(f'cuda:{2}')
else:
    device = torch.device('cpu')

model_num = 250
config = dd.get_model_config(model_num)

model = mm.load_model(config=config, text='last')
state = mm.get_state(config=config, text='last')
optimizer = torch.optim.Adam(model.parameters(), lr=config['transformer']['training']['learning_rate']) #
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['transformer']['training']['step_size'], gamma=config['transformer']['training']['decreas_factor']) #

warmup = config['transformer']['training'].get('warm_up', False)

if warmup:
    lr = config['transformer']['training']['learning_rate']
    warmup_epochs = 10
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs 
        else:
            return 1
        
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


scheduler.load_state_dict(state['scheduler_state_dict'])
initial_epoch = config['transformer']['results']['current_epoch'] + 1
optimizer.load_state_dict(state['optimizer_state_dict'])
# Move optimizer state to the same device
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)

print(optimizer.state_dict())
print(scheduler.state_dict())


