import sys
from ptflops import get_model_complexity_info
CODE_DIR_1  ='/home/halin/Master/Transformer/'
sys.path.append(CODE_DIR_1)
import models.models as mm    
import evaluate.evaluate as ee 
import dataHandler.datahandler as dd



config = dd.get_model_config(model_num=400)
batch_size = dd.get_value(config, 'batch_size')
seq_len = dd.get_value(config, 'seq_len')
n_ant = dd.get_value(config, 'n_ant')


model = mm.build_encoder_transformer(config['transformer'])


flops = mm.get_FLOPs(model, config)
parameters = mm.get_n_params(model)
print(f"FLOPs: {flops}")
print(f"Parameters: {parameters}")
if dd.get_value(config, 'data_type') == 'chunked':
    in_put = (n_ant, seq_len)
else:
    in_put = (seq_len, n_ant)

out = get_model_complexity_info(model, in_put, as_strings=True, print_per_layer_stat=False, verbose=False)
print(out)
config['transformer']['num of parameters']['num_param'] = parameters
config['transformer']['num of parameters']['FLOPs'] = flops

dd.save_data(config)



