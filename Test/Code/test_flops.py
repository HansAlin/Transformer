import sys
from ptflops import get_model_complexity_info
CODE_DIR_1  ='/home/halin/Master/Transformer/'
sys.path.append(CODE_DIR_1)
import models.models as mm    
import evaluate.evaluate as ee 
import dataHandler.datahandler as dd


models = range(335,336)
save = True

for model_num in models:
    try:
        config = dd.get_model_config(model_num=model_num)
        batch_size = dd.get_value(config, 'batch_size')
        seq_len = dd.get_value(config, 'seq_len')
        n_ant = dd.get_value(config, 'n_ant')


        model = mm.build_encoder_transformer(config)


        flops = mm.get_FLOPs(model, config)
        parameters = mm.get_n_params(model)
        if dd.get_value(config, 'data_type') == 'chunked':
            in_put = (n_ant, seq_len)
        else:
            in_put = (seq_len, n_ant)

        #out = get_model_complexity_info(model, in_put, as_strings=True, print_per_layer_stat=False, verbose=False)

        if 'FLOPs' in config['transformer']['num of parameters']:
            oldFlops = config['transformer']['num of parameters']['FLOPs']
        else:
            oldFlops = 0  
        if 'num_param' in config['transformer']['num of parameters']:
            oldParams = config['transformer']['num of parameters']['num_param']
        else:
            oldParams = 0    

        print(f"Model number: {model_num}, New Flops: {flops/1e6:>7.2f} M, New Parameters: {parameters/1e3:>7.2f} k, Old Flops: {oldFlops/1e6:>7.2f} M, Old Parameters: {oldParams/1e3:>7.2f} k") 


        
        config['transformer']['num of parameters']['num_param'] = parameters
        config['transformer']['num of parameters']['FLOPs'] = flops
        if save:
            dd.save_data(config)
    except Exception as e:
        print(f"Model number: {model_num}, Error: {e}")
        pass




