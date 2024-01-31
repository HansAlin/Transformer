import yaml
# History: 
# Date        model
# 2024-01-15: 24    'final_type' is 'maxpool' instead of 'slim'
# 2024-01-16: 24 changed 'num_parms' to 'num_param'
# 2024-01-16: 24 added 'encoder_type' which makes it possible skip the encoder
# 2024-01-18: 25 removed 'bypass' that is now incorporated in 'encoder_type'
# 2024-01-18: 25 changed final_type to avergae_pool
# 2024-01-18: Changed old values to new values on models
# 2024-01-18: Changed seq_len to 128

config_1 = {'model_name': "Attention is all you need",
            'model_type': "base_encoder",
              'model':None,
              'inherit_model': None, # The model to inherit from
              'encoder_type': 'normal', # Posible options: 'normal', 'none', 'bypass', 'vanilla'
              'embed_type': 'linear', # Posible options: 'lin_relu_drop', 'lin_gelu_drop', 'linear', 
              'pos_enc_type':'Sinusoidal', # Posible options: 'Sinusoidal', 'Relative',  'Learnable''None',
              'final_type':  'd_model_average_linear', # Posible options: 'double_linear', 'single_linear', 'seq_average_linear', 'd_model_average_linear'
              'normalization': 'layer', # Posible options: 'layer', 'batch'
              'loss_function': 'BCEWithLogits', # Posible options: 'BCE', 'BCEWithLogits'
              'activation': 'relu', # Posible options: 'relu', 'gelu'
              'model_num': None,
              'seq_len': 128,
              'd_model': 64, # Have to be dividable by h
              'd_ff': 32,
              'N': 2,
              'h': 2,
              'output_size': 1,
              'dropout': 0.1,
              'num_epochs': None,
              'batch_size': 64,
              "learning_rate": 1e-3,
              "decreas_factor": 0.5,
              "num_param":0,
              'MACs':0,
              "data_path":'',
              "current_epoch":0,
              "global_epoch":0,
              "model_path":'',
              "test_acc":0,
              "early_stop":7,
              "omega": 10000,
              "trained_noise":0,
              "trained_signal":0,
              "data_type": "trigger", # Possible options: 'trigger', 'chunked'
              "n_ant":4,
              "metric":'Efficiency', # Posible options: 'Accuracy', 'Efficiency', 'Precision'
              "Accuracy":0,
              "Efficiency":0,
              "Precission":0,
              'nr_area':0,
              'roc_area':0,
              "trained": False,
              'power':0,
              'training_time':0,
              'energy':0,
         
            }

def get_config(num):
    
    if num == 1:
        config = config_1
        return config
    elif num == 0:
        with open('/home/halin/Master/Transformer/model_configs/config.yaml', 'r') as file:
          config = yaml.safe_load(file)
        return config
    

