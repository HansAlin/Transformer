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

        # get the correct data files
        config['sampling'] = {}
        config['sampling']['filter'] = {}
        config['sampling']['band'] = {}
        config['data_locations'] = {}
        config['data_locations']['tag_folder_veff'] = 'CDF_0.7'

        config['input_length'] = config['transformer']['architecture']['seq_len']
        config['n_ant'] = config['transformer']['architecture']['n_ant']
        config['training']['batch_size'] = config['transformer']['training']['batch_size']
        config['training']['learning_rate'] = config['transformer']['training']['learning_rate']
        config['training']['loss_fn'] = config['transformer']['training']['loss_function']



        # Pre-trigger
        if config['transformer']['architecture']['data_type'] == 'trigger': 
          
          config['data_locations']['pre_trig_signal'] = {}



          # LPDA data
          if config['transformer']['architecture']['antenna_type'] == 'LPDA':
            
            config['data_locations']['high_low_noise'] = {}
            config['data_locations']['high_low_noise']['prod'] = 'prod_2023.03.24'
            config['data_locations']['pre_trig_signal']['prod'] = 'prod_2023.05.16'  

            config['sampling']['band']['high'] = 0.23
            config['sampling']['band']['low'] = 0.08
            config['sampling']['rate'] = 0.5
            config['sampling']['filter']['order_high'] = 10
            config['sampling']['filter']['order_low'] = 5
            config['sampling']['filter']['type'] = 'butter'

            config['data_locations']['tag_folder_veff'] = 'CDF_0.7' #veff prod prod_2024.04.22/CDF_0.7

          # Phased array data
          elif config['transformer']['architecture']['antenna_type'] == 'phased': 

            config['data_locations']['phased_array_noise'] = {}     
            config['data_locations']['phased_array_noise']['prod'] = 'prod_2023.06.07'
            config['data_locations']['pre_trig_signal']['prod'] = 'prod_2023.06.12' 

            config['sampling']['band']['high'] = 0.22
            config['sampling']['band']['low'] = 0.096
            config['sampling']['rate'] = 0.5
            config['sampling']['filter']['order_high'] = 7
            config['sampling']['filter']['order_low'] = 4
            config['sampling']['filter']['type'] = 'cheby1' #TODO is this correct
            config['training']['upsampling'] = 2
            config['training']['start_frac'] = 0.5

            config['data_locations']['tag_folder_veff'] = 'CDF_1.0' #veff prod prod_2024.04.19/CDF_1.0

        # Chunked
        elif config['transformer']['architecture']['data_type'] == 'chunked':
          config['data_locations']['noiseless_signal'] = {}
          config['data_locations']['unbiased_noise'] = {}

          # LPDA data
          if config['transformer']['architecture']['antenna_type'] == 'LPDA':

            config['data_locations']['noiseless_signal']['prod'] = 'prod_2023.05.30'
            config['data_locations']['unbiased_noise']['prod'] = 'prod_2023.02.09'

            config['sampling']['band']['high'] = 0.23
            config['sampling']['band']['low'] = 0.08
            config['sampling']['rate'] = 0.5
            config['sampling']['filter']['order_high'] = 10
            config['sampling']['filter']['order_low'] = 5
            config['sampling']['filter']['type'] = 'butter'

            config['data_locations']['tag_folder_veff'] = 'CDF_0.7' #veff prod prod_2024.04.22/CDF_0.7

          # Phased array data
          elif config['transformer']['architecture']['antenna_type'] == 'phased': 

            config['data_locations']['noiseless_signal']['prod'] = 'prod_2024.03.15'
            config['data_locations']['unbiased_noise']['prod'] = 'prod_2023.05.19'

            config['sampling']['band']['high'] = 0.22
            config['sampling']['band']['low'] = 0.096
            config['sampling']['rate'] = 0.5
            config['sampling']['filter']['order_high'] = 7
            config['sampling']['filter']['order_low'] = 4
            config['sampling']['filter']['type'] = 'cheby1' #TODO is this correct
            config['training']['upsampling'] = 2
            config['training']['start_frac'] = 0.5

            config['data_locations']['tag_folder_veff'] = 'CDF_1.0' #veff prod prod_2024.04.19/CDF_1.0

          

        return config
    

