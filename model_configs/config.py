
config_1 =   config = {'model_name': "Attention is all you need",
            'model_type': "base_encoder",
              'model':None,
              'inherit_model': None, # The model to inherit from
              'embed_type': 'basic', # Posible options: 'relu_drop', 'gelu_drop', 'basic'
              'by_pass': False, # If channels are passed separatly through the model
              'pos_enc_type':'Sinusoidal', # Posible options: 'Sinusoidal', 'Relative', 'None', 'Learnable'
              'final_type': 'slim', # Posible options: 'basic', 'slim'
              'loss_function': 'BCEWithLogits', # Posible options: 'BCE', 'BCEWithLogits'
              'model_num': None,
              'seq_len': 256,
              'd_model': 128, # Have to be dividable by h
              'd_ff': 64,
              'N': 2,
              'h': 16,
              'output_size': 1,
              'dropout': 0.1,
              'num_epochs': None,
              'batch_size': None,
              "learning_rate": 1e-3,
              "decreas_factor": 0.5,
              "num_parms":0,
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
              "data_type": "classic", # Possible options: 'classic', 'chunked'
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
    else:
        return None