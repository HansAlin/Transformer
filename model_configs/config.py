import yaml


config_1 = {
    "data_locations": {
        "detector": "gen2",
        "unbiased_noise": {
            "prod": "prod_2023.02.09"
        },
        "noiseless_signal": {
            "prod": "prod_2023.05.30"
        },
        "veff": {
            "subfolder": "prod_2024.04.22/CDF_0.7"
        },
        "tag_folder_veff": "CDF_0.7"
    },
    "input_length": 256,
    "n_ant": 4,
    "sampling": {
        "band": {
            "high": 0.23,
            "low": 0.08
        },
        "filter": {
            "order_high": 10,
            "order_low": 5,
            "type": "butter"
        },
        "rate": 0.5
    },
    "training": {
        "batch_size": 1024,
        "cdf": 0.7,
        "consecutive_deadtime": 2,
        "extra_gap_per_waveform": 200,
        "forward_skip_size": 199,
        "learning_rate": 0.001,
        "step_size": 7,
        "dropout": 0,
        "decreas_factor": 0.7,
        "warm_up": False,
        "loss_fn": "BCEWithLogits",
        "num_epochs": 100,
        "metric": "Efficiency",
        "early_stop": 100,
        "ns_readout_window": 800,
        "permute_for_RNN_input": False,
        "probabilistic_sampling_ensure_min_signal_fraction": 0.3,
        "probabilistic_sampling_ensure_min_signal_num_bins": 3,
        "probabilistic_sampling_ensure_signal_region": True,
        "probabilistic_sampling_oversampling": 1.0,
        "randomize_batchitem_start_position": -1,
        "set_spurious_signal_to_zero": 100,
        "shift_signal_region_away_from_boundaries": True,
        "start_frac": 0.3,
        "test_frac": 0.1,
        "trigger_time": 200,
        "upsampling": 2,
        "val_frac": 0.1
    },
    "transformer": {
      'architecture': {
          'GSA': False,
          'N': 2,
          'activation': 'relu',
          'antenna_type': 'LPDA',
          'by_pass': False,
          'd_ff': 64,
          'd_model': 32,  # Size of the model, needs to be a multiple of h
          'data_type': 'trigger',
          'embed_type': 'cnn',  # Options: 'cnn', 'ViT', 'linear'
          'encoder_type': 'vanilla',  # Options: 'vanilla' which implements the predefined multihead attention, 'normal' which implements everything from scratch
          'final_type': 'd_model_average_linear',
          'h': 2,
          'inherit_model': None,
          'input_embeddings': {
              'kernel_size': 4,  # For embed_type 'cnn' or 'ViT' When using b4 antennas following options, at least works:
              'stride': 4,  # 'cnn' kernel_size: 3, stride: 1,2 'ViT' kernel_size: 2,4, stride: 2,4
          },
          'max_pool': False,  # Only option if embed_type is 'cnn' or 'ViT'
          'max_relative_position': 32,  # Only option if pos_enc_type is 'Relative'
          'normalization': 'layer',
          'omega': 10000,
          'output_size': 1,
          'pos_enc_type': 'Sinusoidal',  # Options: Sinusoidal, Relative, Learnable, None
          'pre_def_dot_product': False,
          'pretrained': None,
          'projection_type': 'linear',
          'residual_type': 'pre_ln',
      },
      'basic': {
          'model_num': 0,  # Number of the model 
          'model_path': None,
          'model_type': 'base_encoder',
      },
      'num of parameters': {
          'FLOPS': 0,  # Values that updates in pre-trigger training Hans script
          'num_param': 0,
      },
  }
}


base_config = {

    "input_length": 100,
    "n_ant": 6,

    "training": {
        "batch_size": 128,
        "learning_rate": 0.001,
        "step_size": 7,
        "dropout": 0,
        "decreas_factor": 0.7,
        "warm_up": False,
        "loss_fn": "BCEWithLogits",
        "num_epochs": 100,
        "metric": "Efficiency",
        "early_stop": 100,
        "test_frac": 0.1,
        "val_frac": 0.1
    },
    "transformer": {
      'architecture': {
          'GSA': False,
          'N': 2,
          'activation': 'relu',
          'antenna_type': 'LPDA',
          'by_pass': False,
          'd_ff': 32,
          'd_model': 16,  # Size of the model, needs to be a multiple of h
          'data_type': 'any',
          'embed_type': 'linear',  # Options: 'cnn', 'ViT', 'linear'
          'encoder_type': 'vanilla',  # Options: 'vanilla' which implements the predefined multihead attention, 'normal' which implements everything from scratch
          'final_type': 'd_model_average_linear',
          'h': 2,
          'inherit_model': None,
          'input_embeddings': {
              'kernel_size': 4,  # For embed_type 'cnn' or 'ViT' When using b4 antennas following options, at least works:
              'stride': 4,  # 'cnn' kernel_size: 3, stride: 1,2 'ViT' kernel_size: 2,4, stride: 2,4
          },
          'max_pool': False,  # Only option if embed_type is 'cnn' or 'ViT'
          'max_relative_position': 32,  # Only option if pos_enc_type is 'Relative'
          'normalization': 'layer',
          'omega': 10000,
          'output_size': 1,
          'pos_enc_type': 'Sinusoidal',  # Options: Sinusoidal, Relative, Learnable, None
          'pre_def_dot_product': False,
          'pretrained': None,
          'projection_type': 'linear',
          'residual_type': 'pre_ln',
      },
      'basic': {
          'model_num': 0,  # Number of the model 
          'model_path': None,
          'model_type': 'base_encoder',
      },
      'num of parameters': {
          'FLOPS': 0,  # Values that updates in pre-trigger training Hans script
          'num_param': 0,
      },
  }
}


def get_config(num):
    
    if num == 1:
        config = config_1
        return config
    elif num == 2:
        config = base_config
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

        # Pre-trigger
        if config['transformer']['architecture']['data_type'] == 'trigger': 
          
          config['data_locations']['pre_trig_signal'] = {}



          # LPDA data
          if config['transformer']['architecture']['antenna_type'] == 'LPDA':
            config['data_locations']['detector'] = 'gen2'
            config['data_locations']['high_low_noise'] = {}
            config['data_locations']['high_low_noise']['prod'] = 'prod_2023.03.24'
            config['data_locations']['pre_trig_signal']['prod'] = 'prod_2023.05.16'  
            config['data_locations']['detector'] = 'gen2'

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
            config['data_locations']['detector'] = 'rnog'

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
            config['data_locations']['detector'] = 'gen2'

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
            config['data_locations']['detector'] = 'rnog'

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
    

