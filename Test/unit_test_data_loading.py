import sys
import os
import torch
import unittest
import numpy as np
import yaml

current_dir = os.getcwd()
print(f"Current directory: {current_dir}")
sys.path.append(current_dir)

import dataHandler.datahandler as dd
import model_configs.config as mc
import models.models as mm
import plots.plots as pp


class TriggerPhasedDataTest(unittest.TestCase):

    def test_data_loading(self):


        config = mc.get_config(0)    
        config['sampling'] = {}
        config['sampling']['filter'] = {}
        config['sampling']['band'] = {}
        config['data_locations'] = {}
        config['data_locations']['pre_trig_signal'] = {}
        config['data_locations']['tag_folder_veff'] = 'CDF_0.7'

        config['input_length'] = config['transformer']['architecture']['seq_len'] = 128
        config['n_ant'] = config['transformer']['architecture']['n_ant']
        config['training']['batch_size'] = config['transformer']['training']['batch_size'] = 256
        config['training']['learning_rate'] = config['transformer']['training']['learning_rate']
        config['training']['loss_fn'] = config['transformer']['training']['loss_function']
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

        config['transformer']['architecture']['data_type'] = 'trigger'
        config['transformer']['architecture']['antenna_type'] = 'phased'
    
        config['transformer']['architecture']['n_ant'] = 5
        config['n_ant'] = 5

        config['transformer']['architecture']['embed_type'] = 'linear'
    
    
        train_data, val_data, test_data = dd.get_trigger_data(config, subset=True)
        del train_data, val_data
    
        x, y = next(iter(test_data))
        count = 0
        while max(y) == 0 and count < 10:
            x, y = next(iter(test_data))
            count += 1
        del test_data
        output_shape = x.shape
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        n_ant = x.shape[2]
    
        pp.plot_examples(x, y, save_path=f'/home/halin/Master/Transformer/figures/examples/phased_seq_len_{seq_len}_examples.png', 
                         data_type = config['transformer']['architecture']['data_type'],
                         antenna_type=config['transformer']['architecture']['antenna_type'])

        model = mm.build_encoder_transformer(config)
        model.eval()
        with torch.no_grad():
            out = model(x)
            print(f"Model output: {out.shape}")
            has_nan = np.isnan(out.cpu().numpy()).any()
    
        

        print(f"Data shape: {output_shape}, label shape: {y.shape}, output shape: {out.shape}")

        self.assertTrue(out.shape[0] == config['transformer']['training']['batch_size'], "Output shape is not 1")
        self.assertFalse(has_nan, "Model output contains NaN values")
        self.assertEqual(n_ant, config['transformer']['architecture']['n_ant'])
        self.assertEqual(seq_len, config['transformer']['architecture']['seq_len'])
        self.assertEqual(batch_size, config['transformer']['training']['batch_size'])

        print(f"Data shape: {output_shape}")
  
class TriggerLPDADataTest(unittest.TestCase):

    def test_data_loading(self):
        # # trigger
        
        

        config = mc.get_config(0)   

                # get the correct data files
        config['sampling'] = {}
        config['sampling']['filter'] = {}
        config['sampling']['band'] = {}
        config['data_locations'] = {}
        config['data_locations']['tag_folder_veff'] = 'CDF_0.7'

        config['input_length'] = config['transformer']['architecture']['seq_len'] = 128
        config['n_ant'] = config['transformer']['architecture']['n_ant']
        config['training']['batch_size'] = config['transformer']['training']['batch_size'] = 256
        config['training']['learning_rate'] = config['transformer']['training']['learning_rate']
        config['training']['loss_fn'] = config['transformer']['training']['loss_function']
        config['data_locations']['pre_trig_signal'] = {}
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

        config['transformer']['architecture']['data_type'] = 'trigger'
        config['transformer']['architecture']['antenna_type'] = 'LPDA' 

        config['transformer']['architecture']['n_ant'] = 4
        config['n_ant'] = 4

        train_data, val_data, test_data = dd.get_trigger_data(config, subset=True)
        print(f"Data shape: {len(train_data)}, val shape: {len(val_data)}, test shape: {len(test_data)}")
        del train_data, val_data

        model = mm.build_encoder_transformer(config)
        model.eval()

        
        x, y = next(iter(test_data))
        count = 0
        while max(y) == 0 and count < 10:
            x, y = next(iter(test_data))
            count += 1
        del test_data

        output_shape = x.shape
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        n_ant = x.shape[2]

        pp.plot_examples(x, y, save_path=f'/home/halin/Master/Transformer/figures/examples/phased_seq_len_{seq_len}_examples.png', 
                         data_type = config['transformer']['architecture']['data_type'],
                         antenna_type=config['transformer']['architecture']['antenna_type'])

        with torch.no_grad():
            out = model(x)
            print(f"Model output: {out.shape}")
            has_nan = np.isnan(out.cpu().numpy()).any()
    

        print(f"Data shape: {output_shape}, label shape: {y.shape}, output shape: {out.shape}")

        config_ant = dd.get_value(config, 'n_ant')
        config_seq_len = dd.get_value(config, 'seq_len')
        config_batch_size = dd.get_value(config, 'batch_size')

        self.assertTrue(out.shape[0] == config['transformer']['training']['batch_size'], "Output shape is not 1")
        self.assertFalse(has_nan, "Model output contains NaN values")
        self.assertEqual(n_ant, config_ant, "Number of antennas is not correct")
        self.assertEqual(seq_len, config_seq_len, "Sequence length is not correct")
        self.assertEqual(batch_size, config_batch_size, "Batch size is not correct")

        print(f"Data shape: {output_shape}")
        
        
class ChunkedLPDADataTest(unittest.TestCase):

    def test_data_loading(self):

        config = mc.get_config(0)   
        # get the correct data files
        config['sampling'] = {}
        config['sampling']['filter'] = {}
        config['sampling']['band'] = {}
        config['data_locations'] = {}
        config['data_locations']['tag_folder_veff'] = 'CDF_0.7'

        config['input_length'] = config['transformer']['architecture']['seq_len'] = 256
        config['n_ant'] = config['transformer']['architecture']['n_ant']
        config['training']['batch_size'] = config['transformer']['training']['batch_size']
        config['training']['learning_rate'] = config['transformer']['training']['learning_rate']
        config['training']['loss_fn'] = config['transformer']['training']['loss_function']
        config['data_locations']['noiseless_signal'] = {}
        config['data_locations']['unbiased_noise'] = {}

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

        config['transformer']['architecture']['data_type'] = 'chunked'
        config['transformer']['architecture']['antenna_type'] = 'LPDA'


        batch_size = 256
        config['transformer']['architecture']['n_ant'] = 4
        config['transformer']['architecture']['seq_len'] = batch_size
        config['transformer']['training']['batch_size'] = batch_size

        config['n_ant'] = config['transformer']['architecture']['n_ant']
        config['input_length'] = config['transformer']['architecture']['seq_len']
        config['training']['batch_size'] = config['transformer']['training']['batch_size']



        train_data, val_data, test_data = dd.get_chunked_data(config, subset=False)
        
        del train_data, val_data    
        
        model = mm.build_encoder_transformer(config)
        model.eval()

        x, y = next(iter(test_data))
        output_shape = x.shape
        batch_size = x.shape[0]
        seq_len = x.shape[2]
        n_ant = x.shape[1]

        pp.plot_examples(x, y, save_path=f'/home/halin/Master/Transformer/figures/examples/phased_seq_len_{seq_len}_examples.png', 
                         data_type = config['transformer']['architecture']['data_type'],
                         antenna_type=config['transformer']['architecture']['antenna_type'])


        with torch.no_grad():
            out = model(x)
            print(f"Model output: {out.shape}")
            has_nan = np.isnan(out.cpu().numpy()).any()
    
        del test_data

        
        print(f"Data shape: {output_shape}, label shape: {y.shape}, output shape: {out.shape}")



        self.assertTrue(out.shape[0] == config['transformer']['training']['batch_size'], "Output shape is not 1")
        self.assertFalse(has_nan, "Model output contains NaN values")
        self.assertEqual(n_ant, config['transformer']['architecture']['n_ant'], "Number of antennas is not correct")
        self.assertEqual(seq_len, config['transformer']['architecture']['seq_len'], "Sequence length is not correct")
        self.assertEqual(batch_size, config['transformer']['training']['batch_size'], "Batch size is not correct")

        print(f"Data shape: {output_shape}")

class ChunkedPhasedDataTest(unittest.TestCase):

    def test_data_loading(self):

        config = mc.get_config(0)    
        # get the correct data files
        config['sampling'] = {}
        config['sampling']['filter'] = {}
        config['sampling']['band'] = {}
        config['data_locations'] = {}
        config['data_locations']['tag_folder_veff'] = 'CDF_0.7'

        config['input_length'] = config['transformer']['architecture']['seq_len'] = 256
        config['n_ant'] = config['transformer']['architecture']['n_ant']
        config['training']['batch_size'] = config['transformer']['training']['batch_size']
        config['training']['learning_rate'] = config['transformer']['training']['learning_rate']
        config['training']['loss_fn'] = config['transformer']['training']['loss_function']
        config['data_locations']['noiseless_signal'] = {}
        config['data_locations']['unbiased_noise'] = {}
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

        config['transformer']['architecture']['data_type'] = 'chunked'
        config['transformer']['architecture']['antenna_type'] = 'phased'


        batch_size = 256
        config['transformer']['architecture']['n_ant'] = 4
        config['transformer']['architecture']['seq_len'] = batch_size
        config['transformer']['training']['batch_size'] = batch_size

        config['n_ant'] = config['transformer']['architecture']['n_ant']
        config['input_length'] = config['transformer']['architecture']['seq_len']
        config['training']['batch_size'] = config['transformer']['training']['batch_size']



        train_data, val_data, test_data = dd.get_chunked_data(config, subset=False
                                                              )
        del train_data, val_data    
        
        model = mm.build_encoder_transformer(config)
        model.eval()

        x, y = next(iter(test_data))
        output_shape = x.shape
        batch_size = x.shape[0]
        seq_len = x.shape[2]
        n_ant = x.shape[1]

        with torch.no_grad():
            out = model(x)
            print(f"Model output: {out.shape}")
            has_nan = np.isnan(out.cpu().numpy()).any()
    
        del test_data

        pp.plot_examples(x, y, save_path=f'/home/halin/Master/Transformer/figures/examples/phased_seq_len_{seq_len}_examples.png', 
                         data_type = config['transformer']['architecture']['data_type'],
                         antenna_type=config['transformer']['architecture']['antenna_type'])
        
        print(f"Data shape: {output_shape}, label shape: {y.shape}, output shape: {out.shape}")

        self.assertTrue(out.shape[0] == config['transformer']['training']['batch_size'], "Output shape is not 1")
        self.assertFalse(has_nan, "Model output contains NaN values")
        self.assertEqual(n_ant, config['transformer']['architecture']['n_ant'], "Number of antennas is not correct")
        self.assertEqual(seq_len, config['transformer']['architecture']['seq_len'], "Sequence length is not correct")
        self.assertEqual(batch_size, config['transformer']['training']['batch_size'], "Batch size is not correct")

        print(f"Data shape: {output_shape}")



if __name__ == '__main__':
    #TriggerPhasedDataTest().test_data_loading()
    #TriggerLPDADataTest().test_data_loading()
    #ChunkedLPDADataTest().test_data_loading()
    ChunkedPhasedDataTest().test_data_loading()
    print("All tests passed")

        