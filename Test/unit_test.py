import os
import unittest
import torch
import torch.nn as nn
import sys
import copy
import itertools


CODE_DIR_1  ='/home/halin/Master/Transformer/'
sys.path.append(CODE_DIR_1)
from models.models import LayerNormalization, BatchNormalization, ResidualConnection, MultiHeadAttentionBlock, FeedForwardBlock, EncoderBlock, Encoder, InputEmbeddings, PositionalEncoding, FinalBlock, EncoderTransformer, build_encoder_transformer, get_FLOPs
from dataHandler.datahandler import prepare_data, get_chunked_data, get_trigger_data
from model_configs.config import get_config
import lossFunctions.lossFunctions as ll

def count_parameters(layer):
    return sum(p.numel() for p in layer.parameters() if p.requires_grad)

class BaseTest(unittest.TestCase):
    def __init__(self, methodName='runTest', inputs=None, device=None, test_string=None):
        super().__init__(methodName)
        self.config = inputs
        self.device = device
        self.test_string = test_string
    


class TestLayers(BaseTest):

    def test_LayerNormalization(self):
        #torch.manual_seed(0)
        
        normal_layer = LayerNormalization(features=self.d_model, eps=1e-6)
        input_data = torch.ones(self.batch_size,self.seq_len,self.d_model)
        print(f"Shape input to layerNormalization: {input_data.shape}")
        output = normal_layer(input_data)
        batch_size = output.size(dim=0)
        seq_len = output.size(dim=1)
        d_model = output.size(dim=2)
        print(f"Shape output: {output.shape}")
        self.assertEqual([batch_size, seq_len, d_model], [self.batch_size , self.seq_len, self.d_model], "Incorrect size!")

        input_data = torch.ones(self.batch_size,self.seq_len,self.d_model)

        output = normal_layer(input_data)
        self.assertEqual(output[0,0,0], 0, "Incorrect value!")


    def test_BatchNormalization(self):
        #torch.manual_seed(0)
        
        normal_layer = BatchNormalization(features=self.d_model, eps=1e-6)
        input_data = torch.ones(self.batch_size,self.seq_len,self.d_model)
        print(f"Shape input to BatchNormalization: {input_data.shape}")
        output = normal_layer(input_data)
        batch_size = output.size(dim=0)
        seq_len = output.size(dim=1)
        d_model = output.size(dim=2)
        print(f"Shape output: {output.shape}")
        self.assertEqual([batch_size, seq_len, d_model], [self.batch_size , self.seq_len, self.d_model], "Incorrect size!")

        input_data = torch.ones(self.batch_size,self.seq_len,self.d_model)

        output = normal_layer(input_data)
        self.assertEqual(output[0,0,0], 0, "Incorrect value!")

    def test_residual_connection(self):
        torch.manual_seed(0)
        for normalization in self.normalizations:
            for residual_type in self.residual_types:
                self.helper_residual_connection(normalization, residual_type)

    def helper_residual_connection(self, normalization, residual_type):
        #torch.manual_seed(0)

        sublayer = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        residual_connection = ResidualConnection(features=self.d_model, residual_type=residual_type, normalization=normalization, )
        input_data = torch.ones(self.batch_size, self.seq_len, self.d_model)
        output = residual_connection(input_data, sublayer)
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))

        sublayer.weight.data = torch.ones_like(sublayer.weight.data)
        sublayer.bias.data = torch.zeros_like(sublayer.bias.data)

        # Create some input data
        input_data = torch.ones(self.batch_size, self.seq_len, self.d_model)
        print(f"Normalization: {normalization}, Residual type: {residual_type}")
        output = residual_connection(input_data, sublayer)

        # Check if the output has the same shape as the input
        self.assertEqual(output.shape, input_data.shape)

        # Check if the output is not all zeros
        self.assertFalse(torch.all(output == 0))

        # Check if the output does not contain any NaN values
        self.assertFalse(torch.any(torch.isnan(output)))

class TestInputEmbeddings(BaseTest):

    def test_input_embeddings(self):
        d_model = self.config['transformer']['architecture']['d_model']
        n_ant = self.config['transformer']['architecture']['n_ant']
        embed_type = self.config['transformer']['architecture']['embed_type']
        kernel_size = self.config['transformer']['architecture']['input_embeddings']['kernel_size']
        stride = self.config['transformer']['architecture']['input_embeddings']['stride']
        seq_len = self.config['transformer']['architecture']['seq_len']
        batch_size = self.config['transformer']['training']['batch_size']
        max_pool = self.config['transformer']['architecture']['max_pool']
        print()
        print(f"Testing {embed_type} embedding with kernel size {kernel_size} and stride {stride}")

        #torch.manual_seed(0)

        input_embeddings = InputEmbeddings(
            d_model=d_model, 
            n_ant=n_ant, 
            embed_type=embed_type,
            kernel_size=kernel_size,
            stride=stride,
            max_pool=max_pool,
            )
        if embed_type == 'cnn':
            padding  = (kernel_size - 1) // 2
            expected_seq_len = (seq_len + 2 * padding - kernel_size) // stride + 1
            batch_size = batch_size
            d_model = d_model
        elif embed_type == 'ViT':
            if seq_len % kernel_size != 0:
                total_padding = (kernel_size - seq_len % kernel_size) % kernel_size
                expected_seq_len = (seq_len + total_padding) // kernel_size
            else:
                expected_seq_len = seq_len // kernel_size         
            expected_seq_len = expected_seq_len * (n_ant // kernel_size)
            batch_size = batch_size
            d_model = d_model 
        else:
            expected_seq_len = seq_len
            batch_size = batch_size
            d_model = d_model
        if max_pool and embed_type != 'linear':
            expected_seq_len = expected_seq_len // 2
             
        print(f"Number of parameters: {count_parameters(input_embeddings)}") 
        print(f"FLOPs: {get_FLOPs(input_embeddings, self.config, verbose=False)}")   
        input_data = torch.randn(batch_size, seq_len, n_ant)
        output = input_embeddings(input_data)
        print(f"Output shape: {output.shape}")
        self.assertEqual(output.shape, (batch_size, expected_seq_len, d_model))


class TestMultiHeadAttentionBlock(BaseTest):

 

    def test_MultiHeadAttentionBlock(self):
        relative_positional_encoding = self.config['transformer']['architecture']['pos_enc_type']  
        GSA = self.config['transformer']['architecture']['GSA']
        projection_type = self.config['transformer']['architecture']['projection_type']
        batch_size = self.config['transformer']['training']['batch_size']
        seq_len = self.config['transformer']['architecture']['seq_len']
        d_model = self.config['transformer']['architecture']['d_model']
        dropout = self.config['transformer']['training']['dropout']
        max_relative_position = self.config['transformer']['architecture']['max_relative_position']
        h = self.config['transformer']['architecture']['h']

        print(f"Positional encoding: {relative_positional_encoding}, GSA: {GSA}, Projection type: {projection_type}")
        with torch.autograd.set_detect_anomaly(True):
            # Create some input data
            torch.manual_seed(100)
            q = torch.randn(batch_size, seq_len, d_model)
            k = torch.randn(batch_size, seq_len, d_model)
            v = torch.randn(batch_size, seq_len, d_model)
            mask = None # torch.ones(batch_size, 1, seq_len).to(dtype=torch.bool)
            if relative_positional_encoding == 'Relative':
                relative_positional_encoding = True
            else:
                relative_positional_encoding = False
            # Initialize the MultiHeadAttentionBlock layer
            multi_head_attention_block = MultiHeadAttentionBlock(d_model=d_model, 
                                                                h=h, 
                                                                max_seq_len=1024,
                                                                dropout=dropout, 
                                                                max_relative_position=max_relative_position, 
                                                                positional_encoding=relative_positional_encoding,
                                                                GSA=GSA,
                                                                projection_type=projection_type,
                                                                )
            print(f"Number of parameters: {count_parameters(multi_head_attention_block)}")
            print(f"FLOPs: {get_FLOPs(multi_head_attention_block, self.config, verbose=False)}")
            # Apply Xavier initialization to all weights in the MultiHeadAttentionBlock
            multi_head_attention_block.apply(lambda m: nn.init.xavier_uniform_(m.weight) if hasattr(m, 'weight') else None)

            # Forward pass through the MultiHeadAttentionBlock layer
            output = multi_head_attention_block(q, k, v, mask)

            # Check if the output has the same shape as the input
            self.assertEqual(output.shape, q.shape)

            # Check if the output is not all zeros
            self.assertFalse(torch.all(output == 0))

            # Check if the output does not contain any NaN values
            if torch.any(torch.isnan(output)):
                print("NaN values in output")
            self.assertFalse(torch.any(torch.isnan(output)))

class TestEncoder(BaseTest):
    def test_encoder_block(self):

        d_model = get_value(self.config, 'd_model')
        h = get_value(self.config, 'h')
        dropout = get_value(self.config, 'dropout')
        max_relative_position = get_value(self.config, 'max_relative_position')
        relative_positional_encoding = get_value(self.config, 'pos_enc_type')
        GSA = get_value(self.config, 'GSA')
        projection_type = get_value(self.config, 'projection_type')
        activation = get_value(self.config, 'activation')
        d_ff = get_value(self.config, 'd_ff')
        residual_type = get_value(self.config, 'residual_type')
        normalization = get_value(self.config, 'normalization')
        batch_size = get_value(self.config, 'batch_size')
        seq_len = get_value(self.config, 'seq_len')

        # Create a random input tensor
        x = torch.randn(batch_size, seq_len, d_model)
        src_mask = None #torch.ones(self.batch_size, 1, self.seq_len)

        self_attention_block = MultiHeadAttentionBlock(d_model=d_model, 
                                                            h=h, 
                                                            max_seq_len=1024,
                                                            dropout=dropout, 
                                                            max_relative_position=max_relative_position, 
                                                            positional_encoding=relative_positional_encoding,
                                                            GSA=GSA,
                                                            projection_type=projection_type,
                                                            )

 
        feed_forward_block = FeedForwardBlock(d_model=d_model,
                                                d_ff=d_ff,
                                                dropout=dropout, 
                                                activation=activation)
                        # Create the EncoderBlock
        encoder_block = EncoderBlock( features=d_model,
                                        self_attention_block=self_attention_block, 
                                        feed_forward_block=feed_forward_block, 
                                        dropout=dropout, 
                                        residual_type=residual_type,
                                        normalization=normalization, 
                                )

        FLOPs = get_FLOPs(encoder_block, self.config, verbose=False)
        print()
        print(f'Pos. enc.: {relative_positional_encoding}, GSA: {GSA}, Projection type: {projection_type}, Activation: {activation}, Residual type: {residual_type}, Normalization: {normalization}, FLOPs: {FLOPs}')    
    

        # Pass the input tensor through the EncoderBlock
        output = encoder_block(x, src_mask)

        # Check that the output has the same shape as the input
        self.assertEqual(output.shape, x.shape)

        # Check that the output is not all zeros
        self.assertTrue(torch.any(output != 0))

        



class TestDataLoader(BaseTest):

    def test_chunked_data(self):
       
        train_loader, val_loader, test_loader = get_chunked_data(
                                                                batch_size=self.batch_size,
                                                                seq_len=self.seq_len,
                                                                subset=True,
                                                                )
        train_item = next(iter(train_loader))
        val_item = next(iter(val_loader))
        test_item = next(iter(test_loader))
        print(f"Train item: {train_item[0].shape}")
        self.assertEqual(train_item[0].shape, (self.batch_size, self.n_ant, self.seq_len ))
        print(f"Val item: {val_item[0].shape}")
        self.assertEqual(val_item[0].shape, (self.batch_size, self.n_ant,   self.seq_len))
        print(f"Test item: {test_item[0].shape}")
        self.assertEqual(test_item[0].shape, (self.batch_size, self.n_ant, self.seq_len))

    def test_trigger_data(self):
        train_loader, val_loader, test_loader = get_trigger_data(
                                                                batch_size=self.batch_size,
                                                                seq_len=self.seq_len,
                                                                subset=True,
                                                                )
        train_item = next(iter(train_loader))
        val_item = next(iter(val_loader))
        test_item = next(iter(test_loader))
        print(f"Train item: {train_item[0].shape}")
        self.assertEqual(train_item[0].shape, (self.batch_size, self.seq_len, self.n_ant))
        print(f"Val item: {val_item[0].shape}")
        self.assertEqual(val_item[0].shape, (self.batch_size, self.seq_len, self.n_ant))
        print(f"Test item: {test_item[0].shape}")
        self.assertEqual(test_item[0].shape, (self.batch_size, self.seq_len, self.n_ant))

class TestModel(BaseTest):

    def testModel(self):

        config = self.config   
        batch_size = get_value(config, 'batch_size') 
        n_ant = get_value(config, 'n_ant')
        seq_len = get_value(config, 'seq_len')
        max_pool = get_value(config, 'max_pool')
        d_model = get_value(config, 'd_model')

        model = build_encoder_transformer(config['transformer'])
        flops = get_FLOPs(model, config)
    
        print(f"\nFLOP: {flops/(1e6):>6.1f} M, "
        f"Number of parameters: {count_parameters(model)/1000:>6.1f} k, "
        f"Test: {self.test_string}"
        )
        

        model.to(self.device)
        encoder_layer = model.network_blocks[0]
        

        if config['transformer']['architecture']['data_type'] == 'chunked':
            data = torch.randn(batch_size, n_ant, seq_len).to(self.device)
            output = model(data)
            #print(f"Output shape: {output.shape}", end=' ')
            true_out_put_shape = torch.randn(batch_size, 1).shape
            #print(f"True output shape: {true_out_put_shape}")
            self.assertEqual(output.shape, true_out_put_shape)
            output2 = model.obtain_pre_activation(data)
            #print(f"Output2 shape: {output2.shape}", end=' ')
            #print(f"True output shape: {true_out_put_shape}")
            self.assertEqual(output2.shape, true_out_put_shape)
            
        else:
            data = torch.randn(batch_size, seq_len, n_ant).to(self.device)
            output = model(data)

            #print(f"Output shape from encoder: {true_out_put_shape_2}", end=' ')
            true_out_put_shape = torch.Size([batch_size])
            #print(f"True output shape: {true_out_put_shape}")
            self.assertEqual(output.shape, true_out_put_shape)
           
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'network_blocks'))
        self.assertTrue(len(model.network_blocks) > 1)
        self.assertTrue(hasattr(model, 'forward')) 
        self.assertTrue(not torch.any(torch.isnan(output)))
  

class TestLossFunctions(BaseTest):

    def setUp(self):
        self.device = torch.device("cpu")
        self.hinge_loss = ll.HingeLoss(self.device)

    def test_hinge_loss(self):
        y_pred_1 = torch.tensor([-1, 1, -1, 1], device=self.device)
        y_true_1 = torch.tensor([0, 1, 0, 1], device=self.device)
        expected_loss = torch.tensor(0.0, device=self.device)
        loss = self.hinge_loss(y_pred_1, y_true_1)
        self.assertTrue(torch.equal(loss, expected_loss))
        loss = loss = self.hinge_loss.forward(y_pred_1, y_true_1)
        self.assertTrue(torch.equal(loss, expected_loss))

        y_pred_2 = torch.tensor([-2, -2, -2, -2], device=self.device)
        y_true_2 = torch.tensor([0, 0, 0, 0], device=self.device)
        expected_loss = torch.tensor(0.0, device=self.device)
        loss = self.hinge_loss(y_pred_2, y_true_2)
        self.assertTrue(torch.equal(loss, expected_loss))
        loss = loss = self.hinge_loss.forward(y_pred_2, y_true_2)
        self.assertTrue(torch.equal(loss, expected_loss))


        y_pred_3 = torch.tensor([2, 2, 2, 2], device=self.device)
        y_true_3 = torch.tensor([0, 0, 0, 0], device=self.device)
        expected_loss = torch.tensor(3.0, device=self.device)
        loss = self.hinge_loss(y_pred_3, y_true_3)
        self.assertTrue(torch.equal(loss, expected_loss))
        loss = loss = self.hinge_loss.forward(y_pred_3, y_true_3)
        self.assertTrue(torch.equal(loss, expected_loss))








def update_nested_dict(d, key, value):
    for k, v in d.items():
        if isinstance(v, dict):
            update_nested_dict(v, key, value)
        if k == key:
            d[k] = value

def get_value(d, key):
    for k, v in d.items():
        if isinstance(v, dict):
            found = get_value(v, key)
            if found is not None:
                return found
        if k == key:
            return v
    return None   

if __name__ == '__main__':
    if torch.cuda.is_available(): 
      device = torch.device(f'cuda:{0}')
    else:
        device = torch.device("cpu")

    suite = unittest.TestSuite()
    config = get_config(0)
    cnn_configs = [
            {'kernel_size': 3, 'stride': 1},
           # {'kernel_size': 3, 'stride': 2},
        ]
    vit_configs = [
            # {'kernel_size': 2, 'stride': 2},
            # {'kernel_size': 4, 'stride': 4},
        ]
  
    test_dict = {
                # 'GSA': [True, False],
                # 'projection_type': ['linear', 'cnn'], 
                # 'activation': ['relu', 'gelu'],
                # 'normalization': ['layer', 'batch'],
                'embed_type': ['cnn', 'linear'],# 'ViT', ''cnn', 'linear'],
                # 'pos_enc_type':['Relative', 'Sinusoidal', 'Learnable'],
                #   'pre_def_dot_product': [True, False],
                #  'data_type': ['trigger'],
                # 'encoder_type':['vanilla', 'normal'],
                'max_pool': [True, False],

    }
    combinations = list(itertools.product(*test_dict.values()))

    configs = []
    for combination in combinations:
        params = dict(zip(test_dict.keys(), combination))

        if params.get('embed_type') == 'cnn':

            for cnn_config in cnn_configs:
                params_copy = params.copy()
                params_copy.update(cnn_config)
                
                for (test_key, value) in params_copy.items():
                    update_nested_dict(config, test_key, value)
                new_config = copy.deepcopy(config)
                configs.append(new_config)

        elif params.get('embed_type') == 'ViT':

            for vit_config in vit_configs:
                params_copy = params.copy()
                params_copy.update(vit_config)
                params.update(vit_config)
                for (test_key, value) in params_copy.items():
                    update_nested_dict(config, test_key, value)
                new_config = copy.deepcopy(config)
                configs.append(new_config)

        else:
            for (test_key, value) in params.items():
                update_nested_dict(config, test_key, value)
            new_config = copy.deepcopy(config)
            configs.append(new_config)

    for config in configs:

        test_string = ''
        for key, value in test_dict.items():
            test_string += f"{key}: {get_value(config, key)}, "
        if len(cnn_configs) > 1:
            test_string += f"kernel size: {get_value(config, 'kernel_size')} stride: {get_value(config, 'stride')}"
        if len(vit_configs) > 1:
            test_string += f"kernel size: {get_value(config, 'kernel_size')} stride: {get_value(config, 'stride')}"   

        # suite.addTest(TestInputEmbeddings('test_input_embeddings', inputs=config))
        # suite.addTest(TestMultiHeadAttentionBlock('test_MultiHeadAttentionBlock', inputs=config))
        suite.addTest(TestModel('testModel', inputs=config, device=device, test_string=test_string))
        # suite.addTest(TestEncoder('test_encoder_block', inputs=config, device=device))
        # suite.addTest(TestLossFunctions('test_hinge_loss', inputs=config, device=device))

    
    # suite.addTest(TestLayers('test_LayerNormalization'))
    # suite.addTest(TestLayers('test_BatchNormalization'))
    
    
    # 

    # suite.addTest(TestLayers('test_residual_connection'))


    # suite.addTest(TestDataLoader('test_chunked_data'))
    # suite.addTest(TestDataLoader('test_trigger_data'))

    
    

    runner = unittest.TextTestRunner()
    runner.run(suite)