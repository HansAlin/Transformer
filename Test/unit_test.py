import os
import unittest
import torch
import torch.nn as nn
import sys


CODE_DIR_1  ='/home/halin/Master/Transformer/'
sys.path.append(CODE_DIR_1)
from models.models import LayerNormalization, BatchNormalization, ResidualConnection, MultiHeadAttentionBlock, FeedForwardBlock, EncoderBlock, Encoder, InputEmbeddings, PositionalEncoding, FinalBlock, EncoderTransformer, build_encoder_transformer
from dataHandler.datahandler import prepare_data, get_chunked_data, get_trigger_data
from model_configs.config import get_config

def count_parameters(layer):
    return sum(p.numel() for p in layer.parameters() if p.requires_grad)

class BaseTest(unittest.TestCase):
    batch_size = 32
    seq_len = 256
    channels = 4
    d_model = 32
    num_heads = 8
    dropout = 0.1
    d_ff = 512
    max_relative_position = 10
    out_put_size =1
    N = 6
    n_ant = 4
    h = 8
    activations = ['relu', 'gelu']
    normalizations = ['layer', 'batch']
    residual_types = ['post_ln', 'pre_ln']
    pos_enc_types = ['Relative']  # Posible options: 'Sinusoidal', 'Relative',  'Learnable''None',
    GSAs = [True]
    projection_types = ['linear']  # Posible options: 'linear', 'cnn'
    input_embeddings = ['cnn'] # options 'lin_relu_drop', 'lin_gelu_drop', 'linear', 'cnn', 'ViT'

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
        #torch.manual_seed(0)
        for embedding_type in self.input_embeddings:
            self.helper_input_embeddings(embedding_type)

    def helper_input_embeddings(self, embedding_type, kernel_size=3, stride=2, padding=0):
        #torch.manual_seed(0)
        input_embeddings = InputEmbeddings(
            d_model=self.d_model, 
            channels=self.n_ant, 
            embed_type=embedding_type,
            padding=padding,
            kernel_size=kernel_size,
            stride=stride,
            )
        if embedding_type == 'cnn':
            padding  = (kernel_size - self.seq_len % kernel_size) // 2
            expected_seq_len = (self.seq_len + 2 * padding - kernel_size) // stride + 1
            batch_size = self.batch_size
            d_model = self.d_model
        elif embedding_type == 'ViT':
            expected_seq_len = self.seq_len // kernel_size
            batch_size = self.batch_size
            d_model = self.d_model * (self.channels // kernel_size + 1 ) 
        else:
            expected_seq_len = self.seq_len
            batch_size = self.batch_size
            d_model = self.d_model
        print(f"Number of parameters: {count_parameters(input_embeddings)}")    
        input_data = torch.randn(self.batch_size, self.seq_len, self.n_ant)
        output = input_embeddings(input_data)
        self.assertEqual(output.shape, (batch_size, expected_seq_len, d_model))


class TestMultiHeadAttentionBlock(BaseTest):

    def setUp(self) -> None:
        torch.manual_seed(0)

    def tearDown(self):
        torch.cuda.empty_cache()    

    def test_MultiHeadAttentionBlock(self):
        #torch.manual_seed(10)
        for relative_positional_encoding in self.pos_enc_types:
            for GSA in self.GSAs:
                for projection_type in self.projection_types:
                    self.helper_MultiHeadAttentionBlock(relative_positional_encoding, GSA, projection_type)


    def helper_MultiHeadAttentionBlock(self, relative_positional_encoding, GSA, projection_type):
        #torch.manual_seed(0)
        print(f"Positional encoding: {relative_positional_encoding}, GSA: {GSA}, Projection type: {projection_type}")
        with torch.autograd.set_detect_anomaly(True):
            # Create some input data
            torch.manual_seed(100)
            q = torch.randn(self.batch_size, self.seq_len, self.d_model)
            k = torch.randn(self.batch_size, self.seq_len, self.d_model)
            v = torch.randn(self.batch_size, self.seq_len, self.d_model)
            mask = None # torch.ones(self.batch_size, 1, seq_len).to(dtype=torch.bool)
            if relative_positional_encoding == 'Relative':
                relative_positional_encoding = True
            else:
                relative_positional_encoding = False
            # Initialize the MultiHeadAttentionBlock layer
            multi_head_attention_block = MultiHeadAttentionBlock(d_model=self.d_model, 
                                                                h=self.h, 
                                                                max_seq_len=1024,
                                                                dropout=self.dropout, 
                                                                max_relative_position=self.max_relative_position, 
                                                                relative_positional_encoding=relative_positional_encoding,
                                                                GSA=GSA,
                                                                projection_type=projection_type,
                                                                )
            print(f"Number of parameters: {count_parameters(multi_head_attention_block)}")
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

        # Create a random input tensor
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        src_mask = None #torch.ones(self.batch_size, 1, self.seq_len)

        for relative_positional_encoding in self.relative_positional_encodings:
            # Create the blocks
            self_attention_block = MultiHeadAttentionBlock(d_model=self.d_model, 
                                                                    h=self.h, 
                                                                    dropout=self.dropout, 
                                                                    max_relative_position=self.max_relative_position, 
                                                                    relative_positional_encoding=relative_positional_encoding,)


            # Iterate over the combinations of normalization and location
            for activation in self.activations:
                for normalization in self.normalizations:
                    for residual_type in self.residual_types:
                        torch.manual_seed(0)
                        feed_forward_block = FeedForwardBlock(d_model=self.d_model,
                                                d_ff=self.d_ff,
                                                dropout=self.dropout, 
                                                activation=activation)
                        # Create the EncoderBlock
                        encoder_block = EncoderBlock(
                                                    features=self.d_model,
                                                    self_attention_block=self_attention_block, 
                                                    feed_forward_block=feed_forward_block, 
                                                    dropout=self.dropout, 
                                                    residual_type=residual_type,
                                                    normalization=normalization, 
                                                    )

                        # Pass the input tensor through the EncoderBlock
                        output = encoder_block(x, src_mask)

                        # Check that the output has the same shape as the input
                        self.assertEqual(output.shape, x.shape)

                        # Check that the output is not all zeros
                        self.assertTrue(torch.any(output != 0))

    def test_encoder(self):
        # Create a random input tensor
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        mask = None #torch.ones(self.batch_size, 1, self.seq_len)

        
        
        # Iterate over the normalization types and the activation functions
        for relative_positional_encoding in [True, False]:
            for activation in self.activations:
                for normalization in self.normalizations:
                    for residual_type in self.residual_types:
                        torch.manual_seed(0)
                        # Create the blocks
                        self_attention_block = MultiHeadAttentionBlock(d_model=self.d_model,
                                                                    h=self.num_heads,
                                                                    dropout=self.dropout,
                                                                    max_relative_position=self.max_relative_position,
                                                                    relative_positional_encoding=relative_positional_encoding,)
                        feed_forward_block = FeedForwardBlock(self.d_model, self.d_ff, self.dropout, activation)
                        encoder_block = EncoderBlock(features=self.d_model,
                                                    self_attention_block=self_attention_block, 
                                                    feed_forward_block=feed_forward_block, 
                                                    dropout=self.dropout, 
                                                    residual_type=residual_type,
                                                    normalization=normalization, 
                                                        )
                        # Create a ModuleList of EncoderBlocks
                        layers = nn.ModuleList([encoder_block for _ in range(6)])
                        # Create the Encoder
                        encoder = Encoder(layers)

                        # Pass the input tensor through the Encoder
                        output = encoder(x, mask)

                        # Check that the output has the same shape as the input
                        self.assertEqual(output.shape, x.shape)

                        # Check that the output is not all zeros
                        self.assertTrue(torch.any(output != 0))




    def test_encode_encoder(self):
        input_data = torch.randn(self.batch_size, self.seq_len, self.n_ant)
        config = get_config(0)
        config['architecture']['N'] = self.N
        config['architecture']['d_model'] = self.d_model
        config['architecture']['d_ff'] = self.d_ff
        config['architecture']['num_heads'] = self.num_heads
        config['architecture']['dropout'] = self.dropout
        config['architecture']['max_relative_position'] = self.max_relative_position
        config['architecture']['seq_len'] = self.seq_len
        config['architecture']['batch_size'] = self.batch_size
        config['architecture']['output_size'] = self.out_put_size
        if config['architecture']['data_type'] == 'cunked':
            config['architecture']['output_size'] = self.seq_len
        else:
            config['architecture']['output_size'] = 1 

        expected_output_size = torch.Size([self.batch_size])    
        
        for location in ['post', 'pre']:
            for normalization in ['layer', 'batch']:
                for relative_positional_encoding in [True, False]:
                    for activation in ['relu', 'gelu']:
                        for encoding_type in ['normal', 'none', 'bypass']:
                            config['architecture']['normalization'] = normalization
                            config['architecture']['relative_positional_encoding'] = relative_positional_encoding
                            config['architecture']['activation'] = activation
                            config['architecture']['encoding_type'] = encoding_type
                            config['architecture']['location'] = location
                            model = build_encoder_transformer(config)
                            output = model.encode(input_data)
                            self.assertEqual(output.shape, expected_output_size)
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
        config = get_config(0)['transformer']
        config['architecture']['encoder_type'] = 'normal'
        config['architecture']['normalization'] = 'layer'
        config['architecture']['residual_type'] = 'pre_ln'
        config['architecture']['data_type'] = 'chunked'
        config['architecture']['seq_len'] = self.seq_len
        config['training']['batch_size'] = self.batch_size
        config['architecture']['output_size'] = self.out_put_size

        model = build_encoder_transformer(config)

        
        

        if config['architecture']['data_type'] == 'chunked':
            data = torch.randn(self.batch_size, self.n_ant, self.seq_len)
            output = model(data)
            print(f"Output shape: {output.shape}", end=' ')
            true_out_put_shape = torch.randn(self.batch_size, 1).shape
            print(f"True output shape: {true_out_put_shape}")
            self.assertEqual(output.shape, true_out_put_shape)
            output2 = model.obtain_pre_activation(data)
            print(f"Output2 shape: {output2.shape}", end=' ')
            print(f"True output shape: {true_out_put_shape}")
            self.assertEqual(output2.shape, true_out_put_shape)
            
        else:
            data = torch.randn(self.batch_size, self.seq_len, self.n_ant)
            output = model(data)
            print(f"Output shape: {output.shape}", end=' ')
            true_out_put_shape = torch.Size([self.batch_size])
            print(f"True output shape: {true_out_put_shape}")
            self.assertEqual(output.shape, true_out_put_shape)

            

        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'network_blocks'))
        self.assertTrue(len(model.network_blocks) > 1)
        #self.assertTrue(model.network_blocks[1].nam)
        print(f"Tetwork blocks [1]: {model.network_blocks[1]}")

if __name__ == '__main__':
    suite = unittest.TestSuite()
    # suite.addTest(TestLayers('test_LayerNormalization'))
    # suite.addTest(TestLayers('test_BatchNormalization'))
    
    
    # suite.addTest(TestMultiHeadAttentionBlock('test_MultiHeadAttentionBlock'))

    # suite.addTest(TestLayers('test_residual_connection'))

    # suite.addTest(TestEncoder('test_encoder_block'))
    # suite.addTest(TestEncoder('test_encoder'))
    # suite.addTest(TestEncoder('test_encode_encoder'))

    # suite.addTest(TestDataLoader('test_chunked_data'))
    # suite.addTest(TestDataLoader('test_trigger_data'))

    # suite.addTest(TestInputEmbeddings('test_input_embeddings'))
    suite.addTest(TestModel('testModel'))

    runner = unittest.TextTestRunner()
    runner.run(suite)