import os
import unittest
import torch
import torch.nn as nn
import sys


CODE_DIR_1  ='/home/halin/Master/Transformer/'
sys.path.append(CODE_DIR_1)
from models.models import LayerNormalization, BatchNormalization, ResidualConnection, MultiHeadAttentionBlock, FeedForwardBlock, EncoderBlock, Encoder, InputEmbeddings, PositionalEncoding, FinalBlock, EncoderTransformer, build_encoder_transformer
from dataHandler.datahandler import get_data, prepare_data
from model_configs.config import get_config


class BaseTest(unittest.TestCase):
    batch_size = 32
    seq_len = 256
    d_model = 512
    num_heads = 8
    dropout = 0.1
    d_ff = 512
    max_relative_position = 10
    out_put_size =1
    N = 6
    n_ant = 4
    h = 8
    normalizations = ['layer', 'batch']
    locations = ['post', 'pre']
    relative_positional_encodings = [True, False] 

class TestLayers(BaseTest):

    def test_LayerNormalization(self):
   
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
        for normalization in self.normalizations:
            for location in self.locations:
                self.helper_residual_connection(normalization, location)

    def helper_residual_connection(self, normalization, location):
  

        sublayer = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        residual_connection = ResidualConnection(features=self.d_model, normalization=normalization, location=location)
        input_data = torch.ones(self.batch_size, self.seq_len, self.d_model)
        output = residual_connection(input_data, sublayer)
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))

        sublayer.weight.data = torch.ones_like(sublayer.weight.data)
        sublayer.bias.data = torch.zeros_like(sublayer.bias.data)

        # Create some input data
        input_data = torch.ones(self.batch_size, self.seq_len, self.d_model)
        print(f"Normalization: {normalization}, Location: {location}")
        output = residual_connection(input_data, sublayer)

        # Check if the output has the same shape as the input
        self.assertEqual(output.shape, input_data.shape)

        # Check if the output is not all zeros
        self.assertFalse(torch.all(output == 0))

        # Check if the output does not contain any NaN values
        self.assertFalse(torch.any(torch.isnan(output)))

    def test_MultiHeadAttentionBlock(self):
        for relative_positional_encoding in self.relative_positional_encodings:
            self.helper_MultiHeadAttentionBlock(relative_positional_encoding)

    def helper_MultiHeadAttentionBlock(self, relative_positional_encoding):

        # Create some input data
        q = torch.randn(self.batch_size, self.seq_len, self.d_model)
        k = torch.randn(self.batch_size, self.seq_len, self.d_model)
        v = torch.randn(self.batch_size, self.seq_len, self.d_model)
        mask = None # torch.ones(self.batch_size, 1, seq_len).to(dtype=torch.bool)

        # Initialize the MultiHeadAttentionBlock layer
        multi_head_attention_block = MultiHeadAttentionBlock(self.d_model, self.h, self.dropout, self.max_relative_position, relative_positional_encoding)

        # Forward pass through the MultiHeadAttentionBlock layer
        output = multi_head_attention_block(q, k, v, mask)

        # Check if the output has the same shape as the input
        self.assertEqual(output.shape, q.shape)

        # Check if the output is not all zeros
        self.assertFalse(torch.all(output == 0))

        # Check if the output does not contain any NaN values
        self.assertFalse(torch.any(torch.isnan(output)))

        # Check if the output changes in response to changes in the input
        q_prime = q + torch.randn_like(q) * 0.01
        output_prime = multi_head_attention_block(q_prime, k, v, mask)
        self.assertFalse(torch.allclose(output, output_prime))    


class TestEncoder(BaseTest):
    def test_encoder_block(self):

        dropout = 0.1
 

        # Create a random input tensor
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        src_mask = None #torch.ones(self.batch_size, 1, self.seq_len)

        # Create the blocks
        self_attention_block = MultiHeadAttentionBlock(d_model=self.d_model,
                                                       h=self.num_heads)


        # Iterate over the combinations of normalization and location
        for activation in ['relu', 'gelu']:
            for normalization in ['layer', 'batch']:
                for location in ['post', 'pre']:
                    feed_forward_block = FeedForwardBlock(d_model=self.d_model,
                                              d_ff=self.d_ff,
                                             dropout=dropout, 
                                             activation=activation)
                    # Create the EncoderBlock
                    encoder_block = EncoderBlock(self.d_model, 
                                                 self_attention_block, 
                                                 feed_forward_block, 
                                                 dropout, 
                                                 normalization, 
                                                 location)

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
            for activation in ['relu', 'gelu']:
                for normalization in ['layer', 'batch']:

                    # Create the blocks
                    self_attention_block = MultiHeadAttentionBlock(d_model=self.d_model,
                                                                h=self.num_heads,
                                                                dropout=self.dropout,
                                                                max_relative_position=self.max_relative_position,
                                                                relative_positional_encoding=relative_positional_encoding,)
                    feed_forward_block = FeedForwardBlock(self.d_model, self.d_ff, self.dropout, activation)
                    encoder_block = EncoderBlock(self.d_model, self_attention_block, feed_forward_block, self.dropout)
                    # Create a ModuleList of EncoderBlocks
                    layers = nn.ModuleList([encoder_block for _ in range(6)])
                    # Create the Encoder
                    encoder = Encoder(self.d_model, layers, normalization)

                    # Pass the input tensor through the Encoder
                    output = encoder(x, mask)

                    # Check that the output has the same shape as the input
                    self.assertEqual(output.shape, x.shape)

                    # Check that the output is not all zeros
                    self.assertTrue(torch.any(output != 0))

                    # Check that the correct normalization layer was used
                    if normalization == 'layer':
                        self.assertIsInstance(encoder.norm, LayerNormalization)
                    else:
                        self.assertIsInstance(encoder.norm, BatchNormalization)


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
            




if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestLayers('test_LayerNormalization'))
    suite.addTest(TestLayers('test_BatchNormalization'))
    suite.addTest(TestLayers('test_residual_connection'))
    suite.addTest(TestLayers('test_MultiHeadAttentionBlock'))



    suite.addTest(TestEncoder('test_encoder_block'))
    suite.addTest(TestEncoder('test_encoder'))
    suite.addTest(TestEncoder('test_encode_encoder'))


    runner = unittest.TextTestRunner()
    runner.run(suite)