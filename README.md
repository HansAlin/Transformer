# Transformer model for detecting High Energy Neutrinos
This is part of my master's thesis in particle physics at Uppsala University 2024

## Abstract
When Ultra-High-Energy (UHE) ( Energy > 1016 eV) neutrinos interact with ice, they produce
detectable radio waves. Using ice as a detection medium is advantageous because it is transparent
to radio waves and is abundantly available in the Arctic and Antarctic regions. Additionally,
the cold temperature of the ice reduces thermal noise. The large availability of ice is crucial
since the expected flux of neutrinos at these energies is low, and the interaction cross-section is
small, requiring a very large detector. Ice’s transparency to radio waves allows for cost-efficient
instrumentation of vast volumes.
Even the planned IceCube-Gen2-Radio, with a coverage area of 500 km2, will only detect a
few neutrinos per year with current triggering systems. Therefore, there is a huge potential to
increase the scientific output by improved triggers that enhance the neutrino detection rate.
This thesis investigates the potential of deep learning and transformer models to improve UHE
neutrino detection. I demonstrate that transformer models, used as second-stage triggers, can
enhance the detection rate of UHE neutrinos by 50% compared to existing systems. These models
are trained on data pre-triggered by a simpler system and are exposed to both signal and noise
events that fulfill the preselection requirements.
Furthermore, transformer models used as first-stage triggers with continuous data achieve a
128% increase in detection rate over current systems. Since models trained on continuous data are
exposed to a continuous stream of data it has to perform more computation than models trained
on pre-triggered data. This makes the first-stage trigger models more computationally expensive
than the second-stage trigger models.

## Get started
In main.py one have to specify model_path, where models are suposed to be stored togheter with plots and so on. 
The script is mad for specific radio signals from neutrino interactions in ice. However, it is possible to choose other kinds of data, eventhoug the plots might have strange labels. When creating costum made data follow the structure in dataHandler/datahandler.py get_any_data() which currently is just toy data. 
In order to run originla data one needs to download nuradio-analysis from Alan Colemans Git repository. Further more one needs to uncumment the nuradio-analysis dependensies in: dataHandler/datahandler.py, evaluate/evaluate.py, Test/unit_test_data_loading.py, Test/unit_test_transformer.py, Test/unit_test.py
The main.py script have the option to create many config files with different hyperparameters and the the script takes care of train and test all the models.         

### Transformer Configurations

## Input Embeddings
There are three options for input embeddings, set in the `embed_type` in the config file.
- **embed_type=linear**  
  Linear embedding is a linear transformation.
- **embed_type=cnn**  
  Contains a CNN layer with a kernel size of (3,1) added with a kernel (1,3). Currently, only `kernel_size=3` is tested. The stride can be set to `stride=1` or `stride=2`. Additional max pooling is an option for this layer.
- **embed_type=ViT**  
  Contains a layer with a kernel size of (4,4) if using 4 channels, or (5,5) if using 5 channels. For 4 channels, it is possible to use `kernel_size=2` or `kernel_size=4`. However, using 5 channels is more restricted, and the available option is `kernel_size=5`. Stride options include `stride=1`, `stride=2`,`stride=4`, and `stride=5`.

## Positional Encoding
Available options for positional encoding:
- **pos_enc_type=Sinusoidal**  
  Sinusoidal, as used in the original implementation in "Attention Is All You Need". There is an option to adjust omega, a parameter in the function. The default value is 10000, which is also recommended. The maximum possible sequence length is hardcoded in `Transformer.py` and is currently 1024.
- **pos_enc_type=Relative**  
  Relative, applied inside the MHA, which requires `encoder_type=normal`. One option is to set the maximum relative distance of interest. For example, setting `max_relative_position=32` means that everything further away than ±32 will have the same values, or only values within ±32 are considered interesting. Note that if one uses CNN or ViT with larger strides than 1, this must be considered when deciding `max_relative_position`, as the effective sequence length gets shortened. The same applies when using max pooling.
- **pos_enc_type=Learnable**  
  Similar to Sinusoidal but with learnable parameters.

## Multi-Head Attention (MHA)
Here, there is an option to use the predefined PyTorch `MultiheadAttention`, though this restricts other options inside MHA.
- **encoder_type=vanilla**  
  The predefined MHA from PyTorch.
- **encoder_type=normal**  
  Custom implemented MHA. This option is required if using relative position. One option is to still use the predefined dot product "scaled_dot_product_attention", which is set in the config file with `pre_def_dot_product=True`. Other options include CNN layers for the normal linear projections inside MHA, `projection_type=cnn`, which is a CNN with kernel size (1,1), or `projection_type=linear`, which is the standard implementation. Additionally, Gaussian Scaled Attention (GSA) can be applied over the attention scores, multiplying the attention score with a Gaussian distance distribution (`GSA=True` or `False`). The default value is `False`.

## Final Block
- **final_type=d_model_average_linear**  
  The standard implementation where averaging occurs over the `d_model` dimension, independent of sequence length.
- **final_type=seq_average_linear**  
  Averaging occurs over `seq_len` instead.
- **final_type=single_linear**  
  Flattens the `d_model` and `seq_len` dimensions and performs a single linear transformation.
- **final_type=double_linear**  
  Two linear transformations in sequence.

## Other Options
- **residual_type=pre_ln**  
  This is the default implementation, where normalization occurs before the layer and the residual connection follows the layer.
- **residual_type=post_ln**  
  Residual connections as done in the original implementation "Attention Is All You Need".
- **by_pass=False**  
  Default setting.
- **by_pass=True**  
  Means that all the channels are treated individually through the encoder and then reassembled in the final block.

## Width and Depth of the Model
- **N**  
  Number of encoder layers in sequence.
- **d_model**  
  The value must be evenly divisible by `h`.
- **d_ff**  
  The dimension of the feed-forward layer.
- **h**  
  Number of heads in the MHA.    

