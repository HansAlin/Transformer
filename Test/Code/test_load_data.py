import argparse
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
import os
import time
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys

CODE_DIR_1  ='/home/acoleman/software/NuRadioMC/'
sys.path.append(CODE_DIR_1)
CODE_DIR_2 = '/home/acoleman/work/rno-g/'
sys.path.append(CODE_DIR_2)
type(sys.path)
for path in sys.path:
   print(path)


from NuRadioReco.utilities import units

from analysis_tools import data_locations
from analysis_tools.config import GetConfig
from analysis_tools.data_loaders import DatasetContinuousStreamStitchless
from analysis_tools.Filters import GetRMSNoise
from analysis_tools.model_loaders import ConstructModelFromConfig, LoadModelFromConfig

# from networks.Chunked import CountModelParameters, ChunkedTrainingLoop
# from networks.Cnn import CnnTrainingLoop

# plt.rcParams["font.family"] = "serif"
# plt.rcParams["mathtext.fontset"] = "dejavuserif"
ABS_PATH_HERE = str(os.path.dirname(os.path.realpath(__file__)))

# parser = argparse.ArgumentParser()
# parser.add_argument("--config", required=True, help="Yaml file with the configuration")
# parser.add_argument("--seed", type=int, default=123, help="Numpy files to train on")
# parser.add_argument("--test_frac", type=float, default=0.15, help="Fraction of waveforms to use for testing")
# parser.add_argument("--n_epochs", type=int, default=50, help="Number of epoch to train on")
# parser.add_argument("--cuda_core", type=str, default="", help="Which core to run on")
# args = parser.parse_args()

# Read the configuration for this training/network
config_path = '/home/halin/Master/Transformer/Test/Code/config.yaml'
config = GetConfig(config_path)
model_string = config["name"]

band_flow = config["sampling"]["band"]["low"]
band_fhigh = config["sampling"]["band"]["high"]
sampling_rate = config["sampling"]["rate"]
wvf_length = config["input_length"]

random_seed = 100
np_rng = np.random.default_rng(random_seed)

waveform_filenames = data_locations.NoiselessSigFiles(
    cdf=config["training"]["cdf"], config=config, nu="*", inter="cc", lgE="1?.??"
)
print(f"\tFound {len(waveform_filenames)} signal files")


waveforms = None
signal_labels = None

print("\tReading in signal waveforms")
t0 = time.time()
# Precalculate how much space will be needed to read in all waveforms
total_len = 0
print("\t\tCalculating required size")
for filename in waveform_filenames:
    shape = np.load(filename, mmap_mode="r")["wvf"].shape
    total_len += shape[0]
waveforms = np.zeros((total_len, shape[1], shape[2]))
signal_labels = np.zeros((total_len, shape[1], shape[2]))
snrs = np.zeros((total_len, shape[1]))

total_len = 0
print("\t\tReading into RAM")
for filename in tqdm(waveform_filenames):
    this_dat = np.load(filename)
    waveforms[total_len : total_len + len(this_dat["wvf"])] = this_dat["wvf"]
    signal_labels[total_len : total_len + len(this_dat["wvf"])] = this_dat["label"]
    snrs[total_len : total_len + len(this_dat["wvf"])] = this_dat["snr"]
    total_len += len(this_dat["wvf"])

assert len(waveforms)
assert waveforms.shape == signal_labels.shape
print(
    f"\t\tLoaded signal data of shape {waveforms.shape} --> {waveforms.shape[0] * waveforms.shape[-1] / sampling_rate / units.s:0.3f} s of data"
)




if np.any(signal_labels > 1):
    print("Labels too big")
if np.any(signal_labels < 0):
    print("Labels too small")
if np.any(snrs <= 0):
    print("BAD SNR")

if np.any(np.isnan(snrs)):
    print("Found NAN SNR")
    index = np.argwhere(np.isnan(snrs))
    snrs = np.delete(snrs, index, axis=0)
    waveforms = np.delete(waveforms, index, axis=0)
    signal_labels = np.delete(signal_labels, index, axis=0)

if np.any(np.isnan(waveforms)):
    print("Found NAN WVF")
    index = np.argwhere(np.isnan(waveforms))
    print(index)

if np.any(np.isnan(signal_labels)):
    print("NAN!!")
    print(np.argwhere(np.isnan(signal_labels)))

del snrs

print(f"\t--->Read-in took {time.time() - t0:0.1f} seconds")



#######################################################
############## Renormalizing the data into units of SNR
#######################################################

rms_noise = GetRMSNoise(float(band_flow), float(band_fhigh), sampling_rate, 300 * units.kelvin)
print(f"Scaling all values by 1 / {rms_noise / (1e-6 * units.volt):0.4f} uV to normalize to SNR")
waveforms /= rms_noise


#####################################################
############## Permuting everything
#####################################################

print("Performing initial scramble")
p_data = np_rng.permutation(len(waveforms))
waveforms = waveforms[p_data]
signal_labels = signal_labels[p_data]

# Make a plot of a waveform with the labels
#PlotWaveformExample(waveforms[0], signal_labels[0], f"{output_plot_dir}/{base_output}_Labels.pdf")

#####################################################
############## Join the labels across channels
#####################################################

print("Joining the label windows")
signal_labels = np.max(signal_labels, axis=1)
for i in range(len(signal_labels)):
    ones = np.where(signal_labels[i] > 0)[0]
    if len(ones):
        signal_labels[i, min(ones) : max(ones)] = 1



###########################
### Setting up data sets
###########################

batch_size = config["training"]["batch_size"]  # Number of "mixtures" of signal/noise
n_features = config["n_ant"]  # Number of antennas
wvf_length = config["input_length"]

mixture = np.linspace(0.0, 1.0, batch_size)  # Percentage of background waveforms in each batch

# Where to split the dataset into training/validation/testing
train_fraction = 0.8
val_fraction = 0.1
train_split= int(train_fraction * len(waveforms))
val_split = int(val_fraction * len(waveforms))

x_train = waveforms[:train_split]
x_val = waveforms[train_split:train_split + val_split]
x_test = waveforms[train_split + val_split:]
y_train = signal_labels[:train_split]
y_val = signal_labels[train_split:train_split + val_split]
y_test = signal_labels[train_split + val_split:]

x_test_data = x_test[:100,:,:]
with open('/home/halin/Master/Transformer/Test/data/test_100_data.npy', 'wb') as f:
    np.save(f, x_test[:100])
    np.save(f, y_test[:100])

print(f"Training on {len(x_train)} waveforms")
print(f"Testing on {len(x_test)} waveforms")
print(f"Number of signals in test set: {np.sum(y_test)}")

del waveforms
del signal_labels


