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
test_data = waveforms[:100,:,:]
with open('/home/halin/Master/Transformer/Test/data/test_100_data.npy', 'wb') as f:
    np.save(f, test_data)

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







