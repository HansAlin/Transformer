import argparse
import glob
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import time
import torch
from torch import nn
from tqdm import tqdm
import sys

CODE_DIR_1  ='/home/acoleman/software/NuRadioMC/'
sys.path.append(CODE_DIR_1)
CODE_DIR_2 = '/home/acoleman/work/rno-g/'
sys.path.append(CODE_DIR_2)
CODE_DIR_3 = '/home/acoleman/work/rno-g/trigger-dev/'
sys.path.append(CODE_DIR_3)
type(sys.path)
for path in sys.path:
   print(path)

from NuRadioReco.utilities import units, fft
from analysis_tools import data_locations
from analysis_tools.config import GetConfig, SaveConfig
from analysis_tools.data_loaders import DatasetSnapshot
from analysis_tools.Filters import GetRMSNoise
from analysis_tools.model_loaders import ConstructModelFromConfig

from networks.Cnn import CnnTrainingLoop
from networks.Rnn import CountModelParameters

ABS_PATH_HERE = str(os.path.dirname(os.path.realpath(__file__)))
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Yaml file with the configuration")
parser.add_argument("--seed", type=int, default=123, help="Numpy files to train on")
parser.add_argument("--test_frac", type=float, default=0.25, help="Fraction of waveforms to use for testing")
parser.add_argument("--n_epochs", type=int, default=50, help="Number of epoch to train on")
parser.add_argument("--base_output", type=str, default="TrainCnn", help="Prefix for the file names")
parser.add_argument("--trigger_time", type=float, default=200 * units.ns, help="Location of the pulse in the waveform")
parser.add_argument("--cuda_core", type=str, default="", help="Which core to run on")
args = parser.parse_args()

# Read the configuration for this training/network
config = GetConfig(args.config)
model_string = config["name"]

band_flow = config["sampling"]["band"]["low"]
band_fhigh = config["sampling"]["band"]["high"]
sampling_rate = config["sampling"]["rate"]
wvf_length = config["input_length"]

#####################################################
############## Set up naming scheme
#####################################################

freq_string = data_locations.SamplingString(config=config)

output_plot_dir = data_locations.PlotDir(config)
if not os.path.isdir(output_plot_dir):
    os.makedirs(output_plot_dir)

args.base_output = args.base_output + "_" + model_string
print("Will save files with prefix", args.base_output)

model_filename = data_locations.ModelWeightFile(config=config)
output_model_dir = os.path.dirname(model_filename)
if not os.path.isdir(output_model_dir):
    os.mkdir(output_model_dir)
print("Will save the best model as", model_filename)

np_rng = np.random.default_rng(args.seed)

is_cuda = torch.cuda.is_available()
print(f"CUDA is available: {('no', 'yes')[is_cuda]}")
if is_cuda:
    device = torch.device(f"cuda:{args.cuda_core}")
else:
    device = torch.device("cpu")
# torch.autograd.set_detect_anomaly(True)


###########################
### Setting up network size
###########################

if not "training" in config.keys():
    print("WARNING: 'training' parameter not specified in config file. Requested batch size is unknown")
    config["training"] = {}
if not "batch_size" in config["training"].keys():
    config["training"]["batch_size"] = 2000
    print("WARNING: 'batch_size' parameter not set in the 'training' config block, settting to", config["training"]["batch_size"])
if not "upsampling" in config["training"].keys():
    config["training"]["upsampling"] = 1
    print("WARNING: 'upsampling' parameter not set in the 'training' config block, settting to", config["training"]["upsampling"])
if not "start_frac" in config["training"].keys():
    config["training"]["start_frac"] = 0.3
    print("WARNING: 'start_frac' parameter not set in the 'training' config block, settting to", config["training"]["start_frac"])

batch_size = config["training"]["batch_size"]

model = ConstructModelFromConfig(config, device)
loss_fcn = nn.BCELoss(reduction="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
model.to(device)
CountModelParameters(model)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=4, cooldown=1, verbose=True)


#####################################################
############## Load in the data
#####################################################

print("READING IN DATA...")
use_beam = ("beam" in config["training"].keys()) and (config["training"]["beam"])
if use_beam:
    print("\tUsing beamed waveforms!")

waveform_filenames = data_locations.PreTrigSignalFiles(config=config, nu="*", inter="?c", lgE="1?.??", beam=use_beam)
background_filenames = data_locations.HighLowNoiseFiles("3.421", config=config)
if not len(background_filenames):
    background_filenames = data_locations.PhasedArrayNoiseFiles(config, beam=use_beam)
if not len(background_filenames):
    print("No background files found!")
    exit()
# waveform_filenames = data_locations.PreTrigSignalFiles(config=config, nu="e", inter="cc", lgE="1?.0?")
# background_filenames = data_locations.HighLowNoiseFiles("3.421", config=config, nFiles=1)
print(f"\t\tFound {len(waveform_filenames)} signal files and {len(background_filenames)} background files")

waveforms = None
background = None

###################
## Defining bins
###################

frac_into_waveform = config["training"]["start_frac"]  # Trigger location will be put this far into the cut waveform
trig_bin = args.trigger_time * sampling_rate * config["training"]["upsampling"]
cut_low_bin = max(0, int(trig_bin - wvf_length * frac_into_waveform))
cut_high_bin = cut_low_bin + wvf_length
print(f"Cutting waveform sizes to be {wvf_length} bins long, trigger bin: {trig_bin}, bins: {cut_low_bin} to {cut_high_bin}")

###################
## Signal waveforms
###################

print("\tReading in waveforms")
t0 = time.time()
# Precalculate how much space will be needed to read in all waveforms
print("\t\tPrecalculating RAM requirements")
total_len = 0
for filename in tqdm(waveform_filenames):
    shape = np.load(filename, mmap_mode="r")["wvf"].shape
    total_len += shape[0]

print(f"\t\tSize on disk {(total_len, shape[1], shape[2])}")
waveforms = np.zeros((total_len, shape[1], wvf_length), dtype=np.float32)
print(f"\t\tWill load as {waveforms.shape}")

total_len = 0
for filename in tqdm(waveform_filenames):
    this_dat = np.load(filename)
    waveforms[total_len : total_len + len(this_dat["wvf"])] = this_dat["wvf"][:, :, cut_low_bin:cut_high_bin]
    total_len += len(this_dat["wvf"])
    del this_dat

assert len(waveforms)

rms_noise = GetRMSNoise(float(band_flow), float(band_fhigh), sampling_rate, 300 * units.kelvin)
print(f"\t\tWill scale waveforms by values by 1 / {rms_noise / (1e-6 * units.volt):0.4f} uV")
std = np.median(np.std(waveforms[:, :, int(waveforms.shape[-1] * 0.77) :]))
print(f"\t\tFYI: the RMS noise of waveforms is {std / (1e-6 * units.volt):0.4f} uV")
waveforms /= rms_noise

print(
    f"\t\tLoaded signal data of shape {waveforms.shape} --> {waveforms.shape[0] * waveforms.shape[-1] / sampling_rate / units.s:0.3f} s of data"
)

nan_check = np.isnan(waveforms)
if np.any(nan_check):
    print("Found NAN WVF")
    index = np.argwhere(nan_check)
    print(numpy.unique(index[:, 0]))
    print(index)
    exit()

#######################
## Background waveforms
#######################

print("\tReading in background")

print("\t\tPrecalculating RAM requirements")
total_len = 0
for filename in tqdm(background_filenames):
    back_shape = np.load(filename, mmap_mode="r").shape
    total_len += back_shape[0]

print(f"\t\tShape on disk is {(total_len, back_shape[1], back_shape[2])}")
back_shape = (total_len, back_shape[1], wvf_length)
print(f"\t\tWill load as {back_shape}")
background = np.zeros(back_shape, dtype=np.float32)

total_len = 0
for filename in tqdm(background_filenames):
    this_dat = np.load(filename)
    background[total_len : total_len + len(this_dat)] = this_dat[:, :, cut_low_bin:cut_high_bin]
    total_len += len(this_dat)
    del this_dat

assert total_len == len(background)
assert background.shape[1] == waveforms.shape[1]
print(
    f"\t\tLoaded background data of shape {background.shape} --> {background.shape[0] * background.shape[-1] / sampling_rate / units.s:0.3f} s of data"
)
print(f"\t--->Read-in took {time.time() - t0:0.1f} seconds")

#######################################################
############## Renormalizing the data into units of SNR
#######################################################

std = np.std(background)
print(f"Will scale backround by 1 / {rms_noise / (1e-6 * units.volt):0.4f} uV")
print(f"FYI: the RMS noise of the backround is {std / (1e-6 * units.volt):0.4f} uV")
background /= rms_noise

#####################################################
############## Permuting everything
#####################################################

print("Performing initial scramble")
p_data = np_rng.permutation(len(waveforms))
waveforms = waveforms[p_data]
p_background = np_rng.permutation(len(background))
background = background[p_background]


n_events = 3
n_channels = background.shape[1]
ncols = 3
nrows = n_events * n_channels
fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * 8, nrows * 3), gridspec_kw={"wspace": 0.2, "hspace": 0.0})

for ievent in range(n_events):
    for ich in range(n_channels):
        ax[ievent * n_channels + ich, 0].plot(waveforms[ievent, ich], label=f"Signal, evt{ievent+1}, ch{ich+1}")
        ax[ievent * n_channels + ich, 1].plot(background[ievent, ich], label=f"Bkgd, evt{ievent+1}, ch{ich+1}")
        ax[ievent * n_channels + ich, 0].legend()
        ax[ievent * n_channels + ich, 1].legend()

    spec = np.median(np.abs(fft.time2freq(waveforms[ievent], sampling_rate * config["training"]["upsampling"])), axis=0)
    ax[ievent * n_channels, 2].plot(spec, color="k")
    spec = np.median(np.abs(fft.time2freq(background[ievent], sampling_rate * config["training"]["upsampling"])), axis=0)
    ax[ievent * n_channels, 2].plot(spec, color="r")
    ax[ievent * n_channels, 2].set_yscale("log")


for i in range(len(ax)):
    for j in range(len(ax[i])):
        ax[i, j].tick_params(axis="both", which="both", direction="in")
        ax[i, j].yaxis.set_ticks_position("both")
        ax[i, j].xaxis.set_ticks_position("both")

example_plot_name = f"{output_plot_dir}/{args.base_output}_ExampleWaveforms.pdf"
print("Saving", example_plot_name)
fig.savefig(example_plot_name, bbox_inches="tight")
plt.close()


###########################
### Setting up data sets
###########################

background_mixture = np.linspace(0.0, 1.0, batch_size)  # Percentage of background waveforms in each batch
train_fraction = 1 - args.test_frac

n_antennas = waveforms.shape[1]

# Where to split the dataset into training/validation/testing
wvf_split_index = int(train_fraction * len(waveforms))
background_split_index = int(train_fraction * len(background))

train_data = DatasetSnapshot(
    waveforms[:wvf_split_index],
    background[:background_split_index],
    n_antennas,
    batch_size,
    np_rng,
)
val_data = DatasetSnapshot(
    waveforms[wvf_split_index:],
    background[background_split_index:],
    n_antennas,
    batch_size,
    np_rng,
)
val_data.scramble_warn = False
train_data.Scramble()

print("Data sets are of size:")
print("\tTraining:", len(waveforms[:wvf_split_index]))
print("\tValidation:", len(waveforms[wvf_split_index:]))


del waveforms
del background
###########################
### Set up the training
###########################
base_plotting_filename = f"{output_plot_dir}/{args.base_output}_TrainingStatus.pdf"
cnn_trainer = CnnTrainingLoop(model, loss_fcn, optimizer, scheduler, base_plotting_filename, sampling_rate, device=device)
model, reduction, thresholds = cnn_trainer.train(train_data, val_data, batch_size, args.n_epochs, model_filename=model_filename)
print("Saving model as:", model_filename)
torch.save(model.state_dict(), model_filename)
CountModelParameters(model)
print(f"CNN calulations: Adds {int(model.adds)}, Mult {int(model.multiplys)}, Tot {int(model.adds + model.multiplys)}")

imin = np.argmin(np.abs(reduction - -4))
threshold = thresholds[imin]

config["trigger"] = {}
config["trigger"]["threshold"] = float(f"{threshold:0.6f}")

SaveConfig(args.config, config)