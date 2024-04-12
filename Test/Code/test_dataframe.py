import pandas as pd
import numpy as np
import sys

CODE_DIR_1  ='/home/halin/Master/Transformer/'
sys.path.append(CODE_DIR_1)

df_1 = pd.read_pickle('/home/halin/Master/Transformer/Test/data/epoch_data_model_246_validation_1.pkl')
df_2 = pd.read_pickle('/home/halin/Master/Transformer/Test/data/epoch_data_model_246_validation_2.pkl')

print(df_1.head())
print(df_2.head())

df_1['Epoch'] = pd.to_numeric(df_1['Epoch'], errors='coerce')
df_1 = df_1.dropna(subset=['Epoch'])
df_1['Epoch'] = df_1['Epoch'].astype(int)

df_2['Epoch'] = pd.to_numeric(df_2['Epoch'], errors='coerce')
df_2 = df_2.dropna(subset=['Epoch'])
df_2['Epoch'] = df_2['Epoch'].astype(int)

df_1_sorted = df_1.sort_values(by=['Epoch'])
df_2_sorted = df_2.sort_values(by=['Epoch'])

delta_eff = np.abs(df_1_sorted['Efficiency'] - df_2_sorted['Efficiency'])/df_1_sorted['Efficiency']
delta_thresh = np.abs(df_1_sorted['Threshold'] - df_2_sorted['Threshold'])/df_1_sorted['Threshold']
print(f"Max error: {np.max(delta_eff):<10.5f} {np.max(delta_thresh):<10.5f}")