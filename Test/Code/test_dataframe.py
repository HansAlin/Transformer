import pandas as pd
import numpy as np
import sys

CODE_DIR_1  ='/home/halin/Master/Transformer/'
sys.path.append(CODE_DIR_1)

df_1 = pd.read_pickle('/mnt/md0/halin/Models/model_256/plot/veff_model_256_best_dict.pkl')


print(df_1.head())


