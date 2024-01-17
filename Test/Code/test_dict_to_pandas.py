import pandas as pd

dic_1 = {'model_num': 999,
        'embed_size': 64,
        'seq_len': 100,
        'd_model': 512,
        'd_ff': 2048,
        'N': 8,
        'h': 4,

        'num_epochs': 100,
        'batch_size': 32,
        'pos_encoding': 'sinusoidal',
        'learning_rate': 0.001}

dic_2 = {'model_num':998,
         'embed_size': 64,
        'd_model': 512,
        'd_ff': 2048,
        'N': 8,
        'h': 4,
        'dropout': 0.1,
        'num_epochs': 100,
        'batch_size': 32,
        'pos_encoding': 'learnable',
        'learning_rate': 0.001}
df_1 = pd.DataFrame()
df_2 = pd.DataFrame(dic_2, index=[0])
df = pd.concat([df_1, df_2])
print(df)