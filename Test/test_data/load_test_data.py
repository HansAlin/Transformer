import numpy as np

def load_label_data(data_path='/home/halin/Autoencoder/Data/', train_size=1000, save_path='/home/halin/Master/Transformer/Test/test_data/test_data.npy'):
  """
    This function loads data from ARIANNA group, downloaded localy
    Args:
    data_path: where the data from ARIANNA is stored
    train_size: size of training data, test data is 100
    save_path: where you store the sorted data
    Returns:
    x_train, y_train,  x_test, y_test 
    
  """
  NOISE_URL = data_path + 'trimmed100_data_noise_3.6SNR_1ch_0000.npy'
  noise = np.load(NOISE_URL)

  SIGNAL_URL = data_path + "trimmed100_data_signal_3.6SNR_1ch_0000.npy"
  signal = np.load(SIGNAL_URL)
  
  x = np.vstack((noise, signal))
  y = np.ones(len(x))
  y[:len(noise)] = 0

  test_size = int(np.floor(train_size*0.1/0.9))
  shuffle = np.arange(x.shape[0], dtype=np.int64)
  np.random.shuffle(shuffle)
  x = x[shuffle]
  y = y[shuffle]
  #y = np.expand_dims(y, axis=-1)
  x_train = x[:(train_size)]
  y_train = y[:(train_size)]
  x_test = x[train_size:(train_size + test_size)]
  y_test = y[train_size:(train_size + test_size)]

  with open(save_path, 'wb') as f:
    np.save(f, x_train)
    np.save(f, x_test)
    np.save(f, y_train)
    np.save(f, y_test)

load_label_data(train_size=100000)

  

  