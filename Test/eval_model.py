import matplotlib.pyplot as plt
import numpy as np


with open('/home/halin/Master/Transformer/Test/test_data/test_data.npy', 'rb') as f:

    epochs = np.load(f)
    train_loss = np.load(f)
    val_loss = np.load(f)

plt.plot(epochs, train_loss)  
plt.show()  