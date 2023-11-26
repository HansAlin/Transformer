import matplotlib.pyplot as plt
import numpy as np


with open('/home/halin/Master/Transformer/Test/Models/test_train_data.npy', 'rb') as f:

    epochs = np.load(f)
    train_loss = np.load(f)
    val_loss = np.load(f)
    acc_loss = np.load(f)
epochs = epochs.reshape((-1,1))    
train_loss = train_loss.reshape((-1,1))
val_loss = val_loss.reshape((-1,1))
acc_loss = acc_loss.reshape((-1,1))

plt.plot(epochs, train_loss, label="Train loss")
plt.plot(epochs, val_loss, label="Val. loss")  
plt.legend()
plt.savefig('/home/halin/Master/Transformer/Test/Models/test_loss_plot.png')
plt.cla()
plt.plot(epochs, acc_loss, label="Accuracy")  
plt.legend()
plt.savefig('/home/halin/Master/Transformer/Test/Models/test_acc_plot.png')
