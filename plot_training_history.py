import numpy as np
import matplotlib.pyplot as plt

"""
Load training history generated from 'model.py'
"""
history = np.load("trained_history.npy").item()

"""
Plot loss vs epochs
"""
plt.plot(history['loss'], '-')
plt.plot(history['val_loss'], '--')
plt.title('Training history')
plt.ylabel('Mean-Squared Error loss')
plt.xlabel('Epoch (#)')
plt.legend(['Training set', 'Validation set'], loc='upper right')
plt.show()