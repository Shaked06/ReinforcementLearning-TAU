
import matplotlib.pyplot as plt
import numpy as np

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 14,
        }


def loss_plot(values, model_name, optimizer, learning_rate, batch_size, num_epochs, file_name):
    plt.plot(values, color="darkred")
    plt.legend(['training loss'], loc='upper right')
    plt.xlim((0, num_epochs))
    plt.xticks(np.arange(0, num_epochs+1, 5, dtype=np.int32))
    plt.title(
        f"{model_name} \nOptimizer: {optimizer}      Learning Rate = {learning_rate}\nBatch Size = {batch_size}   Epoch Number = {num_epochs}", fontdict=font)
    plt.xlabel('Epochs', fontdict=font)
    plt.ylabel('Cross Entropy Loss', fontdict=font)
    plt.tight_layout()
    plt.savefig(f"{file_name}.png")
    plt.show()
