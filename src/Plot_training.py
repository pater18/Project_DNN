import matplotlib.pyplot as plt
import numpy as np


num_epochs = 5

def load_training_data(class_to_load, folderPath):
    train_losses = np.load(folderPath + 'train_losses_' + class_to_load + '.npy')
    train_accuracies = np.load(folderPath + 'train_accuracies_' + class_to_load + '.npy')
    valid_losses = np.load(folderPath + 'valid_losses_' + class_to_load + '.npy')
    valid_accuracies = np.load(folderPath + 'valid_accuracies_' + class_to_load + '.npy')

    return train_losses, train_accuracies, valid_losses, valid_accuracies

def plot_data(train_losses, train_accuracies, valid_losses, valid_accuracies, name):
    
    plt.figure(name, figsize=(7, 3),)
    plt.figure

    plt.subplot(1, 2, 1)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    p = plt.plot(train_losses, label='train')
    plt.plot(valid_losses, label='valid')
    plt.ylim(0.0, 1.5)
    plt.yticks(np.arange(0.0, 2.8, 0.1))
    plt.xticks(np.arange(0, num_epochs, 5))
    plt.title(f"Final loss train: {train_losses[-1]}\nFinal loss valid: {valid_losses[-1]}")
    plt.legend()
    plt.grid(linestyle='--')

    plt.subplot(1, 2, 2)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    p = plt.plot(train_accuracies, label='train')
    plt.plot(valid_accuracies, label='valid')
    plt.ylim(0.5, 1.05)
    plt.yticks(np.arange(0.35, 1.75, 0.05))
    plt.xticks(np.arange(0, num_epochs, 5))
    plt.title(f"Final accuracy train: {train_accuracies[-1]}\nFinal accuracy validation: {valid_accuracies[-1]}")
    plt.legend()
    plt.grid(linestyle='--')

    plt.tight_layout()
    plt.show()


tr_l, tr_a, valid_l, valid_a = load_training_data('Material', '../Training 2/')
plot_data(tr_l, tr_a, valid_l, valid_a, 'Material')

tr_l, tr_a, valid_l, valid_a = load_training_data('GS1 Form', '../Training 2/')
plot_data(tr_l, tr_a, valid_l, valid_a, 'GS1 Form')

tr_l, tr_a, valid_l, valid_a = load_training_data('Colour', '../Training 2/')
plot_data(tr_l, tr_a, valid_l, valid_a, 'Colour')

