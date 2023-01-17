import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import time

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

import cv2 as cv
import numpy as np
from tqdm import tqdm


from CusomImageDataset import CustomImageDataset

import cnn


# Hyper parameters
num_epocs = 50
batch_size = 64
learning_rate = 0.001
data_split_percent = 0.2

# Use the GPU if it is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_tensor_size(data_loader):
    nf = 32
    s = 1
    pool = False
    poollayer = nn.MaxPool2d(2,2)
    indt = nn.Identity()
    conv = nn.Conv2d(3, nf, kernel_size=3, stride=1, padding=1)
    conv1 = nn.Conv2d(nf, 2 * nf, kernel_size=3, stride=1, padding=1 )
    conv2 = nn.Conv2d(2 * nf, 4 * nf, kernel_size=3, stride=1, padding=1 )
    conv3 = nn.Conv2d(4 * nf, 8 * nf, kernel_size=3, stride=1, padding=1)
    flat = nn.Flatten()
    lin1 = nn.Linear(256 * 60 * 76, 512)
    nrm1 = nn.BatchNorm2d(nf)
    nrm2 = nn.BatchNorm2d(2 * nf)
    nrm3 = nn.BatchNorm2d(4 * nf)
    nrm4 = nn.BatchNorm2d(8 * nf)


    dataiter = iter(data_loader)
    images_test, labels = dataiter.next()

    print (f"{images_test.shape} : Image shape ")
    x = conv(images_test)
    print(f"{x.shape} : First convolution ")
    x = nrm1(x)
    print(x.shape)
    x = poollayer(x)
    print (f"{x.shape} : First pooling")
    x = conv1(x)
    print(f"{x.shape} : Second convolution ")
    x = nrm2(x)
    print(x.shape)
    x = poollayer(x)
    print (f"{x.shape} : Second pooling")

    x = conv2(x)
    print(f"{x.shape} : Third convolution ")
    x = nrm3(x)
    print(x.shape)
    x = poollayer(x)
    print (f"{x.shape} : Third pooling")

    x = conv3(x)
    print(f"{x.shape} : Fourth convolution ")

    x = nrm4(x)
    print(x.shape)
    x = poollayer(x)
    print (f"{x.shape} : Fourth pooling")
    x = flat(x)
    print (f"{x.shape} : After flatten")

def plot_data(train_losses, train_accuracies, valid_losses, valid_accuracies):
    plt.figure(figsize=(7, 3))

    plt.subplot(1, 2, 1)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    p = plt.plot(train_losses, label='train')
    plt.plot(valid_losses, label='valid')
    plt.ylim(0.0, 1.5)
    plt.yticks(np.arange(0.0, 2.8, 0.1))
    plt.xticks(np.arange(0, num_epocs, 5))
    plt.title(f"Final loss train: {train_losses[-1]}\nFinal loss valid: {valid_losses[-1]}")
    plt.legend()
    plt.grid(linestyle='--')

    plt.subplot(1, 2, 2)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    p = plt.plot(train_accuracies, label='train')
    plt.plot(valid_accuracies, label='valid')
    plt.ylim(0.5, 1.05)
    plt.yticks(np.arange(0.35, 1.05, 0.05))
    plt.xticks(np.arange(0, num_epocs, 5))
    plt.title(f"Final accuracy train: {train_accuracies[-1]}\nFinal accuracy validation: {valid_accuracies[-1]}")
    plt.legend()
    plt.grid(linestyle='--')

    plt.tight_layout()
    plt.show()
    
def one_epoch(model, data_loader, class_to_predict, criterion, optimizer=None):
    
    losses, correct, total = [], 0, 0
    
    train = False if optimizer is None else True
    model.train() if train else model.eval()
    td = tqdm(data_loader)
    for (images, labels) in td:
        
        if (len(labels) != batch_size):
            continue

        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(train):
            output = model(images)
        loss = criterion(output, labels)

        losses.append(loss.item())
        total += len(labels)
        correct += (torch.argmax(output, dim=1) == labels).sum().item()

        if (optimizer is not None):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
 

    # Returns the mean of the loss in a batch and a percentage of correct labelled classes
    return np.mean(losses), correct/total    

def save_training_data(train_losses, train_accuracies, valid_losses, valid_accuracies, class_predicted): 
    train_losses = np.array(train_losses)
    train_accuracies = np.array(train_accuracies)
    valid_losses = np.array(valid_losses)
    valid_accuracies = np.array(valid_accuracies)

    np.save("train_losses_" + class_predicted + ".npy", train_losses) 
    np.save("train_accuracies_" + class_predicted + ".npy", train_accuracies) 
    np.save("valid_losses_" + class_predicted + ".npy", valid_losses) 
    np.save("valid_accuracies_" + class_predicted + ".npy", valid_accuracies)

def load_data(class_to_load, shuffle_data=False):
    print ("Loading data ...")

    # First define the transformer to normalize the images and make them tensors
    tfms_norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = CustomImageDataset(class_to_load, train=True, data_split=data_split_percent, transform=tfms_norm, shuffle=shuffle_data, image_size_factor=0.15)
    if shuffle_data:
        idxes = train_dataset.get_shuffled_index()
    else :
        idxes = None
    validation_dataset = CustomImageDataset(class_to_load, train=False, data_split=data_split_percent, transform=tfms_norm, shuffle=shuffle_data, valid_indexes=idxes, image_size_factor=0.15)

    print(f"Data used for varification {data_split_percent * 100} % = {len(validation_dataset)}")
    print(f"Data used for training {100 - data_split_percent * 100} % = {len(train_dataset)}\n ")

    data_loader_valid = DataLoader(validation_dataset, batch_size, shuffle=True)
    data_loader_train = DataLoader(train_dataset, batch_size, shuffle=True)

    return data_loader_train, data_loader_valid

def load_training_data():
    train_losses = np.load('train_losses_GS1 Form.npy')
    train_accuracies = np.load('train_accuracies_GS1 Form.npy')
    valid_losses = np.load('valid_losses_GS1 Form.npy')
    valid_accuracies = np.load('valid_accuracies_GS1 Form.npy')

    return train_losses, train_accuracies, valid_losses, valid_accuracies

def train_model(model, data_loader_train, data_loader_valid, class_to_predict):

    # Function used = 1 - (number of samples in the class / total number of samples) 
    total_in_dataset = 9986

    if class_to_predict == 'Material':
        class_wights = torch.tensor( [1-(5102/9986), 1-(1778/9986), 1-(782/9986), 1-(2324/9986)])
    if class_to_predict == 'GS1 Form':
        class_wights = torch.tensor( [0.761065, 0.931904, 0.877328, 0.904066, 0.896255, 0.778490, 0.996395,
                                      0.994192, 0.998598, 0.964550, 0.977969, 0.925496, 0.993691])
    if class_to_predict == 'Colour':
        class_wights = torch.tensor ( [0.971660, 0.691468, 0.982876, 0.353996] )

    class_wights = class_wights.to(device)


    # Define the lossfunction 
    # criterion = nn.CrossEntropyLoss(weight=class_wights)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    train_losses, train_accuracies = [], []
    valid_losses, valid_accuracies = [], []
    best_valid_epoch = 0
    best_valid_acc = 0
    patience = 5

    print ("Training " + class_to_predict + " --------------------------------------------------------------------")
    start_time = time.time()
    #t = tqdm(range(num_epocs))
    for epoch in range(num_epocs):
        print (f"Epoch [{epoch + 1} / {num_epocs}] ...")    
        start_epoch_time = time.time()
        
        train_loss, train_acc = one_epoch(model, data_loader_train, class_to_predict, criterion, optimizer)
        valid_loss, valid_acc = one_epoch(model, data_loader_valid, class_to_predict, criterion)
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_acc)
        
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_valid_epoch = epoch 
            torch.save(model.state_dict(), 'weights_' + class_to_predict + '_best.pt')


        train_acc *= 100.0
        valid_acc *= 100.0
        text_to_print = f'{class_to_predict} train_acc: {train_acc:.5f}%, valid_acc: {valid_acc:.5f}% \ntrain_loss: {train_loss}, valid_loss: {valid_loss}'
        print(text_to_print) 

        end_epoch_time = time.time()
        print (f"Epoch number {epoch + 1} took {end_epoch_time-start_epoch_time} seconds\n")

        if best_valid_epoch + patience < epoch:
            break
    
    end_time = time.time()       
    print (f"Training is finished. Took {end_time-start_time}")
    torch.save(model.state_dict(), 'weights_' + class_to_predict + '_end.pt')

    return train_losses, train_accuracies, valid_losses, valid_accuracies
def main():

    # _, testdata = load_data('GS1 Form', shuffle_data=True)

    # test_tensor_size(testdata)

    # # First set up the model that is used and the type that is classified
    model_material = cnn.buildModel('Material').to(device)
    model_gs1_form = cnn.buildModel('GS1 Form').to(device)
    model_colour = cnn.buildModel('Colour').to(device)    

    print(model_colour)
    train_losses, train_accuracies = [], []
    valid_losses, valid_accuracies = [], []

    # Then train the model
    train_losses, train_accuracies, valid_losses, valid_accuracies = train_model(model_material, *load_data('Material', shuffle_data=True), 'Material' )
    save_training_data(train_losses, train_accuracies, valid_losses, valid_accuracies, 'Material')

    train_losses, train_accuracies = [], []
    valid_losses, valid_accuracies = [], []
    
    train_losses, train_accuracies, valid_losses, valid_accuracies = train_model(model_gs1_form, *load_data('GS1 Form', shuffle_data=True), 'GS1 Form' )
    save_training_data(train_losses, train_accuracies, valid_losses, valid_accuracies, 'GS1 Form')
    

    train_losses, train_accuracies = [], []
    valid_losses, valid_accuracies = [], []

    train_losses, train_accuracies, valid_losses, valid_accuracies = train_model(model_colour, *load_data('Colour', shuffle_data=True), 'Colour' )
    save_training_data(train_losses, train_accuracies, valid_losses, valid_accuracies, 'Colour')
    
    
    # torch.save(model_gs1_form.state_dict(), 'weights_gs1.pt')
    # torch.save(model_colour.state_dict(), 'weights_colour.pt')
    # torch.save(model_material.state_dict(), 'weights_material.pt')


    # Plotting the training data and saving them for future reference
    # train_losses, train_accuracies, valid_losses, valid_accuracies= load_training_data()
    plot_data(train_losses, train_accuracies, valid_losses, valid_accuracies)
    #save_training_data(train_losses, train_accuracies, valid_losses, valid_accuracies)
    
    # image = cv.imread("../images/5011428000232.jpg")
    # weights_file_path = "weights_gs1_form.pt"
    # cnn.predict(class_to_predict, weights_file_path, image )
   
    #cnn.predict('Colour', '../Training 2/weights_colour.pt', show_images=True)

if __name__ == "__main__":
    main()