from torch.utils.data import DataLoader, Dataset

import pandas as pd
import numpy as np 
import cv2 as cv
import random
import torch
import os

class CustomImageDataset(Dataset):
    """
    A class to import the data from the training or validation data

    """
    def __init__(self, labels, train=None, transform=None, data_split=0.2, shuffle=True, image_size_factor=0.2, valid_indexes = None ):
        """Initializes the dataset

        Parameters
        ----------
        labels : str 
            Set what kind of label we want to classify

        train : bool
            Sets if the data to be loaded are training data (train=True) or validation (train=False)
            If it is None all the data is loaded
        
        transform : transforms.Compose
            Transforms the data if a transformer is passed
        
        datasplit : float, optional
            Determines the size of the validation data default 20% = 0.2
        
        shuffle : bool, optional
            Set if the data should be shuffled or not. Default it is shuffled
        
        image_size_factor : float, optional
            Sets the factor for resizeing the images. Default is 20% = 0.2 of original image size 
        
        valid_indexes : list, optional
            Is a list of indexes created when taking the training data and the list is shuffled

        """

        # self.csv = pd.read_csv('https://raw.githubusercontent.com/pater18/Project_DNN/main/products.csv')
        self.csv = pd.read_csv('avgDistributionGS1Form.csv')
        self.img_size_factor = image_size_factor
        
        if (labels == 'GS1 Form'):
            self.img_labels = self.csv.get('GS1 Form')
            self.class_map = {'bag': 0, 'can': 1, 'box' : 2, 'jar' : 3, 'sleeve': 4, 'bottle': 5, 'aerosol': 6, 
                              'brick': 7, 'bucket' : 8, 'cup-tub': 9, 'gable-top' : 10, 'tray' : 11, 'tube': 12}

        if (labels == 'Material'):
            self.img_labels = self.csv.get('Material')
            self.class_map = {'plastic': 0, 'fibre-based': 1, 'metal': 2, 'glass': 3}

        if (labels == 'Colour'):
            self.img_labels = self.csv.get('Colour')
            self.class_map = {'black': 0, 'clear' : 1, 'white' : 2, 'other-colours': 3}

        # if multiclassify:
        #     self.img_labels = {'Material' : self.csv.get('Material'), 'GS1 Form' : self.csv.get('GS1 Form'), 'Colour' : self.csv.get('Colour')}
        #     self.class_map = {'Material' : {'plastic': 0, 'fibre-based': 1, 'metal': 2, 'glass': 3},
        #                       'GS1 Form' : {'bag': 0, 'can': 1, 'box' : 2, 'jar' : 3, 'sleeve': 4, 'bottle': 5, 'aerosol': 6, 
        #                                     'brick': 7, 'bucket' : 8, 'cup-tub': 9, 'gable-top' : 10, 'tray' : 11, 'tube': 12},
        #                       'Colour' : {'black': 0, 'clear' : 1, 'white' : 2, 'other-colours': 3}}

        self.img_dir = self.csv.get('Barcode')
        self.transform = transform
        self.data = []

        if train:
            self.idx = list(range(0, len(self.img_labels))) 
        
        if shuffle and not train:
            self.idx = valid_indexes

        if shuffle and train == True:
            random.shuffle(self.idx)



        for i, label in enumerate(self.img_labels):
            if train and self.idx[i] > int(len(self.img_labels) * data_split):
                self.data.append([str(self.img_dir[i]) + '.jpg', str(label)])
            if not train and self.idx[i] <= int(len(self.img_labels) * data_split):
                self.data.append([str(self.img_dir[i]) + '.jpg', str(label)])
            if train is None:
                self.data.append([str(self.img_dir[i]) + '.jpg', str(label)])


    def get_shuffled_index(self): 
        return self.idx

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        read_path = os.path.join("..", 'images', img_path)
        img = cv.imread(read_path)
        # img = cv.imread("../images/" + img_path)
        # resized_img = cv.resize(img, (1280, 1021))
        resized_img = cv.resize(img, (int(1280*self.img_size_factor), int(1021*self.img_size_factor)),interpolation=cv.INTER_AREA)
        resized_img = np.array(resized_img)
        label = torch.tensor(self.class_map[class_name])
        if self.transform:
            resized_img = self.transform(resized_img)
        return resized_img, label

