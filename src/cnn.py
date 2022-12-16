import numpy as np
import pandas as pd 
import cv2 as cv
import PIL
import os
from tqdm import tqdm


import torch
from torch import nn
from torch.nn import functional as F

import torchvision
from torchvision import transforms



class CNN:


    df = None
    imagePaths = None
    images = []

    def  __init__(self, loader='local') -> None:
        self.dataLoaderLocal()



    def dataLoaderLocal(self):
        print ("Running data loader local... ")

        #Read all the target values and put it in a panda dataFrame
        self.df = pd.read_csv('https://raw.githubusercontent.com/pater18/Project_DNN/main/products.csv')
                
        # Read all the images 
        self.imagePaths = os.listdir("../images/")
        for path in tqdm(self.imagePaths):
            self.images.append(cv.imread("../images/" + path))
            

    def train(self):
        pass

    def buildLayer(self , inChannels, outChannels, kernelSize=3, stride=1, padding=1, useBatchnorm=True, pool=True):
        
        return nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernelSize, stride, padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(outChannels) if useBatchnorm else nn.Identity(),
            nn.MaxPool2d(2,2) if pool else nn.Identity()
            )
        

    def buildModel(self, pool=False):

        nf = 32
        s = 1
        return nn.Sequential(
            self.buildLayer(3, nf),
            self.buildLayer(nf, 2 * nf, stride=s, pool=pool),
            self.buildLayer(2 * nf, 4 * nf, stride=s, pool=pool),
            self.buildLayer(4 * nf, 8 * nf, stride=s, pool=pool),
            nn.Flatten(),
            nn.Linear(8 * nf * 4 * 4, 512), nn.ReLU(True), nn.BatchNorm1d(512), 
            nn.Linear(512, 10)
        )

    def predict(self):
        model = self.buildModel().cuda()
        logits = model(self.images[5])
        print (logits)

    # def dataLoaderNotebook(self):
    #     !git clone https://github.com/pater18/Project_DNN.git gitFolder



def main():

    classifier = CNN("")
    classifier.predict()
    print ("running main")
    return 0


if __name__ == "__main__":
    main()