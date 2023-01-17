from torch import nn

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torchvision.transforms as transforms

import pandas as pd 
import cv2 as cv

from CusomImageDataset import CustomImageDataset
from torch.utils.data import DataLoader

from tqdm import tqdm

def buildLayer(inChannels, outChannels, kernelSize=3, stride=1, 
                padding=1, useBatchnorm=True, pool=True, dropout=0):
    
    return nn.Sequential(
        nn.Conv2d(inChannels, outChannels, kernelSize, stride, padding),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(outChannels) if useBatchnorm else nn.Identity(),
        nn.MaxPool2d(2,2) if pool else nn.Identity(),
        nn.Dropout(dropout)
        )

def buildModel(type, pool=True ):

    nf = 32
    s = 1
    n_classes =0
    if type == 'Material':
        n_classes = 4
    elif type == 'GS1 Form':
        n_classes = 13
    elif type == 'Colour':
        n_classes = 4
    else:
        
        raise Exception("The correct type to classifie in buildModel() was not specified. \n \
Types are : \n 'Materials', 'GS1 Form' or 'Colour'" )

    
    return nn.Sequential(
        buildLayer(3, nf),
        buildLayer(nf, 2 * nf, stride=s, pool=pool, dropout=0.0),
        buildLayer(2 * nf, 4 * nf, stride=s, pool=pool, dropout=0.0),
        buildLayer(4 * nf, 8 * nf, stride=s, pool=pool, dropout=0.0),
        nn.Flatten(),
        nn.Dropout(0.5),
        nn.Linear(256 * 9 * 12, 3456), # (256*12*9)/8
        nn.ReLU(True), 
        nn.BatchNorm1d(3456),
        nn.Dropout(0.5),
        nn.Linear(3456, 512),
        nn.ReLU(True), 
        nn.BatchNorm1d(512),
        nn.Linear(512, n_classes)
    )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_class_map(csv, class_to_predict):
    labels = csv.get(class_to_predict)
    if (class_to_predict == 'GS1 Form'):
        return labels, {0 : 'bag', 1: 'can', 2 : 'box', 3: 'jar' , 4: 'sleeve', 5: 'bottle', 6: 'aerosol', 
                        7: 'brick', 8: 'bucket', 9: 'cup-tub', 10: 'gable-top', 11: 'tray', 12: 'tube'}

    if (class_to_predict == 'Material'):
        return labels, {0:'plastic', 1: 'fibre-based', 2 : 'metal', 3 : 'glass'}


    if (class_to_predict == 'Colour'):
        return labels, {0: 'black', 1: 'clear', 2: 'white', 3: 'other-colours'}

def load_data():
    return pd.read_csv('https://raw.githubusercontent.com/pater18/Project_DNN/main/products.csv')

def load_model(type, weights_path):
    model = buildModel(type=type)
    model.load_state_dict(torch.load(weights_path))
    model.to(device)
    model.eval()
    return model

def ready_image(image, show_image=False):
    tfms_norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if show_image:
        cv.imshow("Image to be predicted", image)
    resized_img = cv.resize(image, (int(1280*0.2), int(1021*0.2)),interpolation=cv.INTER_AREA)
    resized_img = tfms_norm(resized_img).unsqueeze(0)
    resized_img = resized_img.to(device)
    return resized_img

def predict(class_to_predict, weights_file_path, image = None, show_images=False):

    model = load_model(class_to_predict, weights_file_path)
    csv = load_data()
    labels, class_map = set_class_map(csv, class_to_predict) 

    if image is None:
        images = csv.get('Barcode')
        total, correct = 0, 0
        td = tqdm(zip(images, labels))
        for (img_name, label) in td:
            key = 0
            with torch.no_grad():
                img = cv.imread("../images/" + str(img_name) + ".jpg")
                img = ready_image(img, show_image=show_images)
                output = model(img)
                predicted = class_map[torch.argmax(output, dim=1).item()]
                if show_images:
                    print (f"\nImage nr: {img_name}")
                    print (f"Predicted output '{predicted}'")
                    print (f"Correct output '{label}'\n")

                    key = cv.waitKey(0)
                    if key =='q' or key == 27:
                        break

                total += 1
                if predicted == label:
                    correct += 1

        print(f"Total images {total}")
        print(f"Images prediced correct {correct}")

        
    else :
        with torch.no_grad():
            img = ready_image(image)
            output = model(img)
            predicted = class_map[torch.argmax(output, dim=1).item()]
            print (f"Predicted output '{predicted}'")
            
            cv.waitKey(0)

