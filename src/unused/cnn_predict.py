import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import pandas as pd 
import cv2 as cv

import cnn


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
    model = cnn.buildModel(type=type)
    model.load_state_dict(torch.load(weights_path))
    model.to(device)
    model.eval()
    return model

def ready_image(image):
    tfms_norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    cv.imshow("Image to be predicted", image)
    resized_img = cv.resize(image, (int(1280*0.2), int(1021*0.2)),interpolation=cv.INTER_AREA)
    resized_img = tfms_norm(resized_img).unsqueeze(0)
    resized_img = resized_img.to(device)
    return resized_img

def predict(class_to_predict, weights_file_path, image = None):
    

    model = load_model(class_to_predict, weights_file_path)
    csv = load_data()
    labels, class_map = set_class_map(csv, class_to_predict) 

    if image is None:
        images = csv.get('Barcode')
        
        for img_name, label in zip(images, labels):
            with torch.no_grad():
                img = cv.imread("../images/" + str(img_name) + ".jpg")
                img = ready_image(img)
                output = model(img)
                predicted = class_map[torch.argmax(output, dim=1).item()]
                print (f"Predicted output '{predicted}'")
                print (f"Correct output '{label}'\n")

                cv.waitKey(0)

    else :
        with torch.no_grad():
            img = ready_image(image)
            output = model(img)
            predicted = class_map[torch.argmax(output, dim=1).item()]
            print (f"Predicted output '{predicted}'")
            
            cv.waitKey(0)
