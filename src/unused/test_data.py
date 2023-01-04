import numpy as np
import csv
import pandas as pd 
import cv2 as cv

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_data(labels):
    csv = pd.read_csv('https://raw.githubusercontent.com/pater18/Project_DNN/main/products.csv')
            
    if (labels == 'GS1 Form'):
        img_labels = csv.get('GS1 Form')
        class_map = {'bag': 0, 'can': 1, 'box' : 2, 'jar' : 3, 'sleeve': 4, 'bottle': 5, 'aerosol': 6, 
                        'brick': 7, 'bucket' : 8, 'cup-tub': 9, 'gable-top' : 10, 'tray' : 11, 'tube': 12}

    if (labels == 'Material'):
        img_labels = csv.get('Material')
        class_map = {'plastic': 0, 'fibre-based': 1, 'metal': 3, 'glass': 4}

    if (labels == 'Colour'):
        img_labels = csv.get('Colour')
        class_map = {'black': 0, 'clear' : 1, 'white' : 2, 'other-colours': 3}

    img_dir = csv.get('Barcode')

    data = []

    for i, label in enumerate(img_labels):
            data.append([str(img_dir[i]) + '.jpg', str(label)])     

    return data, class_map

def get_values(data, class_map, idx):
    img_path, class_name = data[idx]
    img = cv.imread("../images/" + img_path)
    resized_img = cv.resize(img, (1280,1021))
    resized_img = np.array(resized_img)
    label = torch.tensor(class_map[class_name])

    return resized_img, label


data, class_map = get_data('Material')
for i in range(1998):
    img, label = get_values(data, class_map, i)
    if label is None:
        print (i)
        print (label)

'''    
data =[]

with open('../products.csv', 'r') as f:
    csv_reader = csv.reader(f, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
            print ((row[2], row[5], row[6], row[7], row[8]))
        else:
            data.append((row[2], row[5], row[6], row[7], row[8]))


n_color_other = 0
n_color_black = 0
n_color_white = 0
n_color_clear = 0

n_gs1_bag = 0
n_gs1_can = 0
n_gs1_box = 0
n_gs1_jar = 0
n_gs1_sleeve = 0
n_gs1_bottle = 0
n_gs1_aerosol = 0
n_gs1_brick = 0
n_gs1_bucket = 0
n_gs1_cup_tup = 0
n_gs1_gaple_top = 0
n_gs1_tray = 0
n_gs1_tupe = 0


n_material_plastic = 0
n_material_fibre_based = 0 
n_material_metal = 0
n_material_glass = 0


for row in data:
    if (row[1] == 'bag'):
        n_gs1_bag += 1
    elif (row[1] == 'can'):
        n_gs1_can += 1
    elif (row[1] == 'box'):
        n_gs1_box += 1
    elif (row[1] == 'jar'):
        n_gs1_jar += 1
    elif (row[1] == 'sleeve'):
        n_gs1_sleeve += 1
    elif (row[1] == 'bottle'):
        n_gs1_bottle += 1
    elif (row[1] == 'aerosol'):
        n_gs1_aerosol += 1
    elif (row[1] == 'brick'):
        n_gs1_brick += 1
    elif (row[1] == 'bucket'):
        n_gs1_bucket += 1
    elif (row[1] == 'cup-tub'):
        n_gs1_cup_tup += 1
    elif (row[1] == 'gable-top'):
        n_gs1_gaple_top += 1
    elif (row[1] == 'tray'):
        n_gs1_tray += 1
    elif (row[1] == 'tube'):
        n_gs1_tupe += 1
    
    if (row[3] == 'plastic'):
        n_material_plastic += 1
    elif (row[3] == 'fibre-based'):
        n_material_fibre_based += 1
    elif (row[3] == 'metal'):
        n_material_metal += 1
    elif (row[3] == 'glass'):
        n_material_glass += 1

    if (row[4] == 'black'):
        n_color_black += 1
    elif (row[4] == 'clear'):
        n_color_clear += 1
    elif (row[4] == 'white'):
        n_color_white += 1
    elif (row[4] == 'other-colours'):
        n_color_other += 1


print (f"Box: {n_gs1_box}")
print (f"Can: {n_gs1_can}")
print (f"Bottle: {n_gs1_bottle}")
print (f"Bag: {n_gs1_bag}")
print (f"Aerosol: {n_gs1_aerosol}")
print (f"Brick: {n_gs1_brick}")
print (f"Bucket: {n_gs1_bucket}")
print (f"Cup-Tup: {n_gs1_cup_tup}")
print (f"Jar: {n_gs1_jar}")
print (f"Sleeve: {n_gs1_sleeve}")
print (f"Tray: {n_gs1_tray}")
print (f"Tupe: {n_gs1_tupe}")
print (f"Gaple top: {n_gs1_gaple_top}")

n_total_types = n_gs1_bag + n_gs1_can + n_gs1_box + n_gs1_jar + n_gs1_sleeve + n_gs1_bottle + n_gs1_aerosol + n_gs1_brick + n_gs1_bucket + n_gs1_cup_tup + n_gs1_gaple_top + n_gs1_tray + n_gs1_tupe 
print (f"Total number of types = {n_total_types}")

print (f"\nPlastic: { n_material_plastic}")
print (f"Fibre Based: { n_material_fibre_based}")
print (f"Glass: { n_material_glass}")
print (f"Metal: { n_material_metal}")

n_total_materials = n_material_metal + n_material_fibre_based + n_material_glass + n_material_plastic
print(f"Total materials {n_total_materials}")

print (f"\nBlack {n_color_black}")
print(f"White {n_color_white}")
print(f"Clear {n_color_clear}")
print(f"Other {n_color_other}")

n_total_colors = n_color_white + n_color_black + n_color_clear + n_color_other
print(f"Total number of colors {n_total_colors}")

'''