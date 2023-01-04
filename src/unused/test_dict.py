import pandas as pd

csv = pd.read_csv('https://raw.githubusercontent.com/pater18/Project_DNN/main/products.csv')

img_labels = {'Material' : csv.get('Material'), 'GS1 Form' : csv.get('GS1 Form'), 'Colour' : csv.get('Colour')}

labels = img_labels['Material']

print (labels[0])

