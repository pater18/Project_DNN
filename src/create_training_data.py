import pandas as pd
import random
import csv


def createAvgGS1Form():
    data =[]

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
    n_gs1_tube = 0


    while (n_gs1_bucket < 768):

        with open('../products.csv', 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                    
                    continue
                
                if row[5] == 'bag' and n_gs1_bag < 768:
                    data.append(row)
                    n_gs1_bag += 1
                if row[5] == 'aerosol' and n_gs1_aerosol < 768:
                    data.append(row)
                    n_gs1_aerosol += 1
                if row[5] == 'bottle' and n_gs1_bottle < 768:
                    data.append(row)
                    n_gs1_bottle += 1
                if row[5] == 'sleeve' and n_gs1_sleeve < 768:
                    data.append(row)
                    n_gs1_sleeve += 1
                if row[5] == 'box' and n_gs1_box < 768:
                    data.append(row)
                    n_gs1_box += 1
                if row[5] == 'can' and n_gs1_can < 768:
                    data.append(row)
                    n_gs1_can += 1
                if row[5] == 'tube' and n_gs1_tube < 768:
                    data.append(row)
                    n_gs1_tube += 1
                if row[5] == 'tray' and n_gs1_tray < 768:
                    data.append(row)
                    n_gs1_tray += 1
                if row[5] == 'jar' and n_gs1_jar < 768:
                    data.append(row)
                    n_gs1_jar += 1
                if row[5] == 'brick' and n_gs1_brick < 768:
                    data.append(row)
                    n_gs1_brick += 1
                if row[5] == 'bucket' and n_gs1_bucket < 768:
                    data.append(row)
                    n_gs1_bucket += 1
                if row[5] == 'cup-tub' and n_gs1_cup_tup < 768:
                    data.append(row)
                    n_gs1_cup_tup += 1
                if row[5] == 'gable-top' and n_gs1_gaple_top < 768:
                    data.append(row)
                    n_gs1_gaple_top += 1


    for i in range(10):
        random.shuffle(data)
    data = pd.DataFrame(data=data, columns=['id', 'Name', 'Barcode', 'Manufactorer', 'Amount', 'GS1 Form', 'Food-Grade', 'Material', 'Colour'])
    print (data.groupby('GS1 Form').size())
    data.to_csv('avgDistributionGS1Form.csv', index=False)
    print('')

def createAvgMaterial():

    data =[]

    n_material_plastic = 0
    n_material_fibre_based = 0 
    n_material_metal = 0
    n_material_glass = 0

    n_samples = 2496

    while (n_material_metal < n_samples):

        with open('../products.csv', 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                    continue
                
                if row[7] == 'plastic' and n_material_plastic < n_samples:
                    data.append(row)
                    n_material_plastic += 1
                if row[7] == 'fibre-based' and n_material_fibre_based < n_samples:
                    data.append(row)
                    n_material_fibre_based += 1
                if row[7] == 'glass' and n_material_glass < n_samples:
                    data.append(row)
                    n_material_glass += 1
                if row[7] == 'metal' and n_material_metal < n_samples:
                    data.append(row)
                    n_material_metal += 1
    

    for _ in range(10):
        random.shuffle(data)
    data = pd.DataFrame(data=data, columns=['id', 'Name', 'Barcode', 'Manufactorer', 'Amount', 'GS1 Form', 'Food-Grade', 'Material', 'Colour'])
    print (data.groupby('Material').size())
    data.to_csv('avgDistributionMaterial.csv', index=False)
    print('')

def createAvgColour():
    
    data =[]
    n_color_other = 0
    n_color_black = 0
    n_color_white = 0
    n_color_clear = 0

    n_samples = 2496

    while (n_color_white < n_samples):
        with open('../products.csv', 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                    continue
                
                if row[8] == 'white' and n_color_white < n_samples:
                    data.append(row)
                    n_color_white += 1
                if row[8] == 'black' and n_color_black < n_samples:
                    data.append(row)
                    n_color_black += 1
                if row[8] == 'clear' and n_color_clear < n_samples:
                    data.append(row)
                    n_color_clear += 1
                if row[8] == 'other-colours' and n_color_other < n_samples:
                    data.append(row)
                    n_color_other += 1
    
    for _ in range(10):
        random.shuffle(data)

    data = pd.DataFrame(data=data, columns=['id', 'Name', 'Barcode', 'Manufactorer', 'Amount', 'GS1 Form', 'Food-Grade', 'Material', 'Colour'])
    print (data.groupby('Colour').size())
    data.to_csv('avgDistributionColour.csv', index=False)
    print('')

def main():

    createAvgGS1Form()
    createAvgColour()
    createAvgMaterial()




if __name__ == '__main__':
    main()