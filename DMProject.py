'''
DM Project: Rich Gude, Sam Cohen, Luis Ahumada

'''
import os
from zipfile import ZipFile
import csv
from PIL import Image

# Extract the Diabetic zip folder in working directory
if 'dataDia' not in os.listdir():
    with ZipFile('Diabetic.zip', 'r') as ProjObj1:
        ProjObj1.extractall()  # If we provide no arguments, the command will extract from the current folder all files
                          # into the directory.  Since the current folder comes out as a separate folder, 'Diabetic',
                          # this folder with be our data folder.
    os.rename('Diabetic', 'dataDia')  # rename created folder to 'data' folder

diaFiles = os.listdir('dataDia/Clear')
print(diaFiles)       # to show csv file names for reference and verify filefolders is drawing correctly

# Extract the Myopia zip folder in working directory
if 'dataMyo' not in os.listdir():
    with ZipFile('Myopia.zip', 'r') as ProjObj2:
        ProjObj2.extractall()  # If we provide no arguments, the command will extract from the current folder all files
                          # into the directory.  Since the current folder comes out as a separate folder, 'Myopia',
                          # this folder with be our data folder.
    os.rename('Myopia', 'dataMyo')  # rename created folder to 'data' folder

myoFiles = os.listdir('dataMyo/Clear')
#print(file_headers)       # to show csv file names for reference and verify filefolders is drawing correctly

# Extract the Normal zip folder in working directory
if 'dataNorm' not in os.listdir():
    with ZipFile('Normals.zip', 'r') as ProjObj3:
        ProjObj3.extractall()  # If we provide no arguments, the command will extract from the current folder all files
                          # into the directory.  Since the current folder comes out as a separate folder, 'Myopia',
                          # this folder with be our data folder.
    os.rename('Normals', 'dataNorm')  # rename created folder to 'data' folder

normFiles = os.listdir('dataNorm/Clear')
#print(file_headers)       # to show csv file names for reference and verify filefolders is drawing correctly

'''
for files in file_headers:
    with open(('dataDia\\Clear\\' + file_headers[i]), 'r') as f:        # append 'data' to the file name to access the data folder
        reader = csv.DictReader(f)
'''
diaImages = []
myoImages = []
normImages = []
for i in range(2):
    diaImages.append(Image.open('dataDia\\Clear\\' + diaFiles[i]))
    myoImages.append(Image.open('dataMyo\\Clear\\' + myoFiles[i]))
    normImages.append(Image.open('dataNorm\\Clear\\' + normFiles[i]))
diaImages[1].show()
myoImages[1].show()
normImages[1].show()
