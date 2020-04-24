'''
DM Project: Rich Gude

'''
import os                       # Used to manipulate and pull in data files
from zipfile import ZipFile     # Used to import zipped file folders
import cv2                      # Used to store jpg files
import pandas as pd             # Used for dataframe manipulation
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors


# Extract the Diabetic zip folder in working directory
if 'dataDia' not in os.listdir():
    with ZipFile('Diabetic.zip', 'r') as ProjObj1:
        ProjObj1.extractall()  # If we provide no arguments, the command will extract from the current folder all files
                          # into the directory.  Since the current folder comes out as a separate folder, 'Diabetic',
                          # this folder with be our data folder.
    os.rename('Diabetic', 'dataDia')  # rename created folder to 'data' folder

diaFiles = os.listdir('dataDia/Clear')
#print(diaFiles)       # to show csv file names for reference and verify filefolders is drawing correctly

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
Fill in lists with PIL images for each diagnosis and crop before inserting
'''
diaImages = []
for i in range(len(diaFiles)):
    # Prior to input, crop the images to take off bottom tag with unnecessary details (orig image is square)
    preIm = cv2.imread('dataDia\\Clear\\' + diaFiles[i])
    height, width, channels = preIm.shape
    input = preIm[0:width, 0:width]
    diaImages.append(input)

myoImages = []
for i in range(len(myoFiles)):
    # Prior to input, crop the images to take off bottom tag with unnecessary details (orig image is square)
    preIm = cv2.imread('dataMyo\\Clear\\' + myoFiles[i])
    height, width, channels = preIm.shape
    input = preIm[0:width, 0:width]
    myoImages.append(input)

normImages = []
for i in range(len(normFiles)):
    # Prior to input, crop the images to take off bottom tag with unnecessary details (orig image is square)
    preIm = cv2.imread('dataNorm\\Clear\\' + normFiles[i])
    height, width, channels = preIm.shape
    input = preIm[0:width, 0:width]
    normImages.append(input)


'''
Save each image list as a dataframe with two columns: the image and it's classification (dia, myo, or norm)
'''
dfDia = pd.DataFrame({'Image': diaImages, 'Class': ['dia']*len(diaImages)})
dfMyo = pd.DataFrame({'Image': myoImages, 'Class': ['myo']*len(myoImages)})
dfNorm = pd.DataFrame({'Image': normImages, 'Class': ['norm']*len(normImages)})
dataDF = dfDia.append(dfMyo).append(dfNorm).reset_index(drop=True)

'''
Testing for K-Nearest Neighbor
'''
# Train-test-split the data
X = np.array(dataDF['Image'])
y = np.array(dataDF['Class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state=0)

# Normalize the input (target values does not need to be normalized
sc = StandardScaler()
sc.fit(X_train)
X_tr_std = sc.transform(X_train)
X_tst = sc.transform(X_test)

# Perform KNN of X variables
knn = KNeighborsClassifier(n_neighbors=5,           # Use 5 closest neighbors
                           p=2,                     # Use Euclidean Distance
                           metric='minkowski')      # Standard Euclidean distance metric
knn.fit(X_tr_std, y_train)
print(knn.score(X_tr_std, y_test))







