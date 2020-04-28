'''
DM Project: Rich Gude

'''
import os                       # Used to manipulate and pull in data files
from zipfile import ZipFile     # Used to import zipped file folders
import cv2                      # Used to store jpg files
import pandas as pd             # Used for dataframe manipulation
import numpy as np
import matplotlib.pyplot as plt
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
    os.rename('Code/Diabetic', 'dataDia')  # rename created folder to 'data' folder

diaFiles = os.listdir('dataDia/Clear')
#print(diaFiles)       # to show csv file names for reference and verify filefolders is drawing correctly

# Extract the Myopia zip folder in working directory
if 'dataMyo' not in os.listdir():
    with ZipFile('Myopia.zip', 'r') as ProjObj2:
        ProjObj2.extractall()  # If we provide no arguments, the command will extract from the current folder all files
                          # into the directory.  Since the current folder comes out as a separate folder, 'Myopia',
                          # this folder with be our data folder.
    os.rename('Code/Myopia', 'dataMyo')  # rename created folder to 'data' folder

myoFiles = os.listdir('dataMyo/Clear')
#print(file_headers)       # to show csv file names for reference and verify filefolders is drawing correctly

# Extract the Normal zip folder in working directory
if 'dataNorm' not in os.listdir():
    with ZipFile('Normals.zip', 'r') as ProjObj3:
        ProjObj3.extractall()  # If we provide no arguments, the command will extract from the current folder all files
                          # into the directory.  Since the current folder comes out as a separate folder, 'Myopia',
                          # this folder with be our data folder.
    os.rename('Code/Normals', 'dataNorm')  # rename created folder to 'data' folder

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
    input = cv2.cvtColor(preIm[0:width, 0:width], cv2.COLOR_BGR2GRAY)   # crop and convert to gray scale too
    diaImages.append(input)

myoImages = []
for i in range(len(myoFiles)):
    # Prior to input, crop the images to take off bottom tag with unnecessary details (orig image is square)
    preIm = cv2.imread('dataMyo\\Clear\\' + myoFiles[i])
    height, width, channels = preIm.shape
    input = cv2.cvtColor(preIm[0:width, 0:width], cv2.COLOR_BGR2GRAY)   # crop and convert to gray scale too
    myoImages.append(input)

normImages = []
for i in range(len(normFiles)):
    # Prior to input, crop the images to take off bottom tag with unnecessary details (orig image is square)
    preIm = cv2.imread('dataNorm\\Clear\\' + normFiles[i])
    height, width, channels = preIm.shape
    input = cv2.cvtColor(preIm[0:width, 0:width], cv2.COLOR_BGR2GRAY)   # crop and convert to gray scale too
    normImages.append(input)


'''
Save each image list as a dataframe with two columns: the image and it's classification (dia, myo, or norm)
'''
dfDia = pd.DataFrame({'Image': diaImages, 'Class': ['dia']*len(diaImages)})
dfMyo = pd.DataFrame({'Image': myoImages, 'Class': ['myo']*len(myoImages)})
dfNorm = pd.DataFrame({'Image': normImages, 'Class': ['norm']*len(normImages)})
dataDF = dfDia.append(dfMyo).append(dfNorm).reset_index(drop=True)

'''
Adding features to dataframe for testing in KNN
'''
# First Feature: compress image size and flatten (in order to use as a feature)
def image_compress(image, size = (64, 64)):
    # resize the image to 64 x 64 and flatten into raw pixel intensities
    return cv2.resize(image, size).flatten()

    # Run feature creation
dataDF['comprImage'] = dataDF.apply(lambda row: image_compress(row.Image), axis=1)

# Second Feature: Color histogram of image
def extract_color_histogram(image):
    # Extract a 3D color hist into 100 bins
    #hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image], [0], None, [100], [0, 100])

    # Assume the user is using opencv3 to normalize the output values:
    cv2.normalize(hist, hist)
    return hist.flatten()

    # Run feature creation
dataDF['colorHist'] = dataDF.apply(lambda row: extract_color_histogram(row.Image), axis=1)

# Third Feature: identification of corners in image
# Possible Resizing Methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
# cv2.INTER_AREA is the best, but still not great...
def corner_image(img, thres, resMethod = cv2.INTER_AREA, size = (64, 64)):   # Thres is used to determine the scale for how contrast in the picture determines a corner
    # find Harris corners
    gray = np.float32(img)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,thres*dst.max(),255,0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

    # Make a blank map
    res = np.hstack((centroids,corners))
    res = np.int0(res)
    img[:,:] = [0]
    img[res[:,1],res[:,0]]=[255]
    # OPTION 1: Resize, normalize, and flatten the returned image into raw pixel intensities
    #img = cv2.resize(img, size, interpolation= resMethod)
    #img[img > 0] = 255
    #return img.flatten()
    # OPTION 2: Resize and flatten the returned image into raw pixel intensities
    return cv2.resize(img, size, interpolation= resMethod).flatten()


# Run feature creation
dataDF['imgCorner'] = dataDF.apply(lambda row: corner_image(row.Image, 0.5), axis=1)

# Fourth Feature: perform a threshold function to change the intensity of pixel values
def imageThres(image, thres = 70):
    ret, imgThres = cv2.threshold(image, thres, 255, cv2.THRESH_BINARY_INV)
    return imgThres.flatten()

# Run feature creation (threshold of 70 gave best results for accuracy in testing)
dataDF['thresImg'] = dataDF.apply(lambda row: imageThres(row.comprImage, 70), axis=1)
'''
If we want to check what is the best single variable to use for a feature, use this for loop
'''
#print('For given thresholds - 30, 70, 120, 170, 220')
#threshold = [30, 70, 120, 170, 220]
#for i in threshold:

########

#Testing for K-Nearest Neighbor
# Train-test-split the data
X = np.array(list(dataDF['imgCorner']))
y = np.array(list(dataDF['Class']))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state=0)

# Normalize the input (target values does not need to be normalized
#sc = StandardScaler()
#sc.fit(X_train)
#X_tr_std = sc.transform(X_train)
#X_tst = sc.transform(X_test)

# Perform KNN of X variables
knn = KNeighborsClassifier(n_neighbors=5,             # Use 5 closest neighbors
                           p=2,                     # Use Euclidean Distance
                           metric='minkowski')      # Standard Euclidean distance metric
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))






