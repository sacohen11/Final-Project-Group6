'''
DM Project: Rich Gude, Sam Cohen, Luis Ahumada

'''
import os
from zipfile import ZipFile
import csv
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
import cv2
# Extract the Diabetic zip folder in working directory
if 'dataDia' not in os.listdir():
    with ZipFile('Diabetic.zip', 'r') as ProjObj1:
        ProjObj1.extractall()  # If we provide no arguments, the command will extract from the current folder all files
                          # into the directory.  Since the current folder comes out as a separate folder, 'Diabetic',
                          # this folder with be our data folder.
        listOfiles = ProjObj1.namelist()
        # Iterate over the list of file names in given list & print them
        for elem in listOfiles:
            print(elem)
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
for files in file_headers:
    with open(('dataDia\\Clear\\' + file_headers[i]), 'r') as f:        # append 'data' to the file name to access the data folder
        reader = csv.DictReader(f)
'''
diaImages = []
myoImages = []
normImages = []
print(diaFiles[0])
for i in range(25):
    diaImages.append(Image.open(os.getcwd() + r'/dataDia/Clear/' + diaFiles[i]))
    #myoImages.append(Image.open('dataMyo\\Clear\\' + myoFiles[i]))
    #normImages.append(Image.open('dataNorm\\Clear\\' + normFiles[i]))
#diaImages[3].show()
#myoImages[1].show()
#normImages[1].show()
#print(diaImages)
image_gray = diaImages[22]
print("hey")
#plt.imshow(image_gray)
#plt.show()
#blobs_log = blob_log(image_gray, threshold=.4)
#print(len(blobs_log))

img = cv2.imread(os.getcwd() + r'/dataMyo/Clear/Myopia_clear (68).jpg',0)
plt.imshow(img, cmap='gray')
plt.show()
#img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh2 = cv2.threshold(img,70,255,cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(thresh2, contours, -1, (0,255,0), 3)
#ret,thresh2 = cv2.threshold(thresh2,127,255,cv2.THRESH_BINARY_INV)
#img = cv2.Laplacian(img,cv2.CV_64F)
#edges = cv2.Canny(img,100,200)

#circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1.2,200, minRadius=100,maxRadius=300)
#print(circles)     
#circles = np.uint16(np.around(circles))
#for i in circles[0,:]:
    # draw the outer circle
    #cv2.circle(img,(i[0],i[1]),int(round(i[2]/3)),(0,255,0),2)
    # draw the center of the circle
    #cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

#fig, ax = plt.subplots(1, 1)
#plt.imshow(image_gray)
#for blob in blobs_log:
    #y, x, r, z = blob
    #c = plt.Circle((x, y), r+5, color='lime', linewidth=2, fill=False)
    #ax.add_patch(c)
plt.subplot(121),plt.imshow(thresh2, 'gray')
plt.title('Original Image')
#plt.subplot(122), plt.imshow(edges, cmap='gray')
#plt.title(' new')
plt.show()
#cv2.imshow('new', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()