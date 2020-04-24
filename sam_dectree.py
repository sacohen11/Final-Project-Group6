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
import numpy as np
from sklearn import preprocessing
from skimage.transform import rotate
from skimage.color import rgb2grey
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

cwd = os.getcwd()
print(cwd)
image = cv2.imread(cwd + "/dataNorm/Marked/Normal_marked (1).jpg")


#Preprocessing
print("Starting images and label pre-processing...")
label_data = []
images_data = []

for subdir, dirs, files in os.walk(cwd):
    for file in files:
        if "Clear" in subdir:
            # Image preprocessing
            image_path = os.path.join(subdir, file)
            img = cv2.imread(image_path,0)
            img = cv2.resize(img, (400, 400))
        #img.convertTo(img, cv2.CV_32F, 1.0 / 255.0)

            #img = cv2.imread(os.getcwd() + r'/dataMyo/Clear/Myopia_clear (68).jpg',0)

            # 35 is the major number to change
            ret, thresh2 = cv2.threshold(img,30,255,cv2.THRESH_BINARY_INV)
            #thresh2 = cv2.cvtColor(thresh2, cv2.COLOR_GRAY2BGR)
            contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # find the biggest countour (c) by the area
            c = max(contours, key=cv2.contourArea)

            #x, y, w, h = cv2.boundingRect(c)
            # draw the biggest contour (c) in green
            thresh2 = cv2.drawContours(thresh2, [c], 0, (255,255,255), thickness=28, lineType=cv2.LINE_AA )
            images_data.append(thresh2)
            #result_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            #thresh2 = cv2.threshold(thresh2,40,255,cv2.THRESH_BINARY_INV)
            #cv2.rectangle(thresh2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #cv2.drawContours(thresh2, contours, -1, (0,255,0), 3)
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
            #cv2.imshow('new', img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            # Labels preprocessing
            label = (subdir.split("eye-miner/")[1])
            label = (label.split("/")[0])
            label_data.append(label)
           # plt.subplot(121), plt.imshow(thresh2)
           # plt.title(label)

           # plt.show()

print("Images and labels successfully preprocessed!")
print("")
print("-"*50)

# look at labels and images shape
label_data = np.array(label_data)
print("Labels shape:", label_data.shape)
images_data = np.array(images_data)
print("Images shape:", images_data.shape)

le = preprocessing.LabelEncoder()
le.fit(label_data)
integer_labels = le.transform(label_data)

#Confirm we have 3 unique classes
print('Unique classes:',le.classes_)

#Save the images and labels
np.save("x_train.npy", images_data)
np.save("y_train.npy", integer_labels)

print("")
print("-"*50)

x, y = np.load("x_train.npy", allow_pickle=True), np.load("y_train.npy", allow_pickle=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 40, test_size = 0.2, stratify = y)
# Preprocessing: reshape the image data into rows
x_train = np.reshape(x_train, (x_train.shape[0], -1))
print('Training data shape: ', x_train.shape)

x_test = np.reshape(x_test, (x_test.shape[0], -1))
print('Test data shape: ', x_test.shape)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)

# perform training with entropy.
# Decision tree with entropy
clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)

# Performing training
clf_entropy.fit(x_train, y_train)
# predicton on test using entropy
y_pred_entropy = clf_entropy.predict(x_test)
# calculate metrics entropy model
print("\n")
print("Results Using Entropy: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred_entropy))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_entropy) * 100)
print ('-'*80 + '\n')
print("Confusion Matrix")
cmx = confusion_matrix(y_test, y_pred_entropy)
print(cmx)