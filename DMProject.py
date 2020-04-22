'''
DM Project: Rich Gude, Sam Cohen, Luis Ahumada

'''

import numpy as np
from sklearn import preprocessing
from skimage.transform import rotate
from skimage.color import rgb2grey
import matplotlib.pyplot as plt
import cv2
import os
import random

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output



cwd = os.getcwd()
# image = cv2.imread(cwd + "/dataNorm/Marked/Normal_marked (1).jpg")

#Preprocessing
print("Starting images and label pre-processing...")
label_data = []
images_data = []

for subdir, dirs, files in os.walk(cwd):
    for file in files:
        if "Clear" in subdir:
            #Image preprocessing
            image_path = os.path.join(subdir, file)
            img = cv2.imread(image_path)
            img = image_resize(img, height=100)
            img = img[0:95, :]
            images_data.append(img)

            #Labels preprocessing
            label = (subdir.split("eye-miner/")[1])
            label = (label.split("/")[0])
            label_data.append(label)

print("Images and labels successfully preprocessed!")
print("")
print("-"*50)

# look at labels and images shape
label_data = np.array(label_data)
print("Labels shape:", label_data.shape)
images_data = np.array(images_data)
print("Images shape:", images_data.shape)

#304 images of 400x400 pixels, and 3 channels (RGB)

#One-hot encoding: Convert text-based labels to numbers
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

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix

x, y = np.load("x_train.npy"), np.load("y_train.npy")

print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 40, test_size = 0.20, stratify = y)

#data augmentation
a = np.random.randint(0,150)
b = np.random.randint(150,300)
# print(a,b)

x_train2 = []
for i in x_train[a:b]:
      # i = np.fliplr(i)
      # i = sp_noise(i, 0.05)
      # rows, cols = i.shape[:2]
      # M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
      # i = cv2.warpAffine(i, M, (cols, rows))
      x_train2.append(i)

x_train2 = np.array(x_train2)
x_train[a:b] = x_train2

plt.imshow(x_train[150])
plt.show()

# Preprocessing: reshape the image data into rows
x_train = np.reshape(x_train, (x_train.shape[0], -1))
print('Training data shape: ', x_train.shape)

x_test = np.reshape(x_test, (x_test.shape[0], -1))
print('Test data shape: ', x_test.shape)

#Standardizing the features
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)


#Create a svm Classifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OutputCodeClassifier

# clf = svm.LinearSVC(multi_class="ovr")
# clf = svm.SVC(kernel='poly')
# clf = svm.SVC(kernel='rbf')
# clf = svm.SVC(kernel='linear')
# clf = svm.SVC(kernel='sigmoid')
clf = svm.NuSVC(kernel="linear")
# clf = svm.LinearSVC(multi_class='crammer_singer') #, decision_function_shape='ovo')
#clf = OneVsRestClassifier(LinearSVC(random_state=0))
#clf = OneVsOneClassifier(LinearSVC(random_state=0))
#clf = OutputCodeClassifier(LinearSVC(random_state=0),code_size=2, random_state=0)

#Train the model using the training sets
clf.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)

print("")
print("-"*50)
print("Model Accuracy:")
# Model Accuracy: how often is the classifier correct?
print("Accuracy:", round(metrics.accuracy_score(y_test, y_pred), 3))
print("")

print("-"*50)
#Confusion Matrix
print("Confusion Matrix")
cmx = confusion_matrix(y_test, y_pred)
print(cmx)

print("-"*50)
print("Classification Report")
cfrp = classification_report(y_test, y_pred)
print(cfrp)



#Cross Validation Score
from sklearn.model_selection import cross_val_score
print("-"*50)
print("Cross Validation Score")
accuracies = cross_val_score(estimator = svm.NuSVC(kernel="linear"), X = x_train, y = y_train, cv = 10)
print(accuracies)

print("Mean of Accuracies")
print(accuracies.mean())

print("STD of Accuracies")
print(accuracies.std())



#Grid Search (not useful bc we are not changing gamma)
#
# from sklearn.model_selection import GridSearchCV
#
# #Setting parameters for grid
# param_grid = {"C": [0.1,1,10,100,1000], "gamma": [0.1,0.3,0.5,0.7,0.9], "kernel":["rbf"]}
#
# grid = GridSearchCV(estimator = svm.SVC(), param_grid = param_grid, scoring = "accuracy", refit=True, verbose = 3)
#
# #Accuracy Grid
# print("Accuracy Grid")
# print(grid.fit(x_train, y_train))
#
# #Best parameter from Grid
# print("Best Parameter")
# print(grid.best_params_)
#

















#Source: https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python
