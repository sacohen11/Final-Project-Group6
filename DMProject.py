'''
DM Project: Rich Gude, Sam Cohen, Luis Ahumada

'''

import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import cv2
import os

cwd = os.getcwd()
image = cv2.imread(cwd + "/dataNorm/Marked/Normal_marked (1).jpg")


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
            img = cv2.resize(img, (400, 400))

            #cropping the images
            #img = img[0:395, 0:400]

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
from sklearn.metrics import confusion_matrix


x, y = np.load("x_train.npy"), np.load("y_train.npy")

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 40, test_size = 0.2, stratify = y)


#Create a svm Classifier
clf = svm.SVC(kernel="linear", gamma=0.1) #decision_function_shape='ovo'

# Preprocessing: reshape the image data into rows
x_train = np.reshape(x_train, (x_train.shape[0], -1))
print('Training data shape: ', x_train.shape)

x_test = np.reshape(x_test, (x_test.shape[0], -1))
print('Test data shape: ', x_test.shape)

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

print("Confusion Matrix")
cmx = confusion_matrix(y_test, y_pred)
print(cmx)




#Source: https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python
