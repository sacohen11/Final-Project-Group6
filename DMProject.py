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

def augment (x_train, y_train, f):
    unique, count= np.unique(y_train, return_counts=True)
    print("Training set:")
    print("Diabetes:", count[0])
    print("Myopia:", count[1])
    print("Normal:", count[2])
    print("Total:", len(x_train))
    print("-"*50)

    print("Data Augmentation...")

    x_train_new = []
    y_train_new = []

    print("-"*50)
    for i in range(len(le.classes_)):
        if count[i] < f:
            for k in range(len(x_train)):
                if count[y_train[k]] < f:
                        rn = random.randint(0,6)
                        if rn == 0:
                            x_train[k] = np.fliplr(x_train[k])
                            print("Flipped", y_train[k])
                            plt.imshow(x_train[k])
                            plt.show()

                        elif rn ==1:
                            x_train[k] = sp_noise(x_train[k], 0.005)
                            print("Noised", y_train[k])
                            plt.imshow(x_train[k])
                            plt.show()


                        elif rn ==2:
                            rows, cols = x_train[k].shape[:2]
                            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
                            x_train[k] = cv2.warpAffine(x_train[k], M, (cols, rows))
                            print("Rotated 30", y_train[k])
                            plt.imshow(x_train[k])
                            plt.show()


                        elif rn == 3:
                            # shifting the image 100 pixels in both dimensions
                            rows, cols = x_train[k].shape[:2]
                            M = np.float32([[1, 0, -5], [0, 1, -5]])
                            x_train[k] = cv2.warpAffine(x_train[k], M, (cols, rows))
                            print("Shifted", y_train[k])
                            plt.imshow(x_train[k])
                            plt.show()


                        elif rn == 4:
                            rows, cols = x_train[k].shape[:2]
                            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 180, 1)
                            x_train[k] = cv2.warpAffine(x_train[k], M, (cols, rows))
                            print("Rotated 180", y_train[k])
                            plt.imshow(x_train[k])
                            plt.show()

                        elif rn == 5:
                            x_train[k]= cv2.Canny(x_train[k], 200, 600)
                            print("Edge Detection", y_train[k])
                            plt.imshow(x_train[k])
                            plt.show()

                        elif rn == 6:
                            ret, x_train[k] = cv2.threshold(x_train[k], 40, 255, cv2.THRESH_BINARY_INV)   
                            print("Threshold", y_train[k])
                            plt.imshow(x_train[k])
                            plt.show()

                        x_train_new.append(x_train[k])
                        print("New image saved as:", y_train[k])
                        y_train_new.append(y_train[k])
                        print("Labeled as:", y_train[k])
                        count[y_train[k]] += 1
                        print("-" * 50)

    x_train_new = np.array(x_train_new)
    x_train = np.concatenate((x_train, x_train_new))

    y_train_new = np.array(y_train_new)
    y_train = np.concatenate((y_train, y_train_new))
    print("-"*50)

    unique, count= np.unique(y_train, return_counts=True)
    print("Training set after data augmentation:")
    print("Diabetes:", count[0])
    print("Myopia:", count[1])
    print("Normal:", count[2])
    print("Total:", len(x_train))
    print("-"*50)


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
            img = cv2.imread(image_path,0)
            img = image_resize(img, height=400)
            img = img[0:380, :]
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

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 40, test_size = 0.20, stratify = y)

#Data Augmentation
augment(x_train, y_train, 90)

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

import pandas as pd
import seaborn as sns
conf_matrix = confusion_matrix(y_test, y_pred)
class_names = np.unique(label_data)

df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )
plt.figure(figsize=(6,6))
hm = sns.heatmap(df_cm, cmap="Blues", cbar=False,annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
plt.ylabel('True label',fontsize=15)
plt.xlabel('Predicted label',fontsize=15)
# Show heat map
plt.tight_layout()
plt.show()


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


from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# Perform KNN of X variables
knn = KNeighborsClassifier(n_neighbors=16)      # Standard Euclidean distance metric

knn.fit(x_train, y_train)

y_pred_knn = knn.predict(x_test)


# calculate metrics

print("\n")
print("Classification Report: ")
print(classification_report(y_test,y_pred_knn))
print("\n")


print("Accuracy : ", accuracy_score(y_test, y_pred_knn) * 100)
print("\n")



conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
class_names = np.unique(label_data)

df_knn = pd.DataFrame(conf_matrix_knn, index=class_names, columns=class_names )
plt.figure(figsize=(6,6))
hm = sns.heatmap(df_knn, cmap="Blues", cbar=False,annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_knn.columns, xticklabels=df_knn.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
plt.ylabel('True label',fontsize=15)
plt.xlabel('Predicted label',fontsize=15)
# Show heat map
plt.tight_layout()
plt.show()

from sklearn.neighbors import kneighbors_graph
A = kneighbors_graph(x_train, 2, mode='connectivity', include_self=True)
print(A)

from sklearn.model_selection import cross_val_score
import numpy as np
#create a new KNN model
knn_cv = KNeighborsClassifier(n_neighbors=3)
#train model with cv of 5
cv_scores = cross_val_score(knn_cv, x_train, y_train, cv=5)
#print each cv score (accuracy) and average them
print(cv_scores)
print("cv_scores mean:{}".format(np.mean(cv_scores)))


from sklearn.model_selection import GridSearchCV
#create new a knn model
knn2 = KNeighborsClassifier()
#create a dictionary of all values we want to test for n_neighbors
param_grid = {"n_neighbors": np.arange(1, 25)}
#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
#fit model to data
knn_gscv.fit(x_train, y_train)

#check top performing n_neighbors value
print(knn_gscv.best_params_)


from sklearn.tree import DecisionTreeClassifier
# perform training with entropy.
# Decision tree with entropy
clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)

# Performing training
clf_entropy.fit(x_train, y_train)
# predicton on test using entropy
y_pred_entropy = clf_entropy.predict(x_test)
# calculate metrics entropy model
print("DATASET USED IS (GET TYPE TO BE TEXT)")
print("\n")
print("Results Using Entropy: \n")
print("Classification Report: ")
print(classification_report(y_test, y_pred_entropy))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_entropy) * 100)
print('-' * 80 + '\n')
print("Confusion Matrix")
cmx = confusion_matrix(y_test, y_pred_entropy)
print(cmx)







# count = 0
# for i in range(len(x_train)):
#     if y_train[i] == 2:
#         for k in x_train:
#             if count < 17:
#                     # k = np.fliplr(k)
#                     # k = sp_noise(k, 0.05)
#                     rows, cols = k.shape[:2]
#                     M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
#                     k = cv2.warpAffine(k, M, (cols, rows))
#                     x_train_new.append(k)
#                     y_train_new.append(2)
#                     count += 1
#             else:
#                 break






#Source: https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python
