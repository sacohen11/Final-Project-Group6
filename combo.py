'''
DM Project: Rich Gude, Sam Cohen, Luis Ahumada
'''

import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import cv2
import os
import random
import warnings
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
warnings.filterwarnings("ignore")

#::------------------------------------------------------------------------------------
##FUNCTIONS
#::------------------------------------------------------------------------------------

def mode(array):
    '''
        Calculates the mode prediction for each image.
        Searches through each model's predictions and finds the most commmon classification.
        Returns an array of predictions.
        Source: https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/
        '''
    output_array = []
    length_array = len(array)
    for i in range(0,len(array[0])):
        array = np.array(array)
        new_array = array[0:length_array, i]
        count0 = 0;
        count1 = 0;
        count2 = 0;
        for i in new_array:
            if i == 0:
                count0 +=1
            if i == 1:
                count1 +=1
            if i == 2:
                count2 += 1
        maximum = max(count0, count1, count2)
        doubles = 0
        return_value = 3;
        if maximum == count0:
            doubles += 1
            return_value = 0
        if maximum == count1:
            doubles += 1
            return_value = 1
        if maximum == count2:
            doubles +=1
            return_value = 2
        if doubles > 1:
            if maximum == count0 & maximum == count1:
                output_array.append(random.randint(0,2))
            elif maximum == count1 & maximum == count2:
                output_array.append(random.randint(1, 2))
            elif maximum == count0 & maximum == count2:
                output_array.append(random.randrange(0, 2, 2))
            else:
                output_array.append(random.randint(0, 2))
        else:
            output_array.append(return_value)
    return output_array

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    '''
    Resizes images keeping proportions.
    Source: https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
    '''
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
    prob: Probability of the noise.
    Source: https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
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
    '''
    Augment quantities of minority classes to "f".
    New images are either rotated 90, rotated 180, flipped, switched or noised.
    '''

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
                        rn = random.randint(0,4)
                        if rn == 0:
                            x_train[k] = np.fliplr(x_train[k])
                            # print("Flipped", y_train[k])
                            # plt.imshow(x_train[k])
                            # plt.show()

                        elif rn ==1:
                            x_train[k] = sp_noise(x_train[k], 0.005)
                            # print("Noised", y_train[k])
                            # plt.imshow(x_train[k])
                            # plt.show()


                        elif rn ==2:
                            '''
                            Rotates in 90 degrees
                            Source: https://www.programcreek.com/python/example/89459/cv2.getRotationMatrix2D
                            '''
                            rows, cols = x_train[k].shape[:2]
                            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
                            x_train[k] = cv2.warpAffine(x_train[k], M, (cols, rows))
                            # print("Rotated 30", y_train[k])
                            # plt.imshow(x_train[k])
                            # plt.show()


                        elif rn == 3:
                            '''
                            Translation of image
                            Source: http: // wiki.lofarolabs.com / index.php / Translation_of_image
                            '''
                            # shifting the image 100 pixels in both dimensions
                            rows, cols = x_train[k].shape[:2]
                            M = np.float32([[1, 0, -5], [0, 1, -5]])
                            x_train[k] = cv2.warpAffine(x_train[k], M, (cols, rows))
                            # print("Shifted", y_train[k])
                            # plt.imshow(x_train[k])
                            # plt.show()


                        elif rn == 4:
                            '''
                            Rotates in 180 degrees
                            Source: https://www.programcreek.com/python/example/89459/cv2.getRotationMatrix2D
                            '''
                            rows, cols = x_train[k].shape[:2]
                            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 180, 1)
                            x_train[k] = cv2.warpAffine(x_train[k], M, (cols, rows))
                            # print("Rotated 180", y_train[k])
                            # plt.imshow(x_train[k])
                            # plt.show()

                        # elif rn == 5:
                        #     x_train[k]= cv2.Canny(x_train[k], 200, 600)
                        #     print("Edge Detection", y_train[k])
                        #     plt.imshow(x_train[k])
                        #     plt.show()
                        #
                        # elif rn == 6:
                        #     ret, x_train[k] = cv2.threshold(x_train[k], 40, 255, cv2.THRESH_BINARY_INV)
                        #     print("Threshold", y_train[k])
                        #     plt.imshow(x_train[k])
                        #     plt.show()

                        x_train_new.append(x_train[k])
                        # print("New image saved as:", y_train[k])
                        y_train_new.append(y_train[k])
                        # print("Labeled as:", y_train[k])
                        count[y_train[k]] += 1
                        # print("-" * 50)

    x_train_new = np.array(x_train_new)
    x_train = np.concatenate((x_train, x_train_new))

    y_train_new = np.array(y_train_new)
    y_train = np.concatenate((y_train, y_train_new))

    print("Training set after data augmentation:")
    print("Diabetes:", count[0])
    print("Myopia:", count[1])
    print("Normal:", count[2])
    print("Total:", len(x_train))
    return x_train, y_train

def namestr(obj, namespace):
    '''
    Returns the name of an object.
    Source: https://stackoverflow.com/questions/1538342/how-can-i-get-the-name-of-an-object-in-python
    '''
    return [name for name in namespace if namespace[name] is obj]

def no_transformation(img):
    '''
        Returns the original image.
        '''
    return img

def edge_detection(img):
    '''
        Performs a Canny Edge Detection Algorithm.
        Returns an image with only the major edges included.
        More information here: https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html
                '''
    return cv2.Canny(img, 200, 600)

def feature_creation(img):
    '''
        Performs the KAZE feature detection algorithm.
        This algorithm finds the keypoints/features of the image.
        A vector of keypoints is returned.
        Source: https://medium.com/machine-learning-world/feature-extraction-and-similar-image-search-with-opencv-for-newbies-3c59796bf774
            '''
    creator = cv2.KAZE_create()
    # detect
    kps = creator.detect(img)
    vector_size = 32
    kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
    # computing descriptors vector
    kps, img_feature_creation = creator.compute(img, kps)
    # Flatten all of them in one big vector - our feature vector
    img_feature_creation = img_feature_creation.flatten()
    # Making descriptor of same size
    # Descriptor vector size is 64
    needed_size = (vector_size * 64)
    if img_feature_creation.size < needed_size:
        # if we have less the 32 descriptors then just adding zeros at the
        # end of our feature vector
        img_feature_creation = np.concatenate(
            [img_feature_creation, np.zeros(needed_size - img_feature_creation.size)])
    return img_feature_creation

def threshold(img):
    '''
        Inverts color scheme (black = white) based on a threshold.
        The second parameter (first number after img) is the threshold value.
        Any pixel above that value will be black, anything below will be white.
        Returns a black and white image.
            '''
    ret, img_threshold = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY_INV)
    return img_threshold

#::------------------------------------------------------------------------------------
##PREPROCESSING
#::------------------------------------------------------------------------------------

cwd = os.getcwd()

print("Starting images and label pre-processing...")

# create empty arrays
label_data = []
images_data_no_preprocess = []
images_data_no_preprocess_cropped = []
images_data_edge_detect = []
images_data_feature_creation = []
images_data_feature_creation_cropped = []
images_data_threshold = []
images_data_threshold_cropped = []

for subdir, dirs, files in os.walk(cwd):
    for file in files:
        if "Clear" in subdir:
            # Image preprocessing
            image_path = os.path.join(subdir, file)
            # read in image in grayscale
            img = cv2.imread(image_path, 0)
            try:
                # crop out the bottom of the image
                # create two resized images: one original, one zoomed

                # original image
                height, width = img.shape
                img = img[0:width, 0:width]

                # zoomed/cropped image
                img_cropped = img[150:300, 100:300]

                # resize both images
                img_resized = image_resize(img, height=400)
                img_resized_cropped = image_resize(img_cropped, height=400)

                # PREPROCESSING: No transformation
                img_no_preprocess = no_transformation(img_resized)

                # PREPROCESSING: no preprocess cropped
                img_no_preprocess_cropped = no_transformation(img_resized_cropped)

                # PREPROCESSING: edge detect
                img_edge_detect = edge_detection(img_resized)

                # PREPROCESSING: feature creation
                img_feature_creation = feature_creation(img_resized)

                # PREPROCESSING: feature creation cropped
                img_feature_creation_cropped = feature_creation(img_resized_cropped)

                # PREPROCESSING: threshold
                img_threshold = threshold(img_resized)

                # PREPROCESSING: threshold cropped
                img_threshold_cropped = threshold(img_resized_cropped)

                # APPENDING IMAGES TO ARRAYS
                images_data_no_preprocess.append(img_no_preprocess)
                images_data_no_preprocess_cropped.append(img_no_preprocess_cropped)
                images_data_edge_detect.append(img_edge_detect)
                images_data_feature_creation.append(img_feature_creation)
                images_data_feature_creation_cropped.append(img_feature_creation_cropped)
                images_data_threshold.append(img_threshold)
                images_data_threshold_cropped.append(img_threshold_cropped)

                # Labels preprocessing
                label = (subdir.split("Final-Project-Group6/")[1])
                label = (label.split("/")[0])
                label_data.append(label)

            except AttributeError:
                print("shape not found")
            except TypeError:
                print("object is not subscriptable")

print("-"*50)

# look at labels and images shape
label_data = np.array(label_data)
print("Labels shape:", label_data.shape)
no_preprocess = np.array(images_data_no_preprocess)
print("Images No Transformation shape:", no_preprocess.shape)
no_preprocess_cropped = np.array(images_data_no_preprocess_cropped)
print("Images Zoom shape:", no_preprocess_cropped.shape)
feature_creation = np.array(images_data_feature_creation)
print("Images Feature Creation shape:", feature_creation.shape)
feature_creation_cropped = np.array(images_data_feature_creation_cropped)
print("Images Feature Creation Cropped shape:", feature_creation_cropped.shape)
edge_detect = np.array(images_data_edge_detect)
print("Images Edge Detect shape:", edge_detect.shape)
threshold = np.array(images_data_threshold)
print("Images Threshold shape:", threshold.shape)
threshold_cropped = np.array(images_data_threshold_cropped)
print("Images Threshold Cropped shape:", threshold_cropped.shape)
print("")

#One-hot encoding: Convert text-based labels to numbers
le = preprocessing.LabelEncoder()
le.fit(label_data)
integer_labels = le.transform(label_data)

#Confirm we have 3 unique classes
print('Unique classes:',le.classes_)
print("")
print("Images and labels successfully preprocessed!")
print("-"*50)

#::------------------------------------------------------------------------------------
##MODELING
#::------------------------------------------------------------------------------------

voting_array = []
x_test_length = 0;
y_test_ex = 0;
types = [no_preprocess,  no_preprocess_cropped, threshold, threshold_cropped, feature_creation, feature_creation_cropped, edge_detect]

for i in types:
    x = i
    y = integer_labels
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=40, test_size=0.2, stratify=y)
    x_test_length = len(x_test)
    y_test_ex = y_test

    # Data Augmentation
    # only complete data augmentation if it is an image (no feature creation)
    if len(i.shape) > 2:
        augment(x_train, y_train, 87)
        print("Data augmentation completed.")
        print("-" * 50)

    # Reshape the image data into rows
    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    print('Training data shape',namestr(i, globals())[0],":", x_train.shape)
    x_test = np.reshape(x_test, (x_test.shape[0], -1))
    print('Test data shape',namestr(i, globals())[0],":", x_test.shape)

    # Standardizing the features
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    x_train = sc_X.fit_transform(x_train)
    x_test = sc_X.transform(x_test)

    #::------------------------------------------------------------------------------------
    #Support Vector Machine
    #::------------------------------------------------------------------------------------

    # Create a svm Classifier
    clf_svm = svm.NuSVC(kernel="linear")
    # Train the model using the training sets
    clf_svm.fit(x_train, y_train)
    # Predict the response for test dataset
    y_pred = clf_svm.predict(x_test)
    voting_array.append(y_pred)

    print("-" * 80)
    print("Model Results", namestr(i, globals())[0])
    print("-" * 80)
    print("Accuracy SVM",namestr(i, globals())[0],":", round(metrics.accuracy_score(y_test, y_pred), 3))
    print("-")
    print("Confusion Matrix",namestr(i, globals())[0],":")
    cmx_SVM = confusion_matrix(y_test, y_pred)
    print(cmx_SVM)
    print("-")
    print("Classification Report SVM",namestr(i, globals())[0],":")
    cfrp = classification_report(y_test, y_pred)
    print(cfrp)
    print("-")

    #Confusion Matrix Heatmap
    class_names = np.unique(label_data)
    df_cm = pd.DataFrame(cmx_SVM, index=class_names, columns=class_names)
    plt.figure(figsize=(6, 6))
    hm = sns.heatmap(df_cm, cmap="Blues", cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
                     yticklabels=df_cm.columns, xticklabels=df_cm.columns)
    hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
    hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    plt.title(("SVM", namestr(i, globals())[0]))
    # Show heat map
    plt.tight_layout()
    plt.show()

    #Cross Validation Score
    from sklearn.model_selection import cross_val_score
    print("Cross Validation Score", namestr(i, globals())[0],":")
    accuracies = cross_val_score(estimator=svm.NuSVC(kernel="linear"), X=x_train, y=y_train, cv=5)
    print(accuracies)
    print("Mean of Accuracies")
    print(accuracies.mean())

    print("-" * 80)

    # Source: https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python
    #::------------------------------------------------------------------------------------
    #DecisionTree
    #::------------------------------------------------------------------------------------

    # Decision tree with entropy
    clf_dt = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=20, min_samples_leaf=5)
    clf_dt.fit(x_train, y_train)
    X_combined = np.vstack((x_train, x_test))
    y_combined = np.hstack((y_train, y_test))
    y_pred_dt = clf_dt.predict(x_test)
    voting_array.append(y_pred_dt)

    print("Accuracy Tree", namestr(i, globals())[0],":", accuracy_score(y_test, y_pred_dt) * 100)
    print("-")
    print("Confusion Matrix Tree", namestr(i, globals())[0],":")
    conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
    print(conf_matrix_dt)
    print("-")
    print("Classification Report Tree", namestr(i, globals())[0],":")
    print(classification_report(y_test, y_pred_dt))
    print("-")

    #Confusion Matrix Heatmap
    class_names = np.unique(label_data)
    df_dt = pd.DataFrame(conf_matrix_dt, index=class_names, columns=class_names)
    plt.figure(figsize=(6, 6))
    hm = sns.heatmap(df_dt, cmap="Blues", cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
                     yticklabels=df_dt.columns, xticklabels=df_dt.columns)
    hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
    hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    plt.title(("Tree", namestr(i, globals())[0]))
    plt.tight_layout()
    plt.show()

    # Cross validation Tree
    from sklearn.model_selection import cross_val_score
    # train model with cv of 5
    cv_scores = cross_val_score(clf_dt, x_train, y_train, cv=5)
    # print each cv score (accuracy) and average them
    print("Cross Validation Score", namestr(i, globals())[0],":")
    print(cv_scores)
    print("Mean of Accuracies")
    print(format(np.mean(cv_scores)))

    print("-" * 80)

    #::------------------------------------------------------------------------------------
    #KNN
    #::------------------------------------------------------------------------------------

    # Perform KNN of X variables
    knn = KNeighborsClassifier(n_neighbors=17)  # Standard Euclidean distance metric
    knn.fit(x_train, y_train)
    y_pred_knn = knn.predict(x_test)
    voting_array.append(y_pred_knn)

    print("Accuracy KNN", namestr(i, globals())[0],":", accuracy_score(y_test, y_pred_knn) * 100)
    print("-")
    print("Confusion Matrix KNN", namestr(i, globals())[0],":")
    conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
    print(conf_matrix_knn)
    print("-")
    print("Classification Report", namestr(i, globals())[0],":")
    print(classification_report(y_test, y_pred_knn))
    print("-")

    #Confusion Matrix Heatmap
    class_names = np.unique(label_data)
    df_knn = pd.DataFrame(conf_matrix_knn, index=class_names, columns=class_names)
    plt.figure(figsize=(6, 6))
    hm = sns.heatmap(df_knn, cmap="Blues", cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
                     yticklabels=df_knn.columns, xticklabels=df_knn.columns)
    hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
    hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    plt.title(("KNN", namestr(i, globals())[0]))
    # Show heat map
    plt.tight_layout()
    plt.show()

    #Cross validation KNN
    from sklearn.model_selection import cross_val_score
    # train model with cv of 5
    cv_scores = cross_val_score(knn, x_train, y_train, cv=5)
    # print each cv score (accuracy) and average them
    print("Cross Validation Score", namestr(i, globals())[0],":")
    print(cv_scores)
    print("Mean of Accuracies")
    print(format(np.mean(cv_scores)))
    print("-" * 50)

    #Choose the best K
    #Source: https://towardsdatascience.com/building-a-k-nearest-neighbors-k-nn-model-with-scikit-learn-51209555453a
    from sklearn.model_selection import GridSearchCV
    # create new a knn model
    knn2 = KNeighborsClassifier()
    # create a dictionary of all values we want to test for n_neighbors
    param_grid = {"n_neighbors": np.arange(1, 25)}
    # use gridsearch to test all values for n_neighbors
    knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
    # fit model to data
    knn_gscv.fit(x_train, y_train)
    # check top performing n_neighbors value
    print(knn_gscv.best_params_)

    print('-' * 80)
    print('-' * 80 + '\n')

#::------------------------------------------------------------------------------------
##ENSEMBLING METHOD
#::------------------------------------------------------------------------------------

##Accuracy Voting

final_pred = np.array([])
# call the mode function to determine the mode of each images predictions
final_pred = np.append(final_pred, mode(voting_array))

print("Accuracy Voting: ", accuracy_score(y_test_ex, final_pred[0:61]) * 100)
print("Confusion Matrix Voting:")
cmx = confusion_matrix(y_test_ex, final_pred[0:61])
print(cmx)

voting_df = pd.DataFrame(cmx, index=class_names, columns=class_names)
plt.figure(figsize=(6, 6))
hm = sns.heatmap(voting_df, cmap="Blues", cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
                 yticklabels=voting_df.columns, xticklabels=voting_df.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
plt.ylabel('True label', fontsize=15)
plt.xlabel('Predicted label', fontsize=15)
plt.title("Accuracy Voting")
# Show heat map
plt.show()