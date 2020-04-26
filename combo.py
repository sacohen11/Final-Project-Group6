'''
DM Project: Rich Gude, Sam Cohen, Luis Ahumada
'''

import numpy as np
from sklearn import preprocessing
from skimage.transform import rotate
from skimage.color import rgb2grey
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import cv2
import os
import random
import warnings
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
warnings.filterwarnings("ignore")



def mode(array):
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
            else:
                output_array.append(random.randrange(0, 2, 2))
                #output_array.append(2)
        else:
            output_array.append(return_value)
    return output_array

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
                            rows, cols = x_train[k].shape[:2]
                            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
                            x_train[k] = cv2.warpAffine(x_train[k], M, (cols, rows))
                            # print("Rotated 30", y_train[k])
                            # plt.imshow(x_train[k])
                            # plt.show()


                        elif rn == 3:
                            # shifting the image 100 pixels in both dimensions
                            rows, cols = x_train[k].shape[:2]
                            M = np.float32([[1, 0, -5], [0, 1, -5]])
                            x_train[k] = cv2.warpAffine(x_train[k], M, (cols, rows))
                            # print("Shifted", y_train[k])
                            # plt.imshow(x_train[k])
                            # plt.show()


                        elif rn == 4:
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
                        print("-" * 50)

    x_train_new = np.array(x_train_new)
    x_train = np.concatenate((x_train, x_train_new))

    y_train_new = np.array(y_train_new)
    y_train = np.concatenate((y_train, y_train_new))

    unique, count= np.unique(y_train, return_counts=True)
    print("Training set after data augmentation:")
    print("Diabetes:", count[0])
    print("Myopia:", count[1])
    print("Normal:", count[2])
    print("Total:", len(x_train))
    print("-"*50)


cwd = os.getcwd()

#Preprocessing
print("Starting images and label pre-processing...")
label_data_no_preprocess = []
images_data_no_preprocess = []
images_data_no_preprocess_cropped = []
images_data_edge_detect = []
images_data_fir_filter = []
images_data_feature_creation = []
images_data_feature_creation_cropped = []
images_data_threshold = []
images_data_threshold_cropped = []
images_data_contour_filled = []
images_data_contour_filled_cropped = []
label_data_feature_creation_cropped = []


for subdir, dirs, files in os.walk(cwd):
    for file in files:
        if "Clear" in subdir:
            # Image preprocessing
            image_path = os.path.join(subdir, file)
            # read in image in grayscale
            img = cv2.imread(image_path, 0)
            # crop out the bottom of the image
            height, width = img.shape
            img = img[0:width, 0:width]
            # resize the image
            img_resized = image_resize(img, height=400)

            # PREPROCESSING: no Preprocess
            # do nothing but change name
            img_no_preprocess = img_resized

            # PREPROCESSING: edge detect
            # find the edges
            # 200, 600 are parameters
            # has to do with the pixel value
            # more information here: https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html
            # we can add more combinations of different numbers and ensemble them
            img_edge_detect = cv2.Canny(img_resized, 200, 600)

            # PREPROCESSING: threshold
            # inverts color scheme (black = white) based on a threshold
            # the second parameter (first number after img_resized) is the threshold value
            # any pixel above that value will be black, anything below will be white
            ret, img_threshold = cv2.threshold(img_resized, 30, 255, cv2.THRESH_BINARY_INV)

            # PREPROCESSING: contour filled
            # need to do binary inverse threshold first
            # ret, img_contour_filled = cv2.threshold(img_resized, 50, 255, cv2.THRESH_BINARY_INV)
            # finds the contours (curves) in the image
            # selects the maximum contour
            # fills in the area inside the contour
            # contours, hierarchy = cv2.findContours(img_contour_filled, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            # find the biggest countour (c) by the area
            # if len(contours) == 0:
            #     # do nothing
            #     a = 0;
            # else:
            #     c = max(contours, key=cv2.contourArea)

            # draw the biggest contour (c) and fill in the inside
            # img_contour_filled = cv2.drawContours(img_contour_filled, [c], 0, (255, 255, 255), thickness=cv2.FILLED)

            # plot
            # plt.subplot(122), plt.imshow(img_contour_filled)
            # plt.title('Edge Image')
            # plt.show()

            # PREPROCESSING: feature creation
            # I took this function from the below website
            # https://medium.com/machine-learning-world/feature-extraction-and-similar-image-search-with-opencv-for-newbies-3c59796bf774
            creator = cv2.KAZE_create()
            # detect
            kps = creator.detect(img_resized)
            vector_size = 32
            kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
            # computing descriptors vector
            kps, img_feature_creation = creator.compute(img_resized, kps)
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

            # CROPPED
            # crop image
            img_cropped = img[150:300, 100:300]
            # resize the cropped image
            img_resized_cropped = cv2.resize(img_cropped, (400, 400))

            # PREPROCESSING: no preprocess cropped
            img_no_preprocess_cropped = img_resized_cropped

            # PREPROCESSING: threshold cropped
            ret, img_threshold_cropped = cv2.threshold(img_resized_cropped, 30, 255, cv2.THRESH_BINARY_INV)

            # PREPROCESSING: contour filled cropped
            # need to do binary inverse threshold first
            # ret_cropped, img_contour_filled_cropped = cv2.threshold(img_resized_cropped, 50, 255, cv2.THRESH_BINARY_INV)
            # finds the contours (curves) in the image
            # selects the maximum contour
            # fills in the area inside the contour
            # contours_cropped, hierarchy_cropped = cv2.findContours(img_contour_filled_cropped, cv2.RETR_LIST,
            #                                                        cv2.CHAIN_APPROX_SIMPLE)
            # find the biggest countour (c) by the area
            # if len(contours_cropped) == 0:
            #     # do nothing
            #     b = 0;
            # else:
            #     c_cropped = max(contours_cropped, key=cv2.contourArea)

            # draw the biggest contour (c) and fill in the inside
            # img_contour_filled_cropped = cv2.drawContours(img_contour_filled_cropped, [c_cropped], 0, (255, 255, 255),
            #                                               thickness=cv2.FILLED)

            # PREPROCESSING: feature creation cropped
            # I took this function from the below website
            # https://medium.com/machine-learning-world/feature-extraction-and-similar-image-search-with-opencv-for-newbies-3c59796bf774
            creator_cropped = cv2.KAZE_create()
            # detect
            kps_cropped = creator_cropped.detect(img_resized_cropped)
            vector_size = 32
            kps_cropped = sorted(kps_cropped, key=lambda x: -x.response)[:vector_size]
            # computing descriptors vector
            kps_cropped, img_feature_creation_cropped = creator_cropped.compute(img_resized_cropped, kps_cropped)
            # Flatten all of them in one big vector - our feature vector
            img_feature_creation_cropped = img_feature_creation_cropped.flatten()
            # Making descriptor of same size
            # Descriptor vector size is 64
            needed_size = (vector_size * 64)
            if img_feature_creation_cropped.size < needed_size:
                # if we have less the 32 descriptors then just adding zeros at the
                # end of our feature vector
                img_feature_creation_cropped = np.concatenate(
                    [img_feature_creation_cropped, np.zeros(needed_size - img_feature_creation_cropped.size)])

            # APPENDING IMAGES TO ARRAYS
            images_data_no_preprocess.append(img_no_preprocess)
            images_data_edge_detect.append(img_edge_detect)
            # images_data_fir_filter.append(img_fir_filter)
            images_data_feature_creation.append(img_feature_creation)
            images_data_threshold.append(img_threshold)
            # images_data_contour_filled.append(img_contour_filled)
            images_data_no_preprocess_cropped.append(img_no_preprocess_cropped)
            images_data_threshold_cropped.append(img_threshold_cropped)
            # images_data_contour_filled_cropped.append(img_contour_filled_cropped)
            images_data_feature_creation_cropped.append(img_feature_creation_cropped)

            # Labels preprocessing
            label = (subdir.split("eye-miner/")[1])
            label = (label.split("/")[0])

            label_data_no_preprocess.append(label)

print("-"*50)

# look at labels and images shape
label_data = np.array(label_data_no_preprocess)
print("Labels shape:", label_data.shape)
images_data_no_preprocess = np.array(images_data_no_preprocess)
print("Images No Preprocessing shape:", images_data_no_preprocess.shape)
images_data_no_preprocess_cropped = np.array(images_data_no_preprocess_cropped)
print("Images No Preprocessing Cropped shape:", images_data_no_preprocess_cropped.shape)
images_data_feature_creation = np.array(images_data_feature_creation)
print("Images Feature Creation shape:", images_data_feature_creation.shape)
images_data_feature_creation_cropped = np.array(images_data_feature_creation_cropped)
print("Images Feature Creation Cropped shape:", images_data_feature_creation_cropped.shape)
images_data_edge_detect = np.array(images_data_edge_detect)
print("Images Edge Detect shape:", images_data_edge_detect.shape)
images_data_threshold = np.array(images_data_threshold)
print("Images Threshold shape:", images_data_threshold.shape)
images_data_threshold_cropped = np.array(images_data_threshold_cropped)
print("Images Threshold Cropped shape:", images_data_threshold_cropped.shape)
images_data_contour_filled = np.array(images_data_contour_filled)
print("Images Contour Filled shape:", images_data_contour_filled.shape)
images_data_contour_filled_cropped = np.array(images_data_contour_filled_cropped)
print("Images Contour Filled Cropped shape:", images_data_contour_filled_cropped.shape)

print("")

#304 images of 400x400 pixels, and 3 channels (RGB)

#One-hot encoding: Convert text-based labels to numbers
le = preprocessing.LabelEncoder()
le.fit(label_data)
integer_labels = le.transform(label_data)

#Confirm we have 3 unique classes
print('Unique classes:',le.classes_)
print("")
print("Images and labels successfully preprocessed!")
print("-"*50)

#Train/Test Split

voting_array = []
x_test_length = 0;
y_test_ex = 0;
#types = [images_data_no_preprocess, images_data_edge_detect, images_data_feature_creation, images_data_threshold,
         #images_data_contour_filled, images_data_feature_creation_cropped, images_data_no_preprocess_cropped,
         #images_data_threshold_cropped]
types = [images_data_no_preprocess, images_data_threshold, images_data_edge_detect, images_data_no_preprocess_cropped, images_data_threshold_cropped, images_data_feature_creation, images_data_feature_creation_cropped]

for i in types:
    x = i
    y = integer_labels
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=40, test_size=0.2, stratify=y)
    x_test_length = len(x_test)
    y_test_ex = y_test

    # Data Augmentation
    if len(i.shape) > 2:
        augment(x_train, y_train, 90)
        print("Data augmentation completed.")

    # Preprocessing: reshape the image data into rows
    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    print('Training data shape: ', x_train.shape)

    x_test = np.reshape(x_test, (x_test.shape[0], -1))
    print('Test data shape: ', x_test.shape)

    # Standardizing the features
    from sklearn.preprocessing import StandardScaler

    sc_X = StandardScaler()
    x_train = sc_X.fit_transform(x_train)
    x_test = sc_X.transform(x_test)

    # Create a svm Classifier
    from sklearn.svm import NuSVC

    # clf = svm.LinearSVC(multi_class="ovr")
    clf_svm = svm.NuSVC(kernel="linear")

    # Train the model using the training sets
    clf_svm.fit(x_train, y_train)

    # Predict the response for test dataset
    y_pred = clf_svm.predict(x_test)
    voting_array.append(y_pred)
    print("-" * 80)
    print("-" * 80)
    print("Model Results")
    print("-" * 80)
    print("-" * 80)

    # Model Accuracy: how often is the classifier correct?
    print("Accuracy SVM:", round(metrics.accuracy_score(y_test, y_pred), 3))
    print("-" * 50)
    # Confusion Matrix
    print("Confusion Matrix SVM:")
    cmx_SVM = confusion_matrix(y_test, y_pred)
    print(cmx_SVM)
    print("-" * 50)
    print("Classification Report SVM:")
    cfrp = classification_report(y_test, y_pred)
    print(cfrp)
    print("-" * 50)

    class_names = np.unique(label_data)

    df_cm = pd.DataFrame(cmx_SVM, index=class_names, columns=class_names)
    plt.figure(figsize=(6, 6))
    hm = sns.heatmap(df_cm, cmap="Blues", cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
                     yticklabels=df_cm.columns, xticklabels=df_cm.columns)
    hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
    hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    # Show heat map
    plt.tight_layout()
    plt.show()

    # Cross Validation Score
    # from sklearn.model_selection import cross_val_score
    # print("-" * 50)
    # print("Cross Validation Score")
    # accuracies = cross_val_score(estimator=svm.NuSVC(kernel="linear"), X=x_train, y=y_train, cv=10)
    # print(accuracies)
    # print("Mean of Accuracies")
    # print(accuracies.mean())
    # print("STD of Accuracies")
    # print(accuracies.std())

    print("-" * 80)
    print("-" * 80)

    # Decision tree with entropy
    clf_dt = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=20, min_samples_leaf=5)
    # Performing training
    clf_dt.fit(x_train, y_train)
    X_combined = np.vstack((x_train, x_test))
    y_combined = np.hstack((y_train, y_test))

    # prediction on test using entropy
    y_pred_dt = clf_dt.predict(x_test)
    voting_array.append(y_pred_dt)

    # calculate metrics entropy model
    print("Model Accuracy Tree: ", accuracy_score(y_test, y_pred_dt) * 100)
    print('-' * 50)

    # confusion matrix for entropy model
    conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
    print("Confusion Matrix Tree:")
    print(conf_matrix_dt)

    #Clasification report Tree
    print("Classification Report Tree: ")
    print(classification_report(y_test, y_pred_dt))
    print('-' * 50)

    class_names = np.unique(label_data)

    df_dt = pd.DataFrame(conf_matrix_dt, index=class_names, columns=class_names)
    plt.figure(figsize=(6, 6))
    hm = sns.heatmap(df_dt, cmap="Blues", cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
                     yticklabels=df_dt.columns, xticklabels=df_dt.columns)
    hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
    hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    # Show heat map
    plt.tight_layout()
    plt.show()

    print("-" * 80)
    print("-" * 80)



    # Perform KNN of X variables
    knn = KNeighborsClassifier(n_neighbors=7)  # Standard Euclidean distance metric

    knn.fit(x_train, y_train)

    y_pred_knn = knn.predict(x_test)
    voting_array.append(y_pred_knn)

    print("Accuracy KNN: ", accuracy_score(y_test, y_pred_knn) * 100)
    print('-' * 50)

    print("Confusion Matrix KNN:")
    conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
    print(conf_matrix_knn)
    print('-' * 50)

    # Classification Report KNN
    print("Classification Report: ")
    print(classification_report(y_test, y_pred_knn))
    print('-' * 50)


    class_names = np.unique(label_data)

    df_knn = pd.DataFrame(conf_matrix_knn, index=class_names, columns=class_names)
    plt.figure(figsize=(6, 6))
    hm = sns.heatmap(df_knn, cmap="Blues", cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
                     yticklabels=df_knn.columns, xticklabels=df_knn.columns)
    hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
    hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    # Show heat map
    plt.tight_layout()
    plt.show()

    from sklearn.neighbors import kneighbors_graph

    # A = kneighbors_graph(x_train, 2, mode='connectivity', include_self=True)
    # print(A)

    from sklearn.model_selection import cross_val_score

    # # create a new KNN model
    # knn_cv = KNeighborsClassifier(n_neighbors=3)
    # # train model with cv of 5
    # cv_scores = cross_val_score(knn_cv, x_train, y_train, cv=5)
    # # print each cv score (accuracy) and average them
    # print(cv_scores)
    # print("cv_scores mean:{}".format(np.mean(cv_scores)))

    from sklearn.model_selection import GridSearchCV

    # # create new a knn model
    # knn2 = KNeighborsClassifier()
    # # create a dictionary of all values we want to test for n_neighbors
    # param_grid = {"n_neighbors": np.arange(1, 25)}
    # # use gridsearch to test all values for n_neighbors
    # knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
    # # fit model to data
    # knn_gscv.fit(x_train, y_train)
    #
    # # check top performing n_neighbors value
    # print(knn_gscv.best_params_)
    print('-' * 80)
    print('-' * 80)
    print('-' * 80 + '\n')

# print(voting_array)
final_pred = np.array([])
final_pred = np.append(final_pred, mode(voting_array))
                           #  , voting_array[1][i], voting_array[2][i],
                                                       # voting_array[3][i], voting_array[4][i], voting_array[5][i],
                                                      #  voting_array[6][i], voting_array[7][i]]))
# print(final_pred)
#print(final_pred[0:60])
#print(final_pred[0:61])
print("Accuracy Voting: ", accuracy_score(y_test_ex, final_pred[0:61]) * 100)
print('-' * 50)
print("Confusion Matrix Voting:")
cmx = confusion_matrix(y_test_ex, final_pred[0:61])
print(cmx)



#Source: https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python