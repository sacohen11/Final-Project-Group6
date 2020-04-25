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
import random
from scipy import stats
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
label_data_no_preprocess = []
images_data_no_preprocess = []

label_data_no_preprocess_cropped = []
images_data_no_preprocess_cropped = []

label_data_edge_detect = []
images_data_edge_detect = []

label_data_fir_filter = []
images_data_fir_filter = []

label_data_feature_creation = []
images_data_feature_creation = []

label_data_threshold = []
images_data_threshold = []

label_data_threshold_cropped = []
images_data_threshold_cropped = []

label_data_contour_filled = []
images_data_contour_filled = []

label_data_contour_filled_cropped = []
images_data_contour_filled_cropped = []

images_data_feature_creation_cropped = []
label_data_feature_creation_cropped = []

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
                output_array.append(random.randint(0,1))
                print("WHAO")
            else:
                output_array.append(2)
        else:
            output_array.append(return_value)
    return output_array


for subdir, dirs, files in os.walk(cwd):
    for file in files:
        if "Clear" in subdir:
            # Image preprocessing
            image_path = os.path.join(subdir, file)
            # read in image in grayscale
            img = cv2.imread(image_path, 0)
            # crop out the bottom of the image
            img = img[0:415, 0:435]
            # resize the image
            img_resized = cv2.resize(img, (400, 400))

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
            ret, img_threshold = cv2.threshold(img_resized, 50, 255, cv2.THRESH_BINARY_INV)

            # PREPROCESSING: contour filled
            # need to do binary inverse threshold first
            ret, img_contour_filled = cv2.threshold(img_resized, 50, 255, cv2.THRESH_BINARY_INV)
            # finds the contours (curves) in the image
            # selects the maximum contour
            # fills in the area inside the contour
            contours, hierarchy = cv2.findContours(img_contour_filled, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            # find the biggest countour (c) by the area
            if len(contours) ==0:
                # do nothing
                a=0;
            else:
                c = max(contours, key=cv2.contourArea)

            # draw the biggest contour (c) and fill in the inside
            img_contour_filled = cv2.drawContours(img_contour_filled, [c], 0, (255,255,255), thickness=cv2.FILLED)

            # plot
            #plt.subplot(122), plt.imshow(img_contour_filled)
            #plt.title('Edge Image')
            #plt.show()

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
               img_feature_creation = np.concatenate([img_feature_creation, np.zeros(needed_size - img_feature_creation.size)])

            # CROPPED
            # crop image
            img_cropped = img[150:300, 100:300]
            # resize the cropped image
            img_resized_cropped = cv2.resize(img_cropped, (400, 400))

            # PREPROCESSING: no preprocess cropped
            img_no_preprocess_cropped = img_resized_cropped

            # PREPROCESSING: threshold cropped
            ret, img_threshold_cropped = cv2.threshold(img_resized_cropped, 50, 255, cv2.THRESH_BINARY_INV)

            # PREPROCESSING: contour filled cropped
            # need to do binary inverse threshold first
            ret_cropped, img_contour_filled_cropped = cv2.threshold(img_resized_cropped, 50, 255, cv2.THRESH_BINARY_INV)
            # finds the contours (curves) in the image
            # selects the maximum contour
            # fills in the area inside the contour
            contours_cropped, hierarchy_cropped = cv2.findContours(img_contour_filled_cropped, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            # find the biggest countour (c) by the area
            if len(contours_cropped) == 0:
                # do nothing
                b = 0;
            else:
                c_cropped = max(contours_cropped, key=cv2.contourArea)

            # draw the biggest contour (c) and fill in the inside
            img_contour_filled_cropped = cv2.drawContours(img_contour_filled_cropped, [c_cropped], 0, (255, 255, 255),
                                                          thickness=cv2.FILLED)

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
            #images_data_fir_filter.append(img_fir_filter)
            images_data_feature_creation.append(img_feature_creation)
            images_data_threshold.append(img_threshold)
            images_data_contour_filled.append(img_contour_filled)
            images_data_no_preprocess_cropped.append(img_no_preprocess_cropped)
            images_data_threshold_cropped.append(img_threshold_cropped)
            images_data_contour_filled_cropped.append(img_contour_filled_cropped)
            images_data_feature_creation_cropped.append(img_feature_creation_cropped)

            # Labels preprocessing
            label = (subdir.split("eye-miner/")[1])
            label = (label.split("/")[0])

            label_data_no_preprocess.append(label)
            label_data_edge_detect.append(label)
            # label_data_fir_filter.append(img_fir_filter)
            label_data_feature_creation.append(label)
            label_data_threshold.append(label)
            label_data_contour_filled.append(label)
            label_data_no_preprocess_cropped.append(label)
            label_data_threshold_cropped.append(label)
            images_data_contour_filled_cropped.append(label)
            label_data_feature_creation_cropped.append(label)

print("Images and labels successfully preprocessed!")
print("")
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

le = preprocessing.LabelEncoder()
le.fit(label_data)
integer_labels = le.transform(label_data)
print(integer_labels)

#Confirm we have 3 unique classes
print('Unique classes:',le.classes_)

#Save the images and labels
#np.save("x_train.npy", images_data)
#np.save("y_train.npy", integer_labels)

#y = np.load("y_train.npy", allow_pickle=True)
print("")
print("-"*50)
voting_array = []
x_test_length = 0;
y_test_ex = 0;
types = [images_data_no_preprocess, images_data_edge_detect, images_data_feature_creation, images_data_threshold,
         images_data_contour_filled, images_data_feature_creation_cropped, images_data_no_preprocess_cropped,
         images_data_threshold_cropped]
#types = [images_data_feature_creation, images_data_threshold, images_data_contour_filled,
         #images_data_feature_creation_cropped, images_data_no_preprocess_cropped, images_data_threshold_cropped]
for i in types:
    x = i
    y = integer_labels
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=40, test_size=0.2, stratify=y)
    x_test_length = len(x_test)
    y_test_ex = y_test
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
    voting_array.append(y_pred_entropy)
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

print(voting_array)
final_pred = np.array([])
final_pred = np.append(final_pred, mode(voting_array))
                           #  , voting_array[1][i], voting_array[2][i],
                                                       # voting_array[3][i], voting_array[4][i], voting_array[5][i],
                                                      #  voting_array[6][i], voting_array[7][i]]))
print(final_pred)
print(final_pred[0:60])
print(final_pred[0:61])
print("Accuracy : ", accuracy_score(y_test_ex, final_pred[0:61]) * 100)
print('-' * 80 + '\n')
print("Confusion Matrix")
cmx = confusion_matrix(y_test_ex, final_pred[0:61])
print(cmx)