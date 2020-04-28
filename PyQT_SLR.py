#::-------------------------------------------------------------
### Necessary Libraries
#::-------------------------------------------------------------

# Just to Start the Window
import sys
from PyQt5.QtWidgets import QMainWindow, QAction, QMenu, QApplication
from PyQt5.QtGui import QIcon

# For Vertical Layouts
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtWidgets import QSizePolicy

# For Controls and Triggers
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QCheckBox    # Check Box
from PyQt5.QtWidgets import QRadioButton # Radio Buttons
from PyQt5.QtWidgets import QGroupBox    # Group Box
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure     # Figure

# For ALl Models and Image Processing
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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVC
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

#::------------------------------------------------------------------------------------
#:: Initialize pre-processing and augmentation functions for running the rest of the code
#::------------------------------------------------------------------------------------

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
    x_train_new = []
    y_train_new = []
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
                        # print("-" * 50)

    x_train_new = np.array(x_train_new)
    x_train = np.concatenate((x_train, x_train_new))

    y_train_new = np.array(y_train_new)
    y_train = np.concatenate((y_train, y_train_new))
    return x_train, y_train

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

cwd = os.getcwd()

# Import images and Establish Pre-Processing Lists
label_data_no_preprocess = []
images_data_no_preprocess = []
images_data_no_preprocess_cropped = []
images_data_edge_detect = []
images_data_feature_creation = []
images_data_feature_creation_cropped = []
images_data_threshold = []
images_data_threshold_cropped = []
label_data_feature_creation_cropped = []

for subdir, dirs, files in os.walk(cwd):
    for file in files:
        if "Clear" in subdir:
            # Image pre-processing
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
            images_data_feature_creation.append(img_feature_creation)
            images_data_threshold.append(img_threshold)
            images_data_no_preprocess_cropped.append(img_no_preprocess_cropped)
            images_data_threshold_cropped.append(img_threshold_cropped)
            images_data_feature_creation_cropped.append(img_feature_creation_cropped)

            # Labels preprocessing (may need to adjust depending upon file labeling)
            #label = (subdir.split("eye-miner/")[1])
            #label = (label.split("/")[0])
            label = file.split("_")[0]

            label_data_no_preprocess.append(label)

#One-hot encoding: Convert text-based labels to numbers
label_data = np.array(label_data_no_preprocess)
no_preprocess = np.array(images_data_no_preprocess)
no_preprocess_cropped = np.array(images_data_no_preprocess_cropped)
feature_creation = np.array(images_data_feature_creation)
feature_creation_cropped = np.array(images_data_feature_creation_cropped)
edge_detect = np.array(images_data_edge_detect)
threshold = np.array(images_data_threshold)
threshold_cropped = np.array(images_data_threshold_cropped)
le = preprocessing.LabelEncoder()
le.fit(label_data)
integer_labels = le.transform(label_data)

#::------------------------------------------------------------------------------------
#:: Class: SVM Test For Images (appears as a separate window from main window)
#::------------------------------------------------------------------------------------
class SVMtestWindow(QMainWindow):
    # To manage the signals PyQT manages the communication
    send_fig = pyqtSignal(str)

    def __init__(self):
        #::--------------------------------------------------------
        # Initialize the values of the class
        # Here the class inherits all the attributes and methods from the QMainWindow
        #::--------------------------------------------------------
        super(SVMtestWindow, self).__init__()

        self.Title = 'Support Vector Machine (SVM) Modeling'
        self.initUi()

    def initUi(self):
        #::--------------------------------------------------------------
        #  Set up the window organization (vertical layout via QVBoxLayout)
        #::--------------------------------------------------------------
        self.process = "No Pre-Processing"        # Default pre-processing type
        self.dataAug = False                      # Check mark box for whether to data augment or not
        self.images = no_preprocess               # Used to store the pre-processed list of images
        self.accuracy = 'NA'                      # Used to display accuracy of model
        self.confMatrixValue = ''                 # Used to display the confusion matrix
        self.length = str(round(label_data.shape[0]*0.8))
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)
        self.layout = QVBoxLayout(self.main_widget)   # Creates vertical layout


        #::--------------------------------------------------------------
        #  Create the group boxes that appear in the window and separate inputs
        #::--------------------------------------------------------------

        # First Box is used for Pre-processing Option
        self.groupBox1 = QGroupBox('Pre-Processing of Image Options - Modify with:')
        self.groupBox1Layout = QVBoxLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        # Second Box is used to show effect of pre-processing on Images
        self.groupBox2 = QGroupBox('Effect of Pre-Processing on Normal Ocular Image')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        # Third Box is used to see if data augmentation is necessary
        self.groupBox3 = QGroupBox('Supplement Categories with Additional Images via Data Augmentation')
        self.groupBox3Layout = QVBoxLayout()
        self.groupBox3.setLayout(self.groupBox3Layout)

        # Fourth Box is used to show Accuracy and Confusion Matrix of Model
        self.groupBox4 = QGroupBox('Accuracy and Confusion Matrix Results of Selected Model')
        self.groupBox4Layout = QVBoxLayout()
        self.groupBox4.setLayout(self.groupBox4Layout)

        # Create radio buttons for pre-processing group
        self.b1_1 = QRadioButton("No Transformation")
        self.b1_1.setChecked(True)
        self.b1_1.toggled.connect(self.onClicked)
        ###
        self.b1_2 = QRadioButton("Image Zoom")
        self.b1_2.toggled.connect(self.onClicked)
        ###
        self.b1_3 = QRadioButton("Value Threshold")
        self.b1_3.toggled.connect(self.onClicked)
        ###
        self.b1_4 = QRadioButton("Value Threshold and Crop")
        self.b1_4.toggled.connect(self.onClicked)
        ###
        self.b1_5 = QRadioButton("Feature Selection")
        self.b1_5.toggled.connect(self.onClicked)
        ###
        self.b1_6 = QRadioButton("Feature Selection and Crop")
        self.b1_6.toggled.connect(self.onClicked)
        ###
        self.b1_7 = QRadioButton("Edge Detection")
        self.b1_7.toggled.connect(self.onClicked)
        ### (removed due to processing time)
        #self.b1_8 = QRadioButton("All 7 Options and Vote on Best Prediction")
        #self.b1_8.toggled.connect(self.onClicked)

        self.groupBox1Layout.addWidget(self.b1_1)
        self.groupBox1Layout.addWidget(self.b1_2)
        self.groupBox1Layout.addWidget(self.b1_3)
        self.groupBox1Layout.addWidget(self.b1_4)
        self.groupBox1Layout.addWidget(self.b1_5)
        self.groupBox1Layout.addWidget(self.b1_6)
        self.groupBox1Layout.addWidget(self.b1_7)
        #self.groupBox1Layout.addWidget(self.b1_8)

        # Figure and canvas figure to display ocular images in Box 2
        self.fig1 = Figure()
        self.ax1 = self.fig1.add_subplot(111)
        self.canvas1 = FigureCanvas(self.fig1)
        self.canvas1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas1.updateGeometry()

        # Figure and canvas figure to display confusion matrices in Box 4
        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.canvas2 = FigureCanvas(self.fig2)
        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas2.updateGeometry()
        self.accLabel = QLabel("Accuracy of SVM for "+self.process+" : "+self.accuracy)
        self.confMatrix = QLabel('Confusion Matrix:\n'+self.confMatrixValue)

        # Add canvas to the second (Image Viewing) box
        self.groupBox2Layout.addWidget(self.canvas1)
        # Add accuracy prediction and canvas to the fourth (Confusion Matrix) box
        # Added a check mark box to kick off the model production to save processing time and clean up box
        self.chkStartModel = QCheckBox("Check to Start Model:", self)
        self.chkStartModel.setChecked(False)
        self.chkStartModel.stateChanged.connect(self.onClicked)
        self.groupBox4Layout.addWidget(self.chkStartModel)
        self.groupBox4Layout.addWidget(self.accLabel)
        self.groupBox4Layout.addWidget(self.canvas2)       # may add back in with more time
        self.groupBox4Layout.addWidget(self.confMatrix)

        # Adding to the main layout the groupboxes
        self.layout.addWidget(self.groupBox1)
        self.layout.addWidget(self.groupBox2)
        self.layout.addWidget(self.groupBox3)
        self.layout.addWidget(self.groupBox4)

        # Add the data augmentation checkbox to the third group
        self.chkDAug = QCheckBox("Provide Data Augmentation", self)
        self.chkDAug.setChecked(False)
        self.chkDAug.stateChanged.connect(self.onClicked)
        self.chkLabel = QLabel('Training data size: '+self.length)

        self.groupBox3Layout.addWidget(self.chkLabel)
        self.groupBox3Layout.addWidget(self.chkDAug)

        self.setCentralWidget(self.main_widget)       # Creates the window with all the elements
        self.resize(600, 1000)                        # Resize the window
        self.onClicked()

    # The following is used to generate the view in the window.  It's also under this function that SVM Modeling will
    # take place as the two graphs presented in boxes 2 and 4 require the SVM's output.
    def onClicked(self):

        self.ax1.clear()

        # the buttons indicate which pre-processing method to use in model
        # Display a pre-processed image in Box 2 based on the clicked pre-process method:
        if self.b1_1.isChecked():
            self.process = self.b1_1.text()
            self.images = no_preprocess
            self.ax1.imshow(self.images[0])
        if self.b1_2.isChecked():
            self.process = self.b1_2.text()
            self.images = no_preprocess_cropped
            self.ax1.imshow(self.images[0])
        if self.b1_3.isChecked():
            self.process = self.b1_3.text()
            self.images = threshold
            self.ax1.imshow(self.images[0])
        if self.b1_4.isChecked():
            self.process = self.b1_4.text()
            self.images = threshold_cropped
            self.ax1.imshow(self.images[0])
        if self.b1_5.isChecked():
            self.process = self.b1_5.text()
            self.images = feature_creation
            self.ax1.imshow(no_preprocess[0])
        if self.b1_6.isChecked():
            self.process = self.b1_6.text()
            self.images = feature_creation_cropped
            self.ax1.imshow(no_preprocess_cropped[0])
        if self.b1_7.isChecked():
            self.process = self.b1_7.text()
            self.images = edge_detect
            self.ax1.imshow(self.images[0])
        # If all pre-processing is chosen, just use an image of no pre-processing (removed due to processing time)
        #if self.b1_8.isChecked():
        #    self.process = self.b1_8.text()
        #    self.ax1.imshow(images_data_no_preprocess[0])
        #    self.ax1.set_title("Example of Ocular Image with No Pre-Processing, For Reference")

        # Set Image title based on Pre-Processing method
        if (self.b1_5.isChecked() or self.b1_6.isChecked()):
            self.ax1.set_title("Feature Selection has no Image. Non-Pre-Processed Image Shown")
        else:
            self.ax1.set_title("Example of Ocular Image with "+self.process)
        # show the plot
        self.fig1.tight_layout()
        self.fig1.canvas.draw_idle()

        #::--------------------------------------------------------------
        # Based on the results of the pre-process method and dataAug checkbox, perform test-train-split:
        #::--------------------------------------------------------------

        # If anything but 'All 7 Options' is chosen, the test-train-split and run is standard:
        #if (self.b1_8.isChecked() == False):
        x = self.images
        y = integer_labels
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=40, test_size=0.2, stratify=y)

        # if checkbox for data augmentation is checked, perform accordingly
        if (self.chkDAug.isChecked() and len(x.shape) > 2):
            x_train, y_train = augment(x_train, y_train, 87)
            self.length = str(x_train.shape[0])
            self.chkLabel.setText('Training data size: ' + self.length)

        # Preprocessing: reshape the image data into rows
        x_train = np.reshape(x_train, (x_train.shape[0], -1))
        x_test = np.reshape(x_test, (x_test.shape[0], -1))
        sc_X = StandardScaler()
        x_train = sc_X.fit_transform(x_train)
        x_test = sc_X.transform(x_test)

        # Start the Model with the check box marked
        if self.chkStartModel.isChecked():

            # Create SVM Object
            clf_svm = svm.NuSVC(kernel="linear")
            # Train the model using the training sets
            clf_svm.fit(x_train, y_train)
            # Predict the response for test dataset
            y_pred = clf_svm.predict(x_test)

            # Display Accuracy of Model
            self.accuracy = str(round(metrics.accuracy_score(y_test, y_pred), 3)*100)+'%'
            # Display Confusion Matrix
            cmx_SVM = confusion_matrix(y_test, y_pred)
            self.confMatrixValue = str(cmx_SVM)

            self.accLabel.setText("Accuracy of SVM for " + self.process + " : " + self.accuracy)
            self.confMatrix.setText('Confusion Matrix:\n' + self.confMatrixValue)

            # Create a heatmap
            class_names = np.unique(label_data)
            self.ax2.imshow(cmx_SVM, cmap='Blues')
            self.ax2.set_xticks(np.arange(len(class_names)))
            self.ax2.set_yticks(np.arange(len(class_names)))
            # Label X and Y axes
            self.ax2.set_xticklabels(class_names)
            self.ax2.set_yticklabels(class_names)
            # Loop over data dimensions and create text annotations.
            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    text = self.ax2.text(j, i, cmx_SVM[i, j],
                                   ha="center", va="center", color="w")
            self.ax2.set_title("SVM: " + self.process)
            self.fig2.tight_layout()
            self.fig2.canvas.draw_idle()


#::------------------------------------------------------------------------------------
#:: Class: KNN Test For Images (appears as a separate window from main window)
#::------------------------------------------------------------------------------------
class KNNtestWindow(QMainWindow):
    # To manage the signals PyQT manages the communication
    send_fig = pyqtSignal(str)

    def __init__(self):
        #::--------------------------------------------------------
        # Initialize the values of the class
        # Here the class inherits all the attributes and methods from the QMainWindow
        #::--------------------------------------------------------
        super(KNNtestWindow, self).__init__()

        self.Title = 'K-Nearest Neighbor (KNN) Modeling'
        self.initUi()

    def initUi(self):
        #::--------------------------------------------------------------
        #  Set up the window organization (vertical layout via QVBoxLayout)
        #::--------------------------------------------------------------
        self.process = "No Pre-Processing"        # Default pre-processing type
        self.dataAug = False                      # Check mark box for whether to data augment or not
        self.images = no_preprocess               # Used to store the pre-processed list of images
        self.accuracy = 'NA'                      # Used to display accuracy of model
        self.confMatrixValue = ''                 # Used to display the confusion matrix
        self.length = str(round(label_data.shape[0]*0.8))
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)
        self.layout = QVBoxLayout(self.main_widget)   # Creates vertical layout


        #::--------------------------------------------------------------
        #  Create the group boxes that appear in the window and separate inputs
        #::--------------------------------------------------------------

        # First Box is used for Pre-processing Option
        self.groupBox1 = QGroupBox('Pre-Processing of Image Options - Modify with:')
        self.groupBox1Layout = QVBoxLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        # Second Box is used to show effect of pre-processing on Images
        self.groupBox2 = QGroupBox('Effect of Pre-Processing on Normal Ocular Image')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        # Third Box is used to see if data augmentation is necessary
        self.groupBox3 = QGroupBox('Supplement Categories with Additional Images via Data Augmentation')
        self.groupBox3Layout = QVBoxLayout()
        self.groupBox3.setLayout(self.groupBox3Layout)

        # Fourth Box is used to show Accuracy and Confusion Matrix of Model
        self.groupBox4 = QGroupBox('Accuracy and Confusion Matrix Results of Selected Model')
        self.groupBox4Layout = QVBoxLayout()
        self.groupBox4.setLayout(self.groupBox4Layout)

        # Create radio buttons for pre-processing group
        self.b1_1 = QRadioButton("No Transformation")
        self.b1_1.setChecked(True)
        self.b1_1.toggled.connect(self.onClicked)
        ###
        self.b1_2 = QRadioButton("Image Zoom")
        self.b1_2.toggled.connect(self.onClicked)
        ###
        self.b1_3 = QRadioButton("Value Threshold")
        self.b1_3.toggled.connect(self.onClicked)
        ###
        self.b1_4 = QRadioButton("Value Threshold and Crop")
        self.b1_4.toggled.connect(self.onClicked)
        ###
        self.b1_5 = QRadioButton("Feature Selection")
        self.b1_5.toggled.connect(self.onClicked)
        ###
        self.b1_6 = QRadioButton("Feature Selection and Crop")
        self.b1_6.toggled.connect(self.onClicked)
        ###
        self.b1_7 = QRadioButton("Edge Detection")
        self.b1_7.toggled.connect(self.onClicked)
        ### (removed due to processing time)
        #self.b1_8 = QRadioButton("All 7 Options and Vote on Best Prediction")
        #self.b1_8.toggled.connect(self.onClicked)

        self.groupBox1Layout.addWidget(self.b1_1)
        self.groupBox1Layout.addWidget(self.b1_2)
        self.groupBox1Layout.addWidget(self.b1_3)
        self.groupBox1Layout.addWidget(self.b1_4)
        self.groupBox1Layout.addWidget(self.b1_5)
        self.groupBox1Layout.addWidget(self.b1_6)
        self.groupBox1Layout.addWidget(self.b1_7)
        #self.groupBox1Layout.addWidget(self.b1_8)

        # Figure and canvas figure to display ocular images in Box 2
        self.fig1 = Figure()
        self.ax1 = self.fig1.add_subplot(111)
        self.canvas1 = FigureCanvas(self.fig1)
        self.canvas1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas1.updateGeometry()

        # Figure and canvas figure to display confusion matrices in Box 4
        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.canvas2 = FigureCanvas(self.fig2)
        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas2.updateGeometry()
        self.accLabel = QLabel("Accuracy of KNN for "+self.process+" : "+self.accuracy)
        self.confMatrix = QLabel('Confusion Matrix:\n'+self.confMatrixValue)

        # Add canvas to the second (Image Viewing) box
        self.groupBox2Layout.addWidget(self.canvas1)
        # Add accuracy prediction and canvas to the fourth (Confusion Matrix) box
        # Added a check mark box to kick off the model production to save processing time and clean up box
        self.chkStartModel = QCheckBox("Check to Start Model:", self)
        self.chkStartModel.setChecked(False)
        self.chkStartModel.stateChanged.connect(self.onClicked)
        self.groupBox4Layout.addWidget(self.chkStartModel)
        self.groupBox4Layout.addWidget(self.accLabel)
        self.groupBox4Layout.addWidget(self.canvas2)       # may add back in with more time
        self.groupBox4Layout.addWidget(self.confMatrix)

        # Adding to the main layout the groupboxes
        self.layout.addWidget(self.groupBox1)
        self.layout.addWidget(self.groupBox2)
        self.layout.addWidget(self.groupBox3)
        self.layout.addWidget(self.groupBox4)

        # Add the data augmentation checkbox to the third group
        self.chkDAug = QCheckBox("Provide Data Augmentation", self)
        self.chkDAug.setChecked(False)
        self.chkDAug.stateChanged.connect(self.onClicked)
        self.chkLabel = QLabel('Training data size: '+self.length)

        self.groupBox3Layout.addWidget(self.chkLabel)
        self.groupBox3Layout.addWidget(self.chkDAug)

        self.setCentralWidget(self.main_widget)       # Creates the window with all the elements
        self.resize(600, 1000)                        # Resize the window
        self.onClicked()

    # The following is used to generate the view in the window.  It's also under this function that SVM Modeling will
    # take place as the two graphs presented in boxes 2 and 4 require the SVM's output.
    def onClicked(self):

        self.ax1.clear()

        # the buttons indicate which pre-processing method to use in model
        # Display a pre-processed image in Box 2 based on the clicked pre-process method:
        if self.b1_1.isChecked():
            self.process = self.b1_1.text()
            self.images = no_preprocess
            self.ax1.imshow(self.images[0])
        if self.b1_2.isChecked():
            self.process = self.b1_2.text()
            self.images = no_preprocess_cropped
            self.ax1.imshow(self.images[0])
        if self.b1_3.isChecked():
            self.process = self.b1_3.text()
            self.images = threshold
            self.ax1.imshow(self.images[0])
        if self.b1_4.isChecked():
            self.process = self.b1_4.text()
            self.images = threshold_cropped
            self.ax1.imshow(self.images[0])
        if self.b1_5.isChecked():
            self.process = self.b1_5.text()
            self.images = feature_creation
            self.ax1.imshow(no_preprocess[0])
        if self.b1_6.isChecked():
            self.process = self.b1_6.text()
            self.images = feature_creation_cropped
            self.ax1.imshow(no_preprocess_cropped[0])
        if self.b1_7.isChecked():
            self.process = self.b1_7.text()
            self.images = edge_detect
            self.ax1.imshow(self.images[0])
        # If all pre-processing is chosen, just use an image of no pre-processing (removed due to processing time)
        #if self.b1_8.isChecked():
        #    self.process = self.b1_8.text()
        #    self.ax1.imshow(images_data_no_preprocess[0])
        #    self.ax1.set_title("Example of Ocular Image with No Pre-Processing, For Reference")

        # Set Image title based on Pre-Processing method
        if (self.b1_5.isChecked() or self.b1_6.isChecked()):
            self.ax1.set_title("Feature Selection has no Image. Non-Pre-Processed Image Shown")
        else:
            self.ax1.set_title("Example of Ocular Image with "+self.process)
        # show the plot
        self.fig1.tight_layout()
        self.fig1.canvas.draw_idle()

        #::--------------------------------------------------------------
        # Based on the results of the pre-process method and dataAug checkbox, perform test-train-split:
        #::--------------------------------------------------------------

        # If anything but 'All 7 Options' is chosen, the test-train-split and run is standard:
        #if (self.b1_8.isChecked() == False):
        x = self.images
        y = integer_labels
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=40, test_size=0.2, stratify=y)

        # if checkbox for data augmentation is checked, perform accordingly
        if (self.chkDAug.isChecked() and len(x.shape) > 2):
            x_train, y_train = augment(x_train, y_train, 87)
            self.length = str(x_train.shape[0])
            self.chkLabel.setText('Training data size: ' + self.length)

        # Preprocessing: reshape the image data into rows
        x_train = np.reshape(x_train, (x_train.shape[0], -1))
        x_test = np.reshape(x_test, (x_test.shape[0], -1))
        sc_X = StandardScaler()
        x_train = sc_X.fit_transform(x_train)
        x_test = sc_X.transform(x_test)

        # Start the Model with the check box marked
        if self.chkStartModel.isChecked():

            # Set up an KNN object
            knn = KNeighborsClassifier(n_neighbors=17)  # Standard Euclidean distance metric
            # Train the model using the training sets
            knn.fit(x_train, y_train)
            # Predict the response for test dataset
            y_pred_knn = knn.predict(x_test)

            # Display Accuracy of Model
            self.accuracy = str(round(accuracy_score(y_test, y_pred_knn), 3)*100)+'%'
            # Display Confusion Matrix
            conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
            self.confMatrixValue = str(conf_matrix_knn)

            self.accLabel.setText("Accuracy of KNN for " + self.process + " : " + self.accuracy)
            self.confMatrix.setText('Confusion Matrix:\n' + self.confMatrixValue)

            # Create a heatmap
            class_names = np.unique(label_data)
            self.ax2.imshow(conf_matrix_knn, cmap='Blues')
            self.ax2.set_xticks(np.arange(len(class_names)))
            self.ax2.set_yticks(np.arange(len(class_names)))
            # Label X and Y axes
            self.ax2.set_xticklabels(class_names)
            self.ax2.set_yticklabels(class_names)
            # Loop over data dimensions and create text annotations.
            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    text = self.ax2.text(j, i, conf_matrix_knn[i, j],
                                         ha="center", va="center", color="w")
            self.ax2.set_title("SVM: " + self.process)
            self.fig2.tight_layout()
            self.fig2.canvas.draw_idle()

#::------------------------------------------------------------------------------------
#:: Class: Decision Tree Test For Images (appears as a separate window from main window)
#::------------------------------------------------------------------------------------
class DECtestWindow(QMainWindow):
    # To manage the signals PyQT manages the communication
    send_fig = pyqtSignal(str)

    def __init__(self):
        #::--------------------------------------------------------
        # Initialize the values of the class
        # Here the class inherits all the attributes and methods from the QMainWindow
        #::--------------------------------------------------------
        super(DECtestWindow, self).__init__()

        self.Title = 'Decision Tree Modeling'
        self.initUi()

    def initUi(self):
        #::--------------------------------------------------------------
        #  Set up the window organization (vertical layout via QVBoxLayout)
        #::--------------------------------------------------------------
        self.process = "No Pre-Processing"        # Default pre-processing type
        self.dataAug = False                      # Check mark box for whether to data augment or not
        self.images = no_preprocess               # Used to store the pre-processed list of images
        self.accuracy = 'NA'                      # Used to display accuracy of model
        self.confMatrixValue = ''                 # Used to display the confusion matrix
        self.length = str(round(label_data.shape[0]*0.8))
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)
        self.layout = QVBoxLayout(self.main_widget)   # Creates vertical layout

        #::--------------------------------------------------------------
        #  Create the group boxes that appear in the window and separate inputs
        #::--------------------------------------------------------------

        # First Box is used for Pre-processing Option
        self.groupBox1 = QGroupBox('Pre-Processing of Image Options - Modify with:')
        self.groupBox1Layout = QVBoxLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        # Second Box is used to show effect of pre-processing on Images
        self.groupBox2 = QGroupBox('Effect of Pre-Processing on Normal Ocular Image')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        # Third Box is used to see if data augmentation is necessary
        self.groupBox3 = QGroupBox('Supplement Categories with Additional Images via Data Augmentation')
        self.groupBox3Layout = QVBoxLayout()
        self.groupBox3.setLayout(self.groupBox3Layout)

        # Fourth Box is used to show Accuracy and Confusion Matrix of Model
        self.groupBox4 = QGroupBox('Accuracy and Confusion Matrix Results of Selected Model')
        self.groupBox4Layout = QVBoxLayout()
        self.groupBox4.setLayout(self.groupBox4Layout)

        # Create radio buttons for pre-processing group
        self.b1_1 = QRadioButton("No Transformation")
        self.b1_1.setChecked(True)
        self.b1_1.toggled.connect(self.onClicked)
        ###
        self.b1_2 = QRadioButton("Image Zoom")
        self.b1_2.toggled.connect(self.onClicked)
        ###
        self.b1_3 = QRadioButton("Value Threshold")
        self.b1_3.toggled.connect(self.onClicked)
        ###
        self.b1_4 = QRadioButton("Value Threshold and Crop")
        self.b1_4.toggled.connect(self.onClicked)
        ###
        self.b1_5 = QRadioButton("Feature Selection")
        self.b1_5.toggled.connect(self.onClicked)
        ###
        self.b1_6 = QRadioButton("Feature Selection and Crop")
        self.b1_6.toggled.connect(self.onClicked)
        ###
        self.b1_7 = QRadioButton("Edge Detection")
        self.b1_7.toggled.connect(self.onClicked)
        ### (removed due to processing time)
        #self.b1_8 = QRadioButton("All 7 Options and Vote on Best Prediction")
        #self.b1_8.toggled.connect(self.onClicked)

        self.groupBox1Layout.addWidget(self.b1_1)
        self.groupBox1Layout.addWidget(self.b1_2)
        self.groupBox1Layout.addWidget(self.b1_3)
        self.groupBox1Layout.addWidget(self.b1_4)
        self.groupBox1Layout.addWidget(self.b1_5)
        self.groupBox1Layout.addWidget(self.b1_6)
        self.groupBox1Layout.addWidget(self.b1_7)
        #self.groupBox1Layout.addWidget(self.b1_8)

        # Figure and canvas figure to display ocular images in Box 2
        self.fig1 = Figure()
        self.ax1 = self.fig1.add_subplot(111)
        self.canvas1 = FigureCanvas(self.fig1)
        self.canvas1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas1.updateGeometry()

        # Figure and canvas figure to display confusion matrices in Box 4
        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.canvas2 = FigureCanvas(self.fig2)
        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas2.updateGeometry()
        self.accLabel = QLabel("Accuracy of Decision Tree for "+self.process+" : "+self.accuracy)
        self.confMatrix = QLabel('Confusion Matrix:\n'+self.confMatrixValue)

        # Add canvas to the second (Image Viewing) box
        self.groupBox2Layout.addWidget(self.canvas1)
        # Add accuracy prediction and canvas to the fourth (Confusion Matrix) box
        # Added a check mark box to kick off the model production to save processing time and clean up box
        self.chkStartModel = QCheckBox("Check to Start Model:", self)
        self.chkStartModel.setChecked(False)
        self.chkStartModel.stateChanged.connect(self.onClicked)
        self.groupBox4Layout.addWidget(self.chkStartModel)
        self.groupBox4Layout.addWidget(self.accLabel)
        self.groupBox4Layout.addWidget(self.canvas2)       # may add back in with more time
        self.groupBox4Layout.addWidget(self.confMatrix)

        # Adding to the main layout the groupboxes
        self.layout.addWidget(self.groupBox1)
        self.layout.addWidget(self.groupBox2)
        self.layout.addWidget(self.groupBox3)
        self.layout.addWidget(self.groupBox4)

        # Add the data augmentation checkbox to the third group
        self.chkDAug = QCheckBox("Provide Data Augmentation", self)
        self.chkDAug.setChecked(False)
        self.chkDAug.stateChanged.connect(self.onClicked)
        self.chkLabel = QLabel('Training data size: '+self.length)

        self.groupBox3Layout.addWidget(self.chkLabel)
        self.groupBox3Layout.addWidget(self.chkDAug)

        self.setCentralWidget(self.main_widget)       # Creates the window with all the elements
        self.resize(600, 1000)                        # Resize the window
        self.onClicked()

    # The following is used to generate the view in the window.  It's also under this function that SVM Modeling will
    # take place as the two graphs presented in boxes 2 and 4 require the SVM's output.
    def onClicked(self):

        self.ax1.clear()

        # the buttons indicate which pre-processing method to use in model
        # Display a pre-processed image in Box 2 based on the clicked pre-process method:
        if self.b1_1.isChecked():
            self.process = self.b1_1.text()
            self.images = no_preprocess
            self.ax1.imshow(self.images[0])
        if self.b1_2.isChecked():
            self.process = self.b1_2.text()
            self.images = no_preprocess_cropped
            self.ax1.imshow(self.images[0])
        if self.b1_3.isChecked():
            self.process = self.b1_3.text()
            self.images = threshold
            self.ax1.imshow(self.images[0])
        if self.b1_4.isChecked():
            self.process = self.b1_4.text()
            self.images = threshold_cropped
            self.ax1.imshow(self.images[0])
        if self.b1_5.isChecked():
            self.process = self.b1_5.text()
            self.images = feature_creation
            self.ax1.imshow(no_preprocess[0])
        if self.b1_6.isChecked():
            self.process = self.b1_6.text()
            self.images = feature_creation_cropped
            self.ax1.imshow(no_preprocess_cropped[0])
        if self.b1_7.isChecked():
            self.process = self.b1_7.text()
            self.images = edge_detect
            self.ax1.imshow(self.images[0])
        # If all pre-processing is chosen, just use an image of no pre-processing (removed due to processing time)
        #if self.b1_8.isChecked():
        #    self.process = self.b1_8.text()
        #    self.ax1.imshow(images_data_no_preprocess[0])
        #    self.ax1.set_title("Example of Ocular Image with No Pre-Processing, For Reference")

        # Set Image title based on Pre-Processing method
        if (self.b1_5.isChecked() or self.b1_6.isChecked()):
            self.ax1.set_title("Feature Selection has no Image. Non-Pre-Processed Image Shown")
        else:
            self.ax1.set_title("Example of Ocular Image with "+self.process)
        # show the plot
        self.fig1.tight_layout()
        self.fig1.canvas.draw_idle()

        #::--------------------------------------------------------------
        # Based on the results of the pre-process method and dataAug checkbox, perform test-train-split:
        #::--------------------------------------------------------------

        # If anything but 'All 7 Options' is chosen, the test-train-split and run is standard:
        #if (self.b1_8.isChecked() == False):
        x = self.images
        y = integer_labels
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=40, test_size=0.2, stratify=y)

        # if checkbox for data augmentation is checked, perform accordingly
        if (self.chkDAug.isChecked() and len(x.shape) > 2):
            x_train, y_train = augment(x_train, y_train, 87)
            self.length = str(x_train.shape[0])
            self.chkLabel.setText('Training data size: ' + self.length)

        # Preprocessing: reshape the image data into rows
        x_train = np.reshape(x_train, (x_train.shape[0], -1))
        x_test = np.reshape(x_test, (x_test.shape[0], -1))
        sc_X = StandardScaler()
        x_train = sc_X.fit_transform(x_train)
        x_test = sc_X.transform(x_test)

        # Start the Model with the check box marked
        if self.chkStartModel.isChecked():

            # Set up an Decision Tree object
            clf_dt = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=20, min_samples_leaf=5)
            # Train the model using the training sets
            clf_dt.fit(x_train, y_train)
            X_combined = np.vstack((x_train, x_test))
            y_combined = np.hstack((y_train, y_test))
            # Predict the response for test dataset using entropy
            y_pred_dt = clf_dt.predict(x_test)

            # Display Accuracy of Model
            self.accuracy = str(round(accuracy_score(y_test, y_pred_dt), 3)*100)+'%'
            # Display Confusion Matrix
            conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
            self.confMatrixValue = str(conf_matrix_dt)

            self.accLabel.setText("Accuracy of Decision Tree for " + self.process + " : " + self.accuracy)
            self.confMatrix.setText('Confusion Matrix:\n' + self.confMatrixValue)

            # Create a heatmap
            class_names = np.unique(label_data)
            self.ax2.imshow(conf_matrix_dt, cmap='Blues')
            self.ax2.set_xticks(np.arange(len(class_names)))
            self.ax2.set_yticks(np.arange(len(class_names)))
            # Label X and Y axes
            self.ax2.set_xticklabels(class_names)
            self.ax2.set_yticklabels(class_names)
            # Loop over data dimensions and create text annotations.
            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    text = self.ax2.text(j, i, conf_matrix_dt[i, j],
                                         ha="center", va="center", color="w")
            self.ax2.set_title("SVM: " + self.process)
            self.fig2.tight_layout()
            self.fig2.canvas.draw_idle()

#::-------------------------------------------------------------
#:: Definition of a Class for the main menu in the application
#::-------------------------------------------------------------
class Menu(QMainWindow):

    def __init__(self):
        super().__init__()
        # Variables used to set the size of the menu window
        self.left = 100
        self.top = 100
        self.width = 600
        self.height = 800

        #:: Title for the Menu Window
        self.Title = 'Team 6: Image Processing with SVM, KNN, and Decision Trees'

        #:: The initUi is called to create all the necessary elements for the menu
        self.initUI()

    def initUI(self):
        # Creates the menu and the items
        self.setWindowTitle(self.Title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.statusBar()

        # Create the menu bar
        mainMenu = self.menuBar()

        # First Box is used for Introduction Summary
        self.groupBoxTitle = QGroupBox('Introduction to Data')
        self.groupBoxTitleLayout = QVBoxLayout()
        self.groupBoxTitle.setLayout(self.groupBoxTitleLayout)

        self.introLabel = QLabel("Machine learning algorithms are being used in the medical industry to help doctors "+
                                 "diagnose diseases, cancers, and more. This project looks at the usefulness of "+
                                 "applying data mining and machine learning algorithms to diagnose diseases of the "+
                                 "eye. In the eye, different diseases prompt changes to the size and shape of features"+
                                 " in the Foveal Avascular Zone (FAZ). This project aims to analyze images of the FAZ "+
                                 "and classify them into three categories (Diabetes, Myopia, and Normal) based upon "+
                                 "differences inherent in the image. The goal of this research is to help "+
                                 "ophthalmologists diagnose diseases visible from the FAZ quicker and more accurately "+
                                 "than traditional methods.")
        self.groupBoxTitleLayout.addWidget(self.introLabel)

        # Second Box is used to show instructions for the models
        self.groupBoxInstr = QGroupBox('Model Descriptions and Instructions')
        self.groupBoxInstrLayout = QVBoxLayout()
        self.groupBoxInstr.setLayout(self.groupBoxInstrLayout)

        self.instrLabel = QLabel("The given dataset contains roughly 100 images of each diagnosis (107 Diabetic, "+
                                 "109 Myopic, and 88 Normal).\n\nThe interactive GUI contains 7 choices for "+
                                 "pre-processing techniques:\n1. No "+
                                 "Transformation/Pre-processing\n2. No Pre-processing but Zooming in (Cropping) on the "+
                                 "Ocular Center\n3. Adjusting Pixel Values based on a Threshold\n4. Value Thresholding with "+
                                 "Cropping\n5. Selection and Manipulation of Various Image Features\n6. Feature Selection "+
                                 "with Cropping\n7 Detection of Manipulation of Image Edges\n\nThe user is expected to "+
                                 "identify an image pre-processong and whether or not they want want to augment the "+
                                 "given dataset with additional images (via manipulating the current image dataset in "+
                                 "random ways) in order to train on a dataset with equal amounts of each diagnosis.\n\n"+
                                 "The effect of the chosen pre-processing step is shown in the second box in the model "+
                                 "window, and the size of the training dataset (80% of the total dataset - the other 20% "+
                                 "is used for model testing) is shown in the third box in the model window.\n\n"+
                                 "To initiate the model and produce an accuracy measument and confusion matrix for the "+
                                 "model test results, the user must check the test start box in the fourth box.")
        self.groupBoxInstrLayout.addWidget(self.instrLabel)

        # Create an option in the menu bar, mainly to have an exit function
        fileMenu = mainMenu.addMenu('File')
        # Add another option to the menu bar to show different models
        modelWin = mainMenu.addMenu('Models')

        #::--------------------------------------
        # Exit action
        # The following code creates the Exit Action along with all the characteristics associated with the action:
        #::--------------------------------------
        # Sets the exit button
        exitButton = QAction(QIcon('enter.png'), '&Exit', self)
        # Produces a shortcut on the keyboard for the action
        exitButton.setShortcut('Ctrl+Q')
        # Produces a status tip describing what the button does
        exitButton.setStatusTip('Exit application')
        # triggered.connect will indicate what is to be done when the item in the menu is selected
        exitButton.triggered.connect(self.close)
        # This line adds the button (item element ) to the menu
        fileMenu.addAction(exitButton)

        #::----------------------------------------------------
        # Add SVM Model as a Model Menu option
        #::----------------------------------------------------
        svmButton = QAction("SVM Model", self)
        svmButton.setStatusTip("Perform Support Vector Machine Modeling of the Dataset")
        svmButton.triggered.connect(self.SVMtest)   # see below for message function
        # Add the svmButton action to the Model Menu
        modelWin.addAction(svmButton)

        #::----------------------------------------------------
        # Add KNN Model as a Model Menu option
        #::----------------------------------------------------
        knnButton = QAction("KNN Model", self)
        knnButton.setStatusTip("Perform K-Nearest Neighbor Modeling of the Dataset")
        knnButton.triggered.connect(self.KNNtest)  # see below for message function
        # Add the svmButton action to the Model Menu
        modelWin.addAction(knnButton)

        #::----------------------------------------------------
        # Add Decision Tree Model as a Model Menu option
        #::----------------------------------------------------
        decButton = QAction("Decision Tree Model", self)
        decButton.setStatusTip("Perform Decision Tree Modeling of the Dataset")
        decButton.triggered.connect(self.DECtest)  # see below for message function
        # Add the svmButton action to the Model Menu
        modelWin.addAction(decButton)

        # Create an empty list of dialogs to keep track of all the iterations
        self.dialogs = list()

        # Show the windows
        self.show()

    def SVMtest(self):
        dialog = SVMtestWindow()
        self.dialogs.append(dialog)     # Appends the list of dialogs
        dialog.show()                   # Show the window

    def KNNtest(self):
        dialog = KNNtestWindow()
        self.dialogs.append(dialog)     # Appends the list of dialogs
        dialog.show()                   # Show the window

    def DECtest(self):
        dialog = DECtestWindow()
        self.dialogs.append(dialog)     # Appends the list of dialogs
        dialog.show()                   # Show the window

#::------------------------
#:: Main Application
#::------------------------

def main():
    app = QApplication(sys.argv)    # creates the PyQt5 application
    mn = Menu()                     # Creates the menu
    sys.exit(app.exec_())           # Close the application


if __name__ == '__main__':
    main()

