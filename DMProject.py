'''
DM Project: Rich Gude, Sam Cohen, Luis Ahumada

'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

cwd = os.getcwd()
image = cv2.imread(cwd + "/dataNorm/Marked/Normal_marked (1).jpg")

#EDA
print(image)
print(image.shape)

#Preprocessing
print("Starting images and label pre-processing")
label_data = []
images_data = []

for subdir, dirs, files in os.walk(cwd):
    for file in files:
        if "Clear" in subdir:
            #Image preprocessing
            image_path = os.path.join(subdir, file)
            img = cv2.imread(image_path)
            img = cv2.resize(img, (400, 400))
            images_data.append(img)

            #Labels preprocessing
            # if "dataNorm" in subdir:
            #     label = "Normal"
            #     label_data.append(label)
            # elif "dataMyo" in subdir:
            #     label = "Myopia"
            #     label_data.append(label)
            # elif "dataDia" in subdir:
            #     label = "Diabetes"
            #     label_data.append(label)

            #Labels preprocessing
            label = (subdir.split("eye-miner/")[1])
            label = (label.split("/")[0])
            label_data.append(label)

print("Images and labels successfully preprocessed")

# look at labels and images shape
label_data = np.array(label_data)
print("Labels shape:", label_data.shape)
images_data = np.array(images_data)
print("Images shape:", images_data.shape)

#304 images of 400x400 pixels, and 3 channels (RGB)