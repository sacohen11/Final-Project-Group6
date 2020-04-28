# Group 6: Eye-Miner

Here is the structure of the Repository:

1). Folders Diabetic, Myopia, Normals - These folders are where the data resides. Each folder has "Clear" and "Marked" images.
	
2). PyQT_SLR.py - This folder contains our demo. It reads in data, performs pre-processing, runs SVM, KNN, and Decision Tree models, performs data augmentation, and displays the images and results in PyQT. The code necessary to run the demo is in the file.
	
3). enter.png - This file is used by the demo. It is a picture.
	
4). combo.py - This file contains our full code. It reads in data, performs pre-processing, runs SVM, KNN, and Decision Tree models, performs data augmentation, and runs the ensembled model. It prints all confusion matrices in PyCharm. The demo file "PyQT_SLR.py" has a toned down version of our code.
	
5). Group-Proposal - Contains our initial group proposal.
	
6). Final-Group-Presentation - Contains our final presentation in PDF format.
	
7). Final-Group-Project-Report - Contains our final report in PDF format.
	
How to run our code:

First, you need to download PyQT5 and OpenCV. If you do not have the OpenCV package on your computer, this link can help you download and troubleshoot (https://stackoverflow.com/questions/60254766/opencv-giving-an-error-whenever-import-cv2-is-used).

Next, if you would like to see the demo only, run "PyQT_SLR.py". The demo contains a pared-down version of our full code, in order to maximize the computing time of the demo.

Lastly, if you would like to run our full code, run "combo.py". It produces the full results of our analysis/modeling.

