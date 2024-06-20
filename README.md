# Project-Reconnaissance-De-Visage
Reconnaissance-De-Visage is a face recognition program build on python, which inputs frontal faces and recognizes them from the available datasets.

## TechStacks and Libraries Used-
* the program uses OpenCV, to input images
* Currently, this program uses Haar cascade Classifier to derive frontal faces from the input image ,Although,there are more sophisticated methods of Deep neural networks
* the model used for distinguishing faces is KNN (K-nearest Neighbours)

## How to run this program -
*  Download the repository in a folder and create a DATA folder
*  Store the path of the Data folder in 'filepath' in 'cam.py' and in 'dataset_path' of 'loaddata.py'
*  Do make Sure that the 'haarcascade_frontalface_alt.xml' is on the same folder that the entire repository is on.
*  Do make Sure that you have the updated version of python and OpenCV installed on your desktop
### Step-1 Create a DataSet using cam.py
*  open your terminal in the same folder
*  run the cam.py file using 'python cam.py'
*  Enter your name
*  The program will start taking your image (20 Images for one person- It's hardcoded , one can also change it), make sure that you try to show all angles of your frontal face for creating a good DataSet 
### Step-2 Recognition
*  Now, again open your terminal in the same folder
*  run the loaddata.py file using 'python loaddata.py'
*  A window should appear , taking inpur from your camera and labelling the image to the closest Image that has been stored in its database
  
