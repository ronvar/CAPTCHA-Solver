import numpy as np
import os
import scipy.ndimage
from skimage.feature import hog
from skimage import data, color, exposure
from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib

features_list = []
features_label = []
alphabet_list = ['A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z']
# load labeled training / test data
# loop over the 10 directories where each directory stores the images of a digit
for digit in range(2,10):
    label = digit
    training_directory = './extracted_letter_images_old/' + str(label) + '/'
    for filename in os.listdir(training_directory):
        if (filename.endswith('.png')):
            training_digit_image = scipy.misc.imread(training_directory + filename)
            #training_digit_image = color.rgb2gray(training_digit_image)

            # extra digit's Histogram of Gradients (HOG). Divide the image into 5x5 blocks and where block in 10x10
            # pixels
            df= hog(training_digit_image, orientations=8, pixels_per_cell=(10,10), cells_per_block=(5, 5))
       
            features_list.append(df)
            features_label.append(label)
        print(filename)
print('done w first loop')

for digit in alphabet_list:
    label = digit
    training_directory = './extracted_letter_images_old/' + str(label) + '/'
    for filename in os.listdir(training_directory):
        if (filename.endswith('.png')):
            training_digit_image = scipy.misc.imread(training_directory + filename)
            #training_digit_image = color.rgb2gray(training_digit_image)

            # extra digit's Histogram of Gradients (HOG). Divide the image into 4x4 blocks and where block in 10x10
            # pixels
            df= hog(training_digit_image, orientations=8, pixels_per_cell=(10,10), cells_per_block=(5, 5))
       
            features_list.append(df)
            features_label.append(label)
        print(filename)
print('done w second loop')

#print(features_list)
#print(features_label)

# store features array into a numpy array
features  = np.array(features_list, 'float64')
print('done features')

# split the labled dataset into training / test sets
X_train, X_test, y_train, y_test = train_test_split(features, features_label)
print('done split')

# train using K-NN
knn = KNeighborsClassifier(n_neighbors=5)
print('done training')

knn.fit(X_train, y_train)
print('done fitting')

# get the model accuracy
model_score = knn.score(X_test, y_test)
print('model score = ', model_score)

# save trained model
joblib.dump(knn, './knn_model.pkl')









