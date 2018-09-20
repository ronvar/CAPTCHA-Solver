import numpy as np
import os
import scipy.ndimage
from skimage.feature import hog
from skimage import data, color, exposure
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import imutils

knn = joblib.load('./knn_model.pkl')

characters = ['2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N',
              'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def feature_extraction(image):
    return hog(color.rgb2gray(image), orientations=8, pixels_per_cell=(10, 10), cells_per_block=(5, 5))


def predict(df):
    predict = knn.predict(df.reshape(1, -1))[0]
    predict_proba = knn.predict_proba(df.reshape(1, -1))
    # letter_index = predict_proba.tolist().index(predict)
    # print('LETTER INDEX: ', letter_index)
    # print('items in list: ', predict_proba[0].size)
    return predict, predict_proba[0]


# helper
def index_list_builder(prob):
    index_list = []
    for i in range(len(prob)):
        if prob[i] != 0:
            index_list.append(i)
    return index_list


# helper
def print_probs(p_list, i_list):
    print('(', end=' ')
    for i in range(len(i_list)):
        print(characters[i_list[i]], ': p=', p_list[i_list[i]], end='')
        if (i != len(i_list) - 1):
            print(', \n\t\t       ', end='')
    print(' )')


def test(image):
    hog = feature_extraction(image)
    prediction = predict(hog)
    prob_list = prediction[1].tolist()
    index_list = index_list_builder(prob_list)
    print('predicted letter: ', prediction[0], end=' ')
    print_probs(prob_list,index_list)
    return prediction[0]


accuracy = 0
for i in range(1, 23):
    letter = test(scipy.ndimage.imread('./Testing_Extracted/0' + str(i) + '.png'))
    if letter == '2':
        accuracy = accuracy + 1
for i in range(23, 62):
    letter = test(scipy.ndimage.imread('./Testing_Extracted/0' + str(i) + '.png'))
    if letter == '3':
        accuracy = accuracy + 1
for i in range(62, 99):
    letter = test(scipy.ndimage.imread('./Testing_Extracted/0' + str(i) + '.png'))
    if letter == '4':
        accuracy = accuracy + 1
for i in range(99, 140):
    letter = test(scipy.ndimage.imread('./Testing_Extracted/0' + str(i) + '.png'))
    if letter == '5':
        accuracy = accuracy + 1
for i in range(140, 171):
    letter = test(scipy.ndimage.imread('./Testing_Extracted/0' + str(i) + '.png'))
    if letter == '6':
        accuracy = accuracy + 1
for i in range(171, 198):
    letter = test(scipy.ndimage.imread('./Testing_Extracted/0' + str(i) + '.png'))
    if letter == '7':
        accuracy = accuracy + 1
for i in range(198, 238):
    letter = test(scipy.ndimage.imread('./Testing_Extracted/0' + str(i) + '.png'))
    if letter == '8':
        accuracy = accuracy + 1
for i in range(238, 288):
    letter = test(scipy.ndimage.imread('./Testing_Extracted/0' + str(i) + '.png'))
    if letter == '9':
        accuracy = accuracy + 1
for i in range(288, 306):
    letter = test(scipy.ndimage.imread('./Testing_Extracted/0' + str(i) + '.png'))
    if letter == 'A':
        accuracy = accuracy + 1
for i in range(306, 328):
    letter = test(scipy.ndimage.imread('./Testing_Extracted/0' + str(i) + '.png'))
    if letter == 'B':
        accuracy = accuracy + 1
for i in range(328, 345):
    letter = test(scipy.ndimage.imread('./Testing_Extracted/0' + str(i) + '.png'))
    if letter == 'C':
        accuracy = accuracy + 1
for i in range(345, 545):
    letter = test(scipy.ndimage.imread('./Testing_Extracted/0' + str(i) + '.png'))
    if letter == 'D':
        accuracy = accuracy + 1
for i in range(545, 647):
    letter = test(scipy.ndimage.imread('./Testing_Extracted/0' + str(i) + '.png'))
    if letter == 'E':
        accuracy = accuracy + 1
for i in range(647, 671):
    letter = test(scipy.ndimage.imread('./Testing_Extracted/0' + str(i) + '.png'))
    if letter == 'F':
        accuracy = accuracy + 1
for i in range(671, 700):
    letter = test(scipy.ndimage.imread('./Testing_Extracted/0' + str(i) + '.png'))
    if letter == 'G':
        accuracy = accuracy + 1
for i in range(700, 724):
    letter = test(scipy.ndimage.imread('./Testing_Extracted/0' + str(i) + '.png'))
    if letter == 'H':
        accuracy = accuracy + 1
for i in range(724, 748):
    letter = test(scipy.ndimage.imread('./Testing_Extracted/0' + str(i) + '.png'))
    if letter == 'J':
        accuracy = accuracy + 1
for i in range(748, 766):
    letter = test(scipy.ndimage.imread('./Testing_Extracted/0' + str(i) + '.png'))
    if letter == 'K':
        accuracy = accuracy + 1
for i in range(766, 778):
    letter = test(scipy.ndimage.imread('./Testing_Extracted/0' + str(i) + '.png'))
    if letter == 'L':
        accuracy = accuracy + 1
for i in range(778, 790):
    letter = test(scipy.ndimage.imread('./Testing_Extracted/0' + str(i) + '.png'))
    if letter == 'M':
        accuracy = accuracy + 1
for i in range(790, 807):
    letter = test(scipy.ndimage.imread('./Testing_Extracted/0' + str(i) + '.png'))
    if letter == 'N':
        accuracy = accuracy + 1
for i in range(807, 827):
    letter = test(scipy.ndimage.imread('./Testing_Extracted/0' + str(i) + '.png'))
    if letter == 'P':
        accuracy = accuracy + 1
for i in range(827, 844):
    letter = test(scipy.ndimage.imread('./Testing_Extracted/0' + str(i) + '.png'))
    if letter == 'Q':
        accuracy = accuracy + 1
for i in range(844, 861):
    letter = test(scipy.ndimage.imread('./Testing_Extracted/0' + str(i) + '.png'))
    if letter == 'R':
        accuracy = accuracy + 1
for i in range(861, 876):
    letter = test(scipy.ndimage.imread('./Testing_Extracted/0' + str(i) + '.png'))
    if letter == 'S':
        accuracy = accuracy + 1
for i in range(876, 895):
    letter = test(scipy.ndimage.imread('./Testing_Extracted/0' + str(i) + '.png'))
    if letter == 'T':
        accuracy = accuracy + 1
for i in range(895, 909):
    letter = test(scipy.ndimage.imread('./Testing_Extracted/0' + str(i) + '.png'))
    if letter == 'U':
        accuracy = accuracy + 1
for i in range(909, 922):
    letter = test(scipy.ndimage.imread('./Testing_Extracted/0' + str(i) + '.png'))
    if letter == 'V':
        accuracy = accuracy + 1
for i in range(922, 949):
    letter = test(scipy.ndimage.imread('./Testing_Extracted/0' + str(i) + '.png'))
    if letter == 'W':
        accuracy = accuracy + 1
for i in range(949, 967):
    letter = test(scipy.ndimage.imread('./Testing_Extracted/0' + str(i) + '.png'))
    if letter == 'X':
        accuracy = accuracy + 1
for i in range(967, 1000):
    letter = test(scipy.ndimage.imread('./Testing_Extracted/0' + str(i) + '.png'))
    if letter == 'Y':
        accuracy = accuracy + 1
for i in range(1000, 1024):
    letter = test(scipy.ndimage.imread('./Testing_Extracted/0' + str(i) + '.png'))
    if letter == 'Z':
        accuracy = accuracy + 1

accuracy = accuracy / 1024
print('Testing Accuracy: ', accuracy, end=' ')
