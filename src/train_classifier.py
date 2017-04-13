"""
Main project's script.
"""
import glob
import os
import time

import numpy as np

from sklearn.externals import joblib
from sklearn.metrics.classification import precision_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from features import extract_image_set_features


VEHICLES_DATA_PATH = '../data/vehicles/'
NON_VEHICLES_DATA_PATH = '../data/non-vehicles/'

COLORSPACES = ['RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']

FEATURE_EXTRACTOR_PARAMS = {
    'C': [0.0001],
    'spatial_cspace': ['HSV'],
    'spatial_size': [(48, 48)],
    'histogram_cspace': ['HSV'],
    'histogram_nbins': [16],
    'hog_cspace': ['YCrCb'],
    'hog_orient': [12],
    'hog_pix_per_cell': [8],
    'hog_cell_per_block': [2],
    'hog_channels': [(0, 1, 2)]
}


def find_images(images_path):
    """
    Finds all the images in given directory and its subdirectories.

    :rtype: list
    """
    images = []
    for subdirectory in os.listdir(images_path):
        for image in glob.glob('{}{}/*.png'.format(images_path, subdirectory)):
            images.append(image)
    return images


def train_classifier():
    """
    Trains a classifier. Saves the model into a given file.
    """
    car_images = find_images(VEHICLES_DATA_PATH)
    not_car_images = find_images(NON_VEHICLES_DATA_PATH)

    param_grid = ParameterGrid(FEATURE_EXTRACTOR_PARAMS)
    param_set_count = len(param_grid)
    best_classifier_score = 0
    best_classifier_params = None
    best_classifier = None
    best_feature_scaler = None

    for iteration, param_set in enumerate(param_grid):
        start_time = time.time()
        print('Starting iteration {} of {} with params:\n{}'.format(iteration, param_set_count, param_set))
        car_features = extract_image_set_features(car_images, param_set)
        not_car_features = extract_image_set_features(not_car_images, param_set)

        X = np.vstack((car_features, not_car_features)).astype(np.float64)
        X_scaler = StandardScaler().fit(X)
        scaled_X = X_scaler.transform(X)
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(not_car_features))))

        X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2)

        svc = LinearSVC(C=param_set['C'])
        svc.fit(X_train, y_train)

        predictions = svc.predict(X_test)
        accuracy = precision_score(y_test, predictions)

        print('Finished in {} with score of {}\n\n'.format(time.time() - start_time, accuracy))
        if accuracy > best_classifier_score:
            best_classifier_score = accuracy
            best_classifier_params = param_set
            best_classifier = svc
            best_feature_scaler = X_scaler

    print('Best score: {},\n best param set:\n{}'.format(best_classifier_score, best_classifier_params))
    joblib.dump(best_classifier, './car_detector.pkl')
    joblib.dump(best_feature_scaler, './feture_scaler.pkl')

if __name__ == '__main__':
    train_classifier()
