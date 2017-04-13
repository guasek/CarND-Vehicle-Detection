"""
Feature extractor functions.
"""

import cv2
import numpy as np
import matplotlib.image as mpimg

from skimage.feature import hog


def convert_color(image, color_space):
    """
    Converts RGB image to another color space.

    :param np.ndarray image      : RGB image.
    :param str        color_space: Desired color space.

    :rtype: np.ndarray
    """
    if color_space != 'RGB':
        color_flag = getattr(cv2, 'COLOR_RGB2{}'.format(color_space))
        return cv2.cvtColor(image, color_flag)
    return image


def extract_hog_features(
        image, cspace='RGB', orient=9, pix_per_cell=8, cell_per_block=2, hog_channels=(0,), feature_vec=True
):
    """
    Extracts hog features from an image.
    """
    feature_image = convert_color(image, cspace)
    hog_features = []
    for channel in hog_channels:
        channel_hog_features = hog(
            feature_image[:, :, channel],
            orient,
            (pix_per_cell, pix_per_cell),
            (cell_per_block, cell_per_block),
            transform_sqrt=True,
            feature_vector=feature_vec
        )
        hog_features.append(channel_hog_features)
    if not feature_vec:
        return hog_features
    return np.ravel(hog_features)


def extract_spatial_features(image, color_space, size):
    """
    Extracts spatial features of an image.

    :param np.ndarray image      : Original image.
    :param str        color_space: Color space to convert to.
    :param tuple      size       : Desired size.

    :rtype: np.ndarray
    """
    resized_image = cv2.resize(image, size)
    return convert_color(resized_image, color_space).ravel()


def extract_histogram_features(image, color_space, nbins):
    """
    Extracts histogram features from an image.

    :rtype: np.ndarray
    """
    feature_image = convert_color(image, color_space)
    channel_1_bins_range = (0, 256)
    channel_2_bins_range = (0, 256)
    channel_3_bins_range = (0, 256)
    if color_space == 'HSV' or color_space == 'HLS':
        channel_1_bins_range = (0, 180)
    channel1_hist = np.histogram(feature_image[:, :, 0], bins=nbins, range=channel_1_bins_range)
    channel2_hist = np.histogram(feature_image[:, :, 1], bins=nbins, range=channel_2_bins_range)
    channel3_hist = np.histogram(feature_image[:, :, 2], bins=nbins, range=channel_3_bins_range)
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features


def extract_image_set_features(images_set, parameters):
    """
    Extracts features of a given images set.

    :param list images_set: Set of images.
    :param dict parameters: Dict of feature extractor parameters.

    :rtype: list
    """
    features = []
    for image_file in images_set:
        image = mpimg.imread(image_file)
        image_features = extract_image_features(image, parameters)
        features.append(image_features)

    return features


def extract_image_features(image, parameters):
    """
    Extracts features of a single image.

    :param np.ndarray image     : An image.
    :param dict       parameters: Feature extractor parameters.

    :rtype: np.ndarray
    """
    spatial_features = extract_spatial_features(image, parameters['spatial_cspace'], parameters['spatial_size'])
    histogram_features = extract_histogram_features(
        image, parameters['histogram_cspace'], parameters['histogram_nbins']
    )
    hog_features = extract_hog_features(
        image,
        parameters['hog_cspace'],
        parameters['hog_orient'],
        parameters['hog_pix_per_cell'],
        parameters['hog_cell_per_block'],
        parameters['hog_channels'],
    )
    return np.concatenate((spatial_features, histogram_features, hog_features))
