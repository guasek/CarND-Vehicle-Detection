"""
Script searching for a cars in a video or image.
"""
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from moviepy.editor import VideoFileClip
from uuid import uuid4
from scipy.ndimage.measurements import label
from scipy.ndimage.measurements import find_objects
from sklearn.externals import joblib
from sklearn.model_selection import ParameterGrid

from train_classifier import FEATURE_EXTRACTOR_PARAMS
from features import extract_image_features
from features import extract_hog_features
from features import extract_histogram_features
from features import extract_spatial_features


param_grid = ParameterGrid(FEATURE_EXTRACTOR_PARAMS)[0]
classifier = joblib.load('./car_detector.pkl')
feature_scaler = joblib.load('./feture_scaler.pkl')
normalizer = joblib.load('./normalizer.pkl')
previous_detections = {}


def slide_window(image, x_start_stop, y_start_stop, xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    x_start = x_start_stop[0] if x_start_stop[0] else 0
    x_stop = x_start_stop[1] if x_start_stop[1] else image.shape[1]
    y_start = y_start_stop[0] if y_start_stop[0] else 0
    y_stop = y_start_stop[1] if y_start_stop[1] else image.shape[0]

    xspan = x_stop - x_start
    yspan = y_stop - y_start
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    nx_windows = np.int(xspan / nx_pix_per_step) - 1
    ny_windows = np.int(yspan / ny_pix_per_step) - 1
    window_list = []
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            startx = xs * nx_pix_per_step + x_start
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start
            endy = starty + xy_window[1]
            window_list.append(((int(startx), int(starty)), (int(endx), int(endy))))
    return window_list


def search_windows(image, windows):
    on_windows = []
    for window in windows:
        test_image = cv2.resize(image[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        test_image = cv2.normalize(test_image, test_image, 0, 1, cv2.NORM_MINMAX, cv2.CV_32FC1)
        features = extract_image_features(test_image, param_grid).reshape(1, -1)
        try:
            test_features = feature_scaler.transform(features)
        except:
            continue
        prediction = classifier.predict(test_features)
        if prediction == 1:
            on_windows.append(window)
    return on_windows


def windows_sized(min_size, max_size, size_count):
    x_step = (max_size[0] - min_size[0]) / size_count
    y_step = (max_size[1] - min_size[1]) / size_count
    window_sizes = []
    for size in range(size_count + 1):
        window_sizes.append((int(max_size[0] - size * x_step), int(max_size[1] - size * y_step)))
    return window_sizes


def draw_boxes(image, bboxes, color=(0, 0, 255), thick=6):
    """
    Draws bounding boxes in an image

    :rtype: np.ndarray
    """
    imcopy = np.copy(image)
    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy


def construct_heatmap(image, bounding_boxes_list):
    """
    Constructs a heatmap basing on list of bounding boxes and previous detections.
    I account for previous detections of cars so the heatmap get hotter in area of previous detection.

    :param np.ndarray image              : Image.
    :param list       bounding_boxes_list: List of bounding boxes.

    :return:
    """
    heatmap = np.zeros_like(image[:, :, 0]).astype(np.float)
    for box in bounding_boxes_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Increase heatmap value in areas where previous sound detections were made.
    for car_nr in previous_detections.keys():
        if len(previous_detections[car_nr]) > 7:
            xspan = previous_detections[car_nr][-1]['bbox'][0].stop - previous_detections[car_nr][-1]['bbox'][0].start
            yspan = previous_detections[car_nr][-1]['bbox'][1].stop - previous_detections[car_nr][-1]['bbox'][1].start

            xstart = int(previous_detections[car_nr][-1]['bbox'][0].start - xspan * 0.2)
            if xstart < 0:
                xstart = 0
            xstop = int(previous_detections[car_nr][-1]['bbox'][0].stop + xspan * 0.2)
            if xstop > 1280:
                xstop = 1280

            ystart = int(previous_detections[car_nr][-1]['bbox'][1].start - yspan * 0.2)
            if ystart < 0:
                ystart = 0
            ystop = int(previous_detections[car_nr][-1]['bbox'][1].stop + yspan * 0.2)
            if ystop > 720:
                ystop = 720

            xslice = slice(xstart, xstop)
            yslice = slice(ystart, ystop)
            heatmap[xslice, yslice] += 2
    return heatmap


def apply_threshold(heatmap, threshold):
    """
    Applies a threshold to a heatmap.

    :param np.ndarrat heatmap  : Heatmap to be thresholded.
    :param int        threshold: Threshold value.

    :rtype: np.ndarray
    """
    heatmap[heatmap <= threshold] = 0
    return heatmap


def draw_labeled_bboxes(image, labels):
    """
    Draws bounding boxes on an image.

    :param np.ndarray image : Image to draw on.
    :param tuple      labels: Cars labels.

    :rtype: np.ndarray
    """
    for car_number in labels[1]:
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        cv2.rectangle(image, bbox[0], bbox[1], (0, 0, 255), 6)
    return image


def aggregate_detections(new_detections):
    """
    Aggregates new detections with the previous ones. I keep track of last 8 detections.

    :param tuple new_detections: List of new detections.

    :rtype: tuple
    """
    # Discard detections that stayed too long in an object tracker.
    for car_nr in previous_detections.keys():
        previous_detections[car_nr] = [
            {'in_detector': car_detection['in_detector'] + 1, 'bbox': car_detection['bbox']}
            for car_detection in previous_detections[car_nr] if car_detection['in_detector'] < 8
            ]

    # Match new detections with previously detected car. Create new object in tracker when in doesnt match.
    new_cars = []
    for new_detection in find_objects(new_detections[0]):
        for car_nr in previous_detections.keys():
            xmiddle = new_detection[0].start + (new_detection[0].stop - new_detection[0].start) / 2
            ymiddle = new_detection[1].start + (new_detection[1].stop - new_detection[1].start) / 2
            if (previous_detections[car_nr] and
                previous_detections[car_nr][-1]['bbox'][0].start < xmiddle < previous_detections[car_nr][-1]['bbox'][0].stop and
                previous_detections[car_nr][-1]['bbox'][1].start < ymiddle < previous_detections[car_nr][-1]['bbox'][1].stop
            ):
                previous_detections[car_nr].append({'in_detector': 1, 'bbox': new_detection})
                break
        else:
            new_cars.append(new_detection)

    for new_car in new_cars:
        previous_detections[uuid4()] = [{'in_detector': 1, 'bbox': new_car}]

    # Return only confirmed detections.
    confirmed_detections = np.zeros_like(new_detections[0])
    detection_indices = []
    for car_nr, detections in enumerate(previous_detections.values()):
        if len(detections) > 3:
            confirmed_detections[detections[-1]['bbox']] = car_nr + 1
            detection_indices.append(car_nr + 1)
    return confirmed_detections, detection_indices


def filter_labels(labels):
    """
    Filters detections so improbable ones are discarded.

    :param list labels: Labeled detections.

    :rtype: tuple
    """
    filtered_labels = []
    for label_ind, current_label in enumerate(find_objects(labels[0])):
        if not current_label:
            continue
        xsize = current_label[0].stop - current_label[0].start
        ysize = current_label[1].stop - current_label[1].start
        if xsize < 40 or xsize > 350 or ysize < 40 or ysize > 350:
            continue
        filtered_labels.append(label_ind + 1)
    return labels[0], filtered_labels


def detect(image, scale):
    """
    Looks for cars in an image at particular scale.

    :param np.ndarray image: Image to operate on.
    :param float      scale: We use different scales in order to make faster hog feature extraction.

    :rtype: list
    """
    image = image.astype(np.float32) / 255
    ystart = 400
    ystop = 650
    region_of_interest = image[ystart:ystop, :, :]
    if scale != 1:
        imshape = region_of_interest.shape
        region_of_interest = cv2.resize(region_of_interest, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    nxblocks = (region_of_interest.shape[1] // param_grid['hog_pix_per_cell']) - 1
    nyblocks = (region_of_interest.shape[0] // param_grid['hog_pix_per_cell']) - 1

    window_size = 64
    nblocks_per_window = (window_size // param_grid['hog_pix_per_cell']) - 1
    cells_per_step = int(2 * scale)
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    hog_channel_1, hog_channel_2, hog_channel_3 = extract_hog_features(
        region_of_interest, param_grid['hog_cspace'], param_grid['hog_orient'], param_grid['hog_pix_per_cell'],
        param_grid['hog_cell_per_block'], param_grid['hog_channels'], False
    )

    detections = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step

            hog_feat1 = hog_channel_1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog_channel_2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog_channel_3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * param_grid['hog_pix_per_cell']
            ytop = ypos * param_grid['hog_pix_per_cell']

            subimg = cv2.resize(region_of_interest[ytop:ytop + window_size, xleft:xleft + window_size], (64, 64))

            spatial_features = extract_spatial_features(
                subimg, param_grid['spatial_cspace'], param_grid['spatial_size']
            )
            hist_features = extract_histogram_features(
                subimg, param_grid['histogram_cspace'], param_grid['histogram_nbins']
            )

            test_features = feature_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
            )
            test_prediction = classifier.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window_size * scale)
                detections.append(
                    ((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart))
                )

    return detections


def find_cars_in_image(image):
    """
    Looks for a car in a given image.

    :param np.ndarray image: An image to scan for cars.

    :rtype: np.ndarray
    """
    detections = []
    for scale in [0.75, 1, 1.5]:
        detections.extend(detect(image, scale))

    heatmap = construct_heatmap(image, detections)
    heatmap = apply_threshold(heatmap, 2)
    labels = label(heatmap)
    labels = (labels[0], range(1, labels[1] + 1))
    labels = filter_labels(labels)
    detections = aggregate_detections(labels)
    detections = filter_labels(detections)
    image_with_bounding_boxes = draw_labeled_bboxes(np.copy(image), detections)
    return image_with_bounding_boxes


def find_cars_in_video(video_clip):
    """
    Looks for a cars in video clip.

    :param VideoFileClip video_clip: Clip.
    """
    converted_clip = video_clip.fl_image(find_cars_in_image)
    converted_clip.write_videofile('../test.mp4', audio=False)


if __name__ == '__main__':
    # image = mpimg.imread('../test_images/test3.jpg')
    # detected = find_cars_in_image(image)
    # plt.imshow(detected)
    # plt.show()
    video = VideoFileClip('../test_video.mp4')
    find_cars_in_video(video)
