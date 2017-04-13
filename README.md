# Vehicle Detection

In this project, the goal was to write a software pipeline to detect vehicles in a video.

The steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply pply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Normalize features and randomize a selection for training and testing.
* Implement a sliding-window technique and use trained classifier to search for vehicles in images.
* Run pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train the classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.

[image1]: ./output_images/test_1_result.png
[image2]: ./output_images/test_2_result.png

###Histogram of Oriented Gradients (HOG)

I determined hog, histogram and spatial feature extracting functions' parameters in a following way:

1. I created parameter grid which included as many possible values of each param as possible. I've used following grid:

```
COLORSPACES = ['RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']

'C': [0.0001, 0.001, 0.01, 0.1, 1, 10],
'spatial_cspace': COLORSPACES,
'spatial_size': [(16, 16), (24, 24), (32, 32), (48, 48)],
'histogram_cspace': COLORSPACES,
'histogram_nbins': [16, 24, 32, 40, 48, 64],
'hog_cspace': COLORSPACES,
'hog_orient': [6, 7, 8, 9, 10, 11, 12],
'hog_pix_per_cell': [6, 8],
'hog_cell_per_block': [2, 4, 6],
'hog_channels': [(0), (0, 1), (1, 2), (0, 2), (0, 1, 2)]
```

2. I set up a training pipeline including feature extraction, train-test split and scaler creation. LinearSVM with various values of regularization parameter was used.

```
car_features = extract_image_set_features(car_images, param_set)
not_car_features = extract_image_set_features(not_car_images, param_set)

X = np.vstack((car_features, not_car_features)).astype(np.float64)
X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)
y = np.hstack((np.ones(len(car_features)), np.zeros(len(not_car_features))))

X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2)

svc = LinearSVC(C=param_set['C'])
svc.fit(X_train, y_train)
```

3. At first I've tried running full grid search over entire parameter space. It turned out highly inefficient as full search would take nearly half a year.
4. I decided to twiddle each parameter separately. I did this by reducing all but one parameters count to 1, then I ran a search for best performing classifier with just single parameter varying between iterations. After determining best value for a single param I moved to the next one.

5. The best performing classifier with a accuracy of ~99.5% and precision of ~99.4% was obtained when I used following parameters set:

```
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
```

Hog feature extracting function is shown below:

```
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
```

6. I've saved best performing classifier as well as feature scaler in a pickle files using scikit-learn's joblib pickler.

###Sliding Window Search

At first I tried to implement sliding window technique with a varying size of a moving window. Hog feature extraction was run on each window separately. It again turned out to be highly inefficient. Converting entire project video would take nearly 12 hours. The performance optimization technique I found on udacity forums utilizes region of interest resizing so hog features can be computed only once for entire image. I then use subsamples of precomputed features. The scales I chosen were: 0.75, 1 and 1.5. I chose those values because I wanted windows sized (48, 48), (64, 64) and (96, 96) pixels. Then I compute the overlap to be ~0.7 using `cells_per_step = int(2 * scale)`

Code responsible for described actions is contained in `detect` function:

```
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

```
Here are some examples of results my pipeline generated:

![alt text][image1]

![alt text][image2]
-------------------

### Video Implementation

[link to my video result](./result_video.mp4)

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

I also decided to apply following techniques to minimize occurrence of false positive detections:

* in a training process I used precision as an indicator of classifier performance,
* I eliminate bounding box sizes which are improbable for a car to be,
* I track subsequent detections and check whether they appear within a range of previous ones.

The code for the last bullet is shown below:

```
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
```



---

###Discussion

Even though I've implemented a couple of false positive rate reduction techniques, my pipeline still fails to get rid of them all. I believe it would work even worse in a different lighting or weather conditions. Most of the time I spent working on the project was in reducing false positive rate techniques then. I'd continue with some filtering techniques like Kalman Filter to improve precision of pipeline and account for occlusions like a car getting behind another car.
Moreover a lot of attention has to be paid to optimizing performance of the pipeline. Faster, in terms of performance, language like C could be use to improve performance then. Parallelization of computations should also greatly improve performance.
