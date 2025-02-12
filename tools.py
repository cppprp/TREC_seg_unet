import copy
import math

import numpy as np
import os
import tifffile
from tqdm import tqdm
from tqdm import trange
import scipy.ndimage as sc


def load_tif_stack(input_folder, start = None, chunk = None, bit = 16):
    data = []
    exclusion_criteria = ['pre', '._', '.DS', 'overlaps']  # add more strings here for files you don't want to convert
    # reads all file names that end with '.tif' but do not have the exclusion criteria
    file_names = os.listdir(input_folder)
    fnames = sorted([file for file in file_names if
                     not any(exclude_str in file for exclude_str in exclusion_criteria) and file.endswith('.tif')])
    if start is None:
        start = 0
    if chunk is None:
        chunk = len(fnames)
    fnames = fnames[start:start+chunk]
    for fname in tqdm(fnames, desc=f"Loading data from {input_folder} "):
        image_path = os.path.join(input_folder, fname)
        data.append(tifffile.imread(image_path))
    if bit==32:
        data=np.asarray(data, dtype='float32')
    if bit == 16:
        data = np.asarray(data, dtype='uint16')
    if bit == 8:
        data = np.asarray(data, dtype='uint8')
    print(data.shape)
    return data

def create_labels(data, label_data, data_output, label_output):
    os.makedirs(data_output, exist_ok=True)
    os.makedirs(label_output, exist_ok=True)
    data, label_data = crop_to_readable(data, label_data, 48)
    label_data[label_data < 100] = 0
    label_data[label_data >= 100] = 1
    label, num_features = sc.label(label_data > 0)
    for ind in trange(data.shape[0], desc=f"Saving label data to {label_output} "):
        tifffile.imwrite(label_output + "/slice_%04d.tif" % (ind), label[ind].astype('uint8'))
        tifffile.imwrite(data_output + "/slice_%04d.tif" % (ind), data[ind])

def crop_to_readable(data, labels, patch):
    crop = (data.shape[0]%patch,data.shape[1]%patch, data.shape[2]%patch)
    data = data[crop[0]:, crop[1]:, crop[2]:]
    labels = labels[crop[0]:, crop[1]:, crop[2]:]
    return data, labels

def normalise(image):
    image = image.astype("float32")
    # Normalise the image to roughly [0,1]
    minim = np.min(image)
    image = image - minim
    # We don't use the max value here, because there are a few very bright
    # pixels in some images, that would otherwise throw off the normalization.
    # Instead we use the 95th percentile to be robust against these intensity outliers.
    max_value = np.percentile(image, 95)
    image /= max_value
    image[np.isnan(image)] = 0.0
    return image