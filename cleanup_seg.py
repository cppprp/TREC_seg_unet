import skimage.morphology
from skimage.measure import label, regionprops, regionprops_table
import tools as tf
from tqdm import trange
import numpy as np
import tifffile
import time
def clean_labels(label_data, output_path, filter):
    data = tf.load_tif_stack(label_data, bit=8)
    start = time.process_time()
    labels = label(data)
    print("Took to label: %s" %(time.process_time() - start))
    start = time.process_time()
    labels= skimage.morphology.remove_small_objects(labels, filter)
    print("number of lebels:", np.max(labels))
    print("Took to filter: %s" % (time.process_time() - start))
    for i in trange(labels.shape[0]):
        tifffile.imwrite(output_path + "foreground_%04d.tif" % i, labels[i, :, :].astype('uint16'))




clean_labels("D:/Angelika/segmenteru/Prediction/foreground/", "D:/Angelika/segmenteru/Prediction/clean/", 10000)