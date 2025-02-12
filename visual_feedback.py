import napari
import tools as tf
import learning_tools as lt
from torch.utils.data import DataLoader

patch_shape = (48, 48, 48)
batch_size = 1
output_labels = "D:/Angelika/ROI_labels/"
output_data = "D:/Angelika/ROI_Xray/"
xray = tf.normalise(tf.load_tif_stack(output_data, bit=16))
label = tf.load_tif_stack(output_labels, bit = 8)
train_dataset = lt.CustomDataset(xray[:300], label[:300], patch_shape=patch_shape, mask_transform = lt.label_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
im, target = next(iter(train_loader))
viewer = napari.Viewer()
viewer.add_image(im)
viewer.add_image(target)
napari.run()