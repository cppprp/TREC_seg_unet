import tifffile
import tools as tf
import learning_tools as lt
import torch
import os
from torch_em.model import UNet3d
from torch.utils.data import DataLoader
import torchio as tio
import numpy as np

#model_store = "/Volume/asvetlove/ASVETLOVE/segmenteru/model/"
#os.makedirs(model_store, exist_ok=True)
data_dir = "/Volumes/ASVETLOVE/segmenteru/ML_patches/"


xray = []
label =[]
subfolders = [f.path for f in os.scandir(data_dir) if f.is_dir()]
for folder in subfolders:
    print(folder)
    labels =[f.path for f in os.scandir(folder) if 'label' in f.path]
    data = [l.replace('.labels','') for l in labels]
    for i,(im,se) in enumerate(zip(data,labels)):
        xray.append(tf.normalise(tifffile.imread(im)))
        label.append(tifffile.imread(se))
        print("Adding image ", i)

#xray= np.vstack(xray)
#label = np.vstack(label)
#print('Image data sahpe: ', xray.shape)
#print('Label data shape: ', label.shape)

train_val_split = 0.80

#xray = tf.normalise(tf.load_tif_stack(output_data, bit=16))
#label = tf.load_tif_stack(output_labels, bit = 8)

# set a working device, preferentially GPU
'''if torch.cuda.is_available():
    print("GPU is available")
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    print("Running with the Mac silicon chip")
    device = torch.device("mps")
else:
    print("GPU is NOT available. The training will be very slow!")
    device = torch.device("cpu")'''

# setup model
#model = UNet3d(in_channels=1, out_channels=2, initial_features=32,final_activation="Sigmoid")
#model.to(device)
#loss = lt.DiceLoss()
#loss.to(device)
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-10)  #default LR -3
metric = lt.dice_score
patch_shape = (128, 128, 128)
batch_size = 1
n_epochs = 500

#prepare augmentation transforms in 3D
transforms = {
    tio.RandomAffine(
        scales=(0.6,1.2), #random scaling between x and y percent
        degrees=(10,10,10), # random rotation by degrees
        isotropic = True,
        translation= (0.1,0,1), # allow translation for up to x voxels
    ),
    tio.RandomFlip(axes=(0,1,2))
}
# Prepare the data
train_dataset = lt.CustomDataset(xray[:int(len(xray)*train_val_split)], label[:int(len(label)*train_val_split)], patch_shape=patch_shape, transform = None, mask_transform = lt.label_transform)
val_dataset = lt.CustomDataset(xray[int(len(xray)*train_val_split):], label[int(len(label)*train_val_split):], patch_shape=patch_shape, transform = None, mask_transform = lt.label_transform)


train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
im, target = next(iter(train_loader))
print(torch.min(im), torch.max(im), im.dtype)
print(torch.min(target), torch.max(target), target.dtype)
im, target = next(iter(train_loader))
print(torch.min(im), torch.max(im), im.dtype)
print(torch.min(target), torch.max(target), target.dtype)
im, target = next(iter(train_loader))
print(torch.min(im), torch.max(im), im.dtype)
print(torch.min(target), torch.max(target), target.dtype)
im, target = next(iter(train_loader))
print(torch.min(im), torch.max(im), im.dtype)
print(torch.min(target), torch.max(target), target.dtype)
im, target = next(iter(train_loader))
print(torch.min(im), torch.max(im), im.dtype)
print(torch.min(target), torch.max(target), target.dtype)
im, target = next(iter(train_loader))
print(torch.min(im), torch.max(im), im.dtype)
print(torch.min(target), torch.max(target), target.dtype)
im, target = next(iter(train_loader))
print(torch.min(im), torch.max(im), im.dtype)
print(torch.min(target), torch.max(target), target.dtype)
im, target = next(iter(train_loader))
print(torch.min(im), torch.max(im), im.dtype)
print(torch.min(target), torch.max(target), target.dtype)
'''train_losses, train_scores, val_losses, val_scores = lt.run_training(model, train_loader, val_loader, loss, metric, optimizer, n_epochs, device)
save_folder = os.path.join(model_store, "model_eneg10")
os.makedirs(save_folder, exist_ok=True)
lt.plot_training(train_losses, val_losses, train_scores, val_scores, save_folder)

model_name = "UNet3d_xray_eneg10"
model_path = os.path.join(save_folder, model_name)
torch.save(model.state_dict(),model_path)'''


#test_dataset = CustomDataset(xray[300:], anno[300:], patch_shape=patch_shape, mask_transform = label_transform)
#test_loader = DataLoader(test_dataset, batch_size=batch_size)
#test_folder = "D:/Angelika/POR_20to200_20231022_AM_01_epo_02_166bit_crop/"
#test_images = tf.load_tif_stack(test_folder, bit=16)

'''# run the prediction
prediction = predict_with_halo(
    test_images,  # the volume for which you run prediction.
    model,  # your model, e.g. a trained 3D U-net.
    gpu_ids=[0],  # use this if you have a GPU, if you run on a cpu use
    # gpu_ids=["cpu"],  instead
    block_shape=(24, 256, 256),  # this is the siz
    halo=(4, 64, 64),  # this is the 'halo' cropped from each side of the border to avoid boundary artifacts
    preprocess=None,  # This needs to be set to avoid preprocessing by this fucntion.
)
foreground, boundaries= prediction[0], prediction[1]
output_path_foreground = os.path.join(output_folder, "/foreground/foreground_"+"*_.tif")
output_path_boundaries = os.path.join(output_folder, "/boundaries/boundaries_"+"*_.tif")
imageio.imwrite(output_path_foreground, foreground, compression="zlib")
imageio.imwrite(output_path_boundaries, boundaries, compression="zlib")'''