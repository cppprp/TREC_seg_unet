import tools as tf
import learning_tools as lt
import torch
import os
import tifffile
from torch_em.model import UNet3d
from torch.utils.data import DataLoader
from torch_em.util.prediction import predict_with_halo
import imageio.v3 as imageio
model_path="/media/asvetlove/ASVETLOVE/segmenteru/model/model_aug/UNet3d_xray_aug"
output_folder = "/media/asvetlove/ASVETLOVE/segmenteru/137/"
output_path_foreground = os.path.join(output_folder, "foreground/")
output_path_boundaries = os.path.join(output_folder, "boundaries/")
os.makedirs(output_path_boundaries, exist_ok=True)
os.makedirs(output_path_foreground, exist_ok=True)
model = UNet3d(in_channels=1, out_channels=2, initial_features=32,final_activation="Sigmoid")
model.load_state_dict(torch.load(model_path))
test_folder = "/mnt/ximg/2024/p3l-yschwab/RECON/20240501/RAW_DATA/137/recon_111_1/tomo/"
test_images = tf.normalise(tf.load_tif_stack(test_folder, start=526, chunk=200, bit=32))
prediction = predict_with_halo(
    test_images,  # the volume for which you run prediction.
    model,  # your model, e.g. a trained 3D U-net.
    gpu_ids=[0],  # use this if you have a GPU, if you run on a cpu use
    # gpu_ids=["cpu"],  instead
    block_shape=(128, 128, 128),  # this is the siz
    halo=(64, 64, 64),  # this is the 'halo' cropped from each side of the border to avoid boundary artifacts
    preprocess=None,  # This needs to be set to avoid preprocessing by this fucntion.
)
foreground, boundaries = prediction[0], prediction[1]
for s in range(foreground.shape[0]):
    tifffile.imwrite(output_path_foreground+ "foreground_%04d.tif" % s, foreground[s,:,:], photometric='minisblack')
    tifffile.imwrite(output_path_boundaries+ "boundary_%04d.tif" % s, boundaries[s,:,:], photometric='minisblack')