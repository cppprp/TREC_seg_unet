import tifffile
import torch
from torch.utils.data import Dataset
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from skimage.segmentation import find_boundaries
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from tqdm import trange
import torchio as tio

#from train import label


class CustomDataset(Dataset):
    # Here we pass the parameters for creating the dataset:
    # The image data, the labels and the patch shape (= the size of the image patches used for training).
    # mask_transform is a function that is applied only to the label data, in order to convert the cell segmentation
    # we have as labels, which cannot be used for directly training the network, into a different representation
    # transform is an additonal argument that can be used for defining data augmentations (optional exercise)
    def __init__(self, images, labels, patch_shape, mask_transform, transform = None, max_attempts=10):
        self.images = images
        self.labels = labels
        self.patch_shape = patch_shape
        self.transform = transform
        self.mask_transform = mask_transform
        self.max_attempts = max_attempts

        self.num_patch_per_image = self.calculate_patches(images[0].shape, patch_shape)
        self.total_patches = self.num_patch_per_image*len(images)
    def __len__(self):
        return self.total_patches
    def calculate_patches(self, volume_shape, patch_shape):
        num_patches_i = volume_shape[0] / patch_shape[0]
        num_patches_j = volume_shape[1] / patch_shape[1]
        num_patches_k = volume_shape[2] / patch_shape[2]
        return int(num_patches_i * num_patches_j * num_patches_k)

    # The __getitem__ method returns the image data and labels for a given sample index.
    def __getitem__(self, index):
        image_idx = index // self.num_patch_per_image # int division, should give the image number
        patch_idx = index % self.num_patch_per_image
        #print ("Image number, patch number: ", image_idx, ' ,', patch_idx)

        # get the current image and mask (= cell segmentation)
        image = self.images[image_idx]
        mask = self.labels[image_idx]
        assert image.ndim == mask.ndim == 3  # <--- 3 DIMS
        assert image.shape == mask.shape

        num_patches_y = image.shape[0] / self.patch_shape[0]
        num_patches_x = image.shape[1] / self.patch_shape[1]
        num_patches_z = image.shape[2] / self.patch_shape[2]
        # i = int((patch_idx // (num_patches_j * num_patches_k)) % num_patches_i)
        # j = int((patch_idx // num_patches_k) % num_patches_j)
        # z = int(patch_idx % num_patches_k)
        carry = int(patch_idx % (num_patches_x * num_patches_y))
        z = int((patch_idx // (num_patches_x * num_patches_y))*self.patch_shape[2])
        y = int((carry // num_patches_y)*self.patch_shape[0])
        x = int((carry % num_patches_y)*self.patch_shape[1])
        #print(x,y,z)

        counter = 0
        while counter < self.max_attempts:


            image_patch = image[y:y + self.patch_shape[0], x:x + self.patch_shape[1], z:z + self.patch_shape[2]]
            mask_patch = mask[y:y + self.patch_shape[0], x:x + self.patch_shape[1], z:z + self.patch_shape[2]]

            if np.max(mask_patch) != 0:
                break
            else:
                y = np.random.randint(0, image.shape[0]-self.patch_shape[0])
                x = np.random.randint(0, image.shape[1] - self.patch_shape[1])
                z = np.random.randint(0, image.shape[1] - self.patch_shape[1])
                # print("i, j, z: ", i, j, z)
                counter += 1
                #print("Iterating empty patch cycle: ", counter)



        # make sure to add the channel dimension to the image
        image_patch = torch.tensor(image_patch, dtype=torch.float32)
        label_patch = torch.tensor(mask_patch, dtype=torch.uint8)
        #tifffile.imwrite("/home/asvetlove/data/image_before.tif", image_patch.numpy())
        #tifffile.imwrite("/home/asvetlove/data/label_before.tif", label_patch.numpy())
        if image.ndim == 3:
            image_patch = image_patch.unsqueeze(0)
            label_patch = label_patch.unsqueeze(0)

        # Apply transform if it is present.
        if self.transform:
            image_patch=tio.ScalarImage(tensor=image_patch)
            label_patch = tio.LabelMap(tensor=label_patch)
            fam = tio.Subject(image=image_patch, label=label_patch)
            transformed_subject = tio.OneOf(self.transform)(fam)

            # Extract the transformed tensors
            image_patch = transformed_subject.image.tensor
            mask_patch = transformed_subject.label.tensor.squeeze()
            #tifffile.imwrite("/home/asvetlove/data/image_after.tif", image_patch.numpy())
            #tifffile.imwrite("/home/asvetlove/data/label_after.tif", mask_patch.numpy())
        # Apply specific transform for the mask.

        mask_patch = self.mask_transform(mask_patch)
        return image_patch, mask_patch

# Define a loss function based on the dice score.
class DiceLoss(nn.Module):
    def forward(self, input_, target):
        # We have already implemented the dice score in the utils.py file.
        # In order to use it as a loss we have to take 1 - dice.
        # Because a high dice score corresponds to a good result, but we minimize the loss,
        # so a low value for the loss must correspond to a good result.
        return 1. - dice_score(input_, target)

def dice_score(input_, target, eps=1e-7):
    assert input_.shape == target.shape, f"{input_.shape}, {target.shape}"
    # Flatten input and target to have the shape (C, N),
    # where N is the number of samples
    input_ = flatten_samples(torch.sigmoid(input_))
    target = flatten_samples(target)
    # Compute numerator and denominator (by summing over samples and
    # leaving the channels intact)
    numerator = (input_ * target).sum(-1)
    denominator = (input_ * input_).sum(-1) + (target * target).sum(-1)
    channelwise_score = 2 * (numerator / denominator.clamp(min=eps))
    # take the average score over the channels
    score = channelwise_score.mean()

    return score
def flatten_samples(input_):
    # Get number of channels
    num_channels = input_.size(1)
    # Permute the channel axis to first
    permute_axes = list(range(input_.dim()))
    permute_axes[0], permute_axes[1] = permute_axes[1], permute_axes[0]
    # For input shape (say) NCHW, this should have the shape CNHW
    permuted = input_.permute(*permute_axes).contiguous()
    # Now flatten out all but the first axis and return
    flattened = permuted.view(num_channels, -1)
    return flattened


def label_transform(mask):
    mask = np.array(mask)
    fg_target = (mask > 0).astype("float32")
    bd_target = find_boundaries(mask, mode="thick").astype("float32")
    return np.stack([fg_target, bd_target])


def validate(model, loader, loss, metric, device):
    model.eval()
    metric_list, loss_list = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss_value = loss(pred, y).item()
            loss_list.append(loss_value)
            if metric is not None:
                metric_value = metric(pred, y).item()
                metric_list.append(metric_value)

    if metric is not None:
        return np.mean(loss_list), np.mean(metric_list)
    else:
        return np.mean(loss_list), None

def train_epoch(model, loader, loss, metric, optimizer, device):
    model.train()
    metric_list, loss_list = [], []
    for i, (x, y) in enumerate(loader):
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss_value = loss(pred, y)
        loss_value.backward()
        optimizer.step()
        loss_list.append(loss_value.item())
        if metric is not None:
            metric_value = metric(pred, y)
            metric_list.append(metric_value.item())

    if metric is not None:
        return np.mean(loss_list), np.mean(metric_list)
    else:
        return np.mean(loss_list), None

def run_training(model, train_loader, val_loader, loss, metric, optimizer, n_epochs, device):
    train_losses, train_metrics = [], []
    val_losses, val_metrics = [], []
    for epoch in trange(n_epochs):
        train_loss, train_metric = train_epoch(model, train_loader, loss, metric, optimizer, device)
        val_loss, val_metric = validate(model, val_loader, loss, metric, device)

        # save the loss and accuracy for plotting
        train_losses.append(train_loss)
        train_metrics.append(train_metric)
        val_losses.append(val_loss)
        val_metrics.append(val_metric)
    return train_losses, train_metrics, val_losses, val_metrics

def plot_training(train_loss, val_loss, train_score, val_score, save_path):
    epoch = np.arange(len(train_loss)) + 1

    #fig = go.Figure()
    fig = make_subplots(rows = 1, cols = 2, subplot_titles=('Loss vs. Epoch', 'Score vs. Epoch'))
    fig.add_trace(go.Scatter(y=train_loss, x=epoch,
                             mode='lines+markers',
                             name='Training loss'), row = 1, col = 1)
    fig.add_trace(go.Scatter(y=val_loss, x=epoch,
                                mode='lines+markers',
                                name='Validation loss'), row=1, col=1)

    fig.add_trace(go.Scatter(y=train_score, x=epoch,
                                mode='lines+markers',
                                name='Training score'), row=1, col=2)
    fig.add_trace(go.Scatter(y=val_score, x=epoch,
                                mode='lines+markers',
                                name='Validation score'), row=1, col=2)

    fig['layout']['xaxis']['title'] = 'Epoch'
    fig['layout']['yaxis']['title'] = 'Loss'
    fig['layout']['xaxis2']['title'] = 'Epoch'
    fig['layout']['yaxis2']['title'] = 'Dice'
    fig.write_html(save_path + "/training_res.html")
    fig.write_json(save_path + "/training_res.json")