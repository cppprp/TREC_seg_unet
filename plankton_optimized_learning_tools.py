import torch
import numpy as np
from torch.utils.data import Dataset
from skimage.segmentation import find_boundaries
from typing import List, Tuple, Callable, Optional, Dict
from tqdm import tqdm
import torch.nn as nn
from config import ProjectConfig

class OptimizedDataset(Dataset):
    """
    Optimized dataset for 3D image patches with efficient empty patch rejection.
    
    This implementation pre-computes valid patch locations during initialization
    to avoid runtime sampling of empty patches.
    """
    
    def __init__(
        self, 
        images: List[np.ndarray], 
        labels: List[np.ndarray], 
        config: ProjectConfig,
        mask_transform: Callable,
        transform: Optional[List] = None,
        mode: str = 'train'
    ):
        """
        Initialize the dataset with pre-computed valid patch locations.
        
        Args:
            images: List of 3D images
            labels: List of corresponding 3D labels
            config: Project configuration
            mask_transform: Function to transform labels
            transform: Optional data augmentation transforms
            mode: 'train', 'val', or 'test'
        """
        self.images = images
        self.labels = labels
        self.patch_shape = config.data.patch_shape
        self.transform = transform
        self.mask_transform = mask_transform
        self.min_foreground_ratio = config.data.min_foreground_ratio
        self.mode = mode
        
        # Validate inputs
        self._validate_inputs()
        
        # Pre-compute valid patch locations
        self.valid_patches = self._precompute_valid_patches()
        
        print(f"Dataset initialized with {len(self.valid_patches)} valid patches")
        
    def _validate_inputs(self):
        """Validate input data"""
        assert len(self.images) == len(self.labels), \
            f"Number of images ({len(self.images)}) != number of labels ({len(self.labels)})"
        
        for i, (img, lbl) in enumerate(zip(self.images, self.labels)):
            assert img.ndim == 3, f"Image {i} must be 3D, got {img.ndim}D"
            assert lbl.ndim == 3, f"Label {i} must be 3D, got {lbl.ndim}D"
            assert img.shape == lbl.shape, \
                f"Image {i} shape {img.shape} != label shape {lbl.shape}"
            
            # Check if patch fits in image
            for dim, (patch_size, img_size) in enumerate(zip(self.patch_shape, img.shape)):
                assert patch_size <= img_size, \
                    f"Patch size {patch_size} > image size {img_size} in dimension {dim} for image {i}"
    
    def _precompute_valid_patches(self) -> List[Dict]:
        """
        Pre-compute all valid patch locations that contain sufficient foreground pixels.
        
        Returns:
            List of dictionaries containing patch information
        """
        valid_patches = []
        
        for img_idx, (image, label) in enumerate(zip(self.images, self.labels)):
            print(f"Pre-computing valid patches for image {img_idx + 1}/{len(self.images)}")
            
            # Calculate possible patch positions
            z_positions = self._get_patch_positions(image.shape[0], self.patch_shape[0])
            y_positions = self._get_patch_positions(image.shape[1], self.patch_shape[1])
            x_positions = self._get_patch_positions(image.shape[2], self.patch_shape[2])
            
            # Check each possible patch location
            for z_start in tqdm(z_positions, desc=f"Checking patches in image {img_idx}"):
                for y_start in y_positions:
                    for x_start in x_positions:
                        # Extract patch from label
                        z_end = z_start + self.patch_shape[0]
                        y_end = y_start + self.patch_shape[1]
                        x_end = x_start + self.patch_shape[2]
                        
                        label_patch = label[z_start:z_end, y_start:y_end, x_start:x_end]
                        
                        # Check if patch has sufficient foreground
                        if self._is_valid_patch(label_patch):
                            valid_patches.append({
                                'img_idx': img_idx,
                                'z_start': z_start,
                                'y_start': y_start,
                                'x_start': x_start,
                                'z_end': z_end,
                                'y_end': y_end,
                                'x_end': x_end
                            })
        
        if len(valid_patches) == 0:
            raise ValueError("No valid patches found! Consider lowering min_foreground_ratio or checking your data.")
        
        return valid_patches
    
    def _get_patch_positions(self, img_size: int, patch_size: int, overlap: float = 0.5) -> List[int]:
        """
        Get patch start positions with specified overlap.
        
        Args:
            img_size: Size of the image dimension
            patch_size: Size of the patch dimension
            overlap: Overlap ratio between patches (0.5 = 50% overlap)
        
        Returns:
            List of starting positions
        """
        step_size = int(patch_size * (1 - overlap))
        positions = []
        
        for start in range(0, img_size - patch_size + 1, step_size):
            positions.append(start)
        
        # Always include the last possible position to cover the entire image
        if positions[-1] + patch_size < img_size:
            positions.append(img_size - patch_size)
        
        return positions
    
    def _is_valid_patch(self, label_patch: np.ndarray) -> bool:
        """
        Check if a patch contains sufficient foreground pixels.
        
        Args:
            label_patch: The label patch to check
            
        Returns:
            True if patch is valid, False otherwise
        """
        if label_patch.size == 0:
            return False
        
        foreground_pixels = np.sum(label_patch > 0)
        total_pixels = label_patch.size
        foreground_ratio = foreground_pixels / total_pixels
        
        return foreground_ratio >= self.min_foreground_ratio
    
    def __len__(self) -> int:
        """Return the number of valid patches"""
        return len(self.valid_patches)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a patch by index.
        
        Args:
            index: Index of the patch to retrieve
            
        Returns:
            Tuple of (image_patch, label_patch)
        """
        # Get patch information
        patch_info = self.valid_patches[index]
        img_idx = patch_info['img_idx']
        
        # Extract patches
        image = self.images[img_idx]
        label = self.labels[img_idx]
        
        image_patch = image[
            patch_info['z_start']:patch_info['z_end'],
            patch_info['y_start']:patch_info['y_end'],
            patch_info['x_start']:patch_info['x_end']
        ]
        
        label_patch = label[
            patch_info['z_start']:patch_info['z_end'],
            patch_info['y_start']:patch_info['y_end'],
            patch_info['x_start']:patch_info['x_end']
        ]
        
        # Convert to tensors
        image_patch = torch.tensor(image_patch, dtype=torch.float32)
        label_patch = torch.tensor(label_patch, dtype=torch.uint8)
        
        # Add channel dimension
        if image_patch.ndim == 3:
            image_patch = image_patch.unsqueeze(0)
            label_patch = label_patch.unsqueeze(0)
        
        # Apply transforms
        if self.transform is not None:
            image_patch, label_patch = self._apply_transforms(image_patch, label_patch)
        
        # Apply mask transform
        if self.mask_transform is not None:
            label_patch = self.mask_transform(label_patch.squeeze().numpy())
            label_patch = torch.tensor(label_patch, dtype=torch.float32)
        
        return image_patch, label_patch
    
    def _apply_transforms(self, image_patch: torch.Tensor, label_patch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply TorchIO transforms to the patches.
        
        Args:
            image_patch: Image patch tensor
            label_patch: Label patch tensor
            
        Returns:
            Transformed image and label patches
        """
        import torchio as tio
        
        # Create TorchIO subject
        image_tio = tio.ScalarImage(tensor=image_patch)
        label_tio = tio.LabelMap(tensor=label_patch)
        subject = tio.Subject(image=image_tio, label=label_tio)
        
        # Apply transforms
        if isinstance(self.transform, list):
            transform = tio.Compose(self.transform)
        else:
            transform = tio.OneOf(self.transform)
        
        transformed_subject = transform(subject)
        
        # Extract tensors
        image_patch = transformed_subject.image.tensor
        label_patch = transformed_subject.label.tensor
        
        return image_patch, label_patch
    
    def get_patch_info(self, index: int) -> Dict:
        """Get information about a specific patch"""
        return self.valid_patches[index].copy()
    
    def get_image_patch_count(self, img_idx: int) -> int:
        """Get number of valid patches for a specific image"""
        return sum(1 for patch in self.valid_patches if patch['img_idx'] == img_idx)

# Improved label transform function
def enhanced_label_transform(mask: np.ndarray) -> np.ndarray:
    """
    Enhanced label transformation that creates foreground and boundary channels.
    
    Args:
        mask: Input segmentation mask
        
    Returns:
        2-channel array with foreground and boundary maps
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
    
    mask = mask.astype(np.int32)
    
    # Create foreground channel
    fg_target = (mask > 0).astype(np.float32)
    
    # Create boundary channel
    bd_target = find_boundaries(mask, mode="thick").astype(np.float32)
    
    return np.stack([fg_target, bd_target])

# Loss functions
class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    
    def __init__(self, smooth: float = 1e-7):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - dice_score(input_, target, self.smooth)

class CombinedLoss(nn.Module):
    """Combined Dice and Cross-entropy loss"""
    
    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(input_, target)
        ce = self.ce_loss(input_, target)
        return self.dice_weight * dice + self.ce_weight * ce

def dice_score(input_: torch.Tensor, target: torch.Tensor, smooth: float = 1e-7) -> torch.Tensor:
    """
    Calculate Dice score.
    
    Args:
        input_: Predicted tensor
        target: Ground truth tensor
        smooth: Smoothing factor
        
    Returns:
        Dice score
    """
    assert input_.shape == target.shape, f"Shape mismatch: {input_.shape} vs {target.shape}"
    
    # Apply sigmoid to predictions
    input_ = torch.sigmoid(input_)
    
    # Flatten tensors
    input_flat = input_.view(input_.size(0), input_.size(1), -1)
    target_flat = target.view(target.size(0), target.size(1), -1)
    
    # Calculate intersection and union
    intersection = (input_flat * target_flat).sum(dim=2)
    union = input_flat.sum(dim=2) + target_flat.sum(dim=2)
    
    # Calculate Dice score
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    # Average over batch and channels
    return dice.mean()

# Factory function to create datasets
def create_datasets(images: List[np.ndarray], labels: List[np.ndarray], config: ProjectConfig) -> Tuple[OptimizedDataset, OptimizedDataset]:
    """
    Create training and validation datasets.
    
    Args:
        images: List of images
        labels: List of labels
        config: Project configuration
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    from sklearn.model_selection import train_test_split
    
    # Split indices
    indices = list(range(len(images)))
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=1-config.data.train_val_split, 
        random_state=42
    )
    
    # Split data
    train_images = [images[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_images = [images[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    
    # Get transforms
    transforms = config.get_augmentation_transforms()
    
    # Create datasets
    train_dataset = OptimizedDataset(
        train_images, train_labels, config, 
        enhanced_label_transform, transforms, 'train'
    )
    
    val_dataset = OptimizedDataset(
        val_images, val_labels, config, 
        enhanced_label_transform, None, 'val'
    )
    
    return train_dataset, val_dataset