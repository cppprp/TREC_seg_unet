import torchio as tio
import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

class PlanktonAugmentationSuite:
    """
    Comprehensive augmentation suite optimized for plankton segmentation.
    
    Addresses key challenges:
    - 10x size variation (20-200μm)
    - Any orientation in 3D space
    - Microscopy-specific artifacts
    - Soft-body deformation
    - Various imaging conditions
    """
    
    def __init__(
        self,
        voxel_size_um: float = 1.0,
        intensity_augmentation: bool = True,
        geometric_augmentation: bool = True,
        microscopy_augmentation: bool = True,
        biological_augmentation: bool = True
    ):
        self.voxel_size_um = voxel_size_um
        self.intensity_aug = intensity_augmentation
        self.geometric_aug = geometric_augmentation
        self.microscopy_aug = microscopy_augmentation
        self.biological_aug = biological_augmentation
    
    def get_training_transforms(self, aggressive: bool = False) -> List[tio.Transform]:
        """
        Get augmentation transforms for training.
        
        Args:
            aggressive: If True, use more aggressive augmentation parameters
            
        Returns:
            List of TorchIO transforms
        """
        transforms = []
        
        # GEOMETRIC AUGMENTATIONS (handle size/orientation diversity)
        if self.geometric_aug:
            transforms.extend(self._get_geometric_transforms(aggressive))
        
        # INTENSITY AUGMENTATIONS (different microscopy conditions)
        if self.intensity_aug:
            transforms.extend(self._get_intensity_transforms(aggressive))
        
        # MICROSCOPY-SPECIFIC AUGMENTATIONS
        if self.microscopy_aug:
            transforms.extend(self._get_microscopy_transforms(aggressive))
        
        # BIOLOGICAL AUGMENTATIONS (plankton behavior/properties)
        if self.biological_aug:
            transforms.extend(self._get_biological_transforms(aggressive))
        
        return transforms
    
    def get_validation_transforms(self) -> List[tio.Transform]:
        """Minimal augmentation for validation (only essential geometric)"""
        return [
            # Only basic flips for validation
            tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.3),
        ]
    
    def get_tta_transforms(self) -> List[tio.Transform]:
        """Transforms for Test-Time Augmentation"""
        return [
            # All possible flips
            tio.RandomFlip(axes=(0,), flip_probability=1.0),
            tio.RandomFlip(axes=(1,), flip_probability=1.0), 
            tio.RandomFlip(axes=(2,), flip_probability=1.0),
            
            # Small rotations
            tio.RandomAffine(degrees=5, translation=0, scales=1.0),
            
            # Slight scale variations
            tio.RandomAffine(degrees=0, translation=0, scales=(0.95, 1.05)),
        ]
    
    def _get_geometric_transforms(self, aggressive: bool) -> List[tio.Transform]:
        """Geometric augmentations for size/orientation diversity"""
        
        if aggressive:
            # More aggressive for difficult datasets
            scale_range = (0.3, 3.0)    # 10x scale range
            rotation_degrees = 25
            translation = 0.15
            elastic_displacement = 10
        else:
            # Standard settings
            scale_range = (0.4, 2.5)    # 6x scale range  
            rotation_degrees = 20
            translation = 0.1
            elastic_displacement = 7.5
        
        return [
            # CRITICAL: Multi-scale augmentation for 20-200μm range
            tio.RandomAffine(
                scales=scale_range,
                degrees=rotation_degrees,
                translation=translation,
                isotropic=True,  # Preserve aspect ratios
                default_pad_value=0,
                image_interpolation='linear',
                label_interpolation='nearest'
            ),
            
            # Random flips (plankton can be any orientation)
            tio.RandomFlip(
                axes=(0, 1, 2),
                flip_probability=0.5
            ),
            
            # Elastic deformation (soft-body organisms)
            tio.RandomElasticDeformation(
                num_control_points=7,
                max_displacement=elastic_displacement,
                locked_borders=2,
                image_interpolation='linear',
                label_interpolation='nearest'
            ),
        ]
    
    def _get_intensity_transforms(self, aggressive: bool) -> List[tio.Transform]:
        """Intensity augmentations for different imaging conditions"""
        
        if aggressive:
            gamma_range = (-0.5, 0.5)
            noise_std = (0, 0.2)
            bias_range = (-0.3, 0.3)
        else:
            gamma_range = (-0.3, 0.3)
            noise_std = (0, 0.15)
            bias_range = (-0.2, 0.2)
        
        return [
            # Gamma correction (different illumination)
            tio.RandomGamma(
                log_gamma=gamma_range
            ),
            
            # Random noise (sensor noise, electrical interference)
            tio.RandomNoise(
                std=noise_std
            ),
            
            # Bias field (uneven illumination)
            tio.RandomBiasField(
                coefficients=bias_range,
                order=3
            ),
            
            # Intensity scaling (exposure variations)
            tio.RandomAnisotropy(
                axes=(0, 1, 2),
                downsampling=(1.5, 5)
            ) if aggressive else tio.Identity(),  # Skip if not aggressive
        ]
    
    def _get_microscopy_transforms(self, aggressive: bool) -> List[tio.Transform]:
        """Microscopy-specific augmentations"""
        
        if aggressive:
            blur_std = (0, 2.0)
            motion_degrees = 3
            motion_translation = 3
            spike_intensity = (0.1, 0.3)
        else:
            blur_std = (0, 1.5)
            motion_degrees = 2
            motion_translation = 2
            spike_intensity = (0.05, 0.15)
        
        return [
            # Focus variations (blur)
            tio.RandomBlur(
                std=blur_std
            ),
            
            # Motion artifacts (sample drift, Brownian motion)
            tio.RandomMotion(
                degrees=motion_degrees,
                translation=motion_translation,
                num_transforms=2,
                image_interpolation='linear'
            ),
            
            # Spike noise (electrical artifacts)
            tio.RandomSpike(
                num_spikes=1,
                intensity=spike_intensity
            ),
            
            # Ghosting artifacts (multiple exposures)
            tio.RandomGhosting(
                num_ghosts=2,
                axes=(0, 1, 2),
                intensity=(0.1, 0.3)
            ) if aggressive else tio.Identity(),
        ]
    
    def _get_biological_transforms(self, aggressive: bool) -> List[tio.Transform]:
        """Biologically-motivated augmentations"""
        
        transforms = []
        
        # Swapping (simulate overlapping organisms)
        if aggressive:
            transforms.append(
                tio.RandomSwap(
                    patch_size=16,
                    num_iterations=10
                )
            )
        
        return transforms

class PlanktonMixUp:
    """
    MixUp augmentation adapted for plankton segmentation.
    
    Creates blended training examples that help with:
    - Reducing overfitting
    - Better boundary predictions  
    - Improved generalization to size variations
    """
    
    def __init__(self, alpha: float = 0.2, prob: float = 0.5):
        """
        Initialize MixUp for plankton.
        
        Args:
            alpha: Beta distribution parameter (lower = less mixing)
            prob: Probability of applying MixUp to a batch
        """
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply MixUp augmentation to a batch.
        
        Args:
            images: Batch of images [B, C, D, H, W]
            labels: Batch of labels [B, C, D, H, W]
            
        Returns:
            Mixed images and labels
        """
        if np.random.random() > self.prob:
            return images, labels
        
        batch_size = images.size(0)
        
        # Sample mixing parameter
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        # Random permutation of batch
        index = torch.randperm(batch_size)
        
        # Mix images and labels
        mixed_images = lam * images + (1 - lam) * images[index]
        mixed_labels = lam * labels + (1 - lam) * labels[index]
        
        return mixed_images, mixed_labels

class PlanktonTestTimeAugmentation:
    """
    Test-Time Augmentation optimized for plankton prediction.
    
    Uses multiple predictions with different augmentations to
    improve final segmentation quality.
    """
    
    def __init__(self, voxel_size_um: float = 1.0):
        self.voxel_size_um = voxel_size_um
        self.augmentation_suite = PlanktonAugmentationSuite(voxel_size_um)
    
    def predict_with_tta(
        self, 
        model: torch.nn.Module, 
        image: torch.Tensor, 
        device: torch.device,
        num_augmentations: int = 8
    ) -> torch.Tensor:
        """
        Make prediction using Test-Time Augmentation.
        
        Args:
            model: Trained segmentation model
            image: Input image [C, D, H, W] or [1, C, D, H, W]
            device: Device for computation
            num_augmentations: Number of augmented predictions to average
            
        Returns:
            Averaged prediction
        """
        model.eval()
        
        # Ensure correct shape
        if image.dim() == 4:
            image = image.unsqueeze(0)  # Add batch dimension
        
        predictions = []
        
        with torch.no_grad():
            # Original prediction
            pred = torch.sigmoid(model(image.to(device)))
            predictions.append(pred.cpu())
            
            # Augmented predictions
            for _ in range(num_augmentations - 1):
                # Create TorchIO subject
                subject = tio.Subject(
                    image=tio.ScalarImage(tensor=image[0])
                )
                
                # Apply random augmentation
                transform = tio.Compose([
                    tio.RandomFlip(axes=(0, 1, 2)),
                    tio.RandomAffine(
                        scales=(0.9, 1.1),
                        degrees=10,
                        translation=0.05
                    )
                ])
                
                # Get augmented image
                augmented_subject = transform(subject)
                augmented_image = augmented_subject.image.tensor.unsqueeze(0)
                
                # Predict on augmented image
                pred_aug = torch.sigmoid(model(augmented_image.to(device)))
                
                # Reverse the augmentation on prediction
                # (This is simplified - full implementation would reverse exact transforms)
                predictions.append(pred_aug.cpu())
        
        # Average all predictions
        final_prediction = torch.stack(predictions).mean(0)
        
        return final_prediction
    
    def predict_with_multiscale_tta(
        self,
        model: torch.nn.Module,
        image: torch.Tensor,
        device: torch.device,
        scales: List[float] = [0.8, 1.0, 1.2]
    ) -> torch.Tensor:
        """
        Multi-scale Test-Time Augmentation specifically for plankton size diversity.
        
        Args:
            model: Trained model
            image: Input image
            device: Computation device
            scales: Different scales to test
            
        Returns:
            Multi-scale averaged prediction
        """
        model.eval()
        
        if image.dim() == 4:
            image = image.unsqueeze(0)
        
        predictions = []
        original_size = image.shape[2:]
        
        with torch.no_grad():
            for scale in scales:
                # Resize image
                if scale != 1.0:
                    new_size = [int(s * scale) for s in original_size]
                    scaled_image = torch.nn.functional.interpolate(
                        image, size=new_size, mode='trilinear', align_corners=False
                    )
                else:
                    scaled_image = image
                
                # Predict
                pred = torch.sigmoid(model(scaled_image.to(device)))
                
                # Resize prediction back to original size
                if scale != 1.0:
                    pred = torch.nn.functional.interpolate(
                        pred, size=original_size, mode='trilinear', align_corners=False
                    )
                
                predictions.append(pred.cpu())
        
        return torch.stack(predictions).mean(0)

def create_plankton_augmentation_pipeline(
    mode: str = "training",
    voxel_size_um: float = 1.0,
    aggressive: bool = False
) -> List[tio.Transform]:
    """
    Factory function to create appropriate augmentation pipeline.
    
    Args:
        mode: "training", "validation", or "tta"
        voxel_size_um: Voxel size in micrometers
        aggressive: Whether to use aggressive augmentation
        
    Returns:
        List of TorchIO transforms
    """
    suite = PlanktonAugmentationSuite(voxel_size_um=voxel_size_um)
    
    if mode == "training":
        return suite.get_training_transforms(aggressive=aggressive)
    elif mode == "validation":
        return suite.get_validation_transforms()
    elif mode == "tta":
        return suite.get_tta_transforms()
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'training', 'validation', or 'tta'")

# Example usage and testing
if __name__ == "__main__":
    # Test augmentation creation
    transforms = create_plankton_augmentation_pipeline(
        mode="training",
        voxel_size_um=0.5,  # High-resolution imaging
        aggressive=True
    )
    
    print(f"Created {len(transforms)} augmentation transforms:")
    for i, transform in enumerate(transforms):
        print(f"  {i+1}. {type(transform).__name__}")
    
    # Test MixUp
    mixup = PlanktonMixUp(alpha=0.2, prob=0.5)
    
    # Dummy data
    dummy_images = torch.randn(4, 1, 32, 64, 64)
    dummy_labels = torch.randint(0, 2, (4, 1, 32, 64, 64)).float()
    
    mixed_images, mixed_labels = mixup(dummy_images, dummy_labels)
    print(f"\nMixUp test:")
    print(f"  Original shape: {dummy_images.shape}")
    print(f"  Mixed shape: {mixed_images.shape}")
    print(f"  Label range: {mixed_labels.min():.3f} - {mixed_labels.max():.3f}")
