import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.measure import label, regionprops
from typing import Tuple, Optional

class PlanktonFocalDiceLoss(nn.Module):
    """
    Enhanced FocalDiceLoss optimized for plankton segmentation.
    
    Key improvements for plankton:
    - Aggressive focal parameters for extreme class imbalance
    - Size-aware weighting to prioritize small plankton
    - Optional multi-scale loss computation
    """
    
    def __init__(
        self, 
        alpha: float = 0.75,           # Higher for extreme imbalance
        gamma: float = 3.0,            # Higher to really focus on hard examples
        dice_weight: float = 0.6,      # Slightly favor shape accuracy
        small_plankton_boost: float = 0.5,  # Extra weight for small plankton
        voxel_size_um: float = 1.0,    # For size calculations
        smooth: float = 1e-7
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.small_plankton_boost = small_plankton_boost
        self.voxel_size_um = voxel_size_um
        self.smooth = smooth
        
    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Enhanced focal loss for plankton"""
        # Standard binary cross entropy
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Calculate p_t (probability of true class)
        pt = torch.exp(-bce)
        
        # Alpha weighting (higher alpha = more focus on foreground/plankton)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        # Focal weighting (gamma controls focus on hard examples)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply weights
        focal_loss = alpha_t * focal_weight * bce
        
        return focal_loss.mean()
    
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute dice loss with small object boost"""
        pred_sigmoid = torch.sigmoid(pred)
        
        # Flatten for computation
        pred_flat = pred_sigmoid.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        # Batch-wise dice
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        base_dice_loss = 1 - dice.mean()
        
        # Add small plankton boost if enabled
        if self.small_plankton_boost > 0:
            small_boost = self._compute_small_plankton_loss(pred_sigmoid, target)
            return base_dice_loss + self.small_plankton_boost * small_boost
        
        return base_dice_loss
    
    def _compute_small_plankton_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute additional loss for small plankton objects"""
        try:
            # Work with CPU numpy for connected components
            target_np = target[0, 0].cpu().numpy() if target.dim() == 5 else target[0].cpu().numpy()
            pred_np = pred[0, 0].cpu().numpy() if pred.dim() == 5 else pred[0].cpu().numpy()
            
            # Find small objects (< 50 micrometers equivalent)
            max_small_size_voxels = int((50 / self.voxel_size_um) ** 3)
            
            labeled_target = label(target_np > 0.5)
            props = regionprops(labeled_target)
            
            small_mask = np.zeros_like(target_np)
            for prop in props:
                if prop.area <= max_small_size_voxels:
                    small_mask[labeled_target == prop.label] = 1
            
            if small_mask.sum() > 0:
                # Convert back to tensor
                small_mask_tensor = torch.tensor(
                    small_mask, dtype=target.dtype, device=target.device
                ).unsqueeze(0).unsqueeze(0)
                
                # Compute dice loss only on small objects
                pred_small = pred * small_mask_tensor
                target_small = target * small_mask_tensor
                
                pred_flat = pred_small.view(-1)
                target_flat = target_small.view(-1)
                
                intersection = (pred_flat * target_flat).sum()
                union = pred_flat.sum() + target_flat.sum()
                
                small_dice = (2 * intersection + self.smooth) / (union + self.smooth)
                return 1 - small_dice
            
            return torch.tensor(0.0, device=target.device)
            
        except Exception:
            # Fallback if connected components analysis fails
            return torch.tensor(0.0, device=target.device)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Forward pass returning loss and components for logging"""
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        
        combined = (1 - self.dice_weight) * focal + self.dice_weight * dice
        
        # Return loss components for logging
        loss_components = {
            'total_loss': combined.item(),
            'focal_component': focal.item(),
            'dice_component': dice.item(),
            'focal_weight': (1 - self.dice_weight),
            'dice_weight': self.dice_weight
        }
        
        return combined, loss_components

class TopKPlanktonLoss(nn.Module):
    """
    TopK loss optimized for plankton - focuses on hardest examples.
    Particularly good for boundary accuracy of small plankton.
    """
    
    def __init__(self, k: float = 0.7, base_loss: Optional[nn.Module] = None):
        super().__init__()
        self.k = k  # Use hardest 70% of pixels
        self.base_loss = base_loss or PlanktonFocalDiceLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Apply TopK selection to base loss"""
        if hasattr(self.base_loss, 'forward') and len(self.base_loss.forward.__code__.co_varnames) > 2:
            # Base loss returns components
            base_loss_value, components = self.base_loss(pred, target)
        else:
            # Simple loss function
            base_loss_value = self.base_loss(pred, target)
            components = {'base_loss': base_loss_value.item()}
        
        # Get per-pixel losses for TopK selection
        with torch.no_grad():
            pixel_losses = F.binary_cross_entropy_with_logits(
                pred, target, reduction='none'
            )
            loss_flat = pixel_losses.view(-1)
            k_samples = int(self.k * loss_flat.size(0))
            
            if k_samples > 0:
                top_k_values, top_k_indices = torch.topk(loss_flat, k_samples)
                # Create mask for hardest pixels
                mask = torch.zeros_like(loss_flat)
                mask[top_k_indices] = 1
                mask = mask.view_as(pixel_losses)
            else:
                mask = torch.ones_like(pixel_losses)
        
        # Apply mask to focus on hard pixels
        masked_pred = pred * mask
        masked_target = target * mask
        
        if torch.sum(mask) > 0:
            # Recompute loss on hard pixels only
            if hasattr(self.base_loss, 'forward') and len(self.base_loss.forward.__code__.co_varnames) > 2:
                topk_loss, topk_components = self.base_loss(masked_pred, masked_target)
            else:
                topk_loss = self.base_loss(masked_pred, masked_target)
                topk_components = {'topk_loss': topk_loss.item()}
            
            # Update components
            components.update({
                'topk_ratio': self.k,
                'hard_pixels_selected': int(torch.sum(mask).item()),
                **topk_components
            })
            
            return topk_loss, components
        else:
            return base_loss_value, components

def dice_score_plankton(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-7) -> torch.Tensor:
    """Optimized dice score computation for plankton evaluation"""
    pred_sigmoid = torch.sigmoid(pred) if pred.max() > 1 else pred
    
    # Flatten
    pred_flat = pred_sigmoid.view(-1)
    target_flat = target.view(-1)
    
    # Calculate dice
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice

# Factory function for easy loss selection
def create_plankton_loss(loss_type: str = "focal_dice", **kwargs):
    """Factory function to create plankton-optimized losses"""
    
    if loss_type == "focal_dice":
        return PlanktonFocalDiceLoss(**kwargs)
    elif loss_type == "topk_focal":
        base_loss = PlanktonFocalDiceLoss(**kwargs)
        return TopKPlanktonLoss(k=kwargs.get('k', 0.7), base_loss=base_loss)
    elif loss_type == "standard_dice":
        # For comparison purposes
        from plankton_optimized_learning_tools import DiceLoss
        return DiceLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from: focal_dice, topk_focal, standard_dice")
