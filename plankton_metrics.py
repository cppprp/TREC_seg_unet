import torch
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import binary_erosion
from scipy.spatial.distance import directed_hausdorff
from typing import Dict, List, Tuple, Optional
import warnings

class PlanktonMetrics:
    """
    Comprehensive metrics suite for plankton segmentation evaluation.
    
    Provides biologically relevant metrics that standard computer vision
    metrics miss, including size-specific performance, count accuracy,
    and ecological relevance measures.
    """
    
    def __init__(self, voxel_size_um: float = 1.0, min_object_size_um: float = 15):
        """
        Initialize plankton metrics calculator.
        
        Args:
            voxel_size_um: Size of one voxel in micrometers
            min_object_size_um: Minimum size to consider a valid plankton
        """
        self.voxel_size_um = voxel_size_um
        self.min_object_size_um = min_object_size_um
        self.min_object_size_voxels = int((min_object_size_um / voxel_size_um) ** 3)
        
        # Size class definitions (biologically relevant)
        self.size_classes = {
            'nano': (0.2, 2),        # Nanoplankton
            'micro': (2, 20),        # Microplankton  
            'small_meso': (20, 50),  # Small mesoplankton (often missed)
            'large_meso': (50, 200), # Large mesoplankton (easier to detect)
            'macro': (200, 2000)     # Macroplankton
        }
    
    def compute_all_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        Compute comprehensive plankton evaluation metrics.
        
        Args:
            pred: Predicted segmentation (logits or probabilities)
            target: Ground truth segmentation
            
        Returns:
            Dictionary of all computed metrics
        """
        # Convert to numpy and ensure binary
        pred_np = self._to_binary_numpy(pred)
        target_np = self._to_binary_numpy(target)
        
        metrics = {}
        
        # Basic metrics
        metrics.update(self._compute_basic_metrics(pred_np, target_np))
        
        # Size-specific metrics (most important for plankton!)
        metrics.update(self._compute_size_specific_metrics(pred_np, target_np))
        
        # Count-based metrics (population studies)
        metrics.update(self._compute_count_metrics(pred_np, target_np))
        
        # Volume/biomass metrics (ecosystem studies)  
        metrics.update(self._compute_volume_metrics(pred_np, target_np))
        
        # Shape/boundary metrics (morphological accuracy)
        metrics.update(self._compute_boundary_metrics(pred_np, target_np))
        
        # Ecological relevance metrics
        metrics.update(self._compute_ecological_metrics(pred_np, target_np))
        
        return metrics
    
    def _to_binary_numpy(self, tensor: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
        """Convert tensor to binary numpy array"""
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu()
            if tensor.max() > 1:  # Logits
                tensor = torch.sigmoid(tensor)
            tensor = tensor.numpy()
        
        # Handle different tensor shapes
        if tensor.ndim == 5:  # (B, C, D, H, W)
            tensor = tensor[0, 0]
        elif tensor.ndim == 4:  # (C, D, H, W) or (B, D, H, W)
            tensor = tensor[0] if tensor.shape[0] == 1 else tensor[0]
        elif tensor.ndim == 3:  # (D, H, W)
            pass
        else:
            raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
            
        return (tensor > threshold).astype(np.uint8)
    
    def _compute_basic_metrics(self, pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """Compute standard segmentation metrics"""
        intersection = np.sum(pred * target)
        union = np.sum(pred) + np.sum(target)
        
        # Dice coefficient
        dice = (2 * intersection) / (union + 1e-7)
        
        # IoU (Jaccard index)
        iou = intersection / (union - intersection + 1e-7)
        
        # Precision and Recall
        tp = intersection
        fp = np.sum(pred) - tp
        fn = np.sum(target) - tp
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        
        return {
            'dice_overall': dice,
            'iou_overall': iou,
            'precision_overall': precision,
            'recall_overall': recall,
            'f1_overall': f1
        }
    
    def _compute_size_specific_metrics(self, pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """
        Compute metrics for different plankton size classes.
        This is CRITICAL for plankton research!
        """
        metrics = {}
        
        try:
            # Label connected components
            target_labeled = label(target)
            pred_labeled = label(pred)
            
            target_props = regionprops(target_labeled)
            
            for size_class, (min_um, max_um) in self.size_classes.items():
                # Convert size range to voxels
                min_voxels = (min_um / self.voxel_size_um) ** 3
                max_voxels = (max_um / self.voxel_size_um) ** 3
                
                # Create mask for this size class in ground truth
                size_mask = np.zeros_like(target)
                size_count = 0
                
                for prop in target_props:
                    if min_voxels <= prop.area <= max_voxels:
                        size_mask[target_labeled == prop.label] = 1
                        size_count += 1
                
                if size_count > 0:
                    # Compute metrics for this size class
                    pred_masked = pred * size_mask
                    target_masked = target * size_mask
                    
                    intersection = np.sum(pred_masked * target_masked)
                    union = np.sum(pred_masked) + np.sum(target_masked)
                    
                    dice = (2 * intersection) / (union + 1e-7)
                    
                    metrics[f'dice_{size_class}'] = dice
                    metrics[f'count_{size_class}_true'] = size_count
                    
                    # Detection rate for this size class
                    detected = self._count_detected_objects_in_mask(pred, target_masked)
                    detection_rate = detected / size_count
                    metrics[f'detection_rate_{size_class}'] = detection_rate
                else:
                    metrics[f'dice_{size_class}'] = 0.0
                    metrics[f'count_{size_class}_true'] = 0
                    metrics[f'detection_rate_{size_class}'] = 0.0
        
        except Exception as e:
            warnings.warn(f"Error computing size-specific metrics: {e}")
            for size_class in self.size_classes.keys():
                metrics[f'dice_{size_class}'] = 0.0
                metrics[f'count_{size_class}_true'] = 0
                metrics[f'detection_rate_{size_class}'] = 0.0
        
        return metrics
    
    def _compute_count_metrics(self, pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """Compute count-based metrics (critical for population studies)"""
        try:
            # Remove small objects (noise) before counting
            pred_clean = self._remove_small_objects(pred)
            target_clean = self._remove_small_objects(target)
            
            # Count objects
            target_labeled = label(target_clean)
            pred_labeled = label(pred_clean)
            
            true_count = len(regionprops(target_labeled))
            pred_count = len(regionprops(pred_labeled))
            
            # Count correctly detected objects
            detected_count = self._count_detected_objects(pred_clean, target_clean)
            
            # Count-based metrics
            count_precision = detected_count / (pred_count + 1e-7)
            count_recall = detected_count / (true_count + 1e-7)
            count_f1 = 2 * (count_precision * count_recall) / (count_precision + count_recall + 1e-7)
            
            # Population density metrics
            count_error = abs(pred_count - true_count) / (true_count + 1e-7)
            count_accuracy = 1 - count_error
            
            return {
                'count_true': true_count,
                'count_predicted': pred_count,
                'count_detected': detected_count,
                'count_precision': count_precision,
                'count_recall': count_recall,
                'count_f1': count_f1,
                'count_accuracy': count_accuracy,
                'count_error': count_error,
                'population_density_ratio': pred_count / (true_count + 1e-7)
            }
            
        except Exception as e:
            warnings.warn(f"Error computing count metrics: {e}")
            return {
                'count_true': 0, 'count_predicted': 0, 'count_detected': 0,
                'count_precision': 0, 'count_recall': 0, 'count_f1': 0,
                'count_accuracy': 0, 'count_error': 1.0, 'population_density_ratio': 0
            }
    
    def _compute_volume_metrics(self, pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """Compute volume/biomass related metrics"""
        # Total volume (proxy for biomass)
        true_volume = np.sum(target) * (self.voxel_size_um ** 3)
        pred_volume = np.sum(pred) * (self.voxel_size_um ** 3)
        
        volume_error = abs(pred_volume - true_volume) / (true_volume + 1e-7)
        volume_ratio = pred_volume / (true_volume + 1e-7)
        
        # Average object size
        try:
            target_labeled = label(target)
            pred_labeled = label(pred)
            
            target_props = regionprops(target_labeled)
            pred_props = regionprops(pred_labeled)
            
            if len(target_props) > 0:
                true_avg_size = np.mean([prop.area for prop in target_props]) * (self.voxel_size_um ** 3)
            else:
                true_avg_size = 0
                
            if len(pred_props) > 0:
                pred_avg_size = np.mean([prop.area for prop in pred_props]) * (self.voxel_size_um ** 3)
            else:
                pred_avg_size = 0
            
            size_bias = (pred_avg_size - true_avg_size) / (true_avg_size + 1e-7)
            
        except Exception:
            true_avg_size = 0
            pred_avg_size = 0
            size_bias = 0
        
        return {
            'volume_true_um3': true_volume,
            'volume_pred_um3': pred_volume,
            'volume_error': volume_error,
            'volume_ratio': volume_ratio,
            'avg_object_size_true_um3': true_avg_size,
            'avg_object_size_pred_um3': pred_avg_size,
            'size_bias': size_bias
        }
    
    def _compute_boundary_metrics(self, pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """Compute boundary accuracy metrics"""
        try:
            # Surface dice (boundary accuracy)
            target_boundary = target - binary_erosion(target)
            pred_boundary = pred - binary_erosion(pred)
            
            boundary_intersection = np.sum(target_boundary * pred_boundary)
            boundary_union = np.sum(target_boundary) + np.sum(pred_boundary)
            
            surface_dice = (2 * boundary_intersection) / (boundary_union + 1e-7)
            
            # Hausdorff distance (max boundary error)
            hausdorff_dist = self._compute_hausdorff_distance(pred, target)
            
            return {
                'surface_dice': surface_dice,
                'hausdorff_distance_um': hausdorff_dist * self.voxel_size_um,
                'boundary_recall': boundary_intersection / (np.sum(target_boundary) + 1e-7),
                'boundary_precision': boundary_intersection / (np.sum(pred_boundary) + 1e-7)
            }
            
        except Exception as e:
            warnings.warn(f"Error computing boundary metrics: {e}")
            return {
                'surface_dice': 0.0,
                'hausdorff_distance_um': float('inf'),
                'boundary_recall': 0.0,
                'boundary_precision': 0.0
            }
    
    def _compute_ecological_metrics(self, pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """Compute ecologically relevant derived metrics"""
        metrics = {}
        
        # Size distribution preservation
        try:
            true_sizes = self._get_object_sizes(target)
            pred_sizes = self._get_object_sizes(pred)
            
            if len(true_sizes) > 0 and len(pred_sizes) > 0:
                # Compare size distributions
                true_mean_size = np.mean(true_sizes)
                pred_mean_size = np.mean(pred_sizes)
                
                size_distribution_bias = (pred_mean_size - true_mean_size) / (true_mean_size + 1e-7)
                
                # Size diversity (coefficient of variation)
                true_size_cv = np.std(true_sizes) / (true_mean_size + 1e-7)
                pred_size_cv = np.std(pred_sizes) / (pred_mean_size + 1e-7) if pred_mean_size > 0 else 0
                
                diversity_preservation = 1 - abs(pred_size_cv - true_size_cv) / (true_size_cv + 1e-7)
                
                metrics.update({
                    'size_distribution_bias': size_distribution_bias,
                    'diversity_preservation': diversity_preservation,
                    'true_size_diversity': true_size_cv,
                    'pred_size_diversity': pred_size_cv
                })
        
        except Exception:
            metrics.update({
                'size_distribution_bias': 0,
                'diversity_preservation': 0,
                'true_size_diversity': 0,
                'pred_size_diversity': 0
            })
        
        # Ecological quality score (weighted combination of key metrics)
        try:
            small_dice = metrics.get('dice_small_meso', 0)  # Most important size class
            count_acc = metrics.get('count_accuracy', 0)
            volume_acc = 1 - metrics.get('volume_error', 1)
            
            # Weighted ecological quality (small plankton are most important)
            ecological_quality = (0.5 * small_dice + 0.3 * count_acc + 0.2 * volume_acc)
            metrics['ecological_quality_score'] = ecological_quality
            
        except Exception:
            metrics['ecological_quality_score'] = 0
        
        return metrics
    
    # Helper methods
    def _remove_small_objects(self, binary_img: np.ndarray) -> np.ndarray:
        """Remove objects smaller than minimum plankton size"""
        from skimage.morphology import remove_small_objects
        return remove_small_objects(binary_img.astype(bool), min_size=self.min_object_size_voxels).astype(np.uint8)
    
    def _count_detected_objects(self, pred: np.ndarray, target: np.ndarray) -> int:
        """Count how many true objects were detected (IoU > 0.1 threshold)"""
        target_labeled = label(target)
        pred_labeled = label(pred)
        
        target_props = regionprops(target_labeled)
        detected = 0
        
        for target_prop in target_props:
            target_mask = (target_labeled == target_prop.label)
            overlap = np.sum(pred * target_mask)
            
            if overlap > 0.1 * target_prop.area:  # 10% overlap threshold
                detected += 1
        
        return detected
    
    def _count_detected_objects_in_mask(self, pred: np.ndarray, target_mask: np.ndarray) -> int:
        """Count detected objects within a specific mask"""
        masked_target = label(target_mask)
        props = regionprops(masked_target)
        
        detected = 0
        for prop in props:
            obj_mask = (masked_target == prop.label)
            overlap = np.sum(pred * obj_mask)
            
            if overlap > 0.1 * prop.area:
                detected += 1
        
        return detected
    
    def _get_object_sizes(self, binary_img: np.ndarray) -> List[float]:
        """Get sizes of all objects in micrometers^3"""
        labeled_img = label(binary_img)
        props = regionprops(labeled_img)
        
        sizes = [prop.area * (self.voxel_size_um ** 3) for prop in props]
        return sizes
    
    def _compute_hausdorff_distance(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute 95th percentile Hausdorff distance"""
        try:
            pred_points = np.argwhere(pred > 0)
            target_points = np.argwhere(target > 0)
            
            if len(pred_points) == 0 or len(target_points) == 0:
                return float('inf')
            
            # Sample points if too many (for performance)
            if len(pred_points) > 1000:
                pred_points = pred_points[::len(pred_points)//1000]
            if len(target_points) > 1000:
                target_points = target_points[::len(target_points)//1000]
            
            dist1 = directed_hausdorff(pred_points, target_points)[0]
            dist2 = directed_hausdorff(target_points, pred_points)[0]
            
            return max(dist1, dist2)
            
        except Exception:
            return float('inf')

# Convenience functions for quick evaluation
def evaluate_plankton_batch(predictions: torch.Tensor, targets: torch.Tensor, 
                           voxel_size_um: float = 1.0) -> Dict[str, float]:
    """Evaluate a batch of plankton predictions"""
    metrics_calculator = PlanktonMetrics(voxel_size_um=voxel_size_um)
    
    batch_metrics = {}
    batch_size = predictions.shape[0]
    
    for i in range(batch_size):
        sample_metrics = metrics_calculator.compute_all_metrics(predictions[i], targets[i])
        
        # Accumulate metrics
        for key, value in sample_metrics.items():
            if key not in batch_metrics:
                batch_metrics[key] = 0
            batch_metrics[key] += value
    
    # Average across batch
    for key in batch_metrics:
        batch_metrics[key] /= batch_size
    
    return batch_metrics

def get_key_plankton_metrics(all_metrics: Dict[str, float]) -> Dict[str, float]:
    """Extract the most important metrics for quick monitoring"""
    key_metrics = {}
    
    # Most critical metrics for plankton research
    important_keys = [
        'dice_small_meso',      # Small plankton detection (most important!)
        'count_accuracy',       # Population count accuracy
        'volume_error',         # Biomass estimation error
        'ecological_quality_score',  # Overall ecological usefulness
        'dice_overall',         # Standard metric for comparison
        'detection_rate_small_meso',  # Small plankton detection rate
        'count_recall',         # How many individuals we find
        'size_distribution_bias'  # Are we biased toward large objects?
    ]
    
    for key in important_keys:
        if key in all_metrics:
            key_metrics[key] = all_metrics[key]
    
    return key_metrics
