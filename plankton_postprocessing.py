import numpy as np
import torch
from skimage.measure import label, regionprops
from skimage.morphology import (
    remove_small_objects, binary_opening, binary_closing, 
    binary_erosion, binary_dilation, disk, ball
)
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage
from typing import Tuple, Optional, Dict, Any
import warnings

class PlanktonPostProcessor:
    """
    Advanced post-processing specifically designed for plankton segmentation.
    
    Handles common plankton-specific challenges:
    - Remove debris and noise smaller than minimum plankton size
    - Separate touching/overlapping plankton
    - Fill holes in semi-transparent organisms
    - Smooth boundaries while preserving biological shapes
    - Handle size-specific morphological operations
    """
    
    def __init__(
        self, 
        voxel_size_um: float = 1.0,
        min_plankton_size_um: float = 15,
        max_plankton_size_um: float = 300,
        fill_holes: bool = True,
        separate_touching: bool = True,
        smooth_boundaries: bool = True
    ):
        """
        Initialize plankton post-processor.
        
        Args:
            voxel_size_um: Size of one voxel in micrometers
            min_plankton_size_um: Minimum size to consider valid plankton
            max_plankton_size_um: Maximum expected plankton size
            fill_holes: Whether to fill holes in organisms
            separate_touching: Whether to separate touching organisms
            smooth_boundaries: Whether to smooth object boundaries
        """
        self.voxel_size_um = voxel_size_um
        self.min_plankton_size_um = min_plankton_size_um
        self.max_plankton_size_um = max_plankton_size_um
        self.fill_holes = fill_holes
        self.separate_touching = separate_touching
        self.smooth_boundaries = smooth_boundaries
        
        # Convert sizes to voxels
        self.min_size_voxels = int((min_plankton_size_um / voxel_size_um) ** 3)
        self.max_size_voxels = int((max_plankton_size_um / voxel_size_um) ** 3)
        
        # Define morphological structure elements for different operations
        self.small_structure = ball(1)    # For fine operations
        self.medium_structure = ball(2)   # For standard operations
        self.large_structure = ball(3)    # For large plankton
    
    def process_prediction(
        self, 
        prediction: np.ndarray, 
        threshold: float = 0.5,
        return_info: bool = False
    ) -> np.ndarray:
        """
        Apply complete post-processing pipeline to a prediction.
        
        Args:
            prediction: Raw model prediction (0-1 or logits)
            threshold: Threshold for binarization
            return_info: Whether to return processing statistics
            
        Returns:
            Cleaned binary segmentation
        """
        # Convert to binary
        if prediction.max() > 1:  # Logits
            binary_pred = torch.sigmoid(torch.tensor(prediction)).numpy() > threshold
        else:
            binary_pred = prediction > threshold
        
        binary_pred = binary_pred.astype(bool)
        
        processing_info = {
            'original_objects': 0,
            'after_size_filter': 0,
            'after_morphology': 0,
            'after_separation': 0,
            'final_objects': 0
        }
        
        # Step 1: Initial object counting
        initial_labeled = label(binary_pred)
        processing_info['original_objects'] = len(regionprops(initial_labeled))
        
        # Step 2: Remove objects that are too small (noise/debris)
        cleaned = self._remove_small_objects(binary_pred)
        size_filtered_labeled = label(cleaned)
        processing_info['after_size_filter'] = len(regionprops(size_filtered_labeled))
        
        # Step 3: Remove objects that are too large (likely artifacts)
        cleaned = self._remove_large_objects(cleaned)
        
        # Step 4: Morphological operations
        if self.smooth_boundaries or self.fill_holes:
            cleaned = self._apply_morphological_operations(cleaned)
            morph_labeled = label(cleaned)
            processing_info['after_morphology'] = len(regionprops(morph_labeled))
        
        # Step 5: Separate touching objects
        if self.separate_touching:
            cleaned = self._separate_touching_objects(cleaned)
            sep_labeled = label(cleaned)
            processing_info['after_separation'] = len(regionprops(sep_labeled))
        
        # Step 6: Final cleanup - remove any new small objects created by separation
        cleaned = self._remove_small_objects(cleaned)
        
        # Final count
        final_labeled = label(cleaned)
        processing_info['final_objects'] = len(regionprops(final_labeled))
        
        if return_info:
            return cleaned.astype(np.uint8), processing_info
        else:
            return cleaned.astype(np.uint8)
    
    def _remove_small_objects(self, binary_img: np.ndarray) -> np.ndarray:
        """Remove objects smaller than minimum plankton size"""
        try:
            return remove_small_objects(
                binary_img, 
                min_size=self.min_size_voxels,
                connectivity=1
            )
        except Exception as e:
            warnings.warn(f"Error removing small objects: {e}")
            return binary_img
    
    def _remove_large_objects(self, binary_img: np.ndarray) -> np.ndarray:
        """Remove objects larger than maximum expected plankton size"""
        try:
            labeled_img = label(binary_img)
            props = regionprops(labeled_img)
            
            # Create mask for objects within size range
            valid_mask = np.zeros_like(binary_img, dtype=bool)
            
            for prop in props:
                if prop.area <= self.max_size_voxels:
                    valid_mask[labeled_img == prop.label] = True
            
            return valid_mask
            
        except Exception as e:
            warnings.warn(f"Error removing large objects: {e}")
            return binary_img
    
    def _apply_morphological_operations(self, binary_img: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean up shapes"""
        try:
            cleaned = binary_img.copy()
            
            # Fill holes (important for semi-transparent plankton)
            if self.fill_holes:
                cleaned = ndimage.binary_fill_holes(cleaned)
            
            # Smooth boundaries while preserving shape
            if self.smooth_boundaries:
                # Opening to remove small protrusions
                cleaned = binary_opening(cleaned, structure=self.small_structure)
                
                # Closing to fill small gaps
                cleaned = binary_closing(cleaned, structure=self.medium_structure)
            
            return cleaned
            
        except Exception as e:
            warnings.warn(f"Error in morphological operations: {e}")
            return binary_img
    
    def _separate_touching_objects(self, binary_img: np.ndarray) -> np.ndarray:
        """Separate touching plankton using watershed algorithm"""
        try:
            # Distance transform
            distance = ndimage.distance_transform_edt(binary_img)
            
            # Find local maxima (centers of objects)
            # Use different minimum distances based on expected plankton sizes
            min_distance_voxels = max(3, int((self.min_plankton_size_um / self.voxel_size_um) / 3))
            
            local_maxima = peak_local_max(
                distance, 
                min_distance=min_distance_voxels,
                threshold_abs=2  # Minimum distance value to consider
            )
            
            if len(local_maxima[0]) == 0:
                return binary_img
            
            # Create markers for watershed
            markers = np.zeros_like(binary_img, dtype=int)
            for i, (z, y, x) in enumerate(zip(local_maxima[0], local_maxima[1], local_maxima[2])):
                markers[z, y, x] = i + 1
            
            # Apply watershed
            # Use negative distance as the landscape (valleys at object centers)
            segmented = watershed(-distance, markers, mask=binary_img)
            
            # Convert back to binary
            return segmented > 0
            
        except Exception as e:
            warnings.warn(f"Error in watershed separation: {e}")
            return binary_img
    
    def analyze_objects(self, binary_img: np.ndarray) -> Dict[str, Any]:
        """Analyze objects in the segmentation for quality assessment"""
        try:
            labeled_img = label(binary_img)
            props = regionprops(labeled_img)
            
            if len(props) == 0:
                return {
                    'num_objects': 0,
                    'total_volume_um3': 0,
                    'avg_size_um3': 0,
                    'size_distribution': {},
                    'shape_metrics': {}
                }
            
            # Size analysis
            sizes_voxels = [prop.area for prop in props]
            sizes_um3 = [size * (self.voxel_size_um ** 3) for size in sizes_voxels]
            
            # Size distribution by categories
            size_distribution = {
                'nano_count': sum(1 for s in sizes_um3 if 0.2 <= s < 2),
                'micro_count': sum(1 for s in sizes_um3 if 2 <= s < 20),
                'small_meso_count': sum(1 for s in sizes_um3 if 20 <= s < 50),
                'large_meso_count': sum(1 for s in sizes_um3 if 50 <= s < 200),
                'macro_count': sum(1 for s in sizes_um3 if s >= 200)
            }
            
            # Shape analysis
            shape_metrics = {}
            if len(props) > 0:
                # Sphericity (how sphere-like are the objects)
                sphericities = []
                for prop in props:
                    if prop.area > 0:
                        # Approximation of sphericity using equivalent diameter vs actual extent
                        equivalent_diameter = 2 * (3 * prop.area / (4 * np.pi)) ** (1/3)
                        max_extent = max(prop.bbox[3] - prop.bbox[0], 
                                       prop.bbox[4] - prop.bbox[1], 
                                       prop.bbox[5] - prop.bbox[2])
                        if max_extent > 0:
                            sphericity = equivalent_diameter / max_extent
                            sphericities.append(min(sphericity, 1.0))
                
                shape_metrics = {
                    'avg_sphericity': np.mean(sphericities) if sphericities else 0,
                    'sphericity_std': np.std(sphericities) if sphericities else 0
                }
            
            return {
                'num_objects': len(props),
                'total_volume_um3': sum(sizes_um3),
                'avg_size_um3': np.mean(sizes_um3),
                'median_size_um3': np.median(sizes_um3),
                'size_std_um3': np.std(sizes_um3),
                'min_size_um3': min(sizes_um3),
                'max_size_um3': max(sizes_um3),
                'size_distribution': size_distribution,
                'shape_metrics': shape_metrics
            }
            
        except Exception as e:
            warnings.warn(f"Error analyzing objects: {e}")
            return {'num_objects': 0, 'error': str(e)}

class BatchPostProcessor:
    """Process multiple predictions efficiently"""
    
    def __init__(self, post_processor: PlanktonPostProcessor):
        self.post_processor = post_processor
    
    def process_batch(
        self, 
        predictions: np.ndarray, 
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a batch of predictions.
        
        Args:
            predictions: Batch of predictions [B, C, D, H, W] or [B, D, H, W]
            threshold: Threshold for binarization
            
        Returns:
            Processed batch and summary statistics
        """
        if predictions.ndim == 5:  # [B, C, D, H, W]
            predictions = predictions[:, 0]  # Remove channel dimension
        
        batch_size = predictions.shape[0]
        processed_batch = np.zeros_like(predictions, dtype=np.uint8)
        
        batch_stats = {
            'total_objects_before': 0,
            'total_objects_after': 0,
            'total_volume_before_um3': 0,
            'total_volume_after_um3': 0,
            'processing_efficiency': 0
        }
        
        for i in range(batch_size):
            # Process individual prediction
            processed, info = self.post_processor.process_prediction(
                predictions[i], threshold=threshold, return_info=True
            )
            processed_batch[i] = processed
            
            # Accumulate statistics
            batch_stats['total_objects_before'] += info['original_objects']
            batch_stats['total_objects_after'] += info['final_objects']
        
        # Calculate processing efficiency
        if batch_stats['total_objects_before'] > 0:
            batch_stats['processing_efficiency'] = (
                batch_stats['total_objects_after'] / batch_stats['total_objects_before']
            )
        
        return processed_batch, batch_stats

class PlanktonTestTimeAugmentationPostProcessor:
    """
    Combine Test-Time Augmentation with post-processing for optimal results.
    """
    
    def __init__(
        self, 
        model: torch.nn.Module,
        post_processor: PlanktonPostProcessor,
        device: torch.device
    ):
        self.model = model
        self.post_processor = post_processor
        self.device = device
    
    def predict_and_process(
        self, 
        image: torch.Tensor,
        num_tta_iterations: int = 8,
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply TTA, average predictions, then post-process.
        
        Args:
            image: Input image [C, D, H, W] or [1, C, D, H, W]
            num_tta_iterations: Number of TTA iterations
            threshold: Threshold for final binarization
            
        Returns:
            Post-processed prediction and analysis
        """
        from plankton_augmentations import PlanktonTestTimeAugmentation
        
        # Apply TTA
        tta_processor = PlanktonTestTimeAugmentation(
            voxel_size_um=self.post_processor.voxel_size_um
        )
        
        # Get TTA prediction
        tta_prediction = tta_processor.predict_with_tta(
            self.model, image, self.device, num_tta_iterations
        )
        
        # Convert to numpy
        if isinstance(tta_prediction, torch.Tensor):
            tta_prediction = tta_prediction.cpu().numpy()
        
        # Remove batch and channel dimensions if present
        if tta_prediction.ndim == 5:  # [1, C, D, H, W]
            tta_prediction = tta_prediction[0, 0]
        elif tta_prediction.ndim == 4:  # [C, D, H, W]
            tta_prediction = tta_prediction[0]
        
        # Post-process
        processed, processing_info = self.post_processor.process_prediction(
            tta_prediction, threshold=threshold, return_info=True
        )
        
        # Analyze final result
        analysis = self.post_processor.analyze_objects(processed)
        
        return processed, {
            'processing_info': processing_info,
            'object_analysis': analysis,
            'tta_iterations': num_tta_iterations
        }

# Utility functions for easy use
def quick_clean_plankton_prediction(
    prediction: np.ndarray,
    voxel_size_um: float = 1.0,
    min_size_um: float = 15,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Quick cleaning of plankton prediction with default settings.
    
    Args:
        prediction: Raw model prediction
        voxel_size_um: Voxel size in micrometers
        min_size_um: Minimum plankton size in micrometers
        threshold: Threshold for binarization
        
    Returns:
        Cleaned binary prediction
    """
    processor = PlanktonPostProcessor(
        voxel_size_um=voxel_size_um,
        min_plankton_size_um=min_size_um
    )
    
    return processor.process_prediction(prediction, threshold=threshold)

def analyze_plankton_segmentation(
    prediction: np.ndarray,
    voxel_size_um: float = 1.0
) -> Dict[str, Any]:
    """
    Quick analysis of plankton segmentation quality.
    
    Args:
        prediction: Binary segmentation
        voxel_size_um: Voxel size in micrometers
        
    Returns:
        Analysis dictionary
    """
    processor = PlanktonPostProcessor(voxel_size_um=voxel_size_um)
    return processor.analyze_objects(prediction)

# Example usage and testing
if __name__ == "__main__":
    # Test post-processing
    print("Testing Plankton Post-Processing...")
    
    # Create dummy prediction with noise and touching objects
    dummy_prediction = np.random.rand(64, 64, 64)
    
    # Add some "plankton" objects
    dummy_prediction[20:30, 20:30, 20:30] = 0.9  # Large object
    dummy_prediction[40:45, 40:45, 40:45] = 0.8  # Small object
    dummy_prediction[25:35, 45:55, 25:35] = 0.85 # Touching object
    
    # Add noise
    noise_mask = np.random.rand(64, 64, 64) > 0.98
    dummy_prediction[noise_mask] = 0.7
    
    # Test processing
    processor = PlanktonPostProcessor(
        voxel_size_um=0.5,
        min_plankton_size_um=10
    )
    
    cleaned, info = processor.process_prediction(
        dummy_prediction, 
        threshold=0.5, 
        return_info=True
    )
    
    print("Processing Results:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test analysis
    analysis = processor.analyze_objects(cleaned)
    print("\nObject Analysis:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")
    
    print("\nPost-processing test completed successfully!")
