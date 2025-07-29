import numpy as np
import torch
from skimage.measure import label, regionprops
from skimage.segmentation import watershed
from skimage.morphology import binary_erosion, binary_dilation, disk, ball
from scipy import ndimage
from typing import Tuple, Dict, List, Any
import warnings

class PlanktonInstanceSegmentator:
    """
    Convert 2-channel predictions (foreground + boundary) into individual plankton instances.
    
    This is crucial for plankton research where you need to:
    1. Count individual organisms (population studies)
    2. Measure each plankton separately (size distributions)
    3. Separate touching/overlapping plankton
    4. Get accurate boundary delineation
    """
    
    def __init__(self, voxel_size_um: float = 1.0):
        self.voxel_size_um = voxel_size_um
    
    def create_instances_from_channels(
        self, 
        foreground: np.ndarray, 
        boundary: np.ndarray,
        method: str = 'boundary_watershed',
        min_size_um: float = 15,
        foreground_threshold: float = 0.5,
        boundary_threshold: float = 0.3
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Create instance segmentation from foreground and boundary predictions.
        
        Args:
            foreground: Foreground probability map [D, H, W]
            boundary: Boundary probability map [D, H, W]  
            method: Instance creation method
            min_size_um: Minimum plankton size in micrometers
            foreground_threshold: Threshold for foreground channel
            boundary_threshold: Threshold for boundary channel
            
        Returns:
            Instance labels and metadata
        """
        print(f"🔬 Creating plankton instances using {method} method...")
        
        # Threshold the predictions
        fg_binary = foreground > foreground_threshold
        bd_binary = boundary > boundary_threshold
        
        info = {
            'method': method,
            'foreground_pixels': np.sum(fg_binary),
            'boundary_pixels': np.sum(bd_binary),
            'foreground_threshold': foreground_threshold,
            'boundary_threshold': boundary_threshold
        }
        
        if method == 'boundary_watershed':
            instances = self._boundary_watershed_method(fg_binary, bd_binary, info)
        elif method == 'boundary_subtraction':
            instances = self._boundary_subtraction_method(fg_binary, bd_binary, info)
        elif method == 'distance_watershed':
            instances = self._distance_watershed_method(fg_binary, info)
        elif method == 'combined_approach':
            instances = self._combined_approach(fg_binary, bd_binary, foreground, boundary, info)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Remove small objects (debris/noise)
        min_size_voxels = int((min_size_um / self.voxel_size_um) ** 3)
        instances = self._remove_small_instances(instances, min_size_voxels)
        
        # Final statistics
        final_labels = np.unique(instances)
        num_instances = len(final_labels) - 1  # Exclude background (0)
        info['final_instances'] = num_instances
        
        print(f"✅ Created {num_instances} plankton instances")
        
        return instances, info
    
    def _boundary_watershed_method(
        self, 
        foreground: np.ndarray, 
        boundary: np.ndarray, 
        info: Dict
    ) -> np.ndarray:
        """
        Use boundary predictions to guide watershed segmentation.
        
        This is the most effective method for plankton as it uses both channels optimally.
        """
        print("  📍 Using boundary-guided watershed...")
        
        # Step 1: Create seeds by removing boundary regions from foreground
        seeds_mask = foreground & (~boundary)
        
        # Step 2: Distance transform on the seeds
        distance = ndimage.distance_transform_edt(seeds_mask)
        
        # Step 3: Find local maxima as watershed markers
        from skimage.feature import peak_local_maxima
        
        # Adaptive minimum distance based on expected plankton size
        min_distance = max(3, int((20 / self.voxel_size_um) / 2))  # Half of 20μm plankton
        
        try:
            local_maxima = peak_local_maxima(
                distance, 
                min_distance=min_distance,
                threshold_abs=2
            )
            
            if len(local_maxima[0]) == 0:
                print("  ⚠️ No local maxima found, falling back to simple labeling")
                return label(foreground)
            
            # Create markers
            markers = np.zeros_like(foreground, dtype=int)
            for i, (z, y, x) in enumerate(zip(local_maxima[0], local_maxima[1], local_maxima[2])):
                markers[z, y, x] = i + 1
            
            # Step 4: Watershed using boundary-informed distance
            # Use inverted boundary as additional landscape information
            boundary_weight = 0.3  # How much to weight boundary information
            landscape = -distance + boundary_weight * boundary
            
            # Apply watershed
            instances = watershed(landscape, markers, mask=foreground)
            
            info['watershed_seeds'] = len(local_maxima[0])
            
            return instances
            
        except Exception as e:
            warnings.warn(f"Boundary watershed failed: {e}, using simple labeling")
            return label(foreground)
    
    def _boundary_subtraction_method(
        self, 
        foreground: np.ndarray, 
        boundary: np.ndarray, 
        info: Dict
    ) -> np.ndarray:
        """
        Subtract boundary from foreground, then label connected components.
        
        Simple but effective for well-separated plankton.
        """
        print("  ✂️ Using boundary subtraction method...")
        
        # Remove boundary pixels from foreground
        cleaned_foreground = foreground & (~boundary)
        
        # Fill small holes that might have been created
        cleaned_foreground = ndimage.binary_fill_holes(cleaned_foreground)
        
        # Label connected components
        instances = label(cleaned_foreground)
        
        info['method_note'] = 'Simple boundary subtraction'
        
        return instances
    
    def _distance_watershed_method(
        self, 
        foreground: np.ndarray, 
        info: Dict
    ) -> np.ndarray:
        """
        Standard distance-based watershed (ignores boundary channel).
        
        Fallback method when boundary predictions are poor.
        """
        print("  📏 Using distance-based watershed...")
        
        # Distance transform
        distance = ndimage.distance_transform_edt(foreground)
        
        # Find local maxima
        from skimage.feature import peak_local_maxima
        
        min_distance = max(3, int((15 / self.voxel_size_um) / 2))
        
        try:
            local_maxima = peak_local_maxima(
                distance, 
                min_distance=min_distance,
                threshold_abs=1.5
            )
            
            if len(local_maxima[0]) == 0:
                return label(foreground)
            
            # Create markers
            markers = np.zeros_like(foreground, dtype=int)
            for i, (z, y, x) in enumerate(zip(local_maxima[0], local_maxima[1], local_maxima[2])):
                markers[z, y, x] = i + 1
            
            # Watershed
            instances = watershed(-distance, markers, mask=foreground)
            
            return instances
            
        except Exception as e:
            warnings.warn(f"Distance watershed failed: {e}")
            return label(foreground)
    
    def _combined_approach(
        self, 
        fg_binary: np.ndarray, 
        bd_binary: np.ndarray,
        fg_prob: np.ndarray,
        bd_prob: np.ndarray,
        info: Dict
    ) -> np.ndarray:
        """
        Advanced method combining multiple approaches.
        
        Best for challenging cases with many touching plankton.
        """
        print("  🎯 Using combined approach...")
        
        # Step 1: Use boundary probabilities to weight distance transform
        # High boundary probability = low distance value (valleys)
        boundary_weight = bd_prob * 0.5
        
        # Step 2: Create enhanced foreground using probability values
        enhanced_fg = fg_prob * (1 - boundary_weight)
        
        # Step 3: Distance transform on binary foreground
        distance = ndimage.distance_transform_edt(fg_binary)
        
        # Step 4: Combine distance with probability information
        combined_landscape = distance * enhanced_fg
        
        # Step 5: Find seeds in high-confidence, low-boundary regions
        seed_mask = (fg_prob > 0.7) & (bd_prob < 0.3)
        seed_distance = ndimage.distance_transform_edt(seed_mask)
        
        # Step 6: Local maxima on seed regions
        from skimage.feature import peak_local_maxima
        
        min_distance = max(2, int((12 / self.voxel_size_um) / 2))
        
        try:
            local_maxima = peak_local_maxima(
                seed_distance,
                min_distance=min_distance,
                threshold_abs=1
            )
            
            if len(local_maxima[0]) == 0:
                return self._boundary_watershed_method(fg_binary, bd_binary, info)
            
            # Create markers
            markers = np.zeros_like(fg_binary, dtype=int)
            for i, (z, y, x) in enumerate(zip(local_maxima[0], local_maxima[1], local_maxima[2])):
                markers[z, y, x] = i + 1
            
            # Final watershed
            instances = watershed(-combined_landscape, markers, mask=fg_binary)
            
            info['combined_seeds'] = len(local_maxima[0])
            info['method_note'] = 'Probability-weighted combined approach'
            
            return instances
            
        except Exception as e:
            warnings.warn(f"Combined approach failed: {e}")
            return self._boundary_watershed_method(fg_binary, bd_binary, info)
    
    def _remove_small_instances(self, instances: np.ndarray, min_size: int) -> np.ndarray:
        """Remove instances smaller than minimum size"""
        props = regionprops(instances)
        
        # Create mask for valid instances
        valid_mask = np.zeros_like(instances, dtype=bool)
        kept_instances = 0
        
        for prop in props:
            if prop.area >= min_size:
                valid_mask[instances == prop.label] = True
                kept_instances += 1
        
        # Relabel to remove gaps in numbering
        if kept_instances > 0:
            cleaned_instances = label(valid_mask)
        else:
            cleaned_instances = np.zeros_like(instances)
        
        print(f"  🧹 Kept {kept_instances}/{len(props)} instances after size filtering")
        
        return cleaned_instances
    
    def analyze_instances(self, instances: np.ndarray) -> Dict[str, Any]:
        """Analyze the created instances"""
        props = regionprops(instances)
        
        if len(props) == 0:
            return {
                'num_instances': 0,
                'total_volume_um3': 0,
                'size_distribution': {},
                'analysis_error': 'No instances found'
            }
        
        # Size analysis
        volumes_voxels = [prop.area for prop in props]
        volumes_um3 = [v * (self.voxel_size_um ** 3) for v in volumes_voxels]
        
        # Equivalent sphere diameters
        diameters_um = [2 * (3 * v / (4 * np.pi)) ** (1/3) for v in volumes_um3]
        
        # Size class distribution
        size_classes = {
            'nano_count': sum(1 for d in diameters_um if 0.2 <= d < 2),
            'micro_count': sum(1 for d in diameters_um if 2 <= d < 20),
            'small_meso_count': sum(1 for d in diameters_um if 20 <= d < 50),
            'large_meso_count': sum(1 for d in diameters_um if 50 <= d < 200),
            'macro_count': sum(1 for d in diameters_um if d >= 200)
        }
        
        return {
            'num_instances': len(props),
            'total_volume_um3': sum(volumes_um3),
            'mean_volume_um3': np.mean(volumes_um3),
            'median_volume_um3': np.median(volumes_um3),
            'volume_std_um3': np.std(volumes_um3),
            'mean_diameter_um': np.mean(diameters_um),
            'diameter_range_um': (min(diameters_um), max(diameters_um)),
            'size_distribution': size_classes,
            'individual_volumes_um3': volumes_um3,
            'individual_diameters_um': diameters_um
        }

def process_two_channel_prediction(
    prediction: np.ndarray, 
    voxel_size_um: float = 1.0,
    method: str = 'boundary_watershed'
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convenience function to process 2-channel prediction into instances.
    
    Args:
        prediction: 2-channel prediction [2, D, H, W] or [D, H, W, 2]
        voxel_size_um: Voxel size in micrometers
        method: Instance creation method
        
    Returns:
        Instance segmentation and analysis
    """
    # Handle different input formats
    if prediction.ndim == 4:
        if prediction.shape[0] == 2:  # [2, D, H, W]
            foreground = prediction[0]
            boundary = prediction[1]
        elif prediction.shape[-1] == 2:  # [D, H, W, 2]
            foreground = prediction[..., 0]
            boundary = prediction[..., 1]
        else:
            raise ValueError(f"Unexpected prediction shape: {prediction.shape}")
    else:
        raise ValueError(f"Expected 4D prediction, got {prediction.ndim}D")
    
    # Create instance segmentator
    segmentator = PlanktonInstanceSegmentator(voxel_size_um)
    
    # Create instances
    instances, info = segmentator.create_instances_from_channels(
        foreground, boundary, method=method
    )
    
    # Analyze results
    analysis = segmentator.analyze_instances(instances)
    
    # Combine info and analysis
    complete_info = {**info, **analysis}
    
    return instances, complete_info

def compare_instance_methods(
    foreground: np.ndarray, 
    boundary: np.ndarray,
    voxel_size_um: float = 1.0
) -> Dict[str, Dict[str, Any]]:
    """
    Compare different instance creation methods.
    
    Useful for understanding which method works best for your data.
    """
    print("🔍 Comparing instance creation methods...")
    
    segmentator = PlanktonInstanceSegmentator(voxel_size_um)
    methods = ['boundary_watershed', 'boundary_subtraction', 'distance_watershed', 'combined_approach']
    
    results = {}
    
    for method in methods:
        print(f"\n--- Testing {method} ---")
        try:
            instances, info = segmentator.create_instances_from_channels(
                foreground, boundary, method=method
            )
            
            # Quick analysis
            analysis = segmentator.analyze_instances(instances)
            
            results[method] = {
                'success': True,
                'num_instances': analysis['num_instances'],
                'mean_volume_um3': analysis.get('mean_volume_um3', 0),
                'size_distribution': analysis.get('size_distribution', {}),
                'method_info': info
            }
            
        except Exception as e:
            results[method] = {
                'success': False,
                'error': str(e),
                'num_instances': 0
            }
    
    # Print comparison
    print(f"\n📊 METHOD COMPARISON:")
    print(f"{'Method':<20} {'Success':<8} {'Instances':<10} {'Mean Vol (μm³)':<15}")
    print("-" * 60)
    
    for method, result in results.items():
        success = "✅" if result['success'] else "❌"
        instances = result['num_instances']
        mean_vol = result.get('mean_volume_um3', 0)
        print(f"{method:<20} {success:<8} {instances:<10} {mean_vol:<15.1f}")
    
    return results

# Integration with existing pipeline
def update_inference_for_instances():
    """
    Example of how to modify the inference pipeline to use both channels.
    """
    code_example = '''
    # In your inference script, after getting model prediction:
    
    # prediction shape: [1, 2, D, H, W] (batch, channels, depth, height, width)
    foreground = prediction[0, 0].cpu().numpy()  # First channel
    boundary = prediction[0, 1].cpu().numpy()    # Second channel
    
    # Create instances using both channels
    instances, info = process_two_channel_prediction(
        np.stack([foreground, boundary]), 
        voxel_size_um=config.data.voxel_size_um,
        method='boundary_watershed'  # Best for most plankton data
    )
    
    # Save both semantic segmentation and instances
    tifffile.imwrite('foreground.tif', (foreground > 0.5).astype(np.uint8))
    tifffile.imwrite('instances.tif', instances.astype(np.uint16))
    
    # Print analysis
    print(f"Found {info['num_instances']} individual plankton")
    print(f"Size distribution: {info['size_distribution']}")
    '''
    print("💡 Here's how to integrate instance segmentation:")
    print(code_example)

if __name__ == "__main__":
    print("🦠 Plankton Instance Segmentation Demo")
    
    # Create synthetic 2-channel prediction
    shape = (64, 64, 64)
    
    # Synthetic foreground (3 overlapping plankton)
    foreground = np.zeros(shape, dtype=np.float32)
    foreground[10:25, 10:25, 10:25] = 0.9  # Large plankton
    foreground[20:30, 20:30, 20:30] = 0.8  # Overlapping plankton
    foreground[45:50, 45:50, 45:50] = 0.85 # Small plankton
    
    # Synthetic boundary (boundaries between objects)
    boundary = np.zeros(shape, dtype=np.float32)
    
    # Add boundaries around objects
    from skimage.segmentation import find_boundaries
    fg_binary = foreground > 0.5
    boundary = find_boundaries(fg_binary, mode='thick').astype(np.float32) * 0.7
    
    # Add some noise to make it realistic
    boundary += np.random.rand(*shape) * 0.1
    
    print(f"Synthetic data created: {shape}")
    print(f"Foreground pixels: {np.sum(foreground > 0.5)}")
    print(f"Boundary pixels: {np.sum(boundary > 0.3)}")
    
    # Test different methods
    results = compare_instance_methods(foreground, boundary, voxel_size_um=1.0)
    
    # Show best method in detail
    best_method = max(results.keys(), key=lambda k: results[k]['num_instances'] if results[k]['success'] else 0)
    print(f"\n🏆 Best method appears to be: {best_method}")
    
    # Process with best method
    prediction_stack = np.stack([foreground, boundary])
    instances, analysis = process_two_channel_prediction(
        prediction_stack, 
        voxel_size_um=1.0, 
        method=best_method
    )
    
    print(f"\n📊 Final Analysis:")
    print(f"  Individual plankton found: {analysis['num_instances']}")
    print(f"  Total volume: {analysis['total_volume_um3']:.1f} μm³")
    print(f"  Size distribution: {analysis['size_distribution']}")
    
    # Show integration example
    update_inference_for_instances()
    
    print("\n✅ Instance segmentation demo completed!")
