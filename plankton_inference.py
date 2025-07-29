#!/usr/bin/env python3
"""
Complete inference pipeline for plankton 3D segmentation.

Features:
- Load trained models with proper configuration
- Process large 3D volumes with sliding window approach
- Apply Test-Time Augmentation for better accuracy
- Advanced post-processing for clean results
- Comprehensive evaluation and visualization
- Export results in multiple formats
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn.functional as F
import numpy as np
import tifffile
from tqdm import tqdm
import json

# Import our plankton modules
from plankton_config import PlanktonProjectConfig, create_plankton_inference_config
from plankton_postprocessing import (
    PlanktonPostProcessor, 
    PlanktonTestTimeAugmentationPostProcessor,
    quick_clean_plankton_prediction,
    analyze_plankton_segmentation
)
from plankton_metrics import PlanktonMetrics
from plankton_augmentations import PlanktonTestTimeAugmentation
import tools as tf

class PlanktonInferenceEngine:
    """
    Complete inference engine for plankton segmentation.
    
    Handles everything from loading models to generating final results.
    """
    
    def __init__(
        self, 
        model_path: str,
        config: Optional[PlanktonProjectConfig] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to trained model file
            config: Project configuration (will create default if None)
            device: Device for computation (will auto-detect if None)
        """
        self.model_path = Path(model_path)
        
        # Setup device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("Using GPU for inference")
            else:
                self.device = torch.device("cpu")
                print("Using CPU for inference")
        else:
            self.device = device
        
        # Setup configuration
        if config is None:
            self.config = create_plankton_inference_config(
                str(self.model_path), 
                str(self.model_path.parent / "inference_results")
            )
        else:
            self.config = config
        
        # Initialize components
        self.model = None
        self.post_processor = None
        self.tta_processor = None
        self.metrics_calculator = None
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Load model
        self.load_model()
        
        # Initialize processors
        self.setup_processors()
    
    def _setup_logging(self):
        """Setup logging"""
        log_file = self.config.paths['output'] / 'inference.log'
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger(__name__)
    
    def load_model(self):
        """Load trained model"""
        try:
            from torch_em.model import UNet3d
            
            # Create model architecture
            self.model = UNet3d(
                in_channels=self.config.model.in_channels,
                out_channels=self.config.model.out_channels,
                initial_features=self.config.model.initial_features,
                final_activation=self.config.model.final_activation
            )
            
            # Load weights
            if self.model_path.suffix == '.pth':
                # Check if it's a full checkpoint or just state dict
                checkpoint = torch.load(self.model_path, map_location='cpu')
                
                if 'model_state_dict' in checkpoint:
                    # Full checkpoint
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.logger.info(f"Loaded model from checkpoint: {self.model_path}")
                    
                    # Try to extract config from checkpoint
                    if 'config' in checkpoint:
                        self.logger.info("Found config in checkpoint")
                        
                else:
                    # Just state dict
                    self.model.load_state_dict(checkpoint)
                    self.logger.info(f"Loaded model state dict: {self.model_path}")
            else:
                raise ValueError(f"Unsupported model file format: {self.model_path.suffix}")
            
            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            # Count parameters
            num_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"Model loaded with {num_params:,} parameters")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def setup_processors(self):
        """Initialize post-processing and evaluation components"""
        # Post-processor
        self.post_processor = PlanktonPostProcessor(
            voxel_size_um=self.config.data.voxel_size_um,
            min_plankton_size_um=self.config.data.min_plankton_size_um,
            max_plankton_size_um=self.config.data.max_plankton_size_um
        )
        
        # TTA + Post-processing combo
        self.tta_processor = PlanktonTestTimeAugmentationPostProcessor(
            model=self.model,
            post_processor=self.post_processor,
            device=self.device
        )
        
        # Metrics calculator
        self.metrics_calculator = PlanktonMetrics(
            voxel_size_um=self.config.data.voxel_size_um,
            min_object_size_um=self.config.data.min_plankton_size_um
        )
        
        self.logger.info("Processors initialized")
    
    def predict_volume(
        self, 
        volume: np.ndarray,
        use_tta: bool = True,
        tta_iterations: int = 8,
        threshold: float = 0.5,
        use_sliding_window: bool = True,
        overlap_ratio: float = 0.25,
        create_instances: bool = True,
        instance_method: str = 'boundary_watershed'
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Predict segmentation for a full 3D volume.
        
        Args:
            volume: Input 3D volume [D, H, W]
            use_tta: Whether to use Test-Time Augmentation
            tta_iterations: Number of TTA iterations
            threshold: Threshold for final binarization
            use_sliding_window: Whether to use sliding window for large volumes
            overlap_ratio: Overlap ratio for sliding window
            create_instances: Whether to create instance segmentation from foreground+boundary
            instance_method: Method for instance creation ('boundary_watershed', 'combined_approach', etc.)
            
        Returns:
            Final segmentation (instances if create_instances=True, otherwise foreground) and metadata
        """
        self.logger.info(f"Predicting volume with shape: {volume.shape}")
        start_time = time.time()
        
        # Normalize volume
        volume_normalized = tf.normalise(volume.astype(np.float32))
        
        # Decide on prediction strategy based on volume size
        patch_shape = self.config.data.patch_shape
        
        if (use_sliding_window and 
            any(v > p * 1.5 for v, p in zip(volume.shape, patch_shape))):
            # Large volume - use sliding window approach
            self.logger.info("Using sliding window approach for large volume")
            prediction = self._predict_large_volume_sliding_window(
                volume_normalized, overlap_ratio
            )
        else:
            # Small volume - predict directly
            self.logger.info("Predicting entire volume at once")
            prediction = self._predict_single_volume(volume_normalized, use_tta, tta_iterations)
        
        # Post-process prediction
        self.logger.info("Processing prediction channels...")
        
        # Handle 2-channel output (foreground + boundary)
        if prediction.ndim == 4 and prediction.shape[0] == 2:
            foreground_pred = prediction[0]
            boundary_pred = prediction[1]
            
            self.logger.info(f"Two-channel prediction detected:")
            self.logger.info(f"  Foreground range: {foreground_pred.min():.3f} - {foreground_pred.max():.3f}")
            self.logger.info(f"  Boundary range: {boundary_pred.min():.3f} - {boundary_pred.max():.3f}")
            
            if create_instances:
                # Create instance segmentation using both channels
                from plankton_instance_segmentation import process_two_channel_prediction
                
                self.logger.info(f"Creating instances using {instance_method} method...")
                prediction_stack = np.stack([foreground_pred, boundary_pred])
                
                final_segmentation, instance_info = process_two_channel_prediction(
                    prediction_stack,
                    voxel_size_um=self.config.data.voxel_size_um,
                    method=instance_method
                )
                
                # Post-process instances if needed
                if hasattr(self.post_processor, 'process_instances'):
                    final_segmentation = self.post_processor.process_instances(final_segmentation)
                
                # Add instance info to metadata
                processing_info = instance_info
                processing_type = "instance_segmentation"
                
            else:
                # Use only foreground channel for semantic segmentation
                final_segmentation = quick_clean_plankton_prediction(
                    foreground_pred,
                    voxel_size_um=self.config.data.voxel_size_um,
                    threshold=threshold
                )
                processing_info = {"note": "Used foreground channel only"}
                processing_type = "semantic_segmentation"
        
        else:
            # Single channel or unexpected format - treat as foreground only
            self.logger.warning(f"Unexpected prediction shape: {prediction.shape}, treating as single channel")
            final_segmentation = quick_clean_plankton_prediction(
                prediction,
                voxel_size_um=self.config.data.voxel_size_um,
                threshold=threshold
            )
            processing_info = {"note": "Single channel processing"}
            processing_type = "semantic_segmentation"
        
        # Analyze results based on segmentation type
        if processing_type == "instance_segmentation":
            # Instance analysis already included in processing_info
            analysis = processing_info
        else:
            # Standard semantic segmentation analysis
            analysis = analyze_plankton_segmentation(
                final_segmentation,
                voxel_size_um=self.config.data.voxel_size_um
            )
        
        prediction_time = time.time() - start_time
        
        metadata = {
            'prediction_time_seconds': prediction_time,
            'input_shape': volume.shape,
            'output_shape': final_segmentation.shape,
            'use_tta': use_tta,
            'tta_iterations': tta_iterations if use_tta else 0,
            'threshold': threshold,
            'processing_type': processing_type,
            'processing_info': processing_info,
            'object_analysis': analysis,
            'voxel_size_um': self.config.data.voxel_size_um,
            'instance_method': instance_method if create_instances else None,
            'channels_used': 'foreground+boundary' if create_instances and prediction.ndim == 4 and prediction.shape[0] == 2 else 'foreground_only'
        }
        
        self.logger.info(f"Prediction completed in {prediction_time:.1f}s")
        self.logger.info(f"Found {analysis['num_objects']} plankton objects")
        
        return final_segmentation, metadata
    
    def _predict_single_volume(
        self, 
        volume: np.ndarray, 
        use_tta: bool = True, 
        tta_iterations: int = 8
    ) -> np.ndarray:
        """Predict a single volume that fits in memory"""
        
        # Prepare input tensor
        volume_tensor = torch.tensor(volume, dtype=torch.float32)
        if volume_tensor.dim() == 3:
            volume_tensor = volume_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        
        with torch.no_grad():
            if use_tta:
                # Use TTA for better accuracy
                tta_engine = PlanktonTestTimeAugmentation(
                    voxel_size_um=self.config.data.voxel_size_um
                )
                prediction = tta_engine.predict_with_tta(
                    self.model, volume_tensor, self.device, tta_iterations
                )
            else:
                # Simple prediction
                volume_tensor = volume_tensor.to(self.device)
                prediction = torch.sigmoid(self.model(volume_tensor))
                prediction = prediction.cpu()
        
        # Convert to numpy and remove batch/channel dimensions
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.numpy()
        
        if prediction.ndim == 5:  # [B, C, D, H, W]
            prediction = prediction[0, 0]
        elif prediction.ndim == 4:  # [C, D, H, W]
            prediction = prediction[0]
        
        return prediction
    
    def _predict_large_volume_sliding_window(
        self, 
        volume: np.ndarray, 
        overlap_ratio: float = 0.25
    ) -> np.ndarray:
        """Predict large volume using sliding window approach"""
        
        # Use torch_em's prediction utility for sliding window
        try:
            from torch_em.util.prediction import predict_with_halo
            
            prediction = predict_with_halo(
                volume,
                self.model,
                gpu_ids=[self.device.index] if self.device.type == 'cuda' else ['cpu'],
                block_shape=self.config.data.block_shape,
                halo=self.config.data.halo,
                preprocess=None,
                postprocess=None
            )
            
            # Handle multi-channel output
            if prediction.ndim == 4:  # [C, D, H, W]
                if self.config.model.out_channels == 2:
                    # Use foreground channel
                    prediction = prediction[0]
                else:
                    prediction = prediction[0]
            
            return prediction
            
        except ImportError:
            self.logger.warning("torch_em not available, using simple patching")
            return self._predict_with_simple_patching(volume, overlap_ratio)
    
    def _predict_with_simple_patching(
        self, 
        volume: np.ndarray, 
        overlap_ratio: float = 0.25
    ) -> np.ndarray:
        """Simple patching implementation as fallback"""
        
        patch_shape = self.config.data.patch_shape
        stride = [int(p * (1 - overlap_ratio)) for p in patch_shape]
        
        # Initialize output
        prediction = np.zeros(volume.shape, dtype=np.float32)
        weight_map = np.zeros(volume.shape, dtype=np.float32)
        
        # Generate patch positions
        positions = []
        for z in range(0, volume.shape[0] - patch_shape[0] + 1, stride[0]):
            for y in range(0, volume.shape[1] - patch_shape[1] + 1, stride[1]):
                for x in range(0, volume.shape[2] - patch_shape[2] + 1, stride[2]):
                    positions.append((z, y, x))
        
        # Add boundary patches
        # ... (implementation would handle edge cases)
        
        self.logger.info(f"Processing {len(positions)} patches...")
        
        with torch.no_grad():
            for z, y, x in tqdm(positions, desc="Processing patches"):
                # Extract patch
                z_end = min(z + patch_shape[0], volume.shape[0])
                y_end = min(y + patch_shape[1], volume.shape[1])
                x_end = min(x + patch_shape[2], volume.shape[2])
                
                patch = volume[z:z_end, y:y_end, x:x_end]
                
                # Pad if necessary
                if patch.shape != tuple(patch_shape):
                    pad_width = [(0, patch_shape[i] - patch.shape[i]) for i in range(3)]
                    patch = np.pad(patch, pad_width, mode='constant', constant_values=0)
                
                # Predict
                patch_tensor = torch.tensor(patch, dtype=torch.float32)
                patch_tensor = patch_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
                
                patch_pred = torch.sigmoid(self.model(patch_tensor))
                patch_pred = patch_pred.cpu().numpy()[0, 0]
                
                # Remove padding if added
                if patch_pred.shape != (z_end - z, y_end - y, x_end - x):
                    patch_pred = patch_pred[:z_end - z, :y_end - y, :x_end - x]
                
                # Accumulate prediction
                prediction[z:z_end, y:y_end, x:x_end] += patch_pred
                weight_map[z:z_end, y:y_end, x:x_end] += 1
        
        # Normalize by weights
        prediction = np.divide(prediction, weight_map, out=np.zeros_like(prediction), where=weight_map != 0)
        
        return prediction
    
    def evaluate_prediction(
        self, 
        prediction: np.ndarray, 
        ground_truth: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate prediction against ground truth.
        
        Args:
            prediction: Binary prediction
            ground_truth: Binary ground truth
            
        Returns:
            Comprehensive evaluation metrics
        """
        metrics = self.metrics_calculator.compute_all_metrics(
            torch.tensor(prediction), 
            torch.tensor(ground_truth)
        )
        
        return metrics
    
    def save_results(
        self, 
        segmentation: np.ndarray,
        metadata: Dict[str, Any],
        output_prefix: str = "plankton_segmentation",
        save_formats: List[str] = ['tiff', 'npy', 'json']
    ):
        """
        Save segmentation results in multiple formats.
        
        Args:
            segmentation: Binary segmentation result
            metadata: Prediction metadata
            output_prefix: Prefix for output files
            save_formats: List of formats to save ('tiff', 'npy', 'json', 'csv')
        """
        output_dir = self.config.paths['output']
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save segmentation
        if 'tiff' in save_formats:
            tiff_path = output_dir / f"{output_prefix}.tif"
            tifffile.imwrite(str(tiff_path), segmentation.astype(np.uint8))
            self.logger.info(f"Saved TIFF: {tiff_path}")
        
        if 'npy' in save_formats:
            npy_path = output_dir / f"{output_prefix}.npy"
            np.save(str(npy_path), segmentation)
            self.logger.info(f"Saved NPY: {npy_path}")
        
        # Save metadata
        if 'json' in save_formats:
            json_path = output_dir / f"{output_prefix}_metadata.json"
            
            # Convert numpy types to native Python types for JSON
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            serializable_metadata = convert_numpy_types(metadata)
            
            with open(json_path, 'w') as f:
                json.dump(serializable_metadata, f, indent=2)
            self.logger.info(f"Saved metadata: {json_path}")
        
        # Save object analysis as CSV
        if 'csv' in save_formats and 'object_analysis' in metadata:
            import pandas as pd
            
            analysis = metadata['object_analysis']
            if 'size_distribution' in analysis:
                csv_path = output_dir / f"{output_prefix}_size_distribution.csv"
                size_dist_df = pd.DataFrame([analysis['size_distribution']])
                size_dist_df.to_csv(csv_path, index=False)
                self.logger.info(f"Saved size distribution: {csv_path}")

def main():
    """Command line interface for plankton inference"""
    parser = argparse.ArgumentParser(description="Plankton 3D segmentation inference")
    parser.add_argument('model_path', type=str, help='Path to trained model')
    parser.add_argument('input_path', type=str, help='Path to input volume (TIFF stack or single file)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--voxel_size', type=float, default=1.0, help='Voxel size in micrometers')
    parser.add_argument('--threshold', type=float, default=0.5, help='Segmentation threshold')
    parser.add_argument('--use_tta', action='store_true', help='Use Test-Time Augmentation')
    parser.add_argument('--tta_iterations', type=int, default=8, help='Number of TTA iterations')
    parser.add_argument('--no_instances', action='store_true', help='Skip instance creation, output semantic segmentation only')
    parser.add_argument('--instance_method', type=str, default='boundary_watershed',
                        choices=['boundary_watershed', 'boundary_subtraction', 'distance_watershed', 'combined_approach'],
                        help='Method for creating instances from foreground+boundary')
    parser.add_argument('--save_channels', action='store_true', help='Save individual foreground and boundary channels')
    parser.add_argument('--save_formats', nargs='+', default=['tiff', 'json'], 
                        choices=['tiff', 'npy', 'json', 'csv'], help='Output formats')
    
    args = parser.parse_args()
    
    # Setup paths
    input_path = Path(args.input_path)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent / 'inference_results'
    
    # Create inference config
    config = create_plankton_inference_config(
        args.model_path, 
        str(output_dir),
        args.voxel_size
    )
    
    # Create inference engine
    engine = PlanktonInferenceEngine(args.model_path, config)
    
    try:
        # Load input volume
        print(f"Loading input volume: {input_path}")
        if input_path.is_dir():
            # Load TIFF stack
            volume = tf.load_tif_stack(str(input_path), bit=32)
            if volume.ndim == 4:  # Remove batch dimension if present
                volume = volume[0]
        else:
            # Load single file
            volume = tifffile.imread(str(input_path))
        
        print(f"Input volume shape: {volume.shape}")
        
        # Run prediction
        segmentation, metadata = engine.predict_volume(
            volume,
            use_tta=args.use_tta,
            tta_iterations=args.tta_iterations,
            threshold=args.threshold,
            use_sliding_window=True,
            create_instances=not args.no_instances,
            instance_method=args.instance_method
        )
        
        # Save results
        output_prefix = input_path.stem
        engine.save_results(
            segmentation, 
            metadata, 
            output_prefix, 
            args.save_formats
        )
        
        print(f"\nInference completed successfully!")
        
        if metadata['processing_type'] == 'instance_segmentation':
            print(f"Found {metadata['object_analysis']['num_instances']} individual plankton")
            print(f"Total volume: {metadata['object_analysis']['total_volume_um3']:.1f} μm³")
            print(f"Size distribution: {metadata['object_analysis']['size_distribution']}")
            print(f"Instance method used: {metadata['instance_method']}")
        else:
            print(f"Found {metadata['object_analysis']['num_objects']} plankton objects")
            print(f"Total volume: {metadata['object_analysis']['total_volume_um3']:.1f} μm³")
        
        print(f"Processing type: {metadata['processing_type']}")
        print(f"Channels used: {metadata['channels_used']}")
        print(f"Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Inference failed: {e}")
        raise

if __name__ == "__main__":
    main()
