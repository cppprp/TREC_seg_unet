import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available. Some interactive features will be disabled.")

from skimage.measure import label, regionprops
from plankton_metrics import PlanktonMetrics

class PlanktonVisualizer:
    """
    Comprehensive visualization tools for plankton segmentation analysis.
    
    Provides both static (matplotlib) and interactive (plotly) visualizations
    for understanding model performance, data characteristics, and results.
    """
    
    def __init__(self, voxel_size_um: float = 1.0, save_dir: Optional[str] = None):
        """
        Initialize visualizer.
        
        Args:
            voxel_size_um: Voxel size in micrometers
            save_dir: Directory to save plots (optional)
        """
        self.voxel_size_um = voxel_size_um
        self.save_dir = Path(save_dir) if save_dir else None
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def visualize_3d_slice_comparison(
        self, 
        image: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
        prediction: Optional[np.ndarray] = None,
        slice_idx: Optional[int] = None,
        title: str = "Plankton Segmentation Comparison"
    ):
        """
        Visualize 2D slices from 3D volumes for comparison.
        
        Args:
            image: Original 3D image [D, H, W]
            ground_truth: Ground truth segmentation [D, H, W] (optional)
            prediction: Model prediction [D, H, W] (optional)
            slice_idx: Which slice to show (middle slice if None)
            title: Plot title
        """
        if slice_idx is None:
            slice_idx = image.shape[0] // 2
        
        # Count number of plots needed
        num_plots = 1  # Always have image
        if ground_truth is not None:
            num_plots += 1
        if prediction is not None:
            num_plots += 1
        
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
        if num_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Original image
        axes[plot_idx].imshow(image[slice_idx], cmap='gray')
        axes[plot_idx].set_title(f'Original Image (slice {slice_idx})')
        axes[plot_idx].axis('off')
        plot_idx += 1
        
        # Ground truth
        if ground_truth is not None:
            axes[plot_idx].imshow(image[slice_idx], cmap='gray', alpha=0.7)
            axes[plot_idx].imshow(ground_truth[slice_idx], cmap='Reds', alpha=0.5)
            axes[plot_idx].set_title('Ground Truth Overlay')
            axes[plot_idx].axis('off')
            plot_idx += 1
        
        # Prediction
        if prediction is not None:
            axes[plot_idx].imshow(image[slice_idx], cmap='gray', alpha=0.7)
            
            if ground_truth is not None:
                # Show comparison: Green = correct, Red = false positive, Blue = false negative
                gt_slice = ground_truth[slice_idx] > 0.5
                pred_slice = prediction[slice_idx] > 0.5
                
                # Create RGB overlay
                overlay = np.zeros((*gt_slice.shape, 3))
                overlay[gt_slice & pred_slice] = [0, 1, 0]      # Green: True positive
                overlay[~gt_slice & pred_slice] = [1, 0, 0]     # Red: False positive
                overlay[gt_slice & ~pred_slice] = [0, 0, 1]     # Blue: False negative
                
                axes[plot_idx].imshow(overlay, alpha=0.6)
                axes[plot_idx].set_title('Prediction vs GT\n(Green=TP, Red=FP, Blue=FN)')
            else:
                axes[plot_idx].imshow(prediction[slice_idx], cmap='Blues', alpha=0.5)
                axes[plot_idx].set_title('Prediction Overlay')
            
            axes[plot_idx].axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if self.save_dir:
            plt.savefig(self.save_dir / f'slice_comparison_{slice_idx}.png', 
                       dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_size_distribution(
        self, 
        segmentation: np.ndarray,
        title: str = "Plankton Size Distribution",
        bins: int = 30
    ):
        """
        Plot size distribution of detected plankton.
        
        Args:
            segmentation: Binary segmentation [D, H, W]
            title: Plot title
            bins: Number of histogram bins
        """
        # Extract object sizes
        labeled_img = label(segmentation)
        props = regionprops(labeled_img)
        
        if len(props) == 0:
            print("No objects found in segmentation")
            return
        
        sizes_voxels = [prop.area for prop in props]
        sizes_um3 = [size * (self.voxel_size_um ** 3) for size in sizes_voxels]
        
        # Convert to equivalent sphere diameter
        equivalent_diameters = [2 * (3 * size / (4 * np.pi)) ** (1/3) for size in sizes_um3]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Volume distribution
        ax1.hist(sizes_um3, bins=bins, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Volume (μm³)')
        ax1.set_ylabel('Count')
        ax1.set_title('Volume Distribution')
        ax1.axvline(np.mean(sizes_um3), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(sizes_um3):.1f} μm³')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Equivalent diameter distribution
        ax2.hist(equivalent_diameters, bins=bins, alpha=0.7, edgecolor='black', color='orange')
        ax2.set_xlabel('Equivalent Diameter (μm)')
        ax2.set_ylabel('Count')
        ax2.set_title('Size Distribution (Equivalent Spheres)')
        ax2.axvline(np.mean(equivalent_diameters), color='red', linestyle='--',
                   label=f'Mean: {np.mean(equivalent_diameters):.1f} μm')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add size class annotations
        size_classes = {
            'Nano': (0.2, 2),
            'Micro': (2, 20), 
            'Small Meso': (20, 50),
            'Large Meso': (50, 200),
            'Macro': (200, 2000)
        }
        
        y_max = ax2.get_ylim()[1]
        colors = ['purple', 'blue', 'green', 'orange', 'red']
        
        for i, (class_name, (min_size, max_size)) in enumerate(size_classes.items()):
            ax2.axvspan(min_size, max_size, alpha=0.1, color=colors[i], label=class_name)
        
        ax2.legend(loc='upper right', fontsize=8)
        
        plt.suptitle(f'{title}\n{len(props)} objects detected', fontsize=14)
        plt.tight_layout()
        
        if self.save_dir:
            plt.savefig(self.save_dir / 'size_distribution.png', dpi=150, bbox_inches='tight')
        
        plt.show()
        
        # Print statistics
        print(f"\n📊 Size Distribution Statistics:")
        print(f"  Total objects: {len(props)}")
        print(f"  Volume range: {min(sizes_um3):.1f} - {max(sizes_um3):.1f} μm³")
        print(f"  Mean volume: {np.mean(sizes_um3):.1f} ± {np.std(sizes_um3):.1f} μm³")
        print(f"  Diameter range: {min(equivalent_diameters):.1f} - {max(equivalent_diameters):.1f} μm")
        print(f"  Mean diameter: {np.mean(equivalent_diameters):.1f} ± {np.std(equivalent_diameters):.1f} μm")
    
    def plot_training_metrics(
        self, 
        training_history: Dict[str, List],
        title: str = "Training Progress"
    ):
        """
        Plot comprehensive training metrics.
        
        Args:
            training_history: Dictionary with training history
            title: Plot title
        """
        epochs = training_history.get('epochs', range(len(training_history['train_loss'])))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(epochs, training_history['train_loss'], label='Train Loss', color='blue')
        if 'val_loss' in training_history:
            axes[0, 0].plot(epochs, training_history['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Learning rate
        if 'learning_rates' in training_history:
            axes[0, 1].plot(epochs, training_history['learning_rates'], color='green')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].set_title('Learning Rate Schedule')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Dice scores (if available)
        if 'train_metrics' in training_history and training_history['train_metrics']:
            train_dice = [m.get('dice', 0) for m in training_history['train_metrics']]
            axes[1, 0].plot(epochs, train_dice, label='Train Dice', color='blue')
            
            if 'val_metrics' in training_history and training_history['val_metrics']:
                val_dice = [m.get('dice_overall', m.get('dice', 0)) for m in training_history['val_metrics']]
                axes[1, 0].plot(epochs, val_dice, label='Validation Dice', color='red')
            
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Dice Score')
            axes[1, 0].set_title('Dice Score Progress')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Small plankton performance (key metric!)
        if 'val_metrics' in training_history and training_history['val_metrics']:
            small_plankton_dice = [m.get('dice_small_meso', 0) for m in training_history['val_metrics']]
            if any(small_plankton_dice):
                axes[1, 1].plot(epochs, small_plankton_dice, 
                               color='purple', linewidth=2, label='Small Plankton Dice')
                axes[1, 1].axhline(y=0.7, color='red', linestyle='--', 
                                  label='Minimum Acceptable (0.7)')
                axes[1, 1].axhline(y=0.85, color='green', linestyle='--', 
                                  label='Excellent (0.85)')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Small Plankton Dice')
                axes[1, 1].set_title('Small Plankton Detection (Critical!)')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].set_ylim(0, 1)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if self.save_dir:
            plt.savefig(self.save_dir / 'training_progress.png', dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_plankton_metrics_radar(
        self, 
        metrics: Dict[str, float],
        title: str = "Plankton Segmentation Performance"
    ):
        """
        Create radar plot of key plankton metrics.
        
        Args:
            metrics: Dictionary of computed metrics
            title: Plot title
        """
        # Select key metrics for radar plot
        key_metrics = {
            'Overall Dice': metrics.get('dice_overall', 0),
            'Small Plankton Dice': metrics.get('dice_small_meso', 0),
            'Count Accuracy': metrics.get('count_accuracy', 0),
            'Volume Accuracy': 1 - metrics.get('volume_error', 1),
            'Detection Recall': metrics.get('count_recall', 0),
            'Detection Precision': metrics.get('count_precision', 0),
        }
        
        # Prepare data for radar plot
        categories = list(key_metrics.keys())
        values = list(key_metrics.values())
        
        # Number of categories
        N = len(categories)
        
        # Angles for each category
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Add first value to end to complete circle
        values += values[:1]
        
        # Create radar plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, label='Performance', color='blue')
        ax.fill(angles, values, alpha=0.25, color='blue')
        
        # Add category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        # Set y-axis limits
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        
        # Add reference lines
        ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Minimum Acceptable')
        ax.axhline(y=0.85, color='green', linestyle='--', alpha=0.5, label='Excellent')
        
        # Add grid
        ax.grid(True)
        
        plt.title(title, pad=20, fontsize=14)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        if self.save_dir:
            plt.savefig(self.save_dir / 'performance_radar.png', dpi=150, bbox_inches='tight')
        
        plt.show()
        
        # Print interpretation
        print(f"\n🎯 Performance Analysis:")
        critical_low = [k for k, v in key_metrics.items() if v < 0.7]
        if critical_low:
            print(f"  🔴 Critical issues: {', '.join(critical_low)}")
            print("  📊 These metrics need immediate attention!")
        
        excellent = [k for k, v in key_metrics.items() if v > 0.85]
        if excellent:
            print(f"  🟢 Excellent performance: {', '.join(excellent)}")
    
    def create_interactive_dashboard(
        self, 
        metrics: Dict[str, float],
        training_history: Optional[Dict[str, List]] = None,
        segmentation: Optional[np.ndarray] = None
    ):
        """
        Create interactive dashboard with plotly (if available).
        
        Args:
            metrics: Performance metrics
            training_history: Training history (optional)
            segmentation: Segmentation result for size analysis (optional)
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Install with: pip install plotly")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Performance Metrics', 'Training Loss', 
                          'Size Distribution', 'Key Metrics Over Time'),
            specs=[[{"type": "bar"}, {"type": "xy"}],
                   [{"type": "histogram"}, {"type": "xy"}]]
        )
        
        # 1. Performance metrics bar chart
        key_metrics = {
            'Overall Dice': metrics.get('dice_overall', 0),
            'Small Plankton': metrics.get('dice_small_meso', 0),
            'Count Accuracy': metrics.get('count_accuracy', 0),
            'Volume Accuracy': 1 - metrics.get('volume_error', 1),
        }
        
        colors = ['green' if v > 0.85 else 'yellow' if v > 0.7 else 'red' 
                 for v in key_metrics.values()]
        
        fig.add_trace(
            go.Bar(x=list(key_metrics.keys()), y=list(key_metrics.values()),
                   marker_color=colors, name='Performance'),
            row=1, col=1
        )
        
        # 2. Training loss (if available)
        if training_history and 'train_loss' in training_history:
            epochs = list(range(len(training_history['train_loss'])))
            
            fig.add_trace(
                go.Scatter(x=epochs, y=training_history['train_loss'],
                          mode='lines', name='Train Loss', line=dict(color='blue')),
                row=1, col=2
            )
            
            if 'val_loss' in training_history:
                fig.add_trace(
                    go.Scatter(x=epochs, y=training_history['val_loss'],
                              mode='lines', name='Val Loss', line=dict(color='red')),
                    row=1, col=2
                )
        
        # 3. Size distribution (if segmentation provided)
        if segmentation is not None:
            labeled_img = label(segmentation)
            props = regionprops(labeled_img)
            
            if len(props) > 0:
                sizes_um3 = [prop.area * (self.voxel_size_um ** 3) for prop in props]
                equivalent_diameters = [2 * (3 * size / (4 * np.pi)) ** (1/3) for size in sizes_um3]
                
                fig.add_trace(
                    go.Histogram(x=equivalent_diameters, nbinsx=20, 
                               name='Size Distribution', marker_color='lightblue'),
                    row=2, col=1
                )
        
        # 4. Key metrics over time (if training history available)
        if training_history and 'val_metrics' in training_history:
            epochs = list(range(len(training_history['val_metrics'])))
            
            # Small plankton dice over time
            small_dice = [m.get('dice_small_meso', 0) for m in training_history['val_metrics']]
            if any(small_dice):
                fig.add_trace(
                    go.Scatter(x=epochs, y=small_dice,
                              mode='lines+markers', name='Small Plankton Dice',
                              line=dict(color='purple', width=3)),
                    row=2, col=2
                )
                
                # Add reference lines
                fig.add_hline(y=0.7, line_dash="dash", line_color="red",
                             annotation_text="Min Acceptable", row=2, col=2)
                fig.add_hline(y=0.85, line_dash="dash", line_color="green",
                             annotation_text="Excellent", row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title_text="Plankton Segmentation Dashboard",
            showlegend=True,
            height=800
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Metric", row=1, col=1)
        fig.update_yaxes(title_text="Score", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Loss", row=1, col=2)
        fig.update_xaxes(title_text="Equivalent Diameter (μm)", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_xaxes(title_text="Epoch", row=2, col=2)
        fig.update_yaxes(title_text="Dice Score", row=2, col=2)
        
        # Show dashboard
        fig.show()
        
        # Save if requested
        if self.save_dir:
            fig.write_html(str(self.save_dir / 'interactive_dashboard.html'))
            print(f"Interactive dashboard saved to: {self.save_dir / 'interactive_dashboard.html'}")

def visualize_model_results(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    voxel_size_um: float = 1.0,
    save_dir: Optional[str] = None
):
    """
    Convenience function to visualize complete model results.
    
    Args:
        image: Original 3D image
        ground_truth: Ground truth segmentation
        prediction: Model prediction
        voxel_size_um: Voxel size in micrometers
        save_dir: Directory to save plots
    """
    visualizer = PlanktonVisualizer(voxel_size_um, save_dir)
    
    # Slice comparison
    visualizer.visualize_3d_slice_comparison(
        image, ground_truth, prediction,
        title="Model Prediction vs Ground Truth"
    )
    
    # Size distributions
    print("Ground Truth Size Distribution:")
    visualizer.plot_size_distribution(ground_truth, "Ground Truth Size Distribution")
    
    print("\nPrediction Size Distribution:")
    visualizer.plot_size_distribution(prediction, "Prediction Size Distribution")
    
    # Performance metrics
    from plankton_metrics import PlanktonMetrics
    metrics_calc = PlanktonMetrics(voxel_size_um=voxel_size_um)
    metrics = metrics_calc.compute_all_metrics(
        torch.tensor(prediction), torch.tensor(ground_truth)
    )
    
    visualizer.plot_plankton_metrics_radar(metrics)

def visualize_training_results(
    training_history_path: str,
    save_dir: Optional[str] = None
):
    """
    Visualize training results from saved history.
    
    Args:
        training_history_path: Path to training history JSON file
        save_dir: Directory to save plots
    """
    with open(training_history_path, 'r') as f:
        training_history = json.load(f)
    
    visualizer = PlanktonVisualizer(save_dir=save_dir)
    visualizer.plot_training_metrics(training_history)

# Example usage
if __name__ == "__main__":
    print("🎨 Plankton Visualization Tools Demo")
    
    # Create synthetic data for demonstration
    volume_shape = (64, 64, 64)
    voxel_size_um = 1.0
    
    # Create synthetic image
    image = np.random.rand(*volume_shape) * 0.3
    
    # Create synthetic ground truth with different sized plankton
    ground_truth = np.zeros(volume_shape, dtype=np.uint8)
    
    # Large plankton
    ground_truth[10:25, 10:25, 10:25] = 1
    ground_truth[40:50, 40:50, 40:50] = 1
    
    # Small plankton  
    ground_truth[15:20, 45:50, 15:20] = 1
    ground_truth[50:54, 15:19, 50:54] = 1
    
    # Add plankton signal to image
    image[ground_truth > 0] = 0.8
    
    # Create synthetic prediction (with some errors)
    prediction = ground_truth.copy()
    prediction[50:54, 15:19, 50:54] = 0  # Miss one small plankton
    prediction[5:8, 55:58, 5:8] = 1      # False positive
    
    # Demonstrate visualization
    print("Demonstrating visualization tools...")
    
    visualizer = PlanktonVisualizer(voxel_size_um=voxel_size_um)
    
    # Slice comparison
    visualizer.visualize_3d_slice_comparison(
        image, ground_truth, prediction,
        title="Demo: Plankton Segmentation Results"
    )
    
    # Size distribution
    visualizer.plot_size_distribution(
        ground_truth, 
        title="Demo: Ground Truth Size Distribution"
    )
    
    # Performance radar
    from plankton_metrics import PlanktonMetrics
    metrics_calc = PlanktonMetrics(voxel_size_um=voxel_size_um)
    metrics = metrics_calc.compute_all_metrics(
        torch.tensor(prediction).float(), 
        torch.tensor(ground_truth).float()
    )
    
    visualizer.plot_plankton_metrics_radar(metrics, "Demo: Performance Analysis")
    
    print("✅ Visualization demo completed!")
