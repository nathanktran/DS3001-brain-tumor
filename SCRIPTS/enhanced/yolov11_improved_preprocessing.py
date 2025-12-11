"""
Enhanced Preprocessing Pipeline for Brain Tumor Detection
Based on Priyadharshini et al. (2025) - Scientific Reports
Implements log transformation, histogram equalization, and edge-based ROI extraction
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedMRIPreprocessor:
    """
    Advanced MRI preprocessing following research best practices
    Reference: "A successive framework for brain tumor interpretation using Yolo variants"
    """
    
    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        self.target_size = target_size
        self.preprocessing_steps = []
        
    def log_transformation(self, image: np.ndarray, c: float = 1.0) -> np.ndarray:
        """
        Apply log transformation for intensity normalization
        Enhances low-intensity tumor regions by compressing high-intensity values
        
        Args:
            image: Input MRI image
            c: Scaling constant (default=1.0)
        
        Returns:
            Log-transformed image
        """
        # Ensure image is float type
        image_float = image.astype(np.float32)
        
        # Add small constant to avoid log(0)
        epsilon = 1e-7
        
        # Apply log transformation: s = c * log(1 + r)
        log_transformed = c * np.log1p(image_float + epsilon)
        
        # Normalize to 0-255 range
        log_transformed = cv2.normalize(log_transformed, None, 0, 255, cv2.NORM_MINMAX)
        
        return log_transformed.astype(np.uint8)
    
    def adaptive_histogram_equalization(self, image: np.ndarray, 
                                       clip_limit: float = 2.0,
                                       tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        Enhances local contrast while preventing over-amplification
        
        Args:
            image: Input image
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of grid for histogram equalization
            
        Returns:
            Contrast-enhanced image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        # Apply CLAHE
        equalized = clahe.apply(gray)
        
        return equalized
    
    def edge_based_roi_extraction(self, image: np.ndarray,
                                  canny_low: int = 50,
                                  canny_high: int = 150,
                                  dilation_kernel_size: int = 3,
                                  dilation_iterations: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract Region of Interest using Canny edge detection
        Helps focus model attention on tumor boundaries
        
        Args:
            image: Input image
            canny_low: Lower threshold for Canny edge detection
            canny_high: Upper threshold for Canny edge detection
            dilation_kernel_size: Size of dilation kernel
            dilation_iterations: Number of dilation iterations
            
        Returns:
            Tuple of (edge image, dilated edge mask)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect edges using Canny
        edges = cv2.Canny(blurred, canny_low, canny_high)
        
        # Create dilation kernel
        kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
        
        # Dilate edges to ensure complete tumor coverage
        dilated_edges = cv2.dilate(edges, kernel, iterations=dilation_iterations)
        
        return edges, dilated_edges
    
    def intensity_normalization(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image intensity to [0, 1] range
        Ensures consistency across different MRI scanners
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        # Convert to float
        image_float = image.astype(np.float32)
        
        # Normalize to 0-1
        normalized = (image_float - image_float.min()) / (image_float.max() - image_float.min() + 1e-7)
        
        return normalized
    
    def apply_full_pipeline(self, image: np.ndarray, 
                           return_intermediate: bool = False) -> Dict[str, np.ndarray]:
        """
        Apply complete preprocessing pipeline as described in research
        
        Pipeline steps:
        1. Resize to target size
        2. Log transformation for intensity normalization
        3. Histogram equalization for contrast enhancement
        4. Edge-based ROI extraction for boundary detection
        5. Intensity normalization
        
        Args:
            image: Input MRI image
            return_intermediate: If True, return intermediate results
            
        Returns:
            Dictionary containing processed images and intermediate steps
        """
        results = {}
        
        # Step 1: Resize
        if image.shape[:2] != self.target_size:
            resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        else:
            resized = image.copy()
        results['resized'] = resized
        
        # Step 2: Log transformation
        log_transformed = self.log_transformation(resized)
        results['log_transformed'] = log_transformed
        
        # Step 3: Histogram equalization (CLAHE)
        equalized = self.adaptive_histogram_equalization(log_transformed)
        results['histogram_equalized'] = equalized
        
        # Step 4: Edge-based ROI extraction
        edges, dilated_edges = self.edge_based_roi_extraction(equalized)
        results['edges'] = edges
        results['dilated_edges'] = dilated_edges
        
        # Step 5: Intensity normalization
        normalized = self.intensity_normalization(equalized)
        results['normalized'] = normalized
        
        # Final processed image (convert back to uint8 for YOLO)
        final_processed = (normalized * 255).astype(np.uint8)
        results['final_processed'] = final_processed
        
        # Add edge information as additional channel if needed
        # This can help YOLO focus on tumor boundaries
        results['edge_enhanced'] = cv2.addWeighted(final_processed, 0.8, dilated_edges, 0.2, 0)
        
        if not return_intermediate:
            return {'final_processed': final_processed, 
                   'edge_enhanced': results['edge_enhanced']}
        
        return results
    
    def process_dataset(self, input_dir: Path, output_dir: Path, 
                       save_intermediate: bool = False) -> int:
        """
        Process entire dataset with enhanced preprocessing
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save processed images
            save_intermediate: Save intermediate preprocessing steps
            
        Returns:
            Number of images processed
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if save_intermediate:
            for step in ['log_transformed', 'histogram_equalized', 'edges', 'final_processed']:
                (output_dir / step).mkdir(exist_ok=True)
        
        image_files = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png'))
        processed_count = 0
        
        logger.info(f"Processing {len(image_files)} images from {input_dir}")
        
        for img_path in image_files:
            try:
                # Read image
                image = cv2.imread(str(img_path))
                if image is None:
                    logger.warning(f"Failed to read {img_path}")
                    continue
                
                # Apply preprocessing
                results = self.apply_full_pipeline(image, return_intermediate=save_intermediate)
                
                # Save final processed image
                output_path = output_dir / img_path.name
                cv2.imwrite(str(output_path), results['final_processed'])
                
                # Save intermediate results if requested
                if save_intermediate:
                    for step, img_data in results.items():
                        if step != 'final_processed' and step != 'normalized':
                            step_dir = output_dir / step
                            cv2.imwrite(str(step_dir / img_path.name), img_data)
                
                processed_count += 1
                
                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count}/{len(image_files)} images")
                    
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue
        
        logger.info(f"Successfully processed {processed_count} images")
        return processed_count


def visualize_preprocessing_steps(image_path: str, save_path: Optional[str] = None):
    """
    Visualize all preprocessing steps for a single image
    
    Args:
        image_path: Path to input image
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image from {image_path}")
    
    # Create preprocessor
    preprocessor = EnhancedMRIPreprocessor()
    
    # Apply preprocessing
    results = preprocessor.apply_full_pipeline(image, return_intermediate=True)
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Enhanced MRI Preprocessing Pipeline', fontsize=16, fontweight='bold')
    
    steps = [
        ('resized', 'Original (Resized)'),
        ('log_transformed', 'Log Transformation'),
        ('histogram_equalized', 'Histogram Equalization (CLAHE)'),
        ('edges', 'Canny Edge Detection'),
        ('dilated_edges', 'Dilated Edges (ROI)'),
        ('final_processed', 'Final Processed'),
        ('edge_enhanced', 'Edge-Enhanced Output'),
        ('normalized', 'Normalized (0-1)')
    ]
    
    for idx, (key, title) in enumerate(steps):
        row = idx // 4
        col = idx % 4
        
        img = results[key]
        if key == 'normalized':
            img = (img * 255).astype(np.uint8)
        
        # Convert BGR to RGB for matplotlib
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[row, col].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        axes[row, col].set_title(title, fontsize=12)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced MRI Preprocessing for Brain Tumor Detection')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory with MRI images')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for processed images')
    parser.add_argument('--visualize', type=str, help='Path to single image for visualization')
    parser.add_argument('--save_intermediate', action='store_true', help='Save intermediate preprocessing steps')
    
    args = parser.parse_args()
    
    if args.visualize:
        visualize_preprocessing_steps(args.visualize, 'preprocessing_visualization.png')
    else:
        preprocessor = EnhancedMRIPreprocessor()
        preprocessor.process_dataset(args.input_dir, args.output_dir, args.save_intermediate)
