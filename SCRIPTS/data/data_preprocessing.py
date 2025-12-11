"""
Data preprocessing and validation utilities for Brain Tumor Detection
This module provides functions for loading, validating, and preprocessing
the brain tumor MRI dataset in YOLO format.

DATASET FORMAT SPECIFICATION:
============================

Image Files (.jpg):
- Brain MRI scan images
- Examples: 00054_145.jpg, 62 (13).jpg
- Linked with corresponding label files by filename

Label Files (.txt):
- YOLO format annotation files
- Examples: 00054_145.txt, 62 (13).txt
- Each line represents one bounding box annotation

YOLO Label Format:
- Format: class x_center y_center width height
- All coordinates are normalized (0.0 to 1.0)

Field Descriptions:
- class: Classifier (0 = negative/no tumor, 1 = positive/tumor present)
- x_center: X-coordinate of bounding box center (normalized)
- y_center: Y-coordinate of bounding box center (normalized)
- width: Width of bounding box (normalized)
- height: Height of bounding box (normalized)

Examples:
- "0 0.344484 0.342723 0.221831 0.176056" (negative case with bounding box)
- "1 0.518779 0.416667 0.150235 0.070423" (positive case with tumor)
"""

import os
import glob
import cv2
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import logging
from typing import Tuple, List, Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrainTumorDataset:
    """Brain tumor dataset handler for YOLO format data"""
    
    def __init__(self, data_path: str):
        """Initialize dataset with data path"""
        self.data_path = Path(data_path)
        self.train_images_path = self.data_path / "images" / "train"
        self.train_labels_path = self.data_path / "labels" / "train"
        self.val_images_path = self.data_path / "images" / "val"
        self.val_labels_path = self.data_path / "labels" / "val"
        
        self.train_stats = {}
        self.val_stats = {}
        
    def validate_dataset(self) -> Dict:
        """Validate dataset structure and integrity"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check if directories exist
        required_dirs = [
            self.train_images_path,
            self.train_labels_path,
            self.val_images_path,
            self.val_labels_path
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                validation_results['errors'].append(f"Directory not found: {dir_path}")
                validation_results['valid'] = False
        
        if not validation_results['valid']:
            return validation_results
        
        # Get file counts
        train_images = list(self.train_images_path.glob("*.jpg"))
        train_labels = list(self.train_labels_path.glob("*.txt"))
        val_images = list(self.val_images_path.glob("*.jpg"))
        val_labels = list(self.val_labels_path.glob("*.txt"))
        
        validation_results['stats'] = {
            'train_images': len(train_images),
            'train_labels': len(train_labels),
            'val_images': len(val_images),
            'val_labels': len(val_labels)
        }
        
        logger.info(f"Found {len(train_images)} training images")
        logger.info(f"Found {len(train_labels)} training labels")
        logger.info(f"Found {len(val_images)} validation images")
        logger.info(f"Found {len(val_labels)} validation labels")
        
        # Check for missing labels or images
        train_image_names = {img.stem for img in train_images}
        train_label_names = {lbl.stem for lbl in train_labels}
        val_image_names = {img.stem for img in val_images}
        val_label_names = {lbl.stem for lbl in val_labels}
        
        # Find mismatches
        train_missing_labels = train_image_names - train_label_names
        train_missing_images = train_label_names - train_image_names
        val_missing_labels = val_image_names - val_label_names
        val_missing_images = val_label_names - val_image_names
        
        if train_missing_labels:
            validation_results['warnings'].extend([
                f"Training images without labels: {list(train_missing_labels)[:5]}"
                + (f" and {len(train_missing_labels)-5} more" if len(train_missing_labels) > 5 else "")
            ])
        
        if val_missing_labels:
            validation_results['warnings'].extend([
                f"Validation images without labels: {list(val_missing_labels)[:5]}"
                + (f" and {len(val_missing_labels)-5} more" if len(val_missing_labels) > 5 else "")
            ])
        
        return validation_results
    
    def analyze_dataset(self) -> Dict:
        """Analyze dataset statistics and class distribution"""
        logger.info("Analyzing dataset statistics...")
        
        # Analyze training set
        train_stats = self._analyze_split('train', 
                                        self.train_images_path, 
                                        self.train_labels_path)
        
        # Analyze validation set
        val_stats = self._analyze_split('val', 
                                      self.val_images_path, 
                                      self.val_labels_path)
        
        return {
            'train': train_stats,
            'validation': val_stats,
            'summary': {
                'total_images': train_stats['total_images'] + val_stats['total_images'],
                'total_positive': train_stats['positive_count'] + val_stats['positive_count'],
                'total_negative': train_stats['negative_count'] + val_stats['negative_count']
            }
        }
    
    def _analyze_split(self, split_name: str, images_path: Path, labels_path: Path) -> Dict:
        """Analyze a single dataset split"""
        stats = {
            'split': split_name,
            'total_images': 0,
            'positive_count': 0,
            'negative_count': 0,
            'bbox_areas': [],
            'bbox_centers_x': [],
            'bbox_centers_y': [],
            'bbox_widths': [],
            'bbox_heights': [],
            'image_sizes': []
        }
        
        image_files = list(images_path.glob("*.jpg"))
        stats['total_images'] = len(image_files)
        
        for img_file in image_files:
            # Get image dimensions
            try:
                with Image.open(img_file) as img:
                    stats['image_sizes'].append(img.size)
            except Exception as e:
                logger.warning(f"Could not read image {img_file}: {e}")
                continue
            
            # Check corresponding label file
            label_file = labels_path / f"{img_file.stem}.txt"
            
            if label_file.exists():
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    if lines and lines[0].strip():
                        # Parse the first annotation to get class
                        parts = lines[0].strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            
                            # Count based on actual class ID
                            if class_id == 0:
                                stats['negative_count'] += 1
                            elif class_id == 1:
                                stats['positive_count'] += 1
                            
                            # Parse all bounding boxes for statistics
                            for line in lines:
                                line_parts = line.strip().split()
                                if len(line_parts) >= 5:
                                    _, x_c, y_c, w, h = map(float, line_parts[:5])
                                    stats['bbox_centers_x'].append(x_c)
                                    stats['bbox_centers_y'].append(y_c)
                                    stats['bbox_widths'].append(w)
                                    stats['bbox_heights'].append(h)
                                    stats['bbox_areas'].append(w * h)
                    else:
                        # Empty label file (negative case)
                        stats['negative_count'] += 1
                except Exception as e:
                    logger.warning(f"Could not read label {label_file}: {e}")
                    stats['negative_count'] += 1
            else:
                # No label file (assuming negative case)
                stats['negative_count'] += 1
        
        # Calculate percentages
        if stats['total_images'] > 0:
            stats['positive_percentage'] = (stats['positive_count'] / stats['total_images']) * 100
            stats['negative_percentage'] = (stats['negative_count'] / stats['total_images']) * 100
        
        return stats
    
    def visualize_statistics(self, stats: Dict, output_dir: str = None):
        """Create visualizations of dataset statistics"""
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Brain Tumor Dataset Analysis', fontsize=16, fontweight='bold')
        
        # 1. Class distribution comparison
        train_pos = stats['train']['positive_count']
        train_neg = stats['train']['negative_count']
        val_pos = stats['validation']['positive_count']
        val_neg = stats['validation']['negative_count']
        
        categories = ['Training', 'Validation']
        positive_counts = [train_pos, val_pos]
        negative_counts = [train_neg, val_neg]
        
        x = np.arange(len(categories))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, positive_counts, width, label='Positive (Tumor)', alpha=0.8)
        axes[0, 0].bar(x + width/2, negative_counts, width, label='Negative (No Tumor)', alpha=0.8)
        axes[0, 0].set_xlabel('Dataset Split')
        axes[0, 0].set_ylabel('Number of Images')
        axes[0, 0].set_title('Class Distribution by Split')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(categories)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Bounding box area distribution
        if stats['train']['bbox_areas']:
            axes[0, 1].hist(stats['train']['bbox_areas'], bins=30, alpha=0.7, 
                          label='Training', density=True)
        if stats['validation']['bbox_areas']:
            axes[0, 1].hist(stats['validation']['bbox_areas'], bins=30, alpha=0.7, 
                          label='Validation', density=True)
        axes[0, 1].set_xlabel('Bounding Box Area (normalized)')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Tumor Size Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Tumor location heatmap (training data)
        if stats['train']['bbox_centers_x'] and stats['train']['bbox_centers_y']:
            x_centers = stats['train']['bbox_centers_x']
            y_centers = stats['train']['bbox_centers_y']
            axes[0, 2].scatter(x_centers, y_centers, alpha=0.6, s=30)
            axes[0, 2].set_xlabel('X Center (normalized)')
            axes[0, 2].set_ylabel('Y Center (normalized)')
            axes[0, 2].set_title('Tumor Location Distribution (Training)')
            axes[0, 2].set_xlim(0, 1)
            axes[0, 2].set_ylim(0, 1)
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Aspect ratio distribution
        train_aspects = []
        val_aspects = []
        
        if stats['train']['bbox_widths'] and stats['train']['bbox_heights']:
            train_aspects = [w/h for w, h in zip(stats['train']['bbox_widths'], 
                                               stats['train']['bbox_heights']) if h > 0]
        
        if stats['validation']['bbox_widths'] and stats['validation']['bbox_heights']:
            val_aspects = [w/h for w, h in zip(stats['validation']['bbox_widths'], 
                                             stats['validation']['bbox_heights']) if h > 0]
        
        if train_aspects:
            axes[1, 0].hist(train_aspects, bins=20, alpha=0.7, label='Training', density=True)
        if val_aspects:
            axes[1, 0].hist(val_aspects, bins=20, alpha=0.7, label='Validation', density=True)
        axes[1, 0].set_xlabel('Aspect Ratio (Width/Height)')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Bounding Box Aspect Ratio Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Dataset size summary
        summary_data = {
            'Split': ['Training', 'Validation', 'Total'],
            'Images': [stats['train']['total_images'], 
                      stats['validation']['total_images'],
                      stats['summary']['total_images']],
            'Positive': [train_pos, val_pos, stats['summary']['total_positive']],
            'Negative': [train_neg, val_neg, stats['summary']['total_negative']]
        }
        
        df = pd.DataFrame(summary_data)
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        table = axes[1, 1].table(cellText=df.values, colLabels=df.columns,
                                cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        axes[1, 1].set_title('Dataset Summary Statistics')
        
        # 6. Positive vs Negative percentage pie chart
        total_pos = stats['summary']['total_positive']
        total_neg = stats['summary']['total_negative']
        labels = ['Positive (Tumor)', 'Negative (No Tumor)']
        sizes = [total_pos, total_neg]
        colors = ['#ff9999', '#66b3ff']
        
        axes[1, 2].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 2].set_title('Overall Class Distribution')
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'dataset_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            logger.info(f"Saved dataset analysis plot to {output_dir}/dataset_analysis.png")
        
        plt.show()
        
        # Print summary statistics
        print("\\n" + "="*60)
        print("BRAIN TUMOR DATASET ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total Images: {stats['summary']['total_images']}")
        print(f"Training Images: {stats['train']['total_images']}")
        print(f"Validation Images: {stats['validation']['total_images']}")
        print("\\nClass Distribution:")
        print(f"  Total Positive (Tumor): {total_pos} ({total_pos/(total_pos+total_neg)*100:.1f}%)")
        print(f"  Total Negative (No Tumor): {total_neg} ({total_neg/(total_pos+total_neg)*100:.1f}%)")
        print("\\nTraining Set:")
        print(f"  Positive: {train_pos} ({stats['train'].get('positive_percentage', 0):.1f}%)")
        print(f"  Negative: {train_neg} ({stats['train'].get('negative_percentage', 0):.1f}%)")
        print("\\nValidation Set:")
        print(f"  Positive: {val_pos} ({stats['validation'].get('positive_percentage', 0):.1f}%)")
        print(f"  Negative: {val_neg} ({stats['validation'].get('negative_percentage', 0):.1f}%)")
        
        if stats['train']['bbox_areas']:
            print(f"\\nTumor Size Statistics (Training):")
            print(f"  Average area: {np.mean(stats['train']['bbox_areas']):.4f}")
            print(f"  Min area: {np.min(stats['train']['bbox_areas']):.4f}")
            print(f"  Max area: {np.max(stats['train']['bbox_areas']):.4f}")
        
        print("="*60)
    
    def check_image_integrity(self) -> Dict:
        """Check for corrupted or invalid images"""
        logger.info("Checking image integrity...")
        
        corrupted_images = []
        valid_images = 0
        
        all_images = list(self.train_images_path.glob("*.jpg")) + list(self.val_images_path.glob("*.jpg"))
        
        for img_path in all_images:
            try:
                with Image.open(img_path) as img:
                    img.verify()
                valid_images += 1
            except Exception as e:
                corrupted_images.append(str(img_path))
                logger.warning(f"Corrupted image: {img_path} - {e}")
        
        return {
            'total_checked': len(all_images),
            'valid_images': valid_images,
            'corrupted_images': corrupted_images,
            'corruption_rate': len(corrupted_images) / len(all_images) * 100 if all_images else 0
        }
    
    def display_format_examples(self):
        """Display examples of the data format with actual files"""
        print("\n" + "="*60)
        print("DATA FORMAT EXAMPLES")
        print("="*60)
        
        # Find examples of each class
        examples_found = {'negative': [], 'positive': []}
        
        # Check training labels for examples
        train_labels = list(self.train_labels_path.glob("*.txt"))[:10]  # Check first 10
        
        for label_file in train_labels:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                img_file = self.train_images_path / f"{label_file.stem}.jpg"
                
                if lines and lines[0].strip():
                    # Parse the annotation
                    parts = lines[0].strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        example_data = {
                            'image_file': img_file.name,
                            'label_file': label_file.name,
                            'class': class_id,
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height,
                            'annotation_line': lines[0].strip()
                        }
                        
                        if class_id == 0 and len(examples_found['negative']) < 3:
                            examples_found['negative'].append(example_data)
                        elif class_id == 1 and len(examples_found['positive']) < 3:
                            examples_found['positive'].append(example_data)
                        
                        if len(examples_found['negative']) >= 3 and len(examples_found['positive']) >= 3:
                            break
            except Exception as e:
                continue
        
        # Display format specification
        print("\nYOLO ANNOTATION FORMAT:")
        print("Format: class x_center y_center width height")
        print("\nField Descriptions:")
        print("- image:    Brain MRI scan (.jpg format)")
        print("- label:    Corresponding annotation file (.txt format)")
        print("- class:    0 = negative (no tumor), 1 = positive (tumor present)")
        print("- x_center: X-coordinate of bounding box center (normalized 0-1)")
        print("- y_center: Y-coordinate of bounding box center (normalized 0-1)")
        print("- width:    Width of bounding box (normalized 0-1)")
        print("- height:   Height of bounding box (normalized 0-1)")
        
        # Display examples
        print("\n" + "-"*50)
        print("NEGATIVE EXAMPLES (Class 0 - No Tumor):")
        print("-"*50)
        for i, example in enumerate(examples_found['negative'], 1):
            print(f"\nExample {i}:")
            print(f"  Image File:    {example['image_file']}")
            print(f"  Label File:    {example['label_file']}")
            print(f"  Annotation:    {example['annotation_line']}")
            print(f"  Parsed Fields:")
            print(f"    Class:       {example['class']} (negative/no tumor)")
            print(f"    X-Center:    {example['x_center']:.6f}")
            print(f"    Y-Center:    {example['y_center']:.6f}")
            print(f"    Width:       {example['width']:.6f}")
            print(f"    Height:      {example['height']:.6f}")
        
        print("\n" + "-"*50)
        print("POSITIVE EXAMPLES (Class 1 - Tumor Present):")
        print("-"*50)
        for i, example in enumerate(examples_found['positive'], 1):
            print(f"\nExample {i}:")
            print(f"  Image File:    {example['image_file']}")
            print(f"  Label File:    {example['label_file']}")
            print(f"  Annotation:    {example['annotation_line']}")
            print(f"  Parsed Fields:")
            print(f"    Class:       {example['class']} (positive/tumor present)")
            print(f"    X-Center:    {example['x_center']:.6f}")
            print(f"    Y-Center:    {example['y_center']:.6f}")
            print(f"    Width:       {example['width']:.6f}")
            print(f"    Height:      {example['height']:.6f}")
        
        print("\n" + "="*60)

def main():
    """Main function to run dataset validation and analysis"""
    # Set up paths
    data_path = "../DATA"
    output_path = "../OUTPUT"
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize dataset
    dataset = BrainTumorDataset(data_path)
    
    # Validate dataset structure
    print("Validating dataset structure...")
    validation_results = dataset.validate_dataset()
    
    if not validation_results['valid']:
        print("Dataset validation failed!")
        for error in validation_results['errors']:
            print(f"ERROR: {error}")
        return
    
    if validation_results['warnings']:
        print("Validation warnings:")
        for warning in validation_results['warnings']:
            print(f"WARNING: {warning}")
    
    print("Dataset validation passed!")
    print(f"Statistics: {validation_results['stats']}")
    
    # Check image integrity
    print("\\nChecking image integrity...")
    integrity_results = dataset.check_image_integrity()
    print(f"Total images checked: {integrity_results['total_checked']}")
    print(f"Valid images: {integrity_results['valid_images']}")
    print(f"Corrupted images: {len(integrity_results['corrupted_images'])}")
    
    if integrity_results['corrupted_images']:
        print("Corrupted images found:")
        for img in integrity_results['corrupted_images'][:5]:
            print(f"  {img}")
        if len(integrity_results['corrupted_images']) > 5:
            print(f"  ... and {len(integrity_results['corrupted_images'])-5} more")
    
    # Analyze dataset statistics
    print("\\nAnalyzing dataset statistics...")
    stats = dataset.analyze_dataset()
    
    # Create visualizations
    dataset.visualize_statistics(stats, output_path)
    
    # Display format examples
    dataset.display_format_examples()
    
    # Save statistics to file
    stats_file = os.path.join(output_path, 'dataset_statistics.txt')
    with open(stats_file, 'w') as f:
        f.write("Brain Tumor Dataset Statistics\\n")
        f.write("="*50 + "\\n")
        f.write(f"Validation Results: {validation_results}\\n\\n")
        f.write(f"Integrity Check: {integrity_results}\\n\\n")
        f.write(f"Dataset Statistics: {stats}\\n")
    
    print(f"\\nAnalysis complete! Results saved to {output_path}")

if __name__ == "__main__":
    main()
