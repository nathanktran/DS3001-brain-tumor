"""
Brain Tumor Detection Tool - Interactive Inference System
This tool provides an easy-to-use interface for brain tumor detection in MRI scans
using the trained YOLO11n model. Displays bounding boxes and confidence scores.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from ultralytics import YOLO
import argparse
import logging
from typing import List, Tuple, Dict
from PIL import Image, ImageDraw, ImageFont
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrainTumorDetector:
    """Brain tumor detection tool with visualization capabilities"""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.7):
        """Initialize detector with model and thresholds"""
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        
        # Class names
        self.class_names = {0: 'negative', 1: 'positive'}
        self.colors = {0: (0, 255, 0), 1: (255, 0, 0)}  # Green for negative, Red for positive
        
    def load_model(self):
        """Load the trained YOLO model"""
        logger.info(f"Loading model from {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            self.model = YOLO(str(self.model_path))
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def detect_single_image(self, image_path: str, save_result: bool = True, 
                          output_dir: str = None) -> Dict:
        """Detect tumors in a single MRI image"""
        
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        logger.info(f"Processing image: {image_path}")
        
        # Run inference
        results = self.model(
            str(image_path),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # Load original image
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process results
        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                detection = {
                    'class_id': int(boxes.cls[i]),
                    'class_name': self.class_names.get(int(boxes.cls[i]), 'unknown'),
                    'confidence': float(boxes.conf[i]),
                    'bbox': boxes.xyxy[i].tolist(),  # [x1, y1, x2, y2]
                    'bbox_normalized': boxes.xywhn[i].tolist()  # [x_center, y_center, width, height] normalized
                }
                detections.append(detection)
        
        # Create visualization
        result_image = self._visualize_detections(image_rgb, detections)
        
        # Prepare result summary
        result_summary = {
            'image_path': str(image_path),
            'image_size': image_rgb.shape,
            'total_detections': len(detections),
            'detections': detections,
            'model_confidence_threshold': self.conf_threshold,
            'model_iou_threshold': self.iou_threshold
        }
        
        # Save results if requested
        if save_result and output_dir:
            self._save_results(result_summary, result_image, image_path, output_dir)
        
        return result_summary, result_image
    
    def detect_batch_images(self, images_dir: str, output_dir: str = None) -> List[Dict]:
        """Detect tumors in a batch of images"""
        
        images_dir = Path(images_dir)
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))
        
        if not image_files:
            logger.warning(f"No image files found in {images_dir}")
            return []
        
        logger.info(f"Processing {len(image_files)} images...")
        
        batch_results = []
        for img_file in image_files:
            try:
                result_summary, result_image = self.detect_single_image(
                    str(img_file), 
                    save_result=True,
                    output_dir=output_dir
                )
                batch_results.append(result_summary)
            except Exception as e:
                logger.error(f"Failed to process {img_file}: {e}")
                continue
        
        # Create batch summary
        self._create_batch_summary(batch_results, output_dir)
        
        return batch_results
    
    def _visualize_detections(self, image_rgb: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Create visualization with bounding boxes and confidence scores"""
        
        # Create a copy of the image
        vis_image = image_rgb.copy()
        
        # Convert to PIL for better text rendering
        pil_image = Image.fromarray(vis_image)
        draw = ImageDraw.Draw(pil_image)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 20)
            small_font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
            small_font = font
        
        # Draw detections
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            class_id = detection['class_id']
            
            # Get color (convert BGR to RGB)
            color = self.colors.get(class_id, (255, 255, 0))
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Prepare text
            text = f"{class_name}: {confidence:.2f}"
            
            # Get text size
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Draw text background
            text_bg = [x1, y1 - text_height - 5, x1 + text_width + 10, y1]
            draw.rectangle(text_bg, fill=color)
            
            # Draw text
            draw.text((x1 + 5, y1 - text_height - 2), text, fill=(255, 255, 255), font=font)
        
        # Add summary information
        summary_text = f"Detections: {len(detections)}"
        if detections:
            max_conf = max(d['confidence'] for d in detections)
            summary_text += f" | Max Confidence: {max_conf:.3f}"
        
        # Draw summary at the top
        summary_bbox = draw.textbbox((0, 0), summary_text, font=small_font)
        summary_width = summary_bbox[2] - summary_bbox[0]
        summary_height = summary_bbox[3] - summary_bbox[1]
        
        draw.rectangle([10, 10, 20 + summary_width, 20 + summary_height], 
                      fill=(0, 0, 0), outline=(255, 255, 255))
        draw.text((15, 15), summary_text, fill=(255, 255, 255), font=small_font)
        
        # Convert back to numpy array
        return np.array(pil_image)
    
    def _save_results(self, result_summary: Dict, result_image: np.ndarray, 
                     original_path: Path, output_dir: str):
        """Save detection results and visualization"""
        
        if not output_dir:
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save visualization
        vis_path = output_dir / f"{original_path.stem}_detected.jpg"
        plt.imsave(vis_path, result_image)
        
        # Save detection data
        json_path = output_dir / f"{original_path.stem}_detections.json"
        with open(json_path, 'w') as f:
            json.dump(result_summary, f, indent=2)
        
        logger.info(f"Results saved: {vis_path}, {json_path}")
    
    def _create_batch_summary(self, batch_results: List[Dict], output_dir: str):
        """Create summary report for batch processing"""
        
        if not output_dir or not batch_results:
            return
        
        output_dir = Path(output_dir)
        
        # Analyze batch results
        total_images = len(batch_results)
        images_with_detections = sum(1 for r in batch_results if r['total_detections'] > 0)
        total_detections = sum(r['total_detections'] for r in batch_results)
        
        # Confidence statistics
        all_confidences = []
        for result in batch_results:
            for detection in result['detections']:
                all_confidences.append(detection['confidence'])
        
        summary_stats = {
            'batch_summary': {
                'total_images_processed': total_images,
                'images_with_detections': images_with_detections,
                'images_without_detections': total_images - images_with_detections,
                'detection_rate': images_with_detections / total_images if total_images > 0 else 0,
                'total_detections': total_detections,
                'average_detections_per_image': total_detections / total_images if total_images > 0 else 0
            },
            'confidence_statistics': {
                'mean_confidence': np.mean(all_confidences) if all_confidences else 0,
                'max_confidence': np.max(all_confidences) if all_confidences else 0,
                'min_confidence': np.min(all_confidences) if all_confidences else 0,
                'std_confidence': np.std(all_confidences) if all_confidences else 0
            },
            'per_image_results': batch_results
        }
        
        # Save batch summary
        summary_path = output_dir / "batch_detection_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        # Create visualization summary
        self._create_batch_visualization(summary_stats, output_dir)
        
        logger.info(f"Batch summary saved: {summary_path}")
    
    def _create_batch_visualization(self, summary_stats: Dict, output_dir: Path):
        """Create visualization for batch processing results"""
        
        batch_summary = summary_stats['batch_summary']
        conf_stats = summary_stats['confidence_statistics']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Brain Tumor Detection - Batch Processing Results', fontsize=16, fontweight='bold')
        
        # 1. Detection rate pie chart
        labels = ['With Detections', 'No Detections']
        sizes = [batch_summary['images_with_detections'], 
                batch_summary['images_without_detections']]
        colors = ['#ff9999', '#66b3ff']
        
        axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title(f'Detection Rate\\n({batch_summary["total_images_processed"]} images)')
        
        # 2. Confidence distribution
        all_confidences = []
        for result in summary_stats['per_image_results']:
            for detection in result['detections']:
                all_confidences.append(detection['confidence'])
        
        if all_confidences:
            axes[0, 1].hist(all_confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 1].axvline(conf_stats['mean_confidence'], color='red', linestyle='--', 
                             label=f'Mean: {conf_stats["mean_confidence"]:.3f}')
            axes[0, 1].set_xlabel('Confidence Score')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Detection Confidence Distribution')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No detections found', ha='center', va='center',
                          transform=axes[0, 1].transAxes, fontsize=12)
            axes[0, 1].set_title('Detection Confidence Distribution')
        
        # 3. Summary statistics table
        stats_data = [
            ['Total Images', batch_summary['total_images_processed']],
            ['Images with Detections', batch_summary['images_with_detections']],
            ['Detection Rate', f"{batch_summary['detection_rate']:.1%}"],
            ['Total Detections', batch_summary['total_detections']],
            ['Avg Detections/Image', f"{batch_summary['average_detections_per_image']:.2f}"],
            ['Mean Confidence', f"{conf_stats['mean_confidence']:.3f}"],
            ['Max Confidence', f"{conf_stats['max_confidence']:.3f}"]
        ]
        
        axes[1, 0].axis('tight')
        axes[1, 0].axis('off')
        table = axes[1, 0].table(cellText=stats_data, 
                               colLabels=['Metric', 'Value'],
                               cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        axes[1, 0].set_title('Summary Statistics')
        
        # 4. Per-image detection counts
        detection_counts = [r['total_detections'] for r in summary_stats['per_image_results']]
        unique_counts = sorted(set(detection_counts))
        count_frequencies = [detection_counts.count(c) for c in unique_counts]
        
        axes[1, 1].bar(unique_counts, count_frequencies, alpha=0.7, color='lightgreen')
        axes[1, 1].set_xlabel('Number of Detections per Image')
        axes[1, 1].set_ylabel('Number of Images')
        axes[1, 1].set_title('Detection Count Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = output_dir / "batch_processing_summary.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Batch visualization saved: {viz_path}")
    
    def interactive_detection(self, image_path: str):
        """Interactive detection with immediate visualization"""
        
        result_summary, result_image = self.detect_single_image(image_path, save_result=False)
        
        # Display results
        plt.figure(figsize=(12, 8))
        plt.imshow(result_image)
        plt.axis('off')
        plt.title(f"Brain Tumor Detection Results\\n{Path(image_path).name}")
        plt.tight_layout()
        plt.show()
        
        # Print results
        print(f"\\n{'='*60}")
        print("DETECTION RESULTS")
        print(f"{'='*60}")
        print(f"Image: {result_summary['image_path']}")
        print(f"Total Detections: {result_summary['total_detections']}")
        print(f"Confidence Threshold: {result_summary['model_confidence_threshold']}")
        
        if result_summary['detections']:
            print("\\nDetected Objects:")
            for i, detection in enumerate(result_summary['detections'], 1):
                print(f"  {i}. {detection['class_name']} (confidence: {detection['confidence']:.3f})")
                x1, y1, x2, y2 = detection['bbox']
                print(f"     Bounding box: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
        else:
            print("\\nNo tumors detected.")
        
        print(f"{'='*60}")
        
        return result_summary

def main():
    """Main function for command line interface"""
    parser = argparse.ArgumentParser(description='Brain Tumor Detection Tool')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.pt file)')
    parser.add_argument('--input', type=str, required=True, 
                       help='Path to input image or directory of images')
    parser.add_argument('--output', type=str, default='./detection_results',
                       help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (0-1)')
    parser.add_argument('--iou', type=float, default=0.7,
                       help='IoU threshold for NMS (0-1)')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode with immediate display')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = BrainTumorDetector(args.model, args.conf, args.iou)
    detector.load_model()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image processing
        if args.interactive:
            detector.interactive_detection(str(input_path))
        else:
            result_summary, _ = detector.detect_single_image(
                str(input_path), 
                save_result=True,
                output_dir=args.output
            )
            print(f"Detection completed. Results saved to {args.output}")
    
    elif input_path.is_dir():
        # Batch processing
        batch_results = detector.detect_batch_images(str(input_path), args.output)
        print(f"Batch processing completed. {len(batch_results)} images processed.")
        print(f"Results saved to {args.output}")
    
    else:
        logger.error(f"Input path not found: {input_path}")

if __name__ == "__main__":
    # For direct execution, provide example usage
    print("Brain Tumor Detection Tool")
    print("="*40)
    print("Usage examples:")
    print("1. Single image: python tumor_detector.py --model best.pt --input image.jpg")
    print("2. Batch processing: python tumor_detector.py --model best.pt --input ./images/ --output ./results/")
    print("3. Interactive mode: python tumor_detector.py --model best.pt --input image.jpg --interactive")
    print("\\nFor help: python tumor_detector.py --help")
    
    # Check if we have a model available for demo
    output_dir = Path("../OUTPUT")
    model_files = list(output_dir.rglob("best.pt"))
    
    if model_files:
        print(f"\\nFound trained model: {model_files[0]}")
        print("You can test with validation images from ../DATA/images/val/")
    else:
        print("\\nNo trained model found. Please train the model first.")
