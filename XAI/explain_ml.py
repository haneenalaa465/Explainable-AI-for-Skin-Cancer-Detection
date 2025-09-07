"""
Wrapper script to run explainability techniques for ML models.
Can be called from the Makefile.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from XAI.run_explainers import analyze_image
from XAI.config import PROCESSED_DATA_DIR, REPORTS_DIR


def main():
    """Main function to run ML explainers"""
    parser = argparse.ArgumentParser(description='Run ML explainability techniques')
    parser.add_argument('--image', type=str, default=None,
                      help='Path to the image file')
    parser.add_argument('--image_dir', type=str, default=None,
                      help='Directory with test images')
    parser.add_argument('--model', type=str, default='RandomForest',
                      help='Model type (DecisionTree or RandomForest)')
    parser.add_argument('--max_images', type=int, default=5,
                      help='Maximum number of images to analyze from directory')
    parser.add_argument('--features', type=str, default=None,
                      help='Path to features pickle file for background data')
    
    args = parser.parse_args()
    
    # Set default features path if not provided
    if args.features is None:
        args.features = PROCESSED_DATA_DIR / "ham10000_features.pkl"
    
    # Create output directory
    output_base_dir = REPORTS_DIR / "explainability" / f"{args.model}_explanations"
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Process a single image
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Error: Image not found at {image_path}")
            return 1
        
        print(f"Analyzing image: {image_path}")
        output_dir = output_base_dir / image_path.stem
        analyze_image(
            image_path, args.model, args.features, output_dir
        )
    
    # Process multiple images from a directory
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        if not image_dir.exists():
            print(f"Error: Directory not found at {image_dir}")
            return 1
        
        # Find all image files
        image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        
        if not image_paths:
            print(f"Error: No images found in {image_dir}")
            return 1
        
        # Limit number of images
        if args.max_images > 0 and len(image_paths) > args.max_images:
            print(f"Found {len(image_paths)} images, limiting to {args.max_images}")
            image_paths = image_paths[:args.max_images]
        
        # Process each image
        for i, image_path in enumerate(image_paths):
            print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            output_dir = output_base_dir / image_path.stem
            
            try:
                analyze_image(
                    image_path, args.model, args.features, output_dir
                )
            except Exception as e:
                print(f"Error analyzing {image_path}: {e}")
                continue
                
    else:
        print("Error: Please provide either --image or --image_dir")
        return 1
    
    print("Explainability analysis complete!")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
