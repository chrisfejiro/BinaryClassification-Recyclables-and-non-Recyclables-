"""
Split DatasetFinal into train/val/test sets (70/15/15 split)
Dataset: 17,677 Recyclables, 16,782 Non-Recyclables
"""

import os
import sys
import shutil
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def split_dataset():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Source directory (parent folder)
    source_dir = 'DatasetFinal'
    output_dir = 'data'

    print("="*70)
    print("WASTE CLASSIFICATION - DATASET SPLITTING")
    print("="*70)

    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"\nERROR: '{source_dir}' folder not found!")
        print("Please ensure DatasetFinal folder is in the project root.")
        return

    # Get all image paths and labels
    image_paths = []
    labels = []

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

    # Load Recyclables (label = 1)
    recyclables_dir = os.path.join(source_dir, 'Recyclables')
    print(f"\nScanning {recyclables_dir}...")
    if os.path.exists(recyclables_dir):
        for img_file in os.listdir(recyclables_dir):
            if Path(img_file).suffix.lower() in valid_extensions:
                image_paths.append(os.path.join(recyclables_dir, img_file))
                labels.append(1)
    else:
        print(f"ERROR: {recyclables_dir} not found!")
        return

    # Load Non-Recyclables (label = 0)
    non_recyclables_dir = os.path.join(source_dir, 'Non-Recyclables')
    print(f"Scanning {non_recyclables_dir}...")
    if os.path.exists(non_recyclables_dir):
        for img_file in os.listdir(non_recyclables_dir):
            if Path(img_file).suffix.lower() in valid_extensions:
                image_paths.append(os.path.join(non_recyclables_dir, img_file))
                labels.append(0)
    else:
        print(f"ERROR: {non_recyclables_dir} not found!")
        return

    image_paths = np.array(image_paths)
    labels = np.array(labels)

    print(f"\n{'='*70}")
    print("DATASET SUMMARY")
    print("="*70)
    print(f"Total images: {len(image_paths)}")
    print(f"Recyclables: {np.sum(labels == 1)} ({np.mean(labels == 1)*100:.1f}%)")
    print(f"Non-Recyclables: {np.sum(labels == 0)} ({np.mean(labels == 0)*100:.1f}%)")

    # Stratified split: 70% train, 15% val, 15% test
    print(f"\n{'='*70}")
    print("SPLITTING DATASET")
    print("="*70)
    print("Split ratio: 70% train, 15% validation, 15% test")

    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, labels, test_size=0.15, stratify=labels, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42
    )

    # Create directories and copy files
    splits = {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }

    for split_name, (paths, split_labels) in splits.items():
        recyclable_dir = os.path.join(output_dir, split_name, 'recyclable')
        non_recyclable_dir = os.path.join(output_dir, split_name, 'non_recyclable')
        
        os.makedirs(recyclable_dir, exist_ok=True)
        os.makedirs(non_recyclable_dir, exist_ok=True)
        
        print(f"\nCopying {split_name} set...")
        for idx, (img_path, label) in enumerate(zip(paths, split_labels)):
            dest_dir = recyclable_dir if label == 1 else non_recyclable_dir
            shutil.copy2(img_path, os.path.join(dest_dir, os.path.basename(img_path)))
            
            if (idx + 1) % 1000 == 0:
                print(f"  Progress: {idx + 1}/{len(paths)} images copied...")
        
        print(f"\n{split_name.upper()} SET CREATED:")
        print(f"  Total: {len(paths)}")
        print(f"  Recyclable: {np.sum(split_labels == 1)} ({np.mean(split_labels == 1)*100:.1f}%)")
        print(f"  Non-Recyclable: {np.sum(split_labels == 0)} ({np.mean(split_labels == 0)*100:.1f}%)")

    print(f"\n{'='*70}")
    print("DATASET SPLIT COMPLETE!")
    print("="*70)
    print(f"\nOutput directory structure:")
    print(f"data/")
    print(f"├── train/")
    print(f"│   ├── recyclable/        ({np.sum(y_train == 1)} images)")
    print(f"│   └── non_recyclable/    ({np.sum(y_train == 0)} images)")
    print(f"├── val/")
    print(f"│   ├── recyclable/        ({np.sum(y_val == 1)} images)")
    print(f"│   └── non_recyclable/    ({np.sum(y_val == 0)} images)")
    print(f"└── test/")
    print(f"    ├── recyclable/        ({np.sum(y_test == 1)} images)")
    print(f"    └── non_recyclable/    ({np.sum(y_test == 0)} images)")

if __name__ == "__main__":
    split_dataset()