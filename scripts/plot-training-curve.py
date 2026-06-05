#!/usr/bin/env python3
"""
LibreYOLO Training Curve Visualization Script
Plots training curves (losses and validation metrics) from LibreYOLO results.csv.

Usage:
  # Plot a specific run
  python scripts/plot-training-curve.py -f outputs/train/v1.person.LibreYOLO9s/results.csv
  
  # Plot and compare multiple runs
  python scripts/plot-training-curve.py -f outputs/train/run1/results.csv outputs/train/run2/results.csv -n Run1 Run2
  
  # Automatically search and plot all results.csv files in the outputs directory
  python scripts/plot-training-curve.py
"""

import os
import csv
import argparse
import glob
import matplotlib.pyplot as plt

def parse_results_csv(csv_path):
    """
    Parses results.csv to extract training and validation metrics.
    """
    data = {
        'epoch': [],
        'train/loss': [],
        'train/box_loss': [],
        'train/cls_loss': [],
        'train/dfl_loss': [],
        'lr': [],
        'map50_epochs': [],
        'metrics/mAP50': [],
        'map50_95_epochs': [],
        'metrics/mAP50-95': []
    }
    
    if not os.path.exists(csv_path):
        print(f"Warning: File not found: {csv_path}")
        return None

    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epoch = int(row['epoch'])
                
                # Check for train loss (must exist to add training progress)
                if row.get('train/loss') and row['train/loss'].strip():
                    data['epoch'].append(epoch)
                    data['train/loss'].append(float(row['train/loss']))
                    
                    if row.get('train/box_loss') and row['train/box_loss'].strip():
                        data['train/box_loss'].append(float(row['train/box_loss']))
                    else:
                        data['train/box_loss'].append(0.0)
                        
                    if row.get('train/cls_loss') and row['train/cls_loss'].strip():
                        data['train/cls_loss'].append(float(row['train/cls_loss']))
                    else:
                        data['train/cls_loss'].append(0.0)
                        
                    if row.get('train/dfl_loss') and row['train/dfl_loss'].strip():
                        data['train/dfl_loss'].append(float(row['train/dfl_loss']))
                    else:
                        data['train/dfl_loss'].append(0.0)
                        
                    if row.get('lr/group0') and row['lr/group0'].strip():
                        data['lr'].append(float(row['lr/group0']))
                    else:
                        data['lr'].append(0.0)
                
                # Check for validation metrics (mAP50 / mAP50-95)
                m50 = row.get('metrics/mAP50') or row.get('metrics/mAP50(B)')
                if m50 and m50.strip():
                    data['map50_epochs'].append(epoch)
                    data['metrics/mAP50'].append(float(m50))
                    
                m50_95 = row.get('metrics/mAP50-95') or row.get('metrics/mAP50-95(B)')
                if m50_95 and m50_95.strip():
                    data['map50_95_epochs'].append(epoch)
                    data['metrics/mAP50-95'].append(float(m50_95))
                    
            except (ValueError, KeyError):
                # Ignore malformed rows or headers
                continue
                
    # Check if we successfully read any training epochs
    if not data['epoch']:
        print(f"Warning: No valid training data found in {csv_path}")
        return None
        
    return data

def plot_metrics(models_data, output_dir, generate_dashboard=True):
    """
    Plots losses and validation metrics for the loaded models using standard matplotlib style.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Total Train Loss
    plt.figure(figsize=(10, 5))
    for model_name, data in models_data.items():
        plt.plot(data['epoch'], data['train/loss'], label=model_name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total Training Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'train_loss.png'))
    plt.close()
    
    # 2. mAP50
    has_m50 = False
    plt.figure(figsize=(10, 5))
    for model_name, data in models_data.items():
        if data['metrics/mAP50']:
            plt.plot(data['map50_epochs'], data['metrics/mAP50'], label=model_name)
            has_m50 = True
    if has_m50:
        plt.xlabel('Epoch')
        plt.ylabel('mAP50')
        plt.title('Validation mAP50')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'map50.png'))
    plt.close()
    
    # 3. mAP50-95
    has_m50_95 = False
    plt.figure(figsize=(10, 5))
    for model_name, data in models_data.items():
        if data['metrics/mAP50-95']:
            plt.plot(data['map50_95_epochs'], data['metrics/mAP50-95'], label=model_name)
            has_m50_95 = True
    if has_m50_95:
        plt.xlabel('Epoch')
        plt.ylabel('mAP50-95')
        plt.title('Validation mAP50-95')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'map50_95.png'))
    plt.close()
    
    # 4. Combined Dashboard (1x3 Grid)
    if generate_dashboard:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
        
        # Left: Train Loss
        for model_name, data in models_data.items():
            axs[0].plot(data['epoch'], data['train/loss'], label=model_name)
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].set_title('Total Training Loss')
        axs[0].grid(True)
        axs[0].legend()
        
        # Middle: mAP50-95
        if has_m50_95:
            for model_name, data in models_data.items():
                if data['metrics/mAP50-95']:
                    axs[1].plot(data['map50_95_epochs'], data['metrics/mAP50-95'], label=model_name)
            axs[1].set_xlabel('Epoch')
            axs[1].set_ylabel('mAP50-95')
            axs[1].set_title('Validation mAP50-95')
            axs[1].grid(True)
            axs[1].legend()
        else:
            axs[1].text(0.5, 0.5, 'No validation mAP50-95 logs found yet', ha='center', va='center')
            axs[1].set_title('Validation mAP50-95')
            axs[1].grid(True)
            
        # Right: Learning Rate
        for model_name, data in models_data.items():
            if data['lr']:
                axs[2].plot(data['epoch'], data['lr'], label=model_name)
        axs[2].set_xlabel('Epoch')
        axs[2].set_ylabel('LR')
        axs[2].set_title('Learning Rate')
        axs[2].grid(True)
        axs[2].legend()
            
        plt.suptitle('LibreYOLO Training Progress Dashboard', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dashboard.png'))
        plt.close()

    print(f"Successfully saved training plots to: {output_dir}")
    print("  - train_loss.png")
    if has_m50:
        print("  - map50.png")
    if has_m50_95:
        print("  - map50_95.png")
    if generate_dashboard:
        print("  - dashboard.png")

def main():
    parser = argparse.ArgumentParser(description="Visualize training curves for LibreYOLO results.csv")
    parser.add_argument('-f', '--files', nargs='*', help="Path to one or more results.csv files.")
    parser.add_argument('-n', '--names', nargs='*', help="Display names for each run. If omitted, uses folder names.")
    parser.add_argument('-o', '--output-dir', default='outputs/plots', help="Directory to save the plots.")
    parser.add_argument('--no-dashboard', action='store_true', help="Do not generate the combined 1x3 dashboard.")
    
    args = parser.parse_args()
    
    log_files = []
    
    # Auto-find logic if no files are supplied
    if not args.files:
        # Check recursively in outputs/
        print("No files specified. Searching recursively for 'results.csv' in outputs/...")
        pattern = os.path.join('outputs', '**', 'results.csv')
        found_files = glob.glob(pattern, recursive=True)
        
        if not found_files:
            print("No results.csv files found under outputs/. Please specify files using -f.")
            return
            
        print(f"Found {len(found_files)} run log(s):")
        for f in found_files:
            print(f"  - {f}")
        log_files = found_files
    else:
        log_files = args.files
        
    # Set up names
    names = []
    if args.names:
        names = args.names
        # Pad with file names if names list is shorter than files list
        while len(names) < len(log_files):
            # Extract parent directory name as name
            idx = len(names)
            parent_dir = os.path.basename(os.path.dirname(log_files[idx]))
            names.append(parent_dir if parent_dir else f"run_{idx}")
    else:
        for f in log_files:
            parent_dir = os.path.basename(os.path.dirname(f))
            names.append(parent_dir if parent_dir else "LibreYOLO")
            
    # Load all models data
    models_data = {}
    for csv_file, name in zip(log_files, names):
        print(f"Loading {name} from {csv_file}...")
        data = parse_results_csv(csv_file)
        if data:
            models_data[name] = data
            
    if not models_data:
        print("Error: No valid training data could be loaded.")
        return
        
    plot_metrics(models_data, args.output_dir, generate_dashboard=not args.no_dashboard)

if __name__ == '__main__':
    main()
