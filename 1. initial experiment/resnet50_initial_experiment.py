import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import json
import warnings
from collections import defaultdict, Counter
import time
from sklearn.metrics import f1_score, precision_recall_fscore_support
import seaborn as sns

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/home/s3844498/data/2nd_fix', help='Path to the herbarium dataset')
parser.add_argument('--output_dir', type=str, default='/home/s3844498/outputs_resnet50_initial_experiment', help='Directory to save outputs')
parser.add_argument('--test_epochs', nargs='+', type=int, default=[5, 10, 15, 20], 
                    help='List of epochs to test (default: 5 10 15 20)')
parser.add_argument('--run_baseline', action='store_true', 
                    help='Run baseline without class imbalance handling')
parser.add_argument('--run_weighted_loss', action='store_true',
                    help='Run with weighted loss only')
parser.add_argument('--run_balanced_sampling', action='store_true',
                    help='Run with balanced sampling only')
parser.add_argument('--run_combined', action='store_true',
                    help='Run with both weighted loss and balanced sampling')
parser.add_argument('--run_all', action='store_true',
                    help='Run all experiments for complete initial experiment')
args = parser.parse_args()

# Set default to run all if nothing specified
if not any([args.run_baseline, args.run_weighted_loss, args.run_balanced_sampling, 
           args.run_combined, args.run_all]):
    args.run_all = True

os.makedirs(args.output_dir, exist_ok=True)

# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# Custom dataset class
class HerbariumDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Get species names directly from folder names
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []
        self.sample_species = []  # Track species name for each sample
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    # Use original filename without extension as sample ID
                    img_id = os.path.splitext(img_name)[0]
                    self.samples.append((img_path, self.class_to_idx[class_name]))
                    self.sample_species.append(img_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def get_sample_info(self, idx):
        """Get detailed information about a sample"""
        img_path, label = self.samples[idx]
        species_id = self.sample_species[idx]
        species_name = self.classes[label]
        return {
            'idx': idx,
            'path': img_path,
            'label': label,
            'species_name': species_name,
            'species_id': species_id
        }

# Dataset analysis and utilities
def analyze_dataset_distribution(dataset):
    """Analyze class distribution and calculate statistics"""
    class_counts = {}
    for _, label in dataset.samples:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    counts_array = np.array(list(class_counts.values()))
    total_samples = counts_array.sum()
    imbalance_ratio = counts_array.max() / counts_array.min()
    
    print(f"\n=== Dataset Statistics ===")
    print(f"Total samples: {total_samples}")
    print(f"Number of classes: {len(class_counts)}")
    print(f"Class distribution: min={counts_array.min()}, max={counts_array.max()}, mean={counts_array.mean():.1f}")
    print(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}")
    
    return class_counts

def calculate_class_weights(dataset_subset, method='effective_num'):
    class_counts = {}
    for idx in range(len(dataset_subset)):
        _, label = dataset_subset[idx]
        class_counts[label] = class_counts.get(label, 0) + 1
    
    if hasattr(dataset_subset, 'dataset'):
        n_classes = len(dataset_subset.dataset.classes)
    else:
        n_classes = len(dataset_subset.classes)
    
    total = sum(class_counts.values())
    weights = torch.zeros(n_classes)
    
    for class_idx, count in class_counts.items():
        if method == 'effective_num':
            beta = 0.9999
            effective_num = (1 - beta**count) / (1 - beta)
            weights[class_idx] = (1 - beta) / effective_num
        else:  # 'inverse'
            weights[class_idx] = total / count
    
    weights = weights / weights.mean()  # Normalize
    return weights

def create_balanced_sampler(subset_dataset, power=2):
    labels = []
    for idx in range(len(subset_dataset)):
        _, label = subset_dataset[idx]
        labels.append(label)
    
    class_counts = Counter(labels)
    max_count = max(class_counts.values())
    
    sample_weights = []
    for label in labels:
        weight = (max_count / class_counts[label]) ** power
        sample_weights.append(weight)
    
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(labels) * 3,  # Triple the data
        replacement=True
    )

# Transforms
def get_transforms():
    return {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

def create_stratified_split(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Create stratified train/val/test split"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Get all labels
    labels = [sample[1] for sample in dataset.samples]
    indices = list(range(len(dataset)))
    
    # First split: separate test set
    train_val_indices, test_indices, train_val_labels, test_labels = train_test_split(
        indices, labels, test_size=test_ratio, stratify=labels, random_state=42
    )
    
    # Second split: separate train and validation from remaining data
    val_size = val_ratio / (train_ratio + val_ratio)  # Adjust val_ratio for remaining data
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_size, stratify=train_val_labels, random_state=42
    )
    
    return train_indices, val_indices, test_indices

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device,
                num_epochs, experiment_name):
    model.to(device)
    best_model_wts = model.state_dict()
    best_acc = 0.0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    print(f"\n=== Training {experiment_name} for {num_epochs} epochs ===")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss, running_corrects, total = 0.0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total += labels.size(0)

        train_losses.append(running_loss / total)
        train_accs.append(running_corrects / total)

        # Validation phase
        model.eval()
        val_running_loss, val_running_corrects, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels).item()
                val_total += labels.size(0)

        val_losses.append(val_running_loss / val_total)
        val_accs.append(val_running_corrects / val_total)
        
        if scheduler:
            scheduler.step(val_losses[-1])

        if val_accs[-1] > best_acc:
            best_acc = val_accs[-1]
            best_model_wts = model.state_dict()

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_losses[-1]:.4f}, "
              f"Train Acc: {train_accs[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, "
              f"Val Acc: {val_accs[-1]:.4f}")

    model.load_state_dict(best_model_wts)
    print(f"Best Val Accuracy: {best_acc:.4f}")

    return model, best_acc, train_losses, val_losses, train_accs, val_accs

# Evaluation function
def evaluate_model(model, test_loader, criterion, device, class_names, experiment_name):
    model.to(device)
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    running_loss, running_corrects, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    test_loss = running_loss / total
    test_acc = running_corrects / total
    
    # Comprehensive metrics for imbalanced data
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    
    # Per-class precision, recall, f1
    precision_macro = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)[0]
    recall_macro = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)[1]
    
    # Class distribution analysis
    class_counts = Counter(all_labels)
    total_samples = len(all_labels)
    
    # Define minority classes (less than 5% of total samples or fewer than 20 samples)
    minority_threshold = min(20, total_samples * 0.05)
    minority_classes = [cls for cls, count in class_counts.items() if count < minority_threshold]
    
    # Calculate minority class performance
    if minority_classes:
        minority_mask = [label in minority_classes for label in all_labels]
        minority_labels = [all_labels[i] for i, is_min in enumerate(minority_mask) if is_min]
        minority_preds = [all_preds[i] for i, is_min in enumerate(minority_mask) if is_min]
        
        if minority_labels:
            minority_f1 = f1_score(minority_labels, minority_preds, average='macro', zero_division=0)
            minority_precision = precision_recall_fscore_support(minority_labels, minority_preds, average='macro', zero_division=0)[0]
            minority_recall = precision_recall_fscore_support(minority_labels, minority_preds, average='macro', zero_division=0)[1]
        else:
            minority_f1 = minority_precision = minority_recall = 0.0
    else:
        minority_f1 = minority_precision = minority_recall = 0.0
    
    # Calculate per-class metrics
    per_class_report = classification_report(all_labels, all_preds, target_names=class_names, 
                                           output_dict=True, zero_division=0)
    
    # Calculate Cohen's Kappa (agreement measure)
    from sklearn.metrics import cohen_kappa_score
    kappa = cohen_kappa_score(all_labels, all_preds)
    
    print(f"\n📊 {experiment_name} Performance:")
    print(f"  🎯 Accuracy: {test_acc:.4f}")
    print(f"  ⚖️  Balanced Accuracy: {balanced_acc:.4f}")
    print(f"  📈 Macro F1: {macro_f1:.4f}")
    print(f"  🔍 Micro F1: {micro_f1:.4f}")
    print(f"  🤝 Cohen's Kappa: {kappa:.4f}")
    print(f"  🔴 Minority F1: {minority_f1:.4f}")

    return {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'balanced_acc': balanced_acc,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'minority_f1': minority_f1,
        'minority_precision': minority_precision,
        'minority_recall': minority_recall,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'kappa': kappa,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'per_class_report': per_class_report
    }

# Main experiment function
def run_experiment(train_dataset, val_dataset, test_dataset, device, output_dir, 
                  use_weighted_loss=False, use_balanced_sampling=False, use_focal_loss=False, 
                  experiment_name="", epochs_list=[5, 10, 15, 20]):
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"Weighted Loss: {use_weighted_loss}")
    print(f"Balanced Sampling: {use_balanced_sampling}")
    print(f"Focal Loss: {use_focal_loss}")
    print(f"Testing epochs: {epochs_list}")
    print(f"{'='*70}")
    
    transforms_dict = get_transforms()
    
    # Set transforms
    train_dataset.dataset.transform = transforms_dict['train']
    val_dataset.dataset.transform = transforms_dict['test']
    test_dataset.dataset.transform = transforms_dict['test']
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Store results for each epoch
    epoch_results = {}
    
    for num_epochs in epochs_list:
        print(f"\n--- Testing {num_epochs} epochs ---")
        
        # Data loaders
        if use_balanced_sampling:
            train_sampler = create_balanced_sampler(train_dataset, power=2)
            train_loader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler, num_workers=2)
        else:
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)

        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
        
        # Model setup
        model = models.resnet50(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(model.fc.in_features, len(train_dataset.dataset.classes))
        
        # Criterion setup
        if use_focal_loss:
            if use_weighted_loss:
                class_weights = calculate_class_weights(train_dataset).to(device)
                criterion = FocalLoss(alpha=class_weights, gamma=2.0)
            else:
                criterion = FocalLoss(alpha=None, gamma=2.0)
        elif use_weighted_loss:
            class_weights = calculate_class_weights(train_dataset).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
        
        # Train model
        exp_name_with_epochs = f"{experiment_name}_{num_epochs}epochs"
        model, best_val_acc, train_losses, val_losses, train_accs, val_accs = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler, device,
            num_epochs, exp_name_with_epochs
        )
        
        # Evaluate model
        eval_results = evaluate_model(
            model, test_loader, criterion, device, train_dataset.dataset.classes, exp_name_with_epochs
        )
        
        # Store results
        epoch_results[num_epochs] = {
            'val_acc': best_val_acc,
            'training_curves': {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs
            },
            **eval_results
        }
    
    return epoch_results

# Plotting functions
def plot_epoch_comparison(all_results, output_dir):
    """Plot comparison across different epochs for all experiments"""
    
    # Metrics to compare - replaced weighted_f1 with kappa
    metrics = ['test_acc', 'balanced_acc', 'macro_f1', 'kappa', 'minority_f1']
    metric_names = ['Test Accuracy', 'Balanced Accuracy', 'Macro F1', "Cohen's Kappa", 'Minority F1']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    experiments = list(all_results.keys())
    epochs_list = list(all_results[experiments[0]].keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(experiments)))
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        if i < len(axes):
            for j, exp_name in enumerate(experiments):
                values = [all_results[exp_name][epoch][metric] for epoch in epochs_list]
                axes[i].plot(epochs_list, values, marker='o', label=exp_name, 
                           color=colors[j], linewidth=2, markersize=6)
            
            axes[i].set_title(metric_name, fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Epochs')
            axes[i].set_ylabel(metric_name)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xticks(epochs_list)
    
    # Remove empty subplot
    if len(metrics) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'epoch_comparison_all_metrics.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_curves_by_epochs(all_results, output_dir):
    """Plot training curves for each experiment, grouped by epochs"""
    
    experiments = list(all_results.keys())
    epochs_list = list(all_results[experiments[0]].keys())
    
    for num_epochs in epochs_list:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        colors = plt.cm.Set3(np.linspace(0, 1, len(experiments)))
        
        for i, exp_name in enumerate(experiments):
            curves = all_results[exp_name][num_epochs]['training_curves']
            epochs_range = range(1, num_epochs + 1)
            
            # Training Loss
            axes[0,0].plot(epochs_range, curves['train_losses'], 
                          label=f'{exp_name}', color=colors[i], linewidth=2)
            
            # Validation Loss
            axes[0,1].plot(epochs_range, curves['val_losses'], 
                          label=f'{exp_name}', color=colors[i], linewidth=2)
            
            # Training Accuracy
            axes[1,0].plot(epochs_range, curves['train_accs'], 
                          label=f'{exp_name}', color=colors[i], linewidth=2)
            
            # Validation Accuracy
            axes[1,1].plot(epochs_range, curves['val_accs'], 
                          label=f'{exp_name}', color=colors[i], linewidth=2)
        
        axes[0,0].set_title('Training Loss')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        axes[0,1].set_title('Validation Loss')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Loss')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        axes[1,0].set_title('Training Accuracy')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('Accuracy')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        axes[1,1].set_title('Validation Accuracy')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Accuracy')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Training Curves - {num_epochs} Epochs', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'training_curves_{num_epochs}_epochs.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

def save_results(all_results, output_dir):
    """Save comprehensive results with proper JSON serialization"""
    
    # Summary results
    summary_data = []
    detailed_data = []
    
    for exp_name, epoch_results in all_results.items():
        for num_epochs, results in epoch_results.items():
            summary_data.append({
                'Experiment': exp_name,
                'Epochs': num_epochs,
                'Val_Acc': float(results['val_acc']),
                'Test_Acc': float(results['test_acc']),
                'Balanced_Acc': float(results['balanced_acc']),
                'Macro_F1': float(results['macro_f1']),
                'Micro_F1': float(results['micro_f1']),
                'Minority_F1': float(results['minority_f1']),
                'Minority_Precision': float(results['minority_precision']),
                'Minority_Recall': float(results['minority_recall']),
                'Precision_Macro': float(results['precision_macro']),
                'Recall_Macro': float(results['recall_macro']),
                'Kappa': float(results['kappa'])
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'experiment_summary_results.csv'), index=False)
    
    # Best results for each experiment
    best_results = []
    for exp_name, epoch_results in all_results.items():
        # Find best epoch based on macro F1
        best_epoch = max(epoch_results.items(), key=lambda x: x[1]['macro_f1'])
        best_results.append({
            'Experiment': exp_name,
            'Best_Epoch': int(best_epoch[0]),
            'Best_Macro_F1': float(best_epoch[1]['macro_f1']),
            'Best_Test_Acc': float(best_epoch[1]['test_acc']),
            'Best_Balanced_Acc': float(best_epoch[1]['balanced_acc']),
            'Best_Minority_F1': float(best_epoch[1]['minority_f1']),
            'Best_Kappa': float(best_epoch[1]['kappa'])
        })
    
    best_df = pd.DataFrame(best_results)
    best_df.to_csv(os.path.join(output_dir, 'best_results_per_experiment.csv'), index=False)
    
    # Helper function to convert numpy types to Python native types
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    # Save complete results as JSON with proper serialization
    json_results = {}
    for exp_name, epoch_results in all_results.items():
        json_results[exp_name] = {}
        for num_epochs, results in epoch_results.items():
            json_results[exp_name][str(num_epochs)] = {}  # Convert epoch to string key
            for key, value in results.items():
                # Skip non-serializable complex objects but keep important metrics
                if key in ['predictions', 'labels', 'probabilities']:
                    # Convert numpy arrays to lists
                    json_results[exp_name][str(num_epochs)][key] = convert_to_serializable(value)
                elif key == 'per_class_report':
                    # Skip per_class_report as it's complex and already saved elsewhere
                    continue
                elif key == 'training_curves':
                    # Convert training curves
                    curves = {}
                    for curve_key, curve_value in value.items():
                        curves[curve_key] = convert_to_serializable(curve_value)
                    json_results[exp_name][str(num_epochs)][key] = curves
                else:
                    # Convert other values
                    json_results[exp_name][str(num_epochs)][key] = convert_to_serializable(value)
    
    # Save JSON with error handling
    try:
        with open(os.path.join(output_dir, 'complete_results.json'), 'w') as f:
            json.dump(json_results, f, indent=2)
        print("✅ Complete results saved to JSON successfully")
    except Exception as e:
        print(f"⚠️ Warning: Could not save complete JSON results: {str(e)}")
        # Save a simplified version without predictions/labels/probabilities
        simplified_results = {}
        for exp_name, epoch_results in all_results.items():
            simplified_results[exp_name] = {}
            for num_epochs, results in epoch_results.items():
                simplified_results[exp_name][str(num_epochs)] = {
                    'val_acc': float(results['val_acc']),
                    'test_acc': float(results['test_acc']),
                    'balanced_acc': float(results['balanced_acc']),
                    'macro_f1': float(results['macro_f1']),
                    'micro_f1': float(results['micro_f1']),
                    'minority_f1': float(results['minority_f1']),
                    'minority_precision': float(results['minority_precision']),
                    'minority_recall': float(results['minority_recall']),
                    'precision_macro': float(results['precision_macro']),
                    'recall_macro': float(results['recall_macro']),
                    'kappa': float(results['kappa']),
                    'training_curves': {
                        'train_losses': [float(x) for x in results['training_curves']['train_losses']],
                        'val_losses': [float(x) for x in results['training_curves']['val_losses']],
                        'train_accs': [float(x) for x in results['training_curves']['train_accs']],
                        'val_accs': [float(x) for x in results['training_curves']['val_accs']]
                    }
                }
        
        with open(os.path.join(output_dir, 'simplified_results.json'), 'w') as f:
            json.dump(simplified_results, f, indent=2)
        print("✅ Simplified results saved to JSON successfully")
    
    print(f"📊 Results saved to {output_dir}")
    print(f"   - experiment_summary_results.csv")
    print(f"   - best_results_per_experiment.csv") 
    print(f"   - complete_results.json (or simplified_results.json)")

def generate_report(all_results, full_dataset, output_dir):
    """Generate comprehensive report"""
    report_path = os.path.join(output_dir, 'initial_experiment_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("Initial Experiment Report\n")
        f.write("=" * 60 + "\n\n")
        
        # Dataset info
        f.write("Dataset Information:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total samples: {len(full_dataset)}\n")
        f.write(f"Number of classes: {len(full_dataset.classes)}\n")
        
        # Class distribution
        class_counts = {}
        for _, label in full_dataset.samples:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        counts_array = np.array(list(class_counts.values()))
        f.write(f"Class distribution: min={counts_array.min()}, max={counts_array.max()}, mean={counts_array.mean():.1f}\n")
        f.write(f"Imbalance ratio: {counts_array.max() / counts_array.min():.2f}\n\n")
        
        # Experiment settings
        f.write("Experiment Settings:\n")
        f.write("-" * 30 + "\n")
        f.write("Model: ResNet-50\n")
        f.write("Data split: 80% train, 10% val, 10% test\n")
        f.write("Batch size: 16\n")
        f.write("Learning rate: 0.0001\n")
        f.write("Optimizer: Adam\n")
        f.write("Scheduler: ReduceLROnPlateau\n\n")
        
        # Results summary
        f.write("Results Summary by Experiment and Epochs:\n")
        f.write("-" * 50 + "\n")
        
        for exp_name, epoch_results in all_results.items():
            f.write(f"\n{exp_name}:\n")
            f.write(f"{'Epochs':<8} {'Test Acc':<10} {'Macro F1':<10} {'Minority F1':<12} {'Kappa':<10}\n")
            f.write("-" * 55 + "\n")
            
            for num_epochs, results in epoch_results.items():
                f.write(f"{num_epochs:<8} {results['test_acc']:<10.4f} {results['macro_f1']:<10.4f} "
                       f"{results['minority_f1']:<12.4f} {results['kappa']:<10.4f}\n")
        
        # Best results analysis
        f.write(f"\nBest Results Analysis:\n")
        f.write("-" * 30 + "\n")
        
        # Find best overall results
        best_macro_f1 = 0
        best_test_acc = 0
        best_minority_f1 = 0
        best_kappa = 0
        best_macro_exp = ""
        best_acc_exp = ""
        best_minority_exp = ""
        best_kappa_exp = ""
        
        for exp_name, epoch_results in all_results.items():
            for num_epochs, results in epoch_results.items():
                if results['macro_f1'] > best_macro_f1:
                    best_macro_f1 = results['macro_f1']
                    best_macro_exp = f"{exp_name} ({num_epochs} epochs)"
                
                if results['test_acc'] > best_test_acc:
                    best_test_acc = results['test_acc']
                    best_acc_exp = f"{exp_name} ({num_epochs} epochs)"
                
                if results['minority_f1'] > best_minority_f1:
                    best_minority_f1 = results['minority_f1']
                    best_minority_exp = f"{exp_name} ({num_epochs} epochs)"
                
                if results['kappa'] > best_kappa:
                    best_kappa = results['kappa']
                    best_kappa_exp = f"{exp_name} ({num_epochs} epochs)"
        
        f.write(f"Best Macro F1: {best_macro_exp} ({best_macro_f1:.4f})\n")
        f.write(f"Best Test Accuracy: {best_acc_exp} ({best_test_acc:.4f})\n")
        f.write(f"Best Minority F1: {best_minority_exp} ({best_minority_f1:.4f})\n")
        f.write(f"Best Cohen's Kappa: {best_kappa_exp} ({best_kappa:.4f})\n")
        
        # Epoch analysis
        f.write(f"\nEpoch Analysis:\n")
        f.write("-" * 20 + "\n")
        
        # Average performance by epochs across all experiments
        epochs_list = list(all_results[list(all_results.keys())[0]].keys())
        
        for num_epochs in epochs_list:
            macro_f1_scores = [all_results[exp][num_epochs]['macro_f1'] for exp in all_results.keys()]
            test_acc_scores = [all_results[exp][num_epochs]['test_acc'] for exp in all_results.keys()]
            kappa_scores = [all_results[exp][num_epochs]['kappa'] for exp in all_results.keys()]
            
            f.write(f"{num_epochs} epochs:\n")
            f.write(f"  Average Macro F1: {np.mean(macro_f1_scores):.4f} ± {np.std(macro_f1_scores):.4f}\n")
            f.write(f"  Average Test Acc: {np.mean(test_acc_scores):.4f} ± {np.std(test_acc_scores):.4f}\n")
            f.write(f"  Average Kappa: {np.mean(kappa_scores):.4f} ± {np.std(kappa_scores):.4f}\n")
        
        # Method comparison
        f.write(f"\nMethod Comparison (Best Epoch for Each):\n")
        f.write("-" * 45 + "\n")
        
        for exp_name, epoch_results in all_results.items():
            # Find best epoch for this experiment based on macro F1
            best_epoch_result = max(epoch_results.items(), key=lambda x: x[1]['macro_f1'])
            best_epoch, best_result = best_epoch_result
            
            f.write(f"\n{exp_name} (Best at {best_epoch} epochs):\n")
            f.write(f"  Macro F1: {best_result['macro_f1']:.4f}\n")
            f.write(f"  Test Accuracy: {best_result['test_acc']:.4f}\n")
            f.write(f"  Balanced Accuracy: {best_result['balanced_acc']:.4f}\n")
            f.write(f"  Minority F1: {best_result['minority_f1']:.4f}\n")
            f.write(f"  Cohen's Kappa: {best_result['kappa']:.4f}\n")
        
        # Training efficiency analysis
        f.write(f"\nTraining Efficiency Analysis:\n")
        f.write("-" * 35 + "\n")
        
        # Find most efficient epoch (good performance with fewer epochs)
        efficiency_scores = []
        for exp_name, epoch_results in all_results.items():
            for num_epochs, results in epoch_results.items():
                efficiency = results['macro_f1'] / num_epochs  # Performance per epoch
                efficiency_scores.append((exp_name, num_epochs, efficiency, results['macro_f1']))
        
        efficiency_scores.sort(key=lambda x: x[2], reverse=True)
        
        f.write("Most efficient training (Performance/Epoch ratio):\n")
        for i, (exp_name, epochs, efficiency, macro_f1) in enumerate(efficiency_scores[:5]):
            f.write(f"  {i+1}. {exp_name} ({epochs} epochs): {efficiency:.6f} (F1: {macro_f1:.4f})\n")
        
        # Performance plateau analysis
        f.write(f"\nPerformance Plateau Analysis:\n")
        f.write("-" * 35 + "\n")
        
        for exp_name, epoch_results in all_results.items():
            f.write(f"\n{exp_name}:\n")
            epochs_sorted = sorted(epoch_results.items())
            prev_f1 = 0
            
            for epoch, results in epochs_sorted:
                improvement = results['macro_f1'] - prev_f1
                f.write(f"  {epoch} epochs: F1={results['macro_f1']:.4f} (Δ+{improvement:.4f})\n")
                prev_f1 = results['macro_f1']
        
        # Recommendations
        f.write(f"\nRecommendations:\n")
        f.write("-" * 20 + "\n")
        f.write(f"1. Best overall method: {best_macro_exp}\n")
        f.write(f"2. For minority classes: {best_minority_exp}\n")
        f.write(f"3. For highest accuracy: {best_acc_exp}\n")
        f.write(f"4. Best agreement (Kappa): {best_kappa_exp}\n")
        
        # Final summary and action items
        f.write(f"\nFinal Summary and Action Items:\n")
        f.write("-" * 40 + "\n")
        f.write("1. Production Deployment:\n")
        f.write(f"   - Use: {best_macro_exp}\n")
        f.write(f"   - Expected Macro F1: {best_macro_f1:.4f}\n")
        f.write(f"   - Expected Test Accuracy: {best_test_acc:.4f}\n")
        f.write(f"   - Expected Kappa: {best_kappa:.4f}\n\n")
        
        f.write("2. For Resource-Constrained Environments:\n")
        most_efficient = efficiency_scores[0]
        f.write(f"   - Use: {most_efficient[0]} ({most_efficient[1]} epochs)\n")
        f.write(f"   - Efficiency Score: {most_efficient[2]:.6f}\n")
        f.write(f"   - Macro F1: {most_efficient[3]:.4f}\n\n")
        
        f.write("3. For Minority Class Focus:\n")
        f.write(f"   - Use: {best_minority_exp}\n")
        f.write(f"   - Expected Minority F1: {best_minority_f1:.4f}\n\n")
        
        f.write("4. Training Duration Recommendations:\n")
        # Find optimal stopping point for each method
        for exp_name, epoch_results in all_results.items():
            epochs_sorted = sorted(epoch_results.items())
            best_f1 = max(results['macro_f1'] for _, results in epochs_sorted)
            
            # Find minimum epochs to achieve 95% of best performance
            threshold = best_f1 * 0.95
            min_epochs_95 = None
            for epoch, results in epochs_sorted:
                if results['macro_f1'] >= threshold:
                    min_epochs_95 = epoch
                    break
            
            f.write(f"   {exp_name}: Minimum {min_epochs_95} epochs for 95% of best performance\n")

def plot_efficiency_analysis(all_results, output_dir):
    """Plot efficiency analysis comparing performance vs training time"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    experiments = list(all_results.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(experiments)))
    
    # Plot 1: Performance vs Epochs
    for i, exp_name in enumerate(experiments):
        epochs_list = list(all_results[exp_name].keys())
        macro_f1_list = [all_results[exp_name][epoch]['macro_f1'] for epoch in epochs_list]
        
        ax1.plot(epochs_list, macro_f1_list, marker='o', label=exp_name, 
                color=colors[i], linewidth=2, markersize=6)
    
    ax1.set_xlabel('Training Epochs')
    ax1.set_ylabel('Macro F1 Score')
    ax1.set_title('Performance vs Training Epochs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Efficiency Score (Performance/Epoch)
    exp_names = []
    efficiency_scores = []
    epoch_labels = []
    
    for exp_name in experiments:
        for epoch in all_results[exp_name].keys():
            macro_f1 = all_results[exp_name][epoch]['macro_f1']
            efficiency = macro_f1 / epoch
            
            exp_names.append(exp_name)
            efficiency_scores.append(efficiency)
            epoch_labels.append(f"{epoch}e")
    
    # Create scatter plot
    for i, exp_name in enumerate(experiments):
        exp_efficiencies = []
        exp_epochs = []
        exp_labels = []
        
        for j, name in enumerate(exp_names):
            if name == exp_name:
                exp_efficiencies.append(efficiency_scores[j])
                exp_epochs.append(list(all_results[exp_name].keys())[j % len(all_results[exp_name])])
                exp_labels.append(epoch_labels[j])
        
        ax2.scatter(exp_epochs, exp_efficiencies, label=exp_name, 
                   color=colors[i], s=60, alpha=0.7)
        
        # Add epoch labels
        for x, y, label in zip(exp_epochs, exp_efficiencies, exp_labels):
            ax2.annotate(label, (x, y), xytext=(3, 3), textcoords='offset points', 
                        fontsize=8, alpha=0.8)
    
    ax2.set_xlabel('Training Epochs')
    ax2.set_ylabel('Efficiency Score (F1/Epoch)')
    ax2.set_title('Training Efficiency Analysis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'efficiency_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_plateau(all_results, output_dir):
    """Plot performance plateau analysis"""
    
    experiments = list(all_results.keys())
    n_experiments = len(experiments)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = plt.cm.Set3(np.linspace(0, 1, n_experiments))
    
    for i, exp_name in enumerate(experiments):
        if i < len(axes):
            epochs_list = sorted(all_results[exp_name].keys())
            macro_f1_list = [all_results[exp_name][epoch]['macro_f1'] for epoch in epochs_list]
            test_acc_list = [all_results[exp_name][epoch]['test_acc'] for epoch in epochs_list]
            minority_f1_list = [all_results[exp_name][epoch]['minority_f1'] for epoch in epochs_list]
            
            # Calculate improvements
            macro_improvements = [0] + [macro_f1_list[j] - macro_f1_list[j-1] for j in range(1, len(macro_f1_list))]
            
            # Plot performance curves
            ax = axes[i]
            ax2 = ax.twinx()
            
            # Performance lines
            line1 = ax.plot(epochs_list, macro_f1_list, 'o-', label='Macro F1', 
                           color='blue', linewidth=2, markersize=6)
            line2 = ax.plot(epochs_list, test_acc_list, 's-', label='Test Acc', 
                           color='green', linewidth=2, markersize=6)
            line3 = ax.plot(epochs_list, minority_f1_list, '^-', label='Minority F1', 
                           color='red', linewidth=2, markersize=6)
            
            # Improvement bars
            bars = ax2.bar(epochs_list, macro_improvements, alpha=0.3, color='orange', 
                          width=0.8, label='F1 Improvement')
            
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Performance Score')
            ax2.set_ylabel('F1 Improvement')
            ax.set_title(f'{exp_name}\nPerformance vs Improvement')
            
            # Combine legends
            lines1 = line1 + line2 + line3
            labels1 = [l.get_label() for l in lines1]
            ax.legend(lines1, labels1, loc='upper left')
            ax2.legend([bars], ['F1 Δ'], loc='upper right')
            
            ax.grid(True, alpha=0.3)
            ax.set_xticks(epochs_list)
    
    # Remove empty subplots
    for i in range(len(experiments), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_plateau_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_table(all_results, output_dir):
    """Create a comprehensive summary table"""
    
    # Prepare data for summary table
    summary_data = []
    
    for exp_name, epoch_results in all_results.items():
        for num_epochs, results in epoch_results.items():
            summary_data.append({
                'Experiment': exp_name,
                'Epochs': num_epochs,
                'Test_Accuracy': results['test_acc'],
                'Balanced_Accuracy': results['balanced_acc'],
                'Macro_F1': results['macro_f1'],
                'Micro_F1': results['micro_f1'],
                'Minority_F1': results['minority_f1'],
                'Minority_Precision': results['minority_precision'],
                'Minority_Recall': results['minority_recall'],
                'Precision_Macro': results['precision_macro'],
                'Recall_Macro': results['recall_macro'],
                'Kappa': results['kappa'],
                'Efficiency_Score': results['macro_f1'] / num_epochs
            })
    
    df = pd.DataFrame(summary_data)
    
    # Create pivot table for better visualization
    pivot_metrics = ['Test_Accuracy', 'Macro_F1', 'Minority_F1', 'Kappa', 'Efficiency_Score']
    
    for metric in pivot_metrics:
        pivot_df = df.pivot(index='Experiment', columns='Epochs', values=metric)
        
        # Save pivot table
        pivot_df.to_csv(os.path.join(output_dir, f'{metric.lower()}_by_experiment_epochs.csv'))
        
        # Create heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_df, annot=True, fmt='.4f', cmap='viridis', 
                   cbar_kws={'label': metric.replace('_', ' ')})
        plt.title(f'{metric.replace("_", " ")} Heatmap: Experiments vs Epochs')
        plt.xlabel('Training Epochs')
        plt.ylabel('Experiment Method')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric.lower()}_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

def generate_final_recommendations(all_results, output_dir):
    """Generate final recommendations with detailed analysis"""
    
    recommendations_path = os.path.join(output_dir, 'final_recommendations.txt')
    
    # Analyze all results
    all_combinations = []
    for exp_name, epoch_results in all_results.items():
        for num_epochs, results in epoch_results.items():
            all_combinations.append({
                'method': exp_name,
                'epochs': num_epochs,
                'macro_f1': results['macro_f1'],
                'test_acc': results['test_acc'],
                'minority_f1': results['minority_f1'],
                'balanced_acc': results['balanced_acc'],
                'kappa': results['kappa'],
                'efficiency': results['macro_f1'] / num_epochs,
                'total_score': (results['macro_f1'] + results['test_acc'] + results['minority_f1'] + results['kappa']) / 4
            })
    
    with open(recommendations_path, 'w') as f:
        f.write("FINAL RECOMMENDATIONS AND DEPLOYMENT GUIDE\n")
        f.write("=" * 60 + "\n\n")
        
        # Best overall
        best_overall = max(all_combinations, key=lambda x: x['total_score'])
        f.write("1. BEST OVERALL CONFIGURATION\n")
        f.write("-" * 35 + "\n")
        f.write(f"Method: {best_overall['method']}\n")
        f.write(f"Optimal Epochs: {best_overall['epochs']}\n")
        f.write(f"Expected Performance:\n")
        f.write(f"  - Macro F1: {best_overall['macro_f1']:.4f}\n")
        f.write(f"  - Test Accuracy: {best_overall['test_acc']:.4f}\n")
        f.write(f"  - Minority F1: {best_overall['minority_f1']:.4f}\n")
        f.write(f"  - Balanced Accuracy: {best_overall['balanced_acc']:.4f}\n")
        f.write(f"  - Cohen's Kappa: {best_overall['kappa']:.4f}\n\n")
        
        # Most efficient
        most_efficient = max(all_combinations, key=lambda x: x['efficiency'])
        f.write("2. MOST EFFICIENT CONFIGURATION\n")
        f.write("-" * 38 + "\n")
        f.write(f"Method: {most_efficient['method']}\n")
        f.write(f"Optimal Epochs: {most_efficient['epochs']}\n")
        f.write(f"Efficiency Score: {most_efficient['efficiency']:.6f}\n")
        f.write(f"Expected Performance:\n")
        f.write(f"  - Macro F1: {most_efficient['macro_f1']:.4f}\n")
        f.write(f"  - Test Accuracy: {most_efficient['test_acc']:.4f}\n\n")
        
        # Best for minorities
        best_minority = max(all_combinations, key=lambda x: x['minority_f1'])
        f.write("3. BEST FOR MINORITY CLASSES\n")
        f.write("-" * 32 + "\n")
        f.write(f"Method: {best_minority['method']}\n")
        f.write(f"Optimal Epochs: {best_minority['epochs']}\n")
        f.write(f"Minority F1: {best_minority['minority_f1']:.4f}\n")
        f.write(f"Overall Macro F1: {best_minority['macro_f1']:.4f}\n\n")
        
        # Best for agreement
        best_kappa = max(all_combinations, key=lambda x: x['kappa'])
        f.write("4. BEST FOR INTER-RATER AGREEMENT\n")
        f.write("-" * 37 + "\n")
        f.write(f"Method: {best_kappa['method']}\n")
        f.write(f"Optimal Epochs: {best_kappa['epochs']}\n")
        f.write(f"Cohen's Kappa: {best_kappa['kappa']:.4f}\n")
        f.write(f"Overall Macro F1: {best_kappa['macro_f1']:.4f}\n\n")
        
        # Deployment scenarios
        f.write("5. DEPLOYMENT SCENARIOS\n")
        f.write("-" * 25 + "\n\n")
        
        f.write("Scenario A - Production Environment (Best Overall Performance):\n")
        f.write(f"  → Use: {best_overall['method']} with {best_overall['epochs']} epochs\n")
        f.write(f"  → Training Time: ~{best_overall['epochs'] * 2} minutes per fold\n")
        f.write(f"  → Expected F1: {best_overall['macro_f1']:.4f}\n")
        f.write(f"  → Expected Kappa: {best_overall['kappa']:.4f}\n\n")
        
        f.write("Scenario B - Resource-Constrained Environment:\n")
        f.write(f"  → Use: {most_efficient['method']} with {most_efficient['epochs']} epochs\n")
        f.write(f"  → Training Time: ~{most_efficient['epochs'] * 2} minutes per fold\n")
        f.write(f"  → Expected F1: {most_efficient['macro_f1']:.4f}\n\n")
        
        f.write("Scenario C - Rare Species Focus:\n")
        f.write(f"  → Use: {best_minority['method']} with {best_minority['epochs']} epochs\n")
        f.write(f"  → Minority Class F1: {best_minority['minority_f1']:.4f}\n")
        f.write(f"  → Overall F1: {best_minority['macro_f1']:.4f}\n\n")
        
        f.write("Scenario D - High Agreement Requirement:\n")
        f.write(f"  → Use: {best_kappa['method']} with {best_kappa['epochs']} epochs\n")
        f.write(f"  → Cohen's Kappa: {best_kappa['kappa']:.4f}\n")
        f.write(f"  → Overall F1: {best_kappa['macro_f1']:.4f}\n\n")
        
        # Implementation notes
        f.write("6. IMPLEMENTATION NOTES\n")
        f.write("-" * 25 + "\n")
        f.write("- Use early stopping based on validation loss plateau\n")
        f.write("- Monitor minority class performance during training\n")
        f.write("- Consider ensemble methods for critical applications\n")
        f.write("- Validate performance on new data before deployment\n")
        f.write("- Regular model retraining recommended as dataset grows\n")
        f.write("- Cohen's Kappa provides insight into classification reliability\n")

# Main function
def main():
    start_time = time.time()
    
    # Load dataset and analyze distribution
    full_dataset = HerbariumDataset(args.data_dir)
    class_counts = analyze_dataset_distribution(full_dataset)
    
    print(f"\nFound {len(full_dataset.classes)} species:")
    for i, species in enumerate(full_dataset.classes[:10]):  # Show first 10
        print(f"  {i+1}. {species}")
    if len(full_dataset.classes) > 10:
        print(f"  ... and {len(full_dataset.classes) - 10} more species")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create stratified split
    print(f"\nCreating stratified 80/10/10 split...")
    train_indices, val_indices, test_indices = create_stratified_split(full_dataset, 0.8, 0.1, 0.1)
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    print(f"Split sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Store all results
    all_results = {}
    
    # Define experiments to run
    experiments = [
        {'name': 'Baseline', 'weighted_loss': False, 'balanced_sampling': False, 'focal_loss': False},
        {'name': 'Weighted Loss', 'weighted_loss': True, 'balanced_sampling': False, 'focal_loss': False},
        {'name': 'Balanced Sampling', 'weighted_loss': False, 'balanced_sampling': True, 'focal_loss': False},
        {'name': 'Focal Loss', 'weighted_loss': False, 'balanced_sampling': False, 'focal_loss': True},
        {'name': 'Focal + Weighted', 'weighted_loss': True, 'balanced_sampling': False, 'focal_loss': True},
        {'name': 'All Combined', 'weighted_loss': True, 'balanced_sampling': True, 'focal_loss': True},
    ]

    # Filter experiments based on arguments
    if args.run_baseline:
        experiments = [exp for exp in experiments if exp['name'] == 'Baseline']
    elif args.run_weighted_loss:
        experiments = [exp for exp in experiments if 'Weighted' in exp['name']]
    elif args.run_balanced_sampling:
        experiments = [exp for exp in experiments if 'Sampling' in exp['name']]
    elif args.run_combined:
        experiments = [exp for exp in experiments if 'Combined' in exp['name']]

    print(f"\nRunning {len(experiments)} experiments with epochs: {args.test_epochs}")
    total_runs = len(experiments) * len(args.test_epochs)
    print(f"Total runs: {total_runs}")

    # Run experiments
    for exp in experiments:
        exp_start_time = time.time()
        
        all_results[exp['name']] = run_experiment(
            train_dataset, val_dataset, test_dataset, device, args.output_dir,
            use_weighted_loss=exp['weighted_loss'],
            use_balanced_sampling=exp['balanced_sampling'],
            use_focal_loss=exp.get('focal_loss', False),
            experiment_name=exp['name'],
            epochs_list=args.test_epochs
        )
        
        exp_duration = time.time() - exp_start_time
        print(f"Experiment '{exp['name']}' completed in {exp_duration/60:.1f} minutes")

    # Results analysis
    print(f"\n{'='*80}")
    print("COMPREHENSIVE RESULTS ANALYSIS")
    print(f"{'='*80}")

    # Find best results across all experiments and epochs
    print("\n🏆 BEST RESULTS ACROSS ALL EXPERIMENTS:")
    
    best_results = []
    for exp_name, epoch_results in all_results.items():
        for num_epochs, results in epoch_results.items():
            best_results.append({
                'experiment': exp_name,
                'epochs': num_epochs,
                'macro_f1': results['macro_f1'],
                'test_acc': results['test_acc'],
                'minority_f1': results['minority_f1'],
                'balanced_acc': results['balanced_acc'],
                'kappa': results['kappa']
            })
    
    # Sort by different metrics
    print("\n🎯 TOP 5 BY MACRO F1:")
    sorted_by_macro = sorted(best_results, key=lambda x: x['macro_f1'], reverse=True)[:5]
    for i, result in enumerate(sorted_by_macro):
        print(f"  {i+1}. {result['experiment']} ({result['epochs']} epochs): {result['macro_f1']:.4f}")
    
    print("\n📊 TOP 5 BY TEST ACCURACY:")
    sorted_by_acc = sorted(best_results, key=lambda x: x['test_acc'], reverse=True)[:5]
    for i, result in enumerate(sorted_by_acc):
        print(f"  {i+1}. {result['experiment']} ({result['epochs']} epochs): {result['test_acc']:.4f}")
    
    print("\n🔴 TOP 5 BY MINORITY F1:")
    sorted_by_minority = sorted(best_results, key=lambda x: x['minority_f1'], reverse=True)[:5]
    for i, result in enumerate(sorted_by_minority):
        print(f"  {i+1}. {result['experiment']} ({result['epochs']} epochs): {result['minority_f1']:.4f}")
    
    print("\n🤝 TOP 5 BY COHEN'S KAPPA:")
    sorted_by_kappa = sorted(best_results, key=lambda x: x['kappa'], reverse=True)[:5]
    for i, result in enumerate(sorted_by_kappa):
        print(f"  {i+1}. {result['experiment']} ({result['epochs']} epochs): {result['kappa']:.4f}")

    # Epoch analysis
    print(f"\n📈 EPOCH ANALYSIS:")
    epochs_performance = {}
    for epoch in args.test_epochs:
        epoch_scores = []
        kappa_scores = []
        for exp_name, epoch_results in all_results.items():
            epoch_scores.append(epoch_results[epoch]['macro_f1'])
            kappa_scores.append(epoch_results[epoch]['kappa'])
        epochs_performance[epoch] = {
            'mean': np.mean(epoch_scores),
            'std': np.std(epoch_scores),
            'max': max(epoch_scores),
            'kappa_mean': np.mean(kappa_scores),
            'kappa_std': np.std(kappa_scores),
            'kappa_max': max(kappa_scores)
        }
    
    print(f"{'Epochs':<8} {'Mean F1':<12} {'Std F1':<12} {'Max F1':<12} {'Mean Kappa':<12} {'Max Kappa':<12}")
    print("-" * 80)
    for epoch, perf in epochs_performance.items():
        print(f"{epoch:<8} {perf['mean']:<12.4f} {perf['std']:<12.4f} {perf['max']:<12.4f} "
              f"{perf['kappa_mean']:<12.4f} {perf['kappa_max']:<12.4f}")
    
    # Method analysis
    print(f"\n🔬 METHOD ANALYSIS (Best Epoch for Each):")
    method_performance = {}
    for exp_name, epoch_results in all_results.items():
        best_epoch_result = max(epoch_results.items(), key=lambda x: x[1]['macro_f1'])
        method_performance[exp_name] = {
            'best_epoch': best_epoch_result[0],
            'macro_f1': best_epoch_result[1]['macro_f1'],
            'test_acc': best_epoch_result[1]['test_acc'],
            'minority_f1': best_epoch_result[1]['minority_f1'],
            'kappa': best_epoch_result[1]['kappa']
        }
    
    print(f"{'Method':<20} {'Best Epoch':<12} {'Macro F1':<12} {'Test Acc':<12} {'Minority F1':<12} {'Kappa':<12}")
    print("-" * 100)
    for method, perf in method_performance.items():
        print(f"{method:<20} {perf['best_epoch']:<12} {perf['macro_f1']:<12.4f} "
              f"{perf['test_acc']:<12.4f} {perf['minority_f1']:<12.4f} {perf['kappa']:<12.4f}")

    # Generate comprehensive visualizations
    print(f"\nGenerating comprehensive visualizations...")
    plot_epoch_comparison(all_results, args.output_dir)
    plot_training_curves_by_epochs(all_results, args.output_dir)
    plot_efficiency_analysis(all_results, args.output_dir)
    plot_performance_plateau(all_results, args.output_dir)
    
    # Create summary tables and heatmaps
    print(f"Creating summary tables and heatmaps...")
    create_summary_table(all_results, args.output_dir)
    
    # Save results
    save_results(all_results, args.output_dir)
    
    # Generate reports
    generate_report(all_results, full_dataset, args.output_dir)
    generate_final_recommendations(all_results, args.output_dir)
    
    # Final recommendations
    print(f"\n{'='*80}")
    print("🎖️ FINAL RECOMMENDATIONS")
    print(f"{'='*80}")
    
    best_overall = max(best_results, key=lambda x: x['macro_f1'])
    best_for_minority = max(best_results, key=lambda x: x['minority_f1'])
    best_for_accuracy = max(best_results, key=lambda x: x['test_acc'])
    best_for_kappa = max(best_results, key=lambda x: x['kappa'])
    
    print(f"1. 🥇 BEST OVERALL (Macro F1): {best_overall['experiment']} with {best_overall['epochs']} epochs")
    print(f"   - Macro F1: {best_overall['macro_f1']:.4f}")
    print(f"   - Test Accuracy: {best_overall['test_acc']:.4f}")
    print(f"   - Minority F1: {best_overall['minority_f1']:.4f}")
    print(f"   - Cohen's Kappa: {best_overall['kappa']:.4f}")
    
    print(f"\n2. 🔴 BEST FOR MINORITY CLASSES: {best_for_minority['experiment']} with {best_for_minority['epochs']} epochs")
    print(f"   - Minority F1: {best_for_minority['minority_f1']:.4f}")
    print(f"   - Cohen's Kappa: {best_for_minority['kappa']:.4f}")
    
    print(f"\n3. 📊 BEST FOR ACCURACY: {best_for_accuracy['experiment']} with {best_for_accuracy['epochs']} epochs")
    print(f"   - Test Accuracy: {best_for_accuracy['test_acc']:.4f}")
    print(f"   - Cohen's Kappa: {best_for_accuracy['kappa']:.4f}")
    
    print(f"\n4. 🤝 BEST FOR AGREEMENT (Kappa): {best_for_kappa['experiment']} with {best_for_kappa['epochs']} epochs")
    print(f"   - Cohen's Kappa: {best_for_kappa['kappa']:.4f}")
    print(f"   - Macro F1: {best_for_kappa['macro_f1']:.4f}")
    
    # Optimal epoch recommendation
    best_epoch = max(epochs_performance.items(), key=lambda x: x[1]['mean'])[0]
    print(f"\n5. 📈 OPTIMAL EPOCH COUNT: {best_epoch} epochs")
    print(f"   - Average Macro F1 across all methods: {epochs_performance[best_epoch]['mean']:.4f}")
    print(f"   - Average Kappa across all methods: {epochs_performance[best_epoch]['kappa_mean']:.4f}")
    
    # Training efficiency insights
    efficiency_analysis = []
    for exp_name, epoch_results in all_results.items():
        for num_epochs, results in epoch_results.items():
            efficiency = results['macro_f1'] / num_epochs
            efficiency_analysis.append({
                'method': exp_name,
                'epochs': num_epochs,
                'efficiency': efficiency,
                'macro_f1': results['macro_f1'],
                'kappa': results['kappa']
            })
    
    most_efficient = max(efficiency_analysis, key=lambda x: x['efficiency'])
    print(f"\n6. ⚡ MOST EFFICIENT: {most_efficient['method']} with {most_efficient['epochs']} epochs")
    print(f"   - Efficiency Score: {most_efficient['efficiency']:.6f}")
    print(f"   - Macro F1: {most_efficient['macro_f1']:.4f}")
    print(f"   - Cohen's Kappa: {most_efficient['kappa']:.4f}")
    
    # Performance plateau insights
    print(f"\n7. 📈 PERFORMANCE PLATEAU INSIGHTS:")
    for exp_name, epoch_results in all_results.items():
        epochs_sorted = sorted(epoch_results.items())
        
        # Find where improvement becomes minimal (<0.01)
        plateau_epoch = None
        for i in range(1, len(epochs_sorted)):
            prev_f1 = epochs_sorted[i-1][1]['macro_f1']
            curr_f1 = epochs_sorted[i][1]['macro_f1']
            improvement = curr_f1 - prev_f1
            
            if improvement < 0.01:  # Less than 1% improvement
                plateau_epoch = epochs_sorted[i][0]
                break
        
        if plateau_epoch:
            print(f"   - {exp_name}: Performance plateaus after {plateau_epoch} epochs")
        else:
            print(f"   - {exp_name}: Still improving at {epochs_sorted[-1][0]} epochs")
    
    total_duration = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"INITIAL EXPERIMENT COMPLETED")
    print(f"{'='*80}")
    print(f"Total execution time: {total_duration/60:.1f} minutes")
    print(f"Results saved to: {args.output_dir}")
    print(f"\nGenerated files:")
    print(f"├── 📊 SUMMARY RESULTS:")
    print(f"│   ├── experiment_summary_results.csv")
    print(f"│   ├── best_results_per_experiment.csv")
    print(f"│   ├── complete_results.json")
    print(f"│   └── [metric]_by_experiment_epochs.csv (pivot tables)")
    print(f"├── 📈 VISUALIZATIONS:")
    print(f"│   ├── epoch_comparison_all_metrics.png (now with Cohen's Kappa)")
    print(f"│   ├── training_curves_[X]_epochs.png (for each epoch)")
    print(f"│   ├── efficiency_analysis.png")
    print(f"│   ├── performance_plateau_analysis.png")
    print(f"│   └── [metric]_heatmap.png (heatmaps for each metric)")
    print(f"├── 📋 REPORTS:")
    print(f"│   ├── initial_experiment_report.txt")
    print(f"│   └── final_recommendations.txt")
    print(f"└── 🎯 KEY INSIGHTS:")
    print(f"    ├── Training efficiency analysis completed")
    print(f"    ├── Performance plateau detection finished")
    print(f"    ├── Cohen's Kappa analysis included")
    print(f"    └── Deployment recommendations generated")
    
    # Create final summary for quick reference
    summary_path = os.path.join(args.output_dir, 'quick_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("QUICK SUMMARY - INITIAL EXPERIMENT RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("🏆 BEST CONFIGURATIONS:\n")
        f.write("-" * 25 + "\n")
        f.write(f"Overall Best: {best_overall['experiment']} ({best_overall['epochs']} epochs)\n")
        f.write(f"  → Macro F1: {best_overall['macro_f1']:.4f}\n")
        f.write(f"  → Test Acc: {best_overall['test_acc']:.4f}\n")
        f.write(f"  → Kappa: {best_overall['kappa']:.4f}\n\n")
        
        f.write(f"Most Efficient: {most_efficient['method']} ({most_efficient['epochs']} epochs)\n")
        f.write(f"  → Efficiency: {most_efficient['efficiency']:.6f}\n")
        f.write(f"  → Macro F1: {most_efficient['macro_f1']:.4f}\n")
        f.write(f"  → Kappa: {most_efficient['kappa']:.4f}\n\n")
        
        f.write(f"Best for Minorities: {best_for_minority['experiment']} ({best_for_minority['epochs']} epochs)\n")
        f.write(f"  → Minority F1: {best_for_minority['minority_f1']:.4f}\n")
        f.write(f"  → Kappa: {best_for_minority['kappa']:.4f}\n\n")
        
        f.write(f"Best Agreement: {best_for_kappa['experiment']} ({best_for_kappa['epochs']} epochs)\n")
        f.write(f"  → Kappa: {best_for_kappa['kappa']:.4f}\n")
        f.write(f"  → Macro F1: {best_for_kappa['macro_f1']:.4f}\n\n")
        
        f.write("📈 EPOCH RECOMMENDATIONS:\n")
        f.write("-" * 25 + "\n")
        f.write(f"Optimal epoch count: {best_epoch} epochs\n")
        f.write(f"Average performance: {epochs_performance[best_epoch]['mean']:.4f}\n")
        f.write(f"Average Kappa: {epochs_performance[best_epoch]['kappa_mean']:.4f}\n\n")
        
        f.write("🎯 DEPLOYMENT RECOMMENDATION:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Use: {best_overall['experiment']} with {best_overall['epochs']} epochs\n")
        f.write(f"Expected Macro F1: {best_overall['macro_f1']:.4f}\n")
        f.write(f"Expected Test Accuracy: {best_overall['test_acc']:.4f}\n")
        f.write(f"Expected Cohen's Kappa: {best_overall['kappa']:.4f}\n")
    
    print(f"\n✅ STUDY COMPLETE! Check {args.output_dir}/quick_summary.txt for key results.")
    print(f"📝 Key Change: Weighted F1 has been replaced with Cohen's Kappa in all visualizations and reports.")

if __name__ == '__main__':
    main()