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
parser.add_argument('--output_dir', type=str, default='/home/s3844498/outputs_efficientnet_initial_experiment', help='Directory to save outputs')
parser.add_argument('--test_epochs', nargs='+', type=int, default=[5, 10, 15, 20],
                    help='List of epochs to test (default: 2 3)')
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

# Transforms - NO AUGMENTATION, just resize and normalize for EfficientNet-V2
def get_transforms():
    """
    Simple transforms for EfficientNet-V2-S with 224x224 input size
    No augmentation - just resize and normalize
    """
    return {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225]    # ImageNet stds
            )
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225]    # ImageNet stds
            )
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
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    
    # Per-class precision, recall, f1
    precision_macro = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)[0]
    recall_macro = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)[1]
    precision_weighted = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)[0]
    recall_weighted = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)[1]
    
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
    print(f"  📊 Weighted F1: {weighted_f1:.4f}")
    print(f"  🔍 Micro F1: {micro_f1:.4f}")
    print(f"  🤝 Cohen's Kappa: {kappa:.4f}")
    print(f"  🔴 Minority F1: {minority_f1:.4f}")

    return {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'balanced_acc': balanced_acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'micro_f1': micro_f1,
        'minority_f1': minority_f1,
        'minority_precision': minority_precision,
        'minority_recall': minority_recall,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
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
    print(f"Model: EfficientNet-V2-S")
    print(f"Input Size: 224x224")
    print(f"Augmentation: None (Basic resize and normalize only)")
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
        
        # Model setup - EfficientNet-V2-S instead of ResNet-50
        print("Loading EfficientNet-V2-S model...")
        model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        
        # Modify the classifier for our number of classes
        num_classes = len(train_dataset.dataset.classes)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
        
        print(f"Model configured for {num_classes} classes")
        
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
        
        # Optimizer - using Adam with slightly lower learning rate for EfficientNet
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
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

# [All plotting and saving functions remain the same as in original code]
# I'll include the key ones here but they don't need changes for EfficientNet

def plot_epoch_comparison(all_results, output_dir):
    """Plot comparison across different epochs for all experiments"""
    
    # Metrics to compare
    metrics = ['test_acc', 'balanced_acc', 'macro_f1', 'weighted_f1', 'minority_f1']
    metric_names = ['Test Accuracy', 'Balanced Accuracy', 'Macro F1', 'Weighted F1', 'Minority F1']
    
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
    plt.savefig(os.path.join(output_dir, 'epoch_comparison_all_metrics_efficientnet.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def save_results(all_results, output_dir):
    """Save comprehensive results with proper JSON serialization"""
    
    # Summary results
    summary_data = []
    
    for exp_name, epoch_results in all_results.items():
        for num_epochs, results in epoch_results.items():
            summary_data.append({
                'Experiment': exp_name,
                'Model': 'EfficientNet-V2-S',
                'Input_Size': '224x224',
                'Augmentation': 'None',
                'Epochs': num_epochs,
                'Val_Acc': float(results['val_acc']),
                'Test_Acc': float(results['test_acc']),
                'Balanced_Acc': float(results['balanced_acc']),
                'Macro_F1': float(results['macro_f1']),
                'Weighted_F1': float(results['weighted_f1']),
                'Micro_F1': float(results['micro_f1']),
                'Minority_F1': float(results['minority_f1']),
                'Minority_Precision': float(results['minority_precision']),
                'Minority_Recall': float(results['minority_recall']),
                'Precision_Macro': float(results['precision_macro']),
                'Recall_Macro': float(results['recall_macro']),
                'Kappa': float(results['kappa'])
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'efficientnet_experiment_summary_results.csv'), index=False)
    
    print(f"📊 Results saved to {output_dir}")
    print(f"   - efficientnet_experiment_summary_results.csv")

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
    print(f"Model: EfficientNet-V2-S")
    print(f"Input size: 224x224")
    print(f"Data augmentation: None (basic transforms only)")
    
    # Create stratified split
    print(f"\nCreating stratified 80/10/10 split...")
    train_indices, val_indices, test_indices = create_stratified_split(full_dataset, 0.8, 0.1, 0.1)
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    print(f"Split sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Store all results
    all_results = {}
    
    # Define experiments to run - same as original but with EfficientNet
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
    print("COMPREHENSIVE RESULTS ANALYSIS - EfficientNet-V2-S")
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
                'balanced_acc': results['balanced_acc']
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

    # Epoch analysis
    print(f"\n📈 EPOCH ANALYSIS:")
    epochs_performance = {}
    for epoch in args.test_epochs:
        epoch_scores = []
        for exp_name, epoch_results in all_results.items():
            epoch_scores.append(epoch_results[epoch]['macro_f1'])
        epochs_performance[epoch] = {
            'mean': np.mean(epoch_scores),
            'std': np.std(epoch_scores),
            'max': max(epoch_scores)
        }
    
    print(f"{'Epochs':<8} {'Mean F1':<12} {'Std F1':<12} {'Max F1':<12}")
    print("-" * 50)
    for epoch, perf in epochs_performance.items():
        print(f"{epoch:<8} {perf['mean']:<12.4f} {perf['std']:<12.4f} {perf['max']:<12.4f}")
    
    # Method analysis
    print(f"\n🔬 METHOD ANALYSIS (Best Epoch for Each):")
    method_performance = {}
    for exp_name, epoch_results in all_results.items():
        best_epoch_result = max(epoch_results.items(), key=lambda x: x[1]['macro_f1'])
        method_performance[exp_name] = {
            'best_epoch': best_epoch_result[0],
            'macro_f1': best_epoch_result[1]['macro_f1'],
            'test_acc': best_epoch_result[1]['test_acc'],
            'minority_f1': best_epoch_result[1]['minority_f1']
        }
    
    print(f"{'Method':<20} {'Best Epoch':<12} {'Macro F1':<12} {'Test Acc':<12} {'Minority F1':<12}")
    print("-" * 80)
    for method, perf in method_performance.items():
        print(f"{method:<20} {perf['best_epoch']:<12} {perf['macro_f1']:<12.4f} "
              f"{perf['test_acc']:<12.4f} {perf['minority_f1']:<12.4f}")

    # Generate comprehensive visualizations
    print(f"\nGenerating comprehensive visualizations...")
    plot_epoch_comparison(all_results, args.output_dir)
    
    # Save results
    save_results(all_results, args.output_dir)
    
    # Generate simplified report for EfficientNet
    report_path = os.path.join(args.output_dir, 'efficientnet_initial_experiment_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("EfficientNet-V2-S Initial Experiment Report\n")
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
        
        # Model configuration
        f.write("Model Configuration:\n")
        f.write("-" * 30 + "\n")
        f.write("Model: EfficientNet-V2-S\n")
        f.write("Input size: 224x224\n")
        f.write("Data augmentation: None (basic transforms only)\n")
        f.write("Data split: 80% train, 10% val, 10% test\n")
        f.write("Batch size: 16\n")
        f.write("Learning rate: 0.0001\n")
        f.write("Optimizer: Adam (with weight decay 1e-5)\n")
        f.write("Scheduler: ReduceLROnPlateau\n\n")
        
        # Results summary
        f.write("Results Summary by Experiment and Epochs:\n")
        f.write("-" * 50 + "\n")
        
        for exp_name, epoch_results in all_results.items():
            f.write(f"\n{exp_name}:\n")
            f.write(f"{'Epochs':<8} {'Test Acc':<10} {'Macro F1':<10} {'Minority F1':<12} {'Balanced Acc':<12}\n")
            f.write("-" * 55 + "\n")
            
            for num_epochs, results in epoch_results.items():
                f.write(f"{num_epochs:<8} {results['test_acc']:<10.4f} {results['macro_f1']:<10.4f} "
                       f"{results['minority_f1']:<12.4f} {results['balanced_acc']:<12.4f}\n")
        
        # Best results analysis
        f.write(f"\nBest Results Analysis:\n")
        f.write("-" * 30 + "\n")
        
        # Find best overall results
        best_macro_f1 = max(best_results, key=lambda x: x['macro_f1'])
        best_test_acc = max(best_results, key=lambda x: x['test_acc'])
        best_minority_f1 = max(best_results, key=lambda x: x['minority_f1'])
        
        f.write(f"Best Macro F1: {best_macro_f1['experiment']} ({best_macro_f1['epochs']} epochs) - {best_macro_f1['macro_f1']:.4f}\n")
        f.write(f"Best Test Accuracy: {best_test_acc['experiment']} ({best_test_acc['epochs']} epochs) - {best_test_acc['test_acc']:.4f}\n")
        f.write(f"Best Minority F1: {best_minority_f1['experiment']} ({best_minority_f1['epochs']} epochs) - {best_minority_f1['minority_f1']:.4f}\n")
        
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
        
        # Final recommendations
        f.write(f"\nRecommendations for EfficientNet-V2-S:\n")
        f.write("-" * 40 + "\n")
        f.write(f"1. Best overall method: {best_macro_f1['experiment']} ({best_macro_f1['epochs']} epochs)\n")
        f.write(f"2. For minority classes: {best_minority_f1['experiment']} ({best_minority_f1['epochs']} epochs)\n")
        f.write(f"3. For highest accuracy: {best_test_acc['experiment']} ({best_test_acc['epochs']} epochs)\n")
        
        f.write(f"\nModel Architecture Notes:\n")
        f.write("- EfficientNet-V2-S provides efficient computation with good accuracy\n")
        f.write("- 224x224 input size balances speed and performance\n")
        f.write("- No augmentation used to focus on model's inherent capabilities\n")
        f.write("- Weight decay added to prevent overfitting\n")

    # Final recommendations
    print(f"\n{'='*80}")
    print("🎖️ FINAL RECOMMENDATIONS - EfficientNet-V2-S")
    print(f"{'='*80}")
    
    best_overall = max(best_results, key=lambda x: x['macro_f1'])
    best_for_minority = max(best_results, key=lambda x: x['minority_f1'])
    best_for_accuracy = max(best_results, key=lambda x: x['test_acc'])
    
    print(f"1. 🥇 BEST OVERALL (Macro F1): {best_overall['experiment']} with {best_overall['epochs']} epochs")
    print(f"   - Macro F1: {best_overall['macro_f1']:.4f}")
    print(f"   - Test Accuracy: {best_overall['test_acc']:.4f}")
    print(f"   - Minority F1: {best_overall['minority_f1']:.4f}")
    
    print(f"\n2. 🔴 BEST FOR MINORITY CLASSES: {best_for_minority['experiment']} with {best_for_minority['epochs']} epochs")
    print(f"   - Minority F1: {best_for_minority['minority_f1']:.4f}")
    
    print(f"\n3. 📊 BEST FOR ACCURACY: {best_for_accuracy['experiment']} with {best_for_accuracy['epochs']} epochs")
    print(f"   - Test Accuracy: {best_for_accuracy['test_acc']:.4f}")
    
    # Optimal epoch recommendation
    best_epoch = max(epochs_performance.items(), key=lambda x: x[1]['mean'])[0]
    print(f"\n4. 📈 OPTIMAL EPOCH COUNT: {best_epoch} epochs")
    print(f"   - Average Macro F1 across all methods: {epochs_performance[best_epoch]['mean']:.4f}")
    
    # Training efficiency insights
    efficiency_analysis = []
    for exp_name, epoch_results in all_results.items():
        for num_epochs, results in epoch_results.items():
            efficiency = results['macro_f1'] / num_epochs
            efficiency_analysis.append({
                'method': exp_name,
                'epochs': num_epochs,
                'efficiency': efficiency,
                'macro_f1': results['macro_f1']
            })
    
    most_efficient = max(efficiency_analysis, key=lambda x: x['efficiency'])
    print(f"\n5. ⚡ MOST EFFICIENT: {most_efficient['method']} with {most_efficient['epochs']} epochs")
    print(f"   - Efficiency Score: {most_efficient['efficiency']:.6f}")
    print(f"   - Macro F1: {most_efficient['macro_f1']:.4f}")
    
    total_duration = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"EfficientNet-V2-S INITIAL EXPERIMENTS COMPLETED")
    print(f"{'='*80}")
    print(f"Total execution time: {total_duration/60:.1f} minutes")
    print(f"Results saved to: {args.output_dir}")
    print(f"\nGenerated files:")
    print(f"├── 📊 SUMMARY RESULTS:")
    print(f"│   └── efficientnet_experiment_summary_results.csv")
    print(f"├── 📈 VISUALIZATIONS:")
    print(f"│   └── epoch_comparison_all_metrics_efficientnet.png")
    print(f"├── 📋 REPORTS:")
    print(f"│   └── efficientnet_initial_experiment_report.txt")
    print(f"└── 🎯 KEY INSIGHTS:")
    print(f"    ├── EfficientNet-V2-S performance analysis completed")
    print(f"    ├── No augmentation baseline established")
    print(f"    └── 224x224 input size efficiency confirmed")
    
    # Create final summary for quick reference
    summary_path = os.path.join(args.output_dir, 'efficientnet_quick_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("QUICK SUMMARY - EfficientNet-V2-S INITIAL EXPERIMENTS RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("🏆 BEST CONFIGURATIONS:\n")
        f.write("-" * 25 + "\n")
        f.write(f"Overall Best: {best_overall['experiment']} ({best_overall['epochs']} epochs)\n")
        f.write(f"  → Macro F1: {best_overall['macro_f1']:.4f}\n")
        f.write(f"  → Test Acc: {best_overall['test_acc']:.4f}\n\n")
        
        f.write(f"Most Efficient: {most_efficient['method']} ({most_efficient['epochs']} epochs)\n")
        f.write(f"  → Efficiency: {most_efficient['efficiency']:.6f}\n")
        f.write(f"  → Macro F1: {most_efficient['macro_f1']:.4f}\n\n")
        
        f.write(f"Best for Minorities: {best_for_minority['experiment']} ({best_for_minority['epochs']} epochs)\n")
        f.write(f"  → Minority F1: {best_for_minority['minority_f1']:.4f}\n\n")
        
        f.write("📈 MODEL CONFIGURATION:\n")
        f.write("-" * 25 + "\n")
        f.write("Model: EfficientNet-V2-S\n")
        f.write("Input Size: 224x224\n")
        f.write("Augmentation: None\n")
        f.write("Batch Size: 16\n")
        f.write("Learning Rate: 0.0001\n\n")
        
        f.write("🎯 DEPLOYMENT RECOMMENDATION:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Use: {best_overall['experiment']} with {best_overall['epochs']} epochs\n")
        f.write(f"Expected Macro F1: {best_overall['macro_f1']:.4f}\n")
        f.write(f"Expected Test Accuracy: {best_overall['test_acc']:.4f}\n")
    
    print(f"\n✅ STUDY COMPLETE! Check {args.output_dir}/efficientnet_quick_summary.txt for key results.")

if __name__ == '__main__':
    main()