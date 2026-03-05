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
parser.add_argument('--output_dir', type=str, default='/home/s3844498/outputs_mobilenet_initial', help='Directory to save outputs')
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
                    help='Run all experiments for complete initial experiments')
parser.add_argument('--pretrained', action='store_true', default=True,
                    help='Use pretrained MobileNet V3 weights (default: True)')
parser.add_argument('--model_size', type=str, default='large', choices=['small', 'large'],
                    help='MobileNet V3 model size: small or large (default: large)')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='Learning rate for MobileNet V3 (default: 0.001)')
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

# Transforms for MobileNet V3 (without augmentation)
def get_transforms():
    return {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),  # MobileNet V3 expects 224x224
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
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

def create_mobilenet_v3_model(num_classes, model_size='large', pretrained=True, dropout_rate=0.2):
    """Create and configure MobileNet V3 model"""
    
    # Load MobileNet V3
    if model_size == 'large':
        if pretrained:
            print("Loading pretrained MobileNet V3 Large...")
            model = models.mobilenet_v3_large(weights='IMAGENET1K_V1')
        else:
            print("Creating MobileNet V3 Large from scratch...")
            model = models.mobilenet_v3_large(weights=None)
    else:  # small
        if pretrained:
            print("Loading pretrained MobileNet V3 Small...")
            model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        else:
            print("Creating MobileNet V3 Small from scratch...")
            model = models.mobilenet_v3_small(weights=None)
    
    # Get the number of input features for the classifier
    num_features = model.classifier[3].in_features
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, model.classifier[0].out_features),  # Keep first linear layer
        nn.Hardswish(inplace=True),  # Keep activation
        nn.Dropout(dropout_rate),  # Add dropout
        nn.Linear(model.classifier[0].out_features, num_classes)  # Final classification layer
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"MobileNet V3 {model_size.capitalize()} Model Info:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    print(f"  Dropout rate: {dropout_rate}")
    print(f"  Output classes: {num_classes}")
    print(f"  Model variant: MobileNet V3 {model_size.capitalize()}")
    
    return model

# Training function for MobileNet V3
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device,
                num_epochs, experiment_name):
    model.to(device)
    best_model_wts = model.state_dict()
    best_acc = 0.0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    print(f"\n=== Training {experiment_name} for {num_epochs} epochs ===")
    print(f"Training on device: {device}")
    print(f"Model: MobileNet V3 {args.model_size.capitalize()}")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss, running_corrects, total = 0.0, 0, 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for inputs, labels in train_bar:
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
            
            # Update progress bar
            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{running_corrects/total:.4f}'
            })

        train_losses.append(running_loss / total)
        train_accs.append(running_corrects / total)

        # Validation phase
        model.eval()
        val_running_loss, val_running_corrects, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels).item()
                val_total += labels.size(0)
                
                # Update progress bar
                val_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{val_running_corrects/val_total:.4f}'
                })

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
    
    print(f"\n📊 {experiment_name} Performance (MobileNet V3 {args.model_size.capitalize()}):")
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
    print(f"EXPERIMENT: {experiment_name} (MobileNet V3 {args.model_size.capitalize()})")
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
        
        # Data loaders with adjusted batch size for MobileNet V3
        batch_size = 32  # Appropriate batch size for MobileNet V3
        
        if use_balanced_sampling:
            train_sampler = create_balanced_sampler(train_dataset, power=2)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # Model setup - MobileNet V3
        model = create_mobilenet_v3_model(
            num_classes=len(train_dataset.dataset.classes),
            model_size=args.model_size,
            pretrained=args.pretrained,
            dropout_rate=0.2
        )
        
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
        
        # Optimizer optimized for MobileNet V3
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)
        
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

# Plotting functions (updated for MobileNet V3)
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
    
    plt.suptitle(f'MobileNet V3 {args.model_size.capitalize()} - Epoch Comparison All Metrics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'mobilenetv3_{args.model_size}_epoch_comparison_all_metrics.png'), 
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
        
        plt.suptitle(f'MobileNet V3 {args.model_size.capitalize()} Training Curves - {num_epochs} Epochs', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'mobilenetv3_{args.model_size}_training_curves_{num_epochs}_epochs.png'), 
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
    summary_df.to_csv(os.path.join(output_dir, f'mobilenetv3_{args.model_size}_experiment_summary_results.csv'), index=False)
    
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
            'Best_Minority_F1': float(best_epoch[1]['minority_f1'])
        })
    
    best_df = pd.DataFrame(best_results)
    best_df.to_csv(os.path.join(output_dir, f'mobilenetv3_{args.model_size}_best_results_per_experiment.csv'), index=False)
    
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
        with open(os.path.join(output_dir, f'mobilenetv3_{args.model_size}_complete_results.json'), 'w') as f:
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
                    'weighted_f1': float(results['weighted_f1']),
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
        
        with open(os.path.join(output_dir, f'mobilenetv3_{args.model_size}_simplified_results.json'), 'w') as f:
            json.dump(simplified_results, f, indent=2)
        print("✅ Simplified results saved to JSON successfully")
    
    print(f"📊 MobileNet V3 {args.model_size.capitalize()} Results saved to {output_dir}")
    print(f"   - mobilenetv3_{args.model_size}_experiment_summary_results.csv")
    print(f"   - mobilenetv3_{args.model_size}_best_results_per_experiment.csv") 
    print(f"   - mobilenetv3_{args.model_size}_complete_results.json (or simplified version)")

def generate_report(all_results, full_dataset, output_dir):
    """Generate comprehensive report"""
    report_path = os.path.join(output_dir, f'mobilenetv3_{args.model_size}_initial_experiment_report.txt')
    
    with open(report_path, 'w') as f:
        f.write(f"MobileNet V3 {args.model_size.capitalize()} Initial Experiment Report\n")
        f.write("=" * 80 + "\n\n")
        
        # Model info
        f.write("Model Information:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Architecture: MobileNet V3 {args.model_size.capitalize()}\n")
        f.write(f"Pretrained: {args.pretrained}\n")
        f.write(f"Learning Rate: {args.learning_rate}\n")
        f.write("Optimizer: Adam with weight decay 1e-4\n")
        f.write("Scheduler: ReduceLROnPlateau\n")
        f.write("Batch Size: 32\n")
        f.write("Data Augmentation: None (basic transforms only)\n")
        
        # MobileNet V3 specific features
        f.write("\nMobileNet V3 Specific Features:\n")
        f.write("- Squeeze-and-Excitation blocks\n")
        f.write("- Hard-Swish activation function\n")
        f.write("- Neural Architecture Search (NAS) optimized\n")
        f.write("- Improved inverted residual blocks\n")
        f.write("- Efficient channel attention mechanism\n\n")
        
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
        f.write(f"Model: MobileNet V3 {args.model_size.capitalize()}\n")
        f.write("Data split: 80% train, 10% val, 10% test\n")
        f.write("Batch size: 32\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write("Optimizer: Adam\n")
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
        best_macro_f1 = 0
        best_test_acc = 0
        best_minority_f1 = 0
        best_macro_exp = ""
        best_acc_exp = ""
        best_minority_exp = ""
        
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
        
        f.write(f"Best Macro F1: {best_macro_exp} ({best_macro_f1:.4f})\n")
        f.write(f"Best Test Accuracy: {best_acc_exp} ({best_test_acc:.4f})\n")
        f.write(f"Best Minority F1: {best_minority_exp} ({best_minority_f1:.4f})\n")
        
        # MobileNet V3 specific insights
        f.write(f"\nMobileNet V3 {args.model_size.capitalize()} Specific Insights:\n")
        f.write("-" * 45 + "\n")
        f.write("1. Architecture Advantages:\n")
        if args.model_size == 'large':
            f.write("   - Optimized for high accuracy applications\n")
            f.write("   - ~5.4M parameters (Large variant)\n")
            f.write("   - Better performance on complex tasks\n")
        else:
            f.write("   - Optimized for efficiency and speed\n")
            f.write("   - ~2.9M parameters (Small variant)\n")
            f.write("   - Ideal for resource-constrained environments\n")
        f.write("   - Neural Architecture Search optimized design\n")
        f.write("   - Advanced activation functions (Hard-Swish)\n")
        f.write("   - Squeeze-and-Excitation attention mechanism\n\n")
        
        f.write("2. Performance Characteristics:\n")
        f.write("   - Improved efficiency over MobileNet V2\n")
        f.write("   - Better accuracy-latency trade-off\n")
        f.write("   - Enhanced feature representation\n")
        f.write("   - Optimized for mobile deployment\n")
        f.write("   - State-of-the-art efficient architecture\n\n")
        
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


def main():
    start_time = time.time()
    
    print("=" * 80)
    print(f"MOBILENET V3 {args.model_size.upper()} INITIAL EXPERIMENT")
    print("=" * 80)
    
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
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
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
    print(f"Model: MobileNet V3 {args.model_size.capitalize()} (Pretrained: {args.pretrained})")
    print(f"Learning Rate: {args.learning_rate}")

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
    print(f"COMPREHENSIVE MOBILENET V3 {args.model_size.upper()} RESULTS ANALYSIS")
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

    # Generate comprehensive visualizations
    print(f"\nGenerating comprehensive visualizations...")
    plot_epoch_comparison(all_results, args.output_dir)
    plot_training_curves_by_epochs(all_results, args.output_dir)
    
    # Save results
    save_results(all_results, args.output_dir)
    
    # Generate reports
    generate_report(all_results, full_dataset, args.output_dir)
    
    # Final recommendations
    print(f"\n{'='*80}")
    print(f"🎖️ FINAL MOBILENET V3 {args.model_size.upper()} RECOMMENDATIONS")
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
    
    # MobileNet V3 specific insights
    print(f"\n4. 📱 MOBILENET V3 {args.model_size.upper()} ADVANTAGES:")
    if args.model_size == 'large':
        print(f"   - Model size: ~21.9 MB (Large variant)")
        print(f"   - Parameters: ~5.4M")
        print(f"   - Optimized for high accuracy")
        print(f"   - Better performance on complex datasets")
    else:
        print(f"   - Model size: ~11.5 MB (Small variant)")
        print(f"   - Parameters: ~2.9M")
        print(f"   - Optimized for efficiency and speed")
        print(f"   - Perfect for resource-constrained devices")
    
    print(f"   - Neural Architecture Search optimized")
    print(f"   - Squeeze-and-Excitation attention")
    print(f"   - Hard-Swish activation functions")
    print(f"   - Improved over MobileNet V2")
    print(f"   - State-of-the-art mobile architecture")
    
    total_duration = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"MOBILENET V3 {args.model_size.upper()} INITIAL EXPERIMENT COMPLETED")
    print(f"{'='*80}")
    print(f"Total execution time: {total_duration/60:.1f} minutes")
    print(f"Results saved to: {args.output_dir}")
    print(f"\nGenerated files:")
    print(f"├── 📊 SUMMARY RESULTS:")
    print(f"│   ├── mobilenetv3_{args.model_size}_experiment_summary_results.csv")
    print(f"│   ├── mobilenetv3_{args.model_size}_best_results_per_experiment.csv")
    print(f"│   └── mobilenetv3_{args.model_size}_complete_results.json")
    print(f"├── 📈 VISUALIZATIONS:")
    print(f"│   ├── mobilenetv3_{args.model_size}_epoch_comparison_all_metrics.png")
    print(f"│   └── mobilenetv3_{args.model_size}_training_curves_[X]_epochs.png")
    print(f"├── 📋 REPORTS:")
    print(f"│   └── mobilenetv3_{args.model_size}_initial_experiment_report.txt")
    print(f"└── 🎯 KEY INSIGHTS:")
    print(f"    ├── Neural Architecture Search optimized design")
    print(f"    ├── Advanced attention mechanisms")
    print(f"    ├── State-of-the-art mobile efficiency")
    print(f"    └── Perfect for mobile and edge deployment")
    
    # Create final summary for quick reference
    summary_path = os.path.join(args.output_dir, f'mobilenetv3_{args.model_size}_quick_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"QUICK SUMMARY - MOBILENET V3 {args.model_size.upper()} INITIAL EXPERIMENT RESULTS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("🏆 BEST CONFIGURATIONS:\n")
        f.write("-" * 25 + "\n")
        f.write(f"Overall Best: {best_overall['experiment']} ({best_overall['epochs']} epochs)\n")
        f.write(f"  → Macro F1: {best_overall['macro_f1']:.4f}\n")
        f.write(f"  → Test Acc: {best_overall['test_acc']:.4f}\n\n")
        
        f.write(f"Best for Minorities: {best_for_minority['experiment']} ({best_for_minority['epochs']} epochs)\n")
        f.write(f"  → Minority F1: {best_for_minority['minority_f1']:.4f}\n\n")
        
        f.write(f"📱 MOBILENET V3 {args.model_size.upper()} ADVANTAGES:\n")
        f.write("-" * 40 + "\n")
        if args.model_size == 'large':
            f.write("• Optimized for high accuracy (~5.4M params)\n")
            f.write("• Better performance on complex tasks\n")
        else:
            f.write("• Ultra-efficient design (~2.9M params)\n")
            f.write("• Perfect for resource-constrained devices\n")
        f.write("• Neural Architecture Search optimized\n")
        f.write("• Squeeze-and-Excitation attention\n")
        f.write("• Hard-Swish activation functions\n")
        f.write("• State-of-the-art mobile architecture\n")
        f.write("• Improved efficiency over MobileNet V2\n\n")
        
        f.write("🎯 DEPLOYMENT RECOMMENDATION:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Use: {best_overall['experiment']} with {best_overall['epochs']} epochs\n")
        f.write(f"Expected Macro F1: {best_overall['macro_f1']:.4f}\n")
        f.write(f"Expected Test Accuracy: {best_overall['test_acc']:.4f}\n")
        f.write(f"Perfect for mobile applications and edge computing!\n")
        f.write(f"MobileNet V3 {args.model_size.capitalize()} offers the best accuracy-efficiency trade-off!\n")
    
    print(f"\n✅ STUDY COMPLETE! Check {args.output_dir}/mobilenetv3_{args.model_size}_quick_summary.txt for key results.")

if __name__ == '__main__':
    main()