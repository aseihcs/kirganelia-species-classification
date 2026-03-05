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
from sklearn.model_selection import train_test_split, StratifiedKFold
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import json
import warnings
from collections import defaultdict, Counter
import time
from sklearn.metrics import f1_score, precision_recall_fscore_support, cohen_kappa_score
import math

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Argument parser
parser = argparse.ArgumentParser(description='ResNet50 Final Training with Cross Validation')
parser.add_argument('--data_dir', type=str, default='/home/s3844498/data/2nd_fix', 
                    help='Path to the herbarium dataset')
parser.add_argument('--output_dir', type=str, default='/home/s3844498/outputs_final_training', 
                    help='Directory to save outputs')
parser.add_argument('--run_baseline', action='store_true')
parser.add_argument('--run_focal_weighted', action='store_true')
parser.add_argument('--run_both', action='store_true')
parser.add_argument('--epochs', type=int, default=20,
                    help='Number of training epochs (default: 20)')
parser.add_argument('--n_folds', type=int, default=5,
                    help='Number of cross-validation folds (default: 5)')
args = parser.parse_args()

# Set default to run both if nothing specified
if not any([args.run_baseline, args.run_focal_weighted, args.run_both]):
    args.run_both = True

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

# Enhanced Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            n_classes = inputs.size(1)
            smoothed_targets = torch.zeros_like(inputs)
            smoothed_targets.fill_(self.label_smoothing / (n_classes - 1))
            smoothed_targets.scatter_(1, targets.unsqueeze(1), 1 - self.label_smoothing)
            
            ce_loss = -torch.sum(smoothed_targets * torch.log_softmax(inputs, dim=1), dim=1)
        else:
            ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Custom dataset class
class HerbariumDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) 
                             if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []
        self.sample_species = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
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

# Dataset analysis
def analyze_dataset_distribution(dataset):
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
    """Calculate class weights for weighted loss"""
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

# Enhanced transforms with more augmentation
def get_transforms():
    return {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

def create_stratified_split(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Create stratified train/val/test split"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    labels = [sample[1] for sample in dataset.samples]
    indices = list(range(len(dataset)))
    
    # First split: separate test set
    train_val_indices, test_indices, train_val_labels, _ = train_test_split(
        indices, labels, test_size=test_ratio, stratify=labels, random_state=42
    )
    
    # Second split: separate train and validation
    val_size = val_ratio / (train_ratio + val_ratio)
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_size, stratify=train_val_labels, random_state=42
    )
    
    return train_indices, val_indices, test_indices

# Warmup scheduler
class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.current_epoch = 0
        
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.current_epoch += 1

def create_model(num_classes, dropout_rate=0.2):
    """Create EfficientNet-V2-S model with configurable dropout"""
    model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
    
    # Modify the classifier for our number of classes
    if dropout_rate > 0:
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
    else:
        # Replace the original classifier
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),  # Keep original dropout structure
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
    
    return model

# Training function with enhanced features
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                warmup_scheduler, device, num_epochs, experiment_name, fold_num,
                gradient_clip_norm=1.0):
    
    model.to(device)
    best_model_wts = model.state_dict()
    best_acc = 0.0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    learning_rates = []

    print(f"\n=== Training {experiment_name} - Fold {fold_num} for {num_epochs} epochs ===")

    for epoch in range(num_epochs):
        # Warmup phase
        if warmup_scheduler and epoch < warmup_scheduler.warmup_epochs:
            warmup_scheduler.step()
        
        # Training phase
        model.train()
        running_loss, running_corrects, total = 0.0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        epoch_train_loss = running_loss / total
        epoch_train_acc = running_corrects / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

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

        epoch_val_loss = val_running_loss / val_total
        epoch_val_acc = val_running_corrects / val_total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        # Record learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # Step scheduler (after warmup)
        if epoch >= (warmup_scheduler.warmup_epochs if warmup_scheduler else 0):
            if hasattr(scheduler, 'step'):
                if 'ReduceLROnPlateau' in str(type(scheduler)):
                    scheduler.step(epoch_val_loss)
                else:
                    scheduler.step()

        # Save best model
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = model.state_dict()

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}, "
              f"LR: {current_lr:.6f}")

    model.load_state_dict(best_model_wts)
    print(f"Best Val Accuracy for Fold {fold_num}: {best_acc:.4f}")

    return {
        'model': model,
        'best_val_acc': best_acc,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'learning_rates': learning_rates
    }

# Comprehensive evaluation function
def evaluate_model(model, test_loader, criterion, device, class_names, 
                  experiment_name, fold_num=None):
    
    model.to(device)
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    all_sample_indices = []  # Track sample indices for misclassification analysis
    running_loss, running_corrects, total = 0.0, 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader, desc="Testing", leave=False)):
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
            
            # Track sample indices (approximation based on batch)
            batch_size = inputs.size(0)
            start_idx = batch_idx * test_loader.batch_size
            all_sample_indices.extend(range(start_idx, start_idx + batch_size))

    test_loss = running_loss / total
    test_acc = running_corrects / total
    
    # Comprehensive metrics
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    kappa = cohen_kappa_score(all_labels, all_preds)
    
    # Per-class metrics
    precision_macro = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)[0]
    recall_macro = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)[1]
    precision_weighted = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)[0]
    recall_weighted = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)[1]
    
    # Detailed per-class analysis
    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    # Misclassification analysis
    misclassification_data = []
    all_probs_array = np.array(all_probs)
    
    for i, (true_label, pred_label, sample_idx) in enumerate(zip(all_labels, all_preds, all_sample_indices)):
        if true_label != pred_label:  # Misclassified sample
            sample_probs = all_probs_array[i]
            
            # Get top 3 predictions
            top3_indices = np.argsort(sample_probs)[-3:][::-1]  # Top 3 in descending order
            top3_probs = sample_probs[top3_indices]
            
            # Confidence of predicted class
            pred_confidence = sample_probs[pred_label] * 100
            
            # Confidence category
            if pred_confidence >= 70:
                confidence_category = 'High'
            elif pred_confidence >= 50:
                confidence_category = 'Medium'
            else:
                confidence_category = 'Low'
            
            misclassification_data.append({
                'sample_index': sample_idx,
                'true_label': true_label,
                'true_species': class_names[true_label],
                'predicted_label': pred_label,
                'predicted_species': class_names[pred_label],
                'prediction_confidence': pred_confidence,
                'confidence_category': confidence_category,
                'top1_species': class_names[top3_indices[0]],
                'top1_confidence': top3_probs[0] * 100,
                'top2_species': class_names[top3_indices[1]] if len(top3_indices) > 1 else '',
                'top2_confidence': top3_probs[1] * 100 if len(top3_indices) > 1 else 0.0,
                'top3_species': class_names[top3_indices[2]] if len(top3_indices) > 2 else '',
                'top3_confidence': top3_probs[2] * 100 if len(top3_indices) > 2 else 0.0,
                'true_species_confidence': sample_probs[true_label] * 100,
                'true_species_rank': np.where(np.argsort(sample_probs)[::-1] == true_label)[0][0] + 1
            })
    
    # Create species-level summary (REMOVED size_category)
    species_summary = []
    class_counts = Counter(all_labels)
    
    for class_idx in range(len(class_names)):
        species_name = class_names[class_idx]
        
        # Calculate per-class accuracy
        class_mask = np.array(all_labels) == class_idx
        class_preds = np.array(all_preds)[class_mask]
        class_labels = np.array(all_labels)[class_mask]
        
        if len(class_labels) > 0:
            class_accuracy = (class_preds == class_labels).mean()
        else:
            class_accuracy = 0.0
        
        # Get test set count for this class
        test_count = class_counts.get(class_idx, 0)
        
        species_info = {
            'species_name': species_name,
            'class_idx': class_idx,
            'test_samples': test_count,
            'precision': per_class_precision[class_idx] if class_idx < len(per_class_precision) else 0.0,
            'recall': per_class_recall[class_idx] if class_idx < len(per_class_recall) else 0.0,
            'f1_score': per_class_f1[class_idx] if class_idx < len(per_class_f1) else 0.0,
            'accuracy': class_accuracy,
            'support': per_class_support[class_idx] if class_idx < len(per_class_support) else 0
        }
        
        species_summary.append(species_info)
    
    # Minority class analysis
    total_samples = len(all_labels)
    minority_threshold = min(20, total_samples * 0.05)
    minority_classes = [cls for cls, count in class_counts.items() if count < minority_threshold]
    
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
    
    fold_text = f"Fold {fold_num}" if fold_num is not None else ""
    print(f"\n📊 {experiment_name} {fold_text} Performance:")
    print(f"  🎯 Accuracy: {test_acc:.4f}")
    print(f"  ⚖️  Balanced Accuracy: {balanced_acc:.4f}")
    print(f"  📈 Macro F1: {macro_f1:.4f}")
    print(f"  📊 Weighted F1: {weighted_f1:.4f}")
    print(f"  🔍 Micro F1: {micro_f1:.4f}")
    print(f"  🤝 Cohen's Kappa: {kappa:.4f}")
    print(f"  🔴 Minority F1: {minority_f1:.4f}")
    print(f"  ❌ Misclassifications: {len(misclassification_data)}/{total} ({len(misclassification_data)/total*100:.1f}%)")

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
        'species_summary': species_summary,
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
        'misclassification_data': misclassification_data,
        'class_names': class_names  # Store class names for confusion matrix plotting
    }

def calculate_cv_statistics(test_results):
    """Calculate cross-validation statistics across folds"""
    metrics = ['test_acc', 'balanced_acc', 'macro_f1', 'weighted_f1', 'micro_f1', 
              'minority_f1', 'kappa']
    
    cv_stats = {}
    for metric in metrics:
        values = [result[metric] for result in test_results]
        cv_stats[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'values': values
        }
    
    return cv_stats

# Visualization functions
def plot_cv_results(all_experiments, output_dir):
    """Plot cross-validation results comparison"""
    
    metrics = ['test_acc', 'balanced_acc', 'macro_f1', 'weighted_f1', 'minority_f1']
    metric_names = ['Test Accuracy', 'Balanced Accuracy', 'Macro F1', 'Weighted F1', 'Minority F1']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    experiment_names = [exp['experiment_name'] for exp in all_experiments]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        if i < len(axes):
            ax = axes[i]
            
            means = []
            stds = []
            
            for exp in all_experiments:
                cv_stats = exp['cv_statistics']
                means.append(cv_stats[metric]['mean'])
                stds.append(cv_stats[metric]['std'])
            
            # Bar plot with error bars
            bars = ax.bar(experiment_names, means, yerr=stds, capsize=5, 
                         color=colors[:len(experiment_names)], alpha=0.7)
            
            # Add value labels on bars
            for j, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                       f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10)
            
            ax.set_title(metric_name, fontsize=12, fontweight='bold')
            ax.set_ylabel(metric_name)
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels if needed
            if len(max(experiment_names, key=len)) > 10:
                ax.tick_params(axis='x', rotation=15)
    
    # Remove empty subplot
    if len(metrics) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_results_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate all visualization functions
    plot_species_performance(all_experiments, output_dir)
    plot_confusion_matrices(all_experiments, output_dir)
    plot_misclassification_analysis(all_experiments, output_dir)

def plot_training_curves_cv(all_experiments, output_dir):
    """Plot training curves for cross-validation results"""
    
    for exp in all_experiments:
        exp_name = exp['experiment_name']
        fold_results = exp['fold_results']
        epochs = exp['epochs']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        colors = plt.cm.Set3(np.linspace(0, 1, len(fold_results)))
        
        for fold_idx, fold_result in enumerate(fold_results):
            curves = fold_result['training_curves']
            epochs_range = range(1, epochs + 1)
            
            # Training Loss
            axes[0,0].plot(epochs_range, curves['train_losses'], 
                          label=f'Fold {fold_result["fold"]}', 
                          color=colors[fold_idx], linewidth=1.5)
            
            # Validation Loss
            axes[0,1].plot(epochs_range, curves['val_losses'], 
                          label=f'Fold {fold_result["fold"]}', 
                          color=colors[fold_idx], linewidth=1.5)
            
            # Training Accuracy
            axes[1,0].plot(epochs_range, curves['train_accs'], 
                          label=f'Fold {fold_result["fold"]}', 
                          color=colors[fold_idx], linewidth=1.5)
            
            # Validation Accuracy
            axes[1,1].plot(epochs_range, curves['val_accs'], 
                          label=f'Fold {fold_result["fold"]}', 
                          color=colors[fold_idx], linewidth=1.5)
            
            # Learning Rate
            axes[0,2].plot(epochs_range, curves['learning_rates'], 
                          label=f'Fold {fold_result["fold"]}', 
                          color=colors[fold_idx], linewidth=1.5)
        
        # Calculate average curves
        avg_train_losses = np.mean([fold['training_curves']['train_losses'] for fold in fold_results], axis=0)
        avg_val_losses = np.mean([fold['training_curves']['val_losses'] for fold in fold_results], axis=0)
        avg_train_accs = np.mean([fold['training_curves']['train_accs'] for fold in fold_results], axis=0)
        avg_val_accs = np.mean([fold['training_curves']['val_accs'] for fold in fold_results], axis=0)
        avg_lrs = np.mean([fold['training_curves']['learning_rates'] for fold in fold_results], axis=0)
        
        # Plot average curves
        axes[0,0].plot(epochs_range, avg_train_losses, 'k--', linewidth=3, label='Average')
        axes[0,1].plot(epochs_range, avg_val_losses, 'k--', linewidth=3, label='Average')
        axes[1,0].plot(epochs_range, avg_train_accs, 'k--', linewidth=3, label='Average')
        axes[1,1].plot(epochs_range, avg_val_accs, 'k--', linewidth=3, label='Average')
        axes[0,2].plot(epochs_range, avg_lrs, 'k--', linewidth=3, label='Average')
        
        # Set titles and labels
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
        
        axes[0,2].set_title('Learning Rate')
        axes[0,2].set_xlabel('Epoch')
        axes[0,2].set_ylabel('Learning Rate')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        axes[0,2].set_yscale('log')
        
        # Performance summary subplot
        test_accs = [fold['test_results']['test_acc'] for fold in fold_results]
        macro_f1s = [fold['test_results']['macro_f1'] for fold in fold_results]
        
        axes[1,2].bar(['Test Acc', 'Macro F1'], 
                     [np.mean(test_accs), np.mean(macro_f1s)],
                     yerr=[np.std(test_accs), np.std(macro_f1s)],
                     capsize=5, alpha=0.7)
        axes[1,2].set_title('Final Performance')
        axes[1,2].set_ylabel('Score')
        axes[1,2].grid(True, alpha=0.3)
        
        plt.suptitle(f'{exp_name} - 5-Fold Cross Validation Training Curves', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{exp_name.lower().replace(" ", "_")}_cv_training_curves.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()


def save_cv_results(all_experiments, output_dir):
    """Save cross-validation results to files - FIXED VERSION"""
    
    # Summary results
    summary_data = []
    detailed_data = []
    species_data = []
    misclassification_data = []
    
    for exp in all_experiments:
        exp_name = exp['experiment_name']
        cv_stats = exp['cv_statistics']
        
        # Summary statistics
        summary_data.append({
            'Experiment': exp_name,
            'Test_Acc_Mean': cv_stats['test_acc']['mean'],
            'Test_Acc_Std': cv_stats['test_acc']['std'],
            'Balanced_Acc_Mean': cv_stats['balanced_acc']['mean'],
            'Balanced_Acc_Std': cv_stats['balanced_acc']['std'],
            'Macro_F1_Mean': cv_stats['macro_f1']['mean'],
            'Macro_F1_Std': cv_stats['macro_f1']['std'],
            'Weighted_F1_Mean': cv_stats['weighted_f1']['mean'],
            'Weighted_F1_Std': cv_stats['weighted_f1']['std'],
            'Minority_F1_Mean': cv_stats['minority_f1']['mean'],
            'Minority_F1_Std': cv_stats['minority_f1']['std'],
            'Kappa_Mean': cv_stats['kappa']['mean'],
            'Kappa_Std': cv_stats['kappa']['std']
        })
        
        # Detailed fold results
        for fold_result in exp['fold_results']:
            test_res = fold_result['test_results']
            detailed_data.append({
                'Experiment': exp_name,
                'Fold': fold_result['fold'],
                'Best_Val_Acc': fold_result['best_val_acc'],
                'Test_Acc': test_res['test_acc'],
                'Balanced_Acc': test_res['balanced_acc'],
                'Macro_F1': test_res['macro_f1'],
                'Weighted_F1': test_res['weighted_f1'],
                'Micro_F1': test_res['micro_f1'],
                'Minority_F1': test_res['minority_f1'],
                'Kappa': test_res['kappa'],
                'Total_Misclassifications': len(test_res['misclassification_data'])
            })
            
            # Species-level results (REMOVED size_category)
            for species_info in test_res['species_summary']:
                species_data.append({
                    'Experiment': exp_name,
                    'Fold': fold_result['fold'],
                    'Species_Name': species_info['species_name'],
                    'Class_Index': species_info['class_idx'],
                    'Test_Samples': species_info['test_samples'],
                    'Precision': species_info['precision'],
                    'Recall': species_info['recall'],
                    'F1_Score': species_info['f1_score'],
                    'Accuracy': species_info['accuracy'],
                    'Support': species_info['support']
                })
            
            # Misclassification results
            for misc_info in test_res['misclassification_data']:
                misclassification_data.append({
                    'Experiment': exp_name,
                    'Fold': fold_result['fold'],
                    'Sample_Index': misc_info['sample_index'],
                    'True_Label': misc_info['true_label'],
                    'True_Species': misc_info['true_species'],
                    'Predicted_Label': misc_info['predicted_label'],
                    'Predicted_Species': misc_info['predicted_species'],
                    'Prediction_Confidence': misc_info['prediction_confidence'],
                    'Confidence_Category': misc_info['confidence_category'],
                    'Top1_Species': misc_info['top1_species'],
                    'Top1_Confidence': misc_info['top1_confidence'],
                    'Top2_Species': misc_info['top2_species'],
                    'Top2_Confidence': misc_info['top2_confidence'],
                    'Top3_Species': misc_info['top3_species'],
                    'Top3_Confidence': misc_info['top3_confidence'],
                    'True_Species_Confidence': misc_info['true_species_confidence'],
                    'True_Species_Rank': misc_info['true_species_rank']
                })
    
    # Save CSV files
    summary_df = pd.DataFrame(summary_data)
    detailed_df = pd.DataFrame(detailed_data)
    species_df = pd.DataFrame(species_data)
    misclassification_df = pd.DataFrame(misclassification_data)
    
    summary_df.to_csv(os.path.join(output_dir, 'cv_summary_results.csv'), index=False)
    detailed_df.to_csv(os.path.join(output_dir, 'cv_detailed_results.csv'), index=False)
    species_df.to_csv(os.path.join(output_dir, 'cv_species_level_results.csv'), index=False)
    misclassification_df.to_csv(os.path.join(output_dir, 'cv_misclassification_analysis.csv'), index=False)
    
    # Create species performance summary across all folds (REMOVED size_category)
    species_summary_stats = []
    
    # Group by experiment and species
    for exp in all_experiments:
        exp_name = exp['experiment_name']
        
        # Collect all species results across folds
        species_results = {}
        for fold_result in exp['fold_results']:
            for species_info in fold_result['test_results']['species_summary']:
                species_name = species_info['species_name']
                if species_name not in species_results:
                    species_results[species_name] = []
                species_results[species_name].append(species_info)
        
        # Calculate average performance per species
        for species_name, species_folds in species_results.items():
            if len(species_folds) > 0:
                # Get consistent info
                first_fold = species_folds[0]
                
                # Calculate averages across folds
                avg_precision = np.mean([s['precision'] for s in species_folds])
                avg_recall = np.mean([s['recall'] for s in species_folds])
                avg_f1 = np.mean([s['f1_score'] for s in species_folds])
                avg_accuracy = np.mean([s['accuracy'] for s in species_folds])
                
                std_precision = np.std([s['precision'] for s in species_folds])
                std_recall = np.std([s['recall'] for s in species_folds])
                std_f1 = np.std([s['f1_score'] for s in species_folds])
                std_accuracy = np.std([s['accuracy'] for s in species_folds])
                
                species_summary_stats.append({
                    'Experiment': exp_name,
                    'Species_Name': species_name,
                    'Class_Index': first_fold['class_idx'],
                    'Avg_Test_Samples': np.mean([s['test_samples'] for s in species_folds]),
                    'Precision_Mean': avg_precision,
                    'Precision_Std': std_precision,
                    'Recall_Mean': avg_recall,
                    'Recall_Std': std_recall,
                    'F1_Mean': avg_f1,
                    'F1_Std': std_f1,
                    'Accuracy_Mean': avg_accuracy,
                    'Accuracy_Std': std_accuracy,
                    'Num_Folds': len(species_folds)
                })
    
    species_summary_df = pd.DataFrame(species_summary_stats)
    species_summary_df.to_csv(os.path.join(output_dir, 'cv_species_summary_stats.csv'), index=False)
    
    # REMOVED: Size category performance summary
    # No more size category analysis
    
    # Create misclassification summary statistics
    misc_summary = []
    for exp in all_experiments:
        exp_name = exp['experiment_name']
        
        # Collect all misclassification data
        all_misc_data = []
        for fold_result in exp['fold_results']:
            all_misc_data.extend(fold_result['test_results']['misclassification_data'])
        
        if all_misc_data:
            # Overall misclassification statistics
            total_misclass = len(all_misc_data)
            confidence_categories = Counter([m['confidence_category'] for m in all_misc_data])
            
            # Most confused species pairs
            confusion_pairs = Counter()
            for misc in all_misc_data:
                pair = (misc['true_species'], misc['predicted_species'])
                confusion_pairs[pair] += 1
            
            misc_summary.append({
                'Experiment': exp_name,
                'Total_Misclassifications': total_misclass,
                'High_Confidence_Errors': confidence_categories.get('High', 0),
                'Medium_Confidence_Errors': confidence_categories.get('Medium', 0),
                'Low_Confidence_Errors': confidence_categories.get('Low', 0),
                'Avg_Prediction_Confidence': np.mean([m['prediction_confidence'] for m in all_misc_data]),
                'Avg_True_Species_Confidence': np.mean([m['true_species_confidence'] for m in all_misc_data]),
                'Avg_True_Species_Rank': np.mean([m['true_species_rank'] for m in all_misc_data])
            })
    
    misc_summary_df = pd.DataFrame(misc_summary)
    misc_summary_df.to_csv(os.path.join(output_dir, 'cv_misclassification_summary.csv'), index=False)
    
    # Save complete results as JSON
    json_results = {}
    for exp in all_experiments:
        exp_name = exp['experiment_name']
        json_results[exp_name] = {
            'experiment_config': exp['experiment_config'],
            'cv_statistics': exp['cv_statistics'],
            'n_folds': exp['n_folds'],
            'epochs': exp['epochs'],
            'fold_results': []
        }
        
        for fold_result in exp['fold_results']:
            fold_data = {
                'fold': fold_result['fold'],
                'best_val_acc': float(fold_result['best_val_acc']),
                'test_results': {
                    key: float(value) if isinstance(value, (int, float, np.number)) else value
                    for key, value in fold_result['test_results'].items()
                    if key not in ['predictions', 'labels', 'probabilities', 'species_summary', 
                                 'confusion_matrix', 'misclassification_data']  # Skip large arrays
                }
            }
            
            # Handle both 'training_curves' and 'training_history' keys
            history_key = 'training_history' if 'training_history' in fold_result else 'training_curves'
            if history_key in fold_result:
                history = fold_result[history_key]
                fold_data['training_curves'] = {
                    'final_train_loss': float(history['train_losses'][-1]),
                    'final_val_loss': float(history['val_losses'][-1]),
                    'final_train_acc': float(history['train_accs'][-1]),
                    'final_val_acc': float(history['val_accs'][-1]),
                    'final_lr': float(history['learning_rates'][-1])
                }
            
            json_results[exp_name]['fold_results'].append(fold_data)
    
    with open(os.path.join(output_dir, 'cv_complete_results.json'), 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"✅ Results saved to {output_dir}")
    print(f"   - cv_summary_results.csv (overall summary)")
    print(f"   - cv_detailed_results.csv (per-fold results)")
    print(f"   - cv_species_level_results.csv (all species results)")
    print(f"   - cv_species_summary_stats.csv (species averages)")
    print(f"   - cv_misclassification_analysis.csv (detailed misclassification data)")
    print(f"   - cv_misclassification_summary.csv (misclassification statistics)")
    print(f"   - cv_complete_results.json (complete results)")
    
def plot_confusion_matrices(all_experiments, output_dir):
    """Create and save confusion matrix visualizations - MODIFIED VERSION"""
    
    for exp in all_experiments:
        exp_name = exp['experiment_name']
        
        # Skip individual fold confusion matrices - REMOVED
        
        # Create averaged confusion matrix across all folds
        all_cms = []
        class_names = None
        
        for fold_result in exp['fold_results']:
            cm = np.array(fold_result['test_results']['confusion_matrix'])
            all_cms.append(cm)
            
            # Get class names from the dataset (should be available in fold results)
            if class_names is None:
                # Get species names from the first fold's test results
                # Assuming you have access to the dataset classes
                # You'll need to pass this information or store it in test_results
                class_names = [name[:20] + '...' if len(name) > 20 else name 
                              for name in fold_result['test_results'].get('class_names', 
                              [f'Class_{i}' for i in range(cm.shape[0])])]
        
        # Calculate average confusion matrix
        avg_cm = np.mean(all_cms, axis=0)
        
        # Create figure for average confusion matrix
        plt.figure(figsize=(max(12, avg_cm.shape[0] * 0.6), max(10, avg_cm.shape[1] * 0.6)))
        sns.heatmap(avg_cm, annot=True, fmt='.1f', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Average Number of Predictions'})
        plt.title(f'{exp_name} - Average Confusion Matrix (5-Fold CV)', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Species', fontsize=12)
        plt.ylabel('True Species', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        
        # Save average confusion matrix
        filename = f'{exp_name.lower().replace(" ", "_")}_average_confusion_matrix.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        # CREATE TEST SET CONFUSION MATRIX - NEW ADDITION
        # Aggregate predictions across all folds for the test set
        all_test_preds = []
        all_test_labels = []
        
        for fold_result in exp['fold_results']:
            test_results = fold_result['test_results']
            all_test_preds.extend(test_results['predictions'])
            all_test_labels.extend(test_results['labels'])
        
        # Create test set confusion matrix
        test_cm = confusion_matrix(all_test_labels, all_test_preds)
        
        plt.figure(figsize=(max(12, test_cm.shape[0] * 0.6), max(10, test_cm.shape[1] * 0.6)))
        sns.heatmap(test_cm, annot=True, fmt='d', cmap='Oranges', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Number of Predictions'})
        plt.title(f'{exp_name} - Test Set Confusion Matrix (All Folds Combined)', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Species', fontsize=12)
        plt.ylabel('True Species', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        
        # Save test set confusion matrix
        filename = f'{exp_name.lower().replace(" ", "_")}_test_confusion_matrix.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

def plot_species_performance(all_experiments, output_dir):
    """Create visualizations for species-level performance - MODIFIED VERSION"""
    
    # Create species performance comparison plots
    for exp in all_experiments:
        exp_name = exp['experiment_name']
        
        # Collect species summary data
        species_data = []
        for fold_result in exp['fold_results']:
            for species_info in fold_result['test_results']['species_summary']:
                species_data.append({
                    'fold': fold_result['fold'],
                    'species_name': species_info['species_name'][:25] + '...' if len(species_info['species_name']) > 25 else species_info['species_name'],
                    'f1_score': species_info['f1_score'],
                    'accuracy': species_info['accuracy'],
                    'precision': species_info['precision'],
                    'recall': species_info['recall'],
                    'test_samples': species_info['test_samples']
                })
        
        species_df = pd.DataFrame(species_data)
        
        # Calculate species averages across folds
        species_avg = species_df.groupby('species_name').agg({
            'f1_score': ['mean', 'std'],
            'accuracy': ['mean', 'std'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'],
            'test_samples': 'mean'
        }).round(4)
        
        species_avg.columns = ['f1_mean', 'f1_std', 'acc_mean', 'acc_std', 
                              'prec_mean', 'prec_std', 'rec_mean', 'rec_std',
                              'test_samples']
        species_avg = species_avg.reset_index()
        
        # REMOVED: Performance by Size Category section
        # No more size category analysis
        
        # 1. Top and Bottom Performers
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # Sort by F1 score
        species_sorted = species_avg.sort_values('f1_mean', ascending=False)
        
        # Top 15 performers
        top_species = species_sorted.head(15)
        ax = axes[0, 0]
        bars = ax.barh(range(len(top_species)), top_species['f1_mean'], 
                      xerr=top_species['f1_std'], capsize=3, alpha=0.7, color='green')
        ax.set_yticks(range(len(top_species)))
        ax.set_yticklabels(top_species['species_name'], fontsize=8)
        ax.set_xlabel('F1 Score')
        ax.set_title('Top 15 Species by F1 Score')
        ax.grid(True, alpha=0.3)
        
        # Bottom 15 performers (excluding zero F1)
        bottom_species = species_sorted[species_sorted['f1_mean'] > 0].tail(15)
        ax = axes[0, 1]
        bars = ax.barh(range(len(bottom_species)), bottom_species['f1_mean'], 
                      xerr=bottom_species['f1_std'], capsize=3, alpha=0.7, color='red')
        ax.set_yticks(range(len(bottom_species)))
        ax.set_yticklabels(bottom_species['species_name'], fontsize=8)
        ax.set_xlabel('F1 Score')
        ax.set_title('Bottom 15 Species by F1 Score')
        ax.grid(True, alpha=0.3)
        
        # Sample size vs Performance scatter
        ax = axes[1, 0]
        scatter = ax.scatter(species_avg['test_samples'], species_avg['f1_mean'], 
                           c=species_avg['test_samples'], cmap='viridis', alpha=0.6, s=50)
        ax.set_xlabel('Test Samples')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score vs Sample Size')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Test Samples')
        
        # Performance distribution
        ax = axes[1, 1]
        ax.hist(species_avg['f1_mean'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(species_avg['f1_mean'].mean(), color='red', linestyle='--', 
                  label=f'Mean: {species_avg["f1_mean"].mean():.3f}')
        ax.axvline(species_avg['f1_mean'].median(), color='orange', linestyle='--', 
                  label=f'Median: {species_avg["f1_mean"].median():.3f}')
        ax.set_xlabel('F1 Score')
        ax.set_ylabel('Number of Species')
        ax.set_title('F1 Score Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{exp_name} - Species Performance Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{exp_name.lower().replace(" ", "_")}_species_analysis.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # REMOVED: Size Category Summary Heatmap
        # No more size category heatmap
    
    # Cross-experiment species comparison (if multiple experiments)
    if len(all_experiments) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Collect data from all experiments
        all_species_data = []
        for exp in all_experiments:
            exp_name = exp['experiment_name']
            
            for fold_result in exp['fold_results']:
                for species_info in fold_result['test_results']['species_summary']:
                    all_species_data.append({
                        'experiment': exp_name,
                        'species_name': species_info['species_name'][:20] + '...' if len(species_info['species_name']) > 20 else species_info['species_name'],
                        'f1_score': species_info['f1_score'],
                        'accuracy': species_info['accuracy'],
                        'test_samples': species_info['test_samples']
                    })
        
        all_species_df = pd.DataFrame(all_species_data)
        
        # Average performance by experiment
        exp_avg = all_species_df.groupby('experiment').agg({
            'f1_score': 'mean',
            'accuracy': 'mean'
        }).reset_index()
        
        # Plot 1: F1 by experiment
        ax = axes[0, 0]
        bars = ax.bar(exp_avg['experiment'], exp_avg['f1_score'], alpha=0.8, color=['#1f77b4', '#ff7f0e'])
        ax.set_title('Average F1 Score by Experiment')
        ax.set_ylabel('F1 Score')
        ax.set_xlabel('Experiment')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=15)
        
        # Add value labels on bars
        for bar, value in zip(bars, exp_avg['f1_score']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 2: Accuracy by experiment
        ax = axes[0, 1]
        bars = ax.bar(exp_avg['experiment'], exp_avg['accuracy'], alpha=0.8, color=['#1f77b4', '#ff7f0e'])
        ax.set_title('Average Accuracy by Experiment')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Experiment')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=15)
        
        # Add value labels on bars
        for bar, value in zip(bars, exp_avg['accuracy']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 3: Performance distribution comparison
        ax = axes[1, 0]
        for exp_name in all_species_df['experiment'].unique():
            exp_data = all_species_df[all_species_df['experiment'] == exp_name]['f1_score']
            ax.hist(exp_data, bins=15, alpha=0.6, label=exp_name, density=True)
        ax.set_xlabel('F1 Score')
        ax.set_ylabel('Density')
        ax.set_title('F1 Score Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Sample size effect comparison
        ax = axes[1, 1]
        for exp_name in all_species_df['experiment'].unique():
            exp_data = all_species_df[all_species_df['experiment'] == exp_name]
            ax.scatter(exp_data['test_samples'], exp_data['f1_score'], 
                      alpha=0.6, label=exp_name, s=30)
        ax.set_xlabel('Test Samples')
        ax.set_ylabel('F1 Score')
        ax.set_title('Sample Size vs F1 Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Cross-Experiment Species Performance Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cross_experiment_species_comparison.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()


def plot_misclassification_analysis(all_experiments, output_dir):
    """Create visualizations for misclassification analysis"""
    
    for exp in all_experiments:
        exp_name = exp['experiment_name']
        
        # Collect all misclassification data
        all_misc_data = []
        for fold_result in exp['fold_results']:
            misc_data = fold_result['test_results']['misclassification_data']
            for misc in misc_data:
                misc['fold'] = fold_result['fold']
                all_misc_data.append(misc)
        
        if not all_misc_data:
            continue
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. Confidence distribution of misclassifications
        ax = axes[0, 0]
        confidences = [misc['prediction_confidence'] for misc in all_misc_data]
        ax.hist(confidences, bins=20, alpha=0.7, color='red', edgecolor='black')
        ax.axvline(np.mean(confidences), color='blue', linestyle='--', 
                  label=f'Mean: {np.mean(confidences):.1f}%')
        ax.set_xlabel('Prediction Confidence (%)')
        ax.set_ylabel('Number of Misclassifications')
        ax.set_title('Confidence Distribution of Misclassifications')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Confidence categories
        ax = axes[0, 1]
        conf_categories = Counter([misc['confidence_category'] for misc in all_misc_data])
        categories = list(conf_categories.keys())
        counts = list(conf_categories.values())
        colors = ['red', 'orange', 'yellow']
        bars = ax.bar(categories, counts, color=colors[:len(categories)], alpha=0.7)
        ax.set_ylabel('Number of Misclassifications')
        ax.set_title('Misclassifications by Confidence Category')
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{count}', ha='center', va='bottom')
        ax.grid(True, alpha=0.3)
        
        # 3. True species rank distribution
        ax = axes[0, 2]
        true_ranks = [misc['true_species_rank'] for misc in all_misc_data]
        ax.hist(true_ranks, bins=range(1, max(true_ranks) + 2), alpha=0.7, 
               color='green', edgecolor='black')
        ax.set_xlabel('True Species Rank in Predictions')
        ax.set_ylabel('Number of Misclassifications')
        ax.set_title('True Species Rank Distribution')
        ax.grid(True, alpha=0.3)
        
        # 4. Most confused species pairs
        ax = axes[1, 0]
        confusion_pairs = Counter()
        for misc in all_misc_data:
            pair = f"{misc['true_species'][:15]}→{misc['predicted_species'][:15]}"
            confusion_pairs[pair] += 1
        
        top_pairs = confusion_pairs.most_common(15)
        if top_pairs:
            pairs, counts = zip(*top_pairs)
            y_pos = range(len(pairs))
            bars = ax.barh(y_pos, counts, alpha=0.7, color='purple')
            ax.set_yticks(y_pos)
            ax.set_yticklabels([p.replace('→', '\n→\n') for p in pairs], fontsize=8)
            ax.set_xlabel('Number of Confusions')
            ax.set_title('Top 15 Most Confused Species Pairs')
            ax.grid(True, alpha=0.3)
        
        # 5. Confidence vs True Species Confidence
        ax = axes[1, 1]
        pred_conf = [misc['prediction_confidence'] for misc in all_misc_data]
        true_conf = [misc['true_species_confidence'] for misc in all_misc_data]
        scatter = ax.scatter(pred_conf, true_conf, alpha=0.6, c=true_ranks, 
                           cmap='viridis', s=30)
        ax.plot([0, 100], [0, 100], 'r--', alpha=0.5, label='Equal Confidence')
        ax.set_xlabel('Prediction Confidence (%)')
        ax.set_ylabel('True Species Confidence (%)')
        ax.set_title('Prediction vs True Species Confidence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='True Species Rank')
        
        # 6. Misclassification rate by fold
        ax = axes[1, 2]
        fold_misc_counts = Counter([misc['fold'] for misc in all_misc_data])
        folds = sorted(fold_misc_counts.keys())
        counts = [fold_misc_counts[fold] for fold in folds]
        
        ax.bar(folds, counts, alpha=0.7, color='orange')
        ax.set_xlabel('Fold Number')
        ax.set_ylabel('Number of Misclassifications')
        ax.set_title('Misclassifications by Fold')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{exp_name} - Misclassification Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save misclassification analysis
        filename = f'{exp_name.lower().replace(" ", "_")}_misclassification_analysis.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create detailed confidence analysis
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # High confidence errors analysis
        high_conf_errors = [misc for misc in all_misc_data if misc['confidence_category'] == 'High']
        if high_conf_errors:
            ax = axes[0]
            # Most frequent high confidence errors
            high_conf_pairs = Counter()
            for misc in high_conf_errors:
                pair = f"{misc['true_species'][:12]}→{misc['predicted_species'][:12]}"
                high_conf_pairs[pair] += 1
            
            top_high_conf = high_conf_pairs.most_common(10)
            if top_high_conf:
                pairs, counts = zip(*top_high_conf)
                y_pos = range(len(pairs))
                ax.barh(y_pos, counts, alpha=0.7, color='red')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(pairs, fontsize=9)
                ax.set_xlabel('Count')
                ax.set_title(f'Top High Confidence Errors\n({len(high_conf_errors)} total)')
                ax.grid(True, alpha=0.3)
        
        # Medium confidence errors
        med_conf_errors = [misc for misc in all_misc_data if misc['confidence_category'] == 'Medium']
        if med_conf_errors:
            ax = axes[1]
            med_conf_pairs = Counter()
            for misc in med_conf_errors:
                pair = f"{misc['true_species'][:12]}→{misc['predicted_species'][:12]}"
                med_conf_pairs[pair] += 1
            
            top_med_conf = med_conf_pairs.most_common(10)
            if top_med_conf:
                pairs, counts = zip(*top_med_conf)
                y_pos = range(len(pairs))
                ax.barh(y_pos, counts, alpha=0.7, color='orange')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(pairs, fontsize=9)
                ax.set_xlabel('Count')
                ax.set_title(f'Top Medium Confidence Errors\n({len(med_conf_errors)} total)')
                ax.grid(True, alpha=0.3)
        
        # Low confidence errors
        low_conf_errors = [misc for misc in all_misc_data if misc['confidence_category'] == 'Low']
        if low_conf_errors:
            ax = axes[2]
            low_conf_pairs = Counter()
            for misc in low_conf_errors:
                pair = f"{misc['true_species'][:12]}→{misc['predicted_species'][:12]}"
                low_conf_pairs[pair] += 1
            
            top_low_conf = low_conf_pairs.most_common(10)
            if top_low_conf:
                pairs, counts = zip(*top_low_conf)
                y_pos = range(len(pairs))
                ax.barh(y_pos, counts, alpha=0.7, color='yellow')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(pairs, fontsize=9)
                ax.set_xlabel('Count')
                ax.set_title(f'Top Low Confidence Errors\n({len(low_conf_errors)} total)')
                ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{exp_name} - Confidence-Based Error Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save confidence analysis
        filename = f'{exp_name.lower().replace(" ", "_")}_confidence_error_analysis.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()


def generate_cv_report(all_experiments, full_dataset, output_dir):
    """Generate comprehensive cross-validation report - FIXED VERSION"""
    
    report_path = os.path.join(output_dir, 'cv_final_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("ResNet50 Final Training - Cross Validation Report\n")
        f.write("=" * 60 + "\n\n")
        
        # Dataset info
        f.write("Dataset Information:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total samples: {len(full_dataset)}\n")
        f.write(f"Number of classes: {len(full_dataset.classes)}\n")
        
        class_counts = {}
        for _, label in full_dataset.samples:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        counts_array = np.array(list(class_counts.values()))
        f.write(f"Class distribution: min={counts_array.min()}, max={counts_array.max()}, mean={counts_array.mean():.1f}\n")
        f.write(f"Imbalance ratio: {counts_array.max() / counts_array.min():.2f}\n\n")
        
        # Experiment configurations
        f.write("Experiment Configurations:\n")
        f.write("-" * 35 + "\n")
        for exp in all_experiments:
            f.write(f"\n{exp['experiment_name']}:\n")
            config = exp['experiment_config']
            for key, value in config.items():
                f.write(f"  {key}: {value}\n")
        
        # Cross-validation results
        f.write(f"\nCross-Validation Results ({all_experiments[0]['n_folds']}-Fold, {all_experiments[0]['epochs']} Epochs):\n")
        f.write("-" * 60 + "\n")
        
        f.write(f"{'Experiment':<20} {'Metric':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}\n")
        f.write("-" * 80 + "\n")
        
        for exp in all_experiments:
            exp_name = exp['experiment_name']
            cv_stats = exp['cv_statistics']
            
            # Key metrics
            key_metrics = ['test_acc', 'macro_f1', 'minority_f1', 'kappa']
            metric_names = ['Test Acc', 'Macro F1', 'Minority F1', 'Kappa']
            
            for metric, metric_name in zip(key_metrics, metric_names):
                stats = cv_stats[metric]
                f.write(f"{exp_name:<20} {metric_name:<15} {stats['mean']:<10.4f} "
                       f"{stats['std']:<10.4f} {stats['min']:<10.4f} {stats['max']:<10.4f}\n")
            f.write("\n")
        
        # Best results comparison
        f.write("Best Results Comparison:\n")
        f.write("-" * 30 + "\n")
        
        # Find best experiment for each metric
        best_experiments = {}
        for metric in ['test_acc', 'macro_f1', 'minority_f1', 'kappa']:
            best_exp = max(all_experiments, key=lambda x: x['cv_statistics'][metric]['mean'])
            best_experiments[metric] = {
                'name': best_exp['experiment_name'],
                'mean': best_exp['cv_statistics'][metric]['mean'],
                'std': best_exp['cv_statistics'][metric]['std']
            }
        
        f.write(f"Best Test Accuracy: {best_experiments['test_acc']['name']} ")
        f.write(f"({best_experiments['test_acc']['mean']:.4f} ± {best_experiments['test_acc']['std']:.4f})\n")
        
        f.write(f"Best Macro F1: {best_experiments['macro_f1']['name']} ")
        f.write(f"({best_experiments['macro_f1']['mean']:.4f} ± {best_experiments['macro_f1']['std']:.4f})\n")
        
        f.write(f"Best Minority F1: {best_experiments['minority_f1']['name']} ");
        f.write(f"({best_experiments['minority_f1']['mean']:.4f} ± {best_experiments['minority_f1']['std']:.4f})\n")
        
        f.write(f"Best Kappa: {best_experiments['kappa']['name']} ");
        f.write(f"({best_experiments['kappa']['mean']:.4f} ± {best_experiments['kappa']['std']:.4f})\n\n")
        
        # Statistical significance analysis
        f.write("Statistical Analysis:\n")
        f.write("-" * 25 + "\n")
        
        if len(all_experiments) == 2:
            from scipy import stats
            
            exp1, exp2 = all_experiments
            f.write(f"Comparing {exp1['experiment_name']} vs {exp2['experiment_name']}:\n\n")
            
            for metric in ['test_acc', 'macro_f1', 'minority_f1']:
                values1 = exp1['cv_statistics'][metric]['values']
                values2 = exp2['cv_statistics'][metric]['values']
                
                # Paired t-test
                t_stat, p_value = stats.ttest_rel(values1, values2)
                
                f.write(f"{metric.upper()}:\n")
                f.write(f"  {exp1['experiment_name']}: {np.mean(values1):.4f} ± {np.std(values1):.4f}\n")
                f.write(f"  {exp2['experiment_name']}: {np.mean(values2):.4f} ± {np.std(values2):.4f}\n")
                f.write(f"  Paired t-test: t={t_stat:.4f}, p={p_value:.4f}\n")
                f.write(f"  Significant (p<0.05): {'Yes' if p_value < 0.05 else 'No'}\n\n")
        
        # Final recommendations
        f.write("Final Recommendations:\n")
        f.write("-" * 25 + "\n")
        
        # Overall best
        overall_best = max(all_experiments, 
                         key=lambda x: (x['cv_statistics']['macro_f1']['mean'] + 
                                      x['cv_statistics']['test_acc']['mean']) / 2)
        
        f.write(f"1. RECOMMENDED CONFIGURATION:\n")
        f.write(f"   Method: {overall_best['experiment_name']}\n")
        f.write(f"   Expected Performance:\n")
        f.write(f"     - Test Accuracy: {overall_best['cv_statistics']['test_acc']['mean']:.4f} ± {overall_best['cv_statistics']['test_acc']['std']:.4f}\n")
        f.write(f"     - Macro F1: {overall_best['cv_statistics']['macro_f1']['mean']:.4f} ± {overall_best['cv_statistics']['macro_f1']['std']:.4f}\n")
        f.write(f"     - Minority F1: {overall_best['cv_statistics']['minority_f1']['mean']:.4f} ± {overall_best['cv_statistics']['minority_f1']['std']:.4f}\n\n")
        
        f.write(f"2. CONFIGURATION DETAILS:\n")
        config = overall_best['experiment_config']
        for key, value in config.items():
            f.write(f"   {key}: {value}\n")
        
        f.write(f"\n3. TRAINING RECOMMENDATIONS:\n")
        f.write(f"   - Use 5-fold cross-validation for model selection\n")
        f.write(f"   - Train for {overall_best['epochs']} epochs with early stopping\n")
        f.write(f"   - Monitor validation loss for plateau detection\n")
        f.write(f"   - Use gradient clipping and warmup for stable training\n")
        f.write(f"   - Expected training time: ~{overall_best['epochs'] * 5 * 2} minutes total\n")

        # Species-level analysis (REMOVED size_category analysis)
        f.write("Species-Level Performance Analysis:\n")
        f.write("-" * 40 + "\n")
        
        for exp in all_experiments:
            exp_name = exp['experiment_name']
            f.write(f"\n{exp_name}:\n")
            
            # Collect all species data across folds
            all_species_results = []
            for fold_result in exp['fold_results']:
                all_species_results.extend(fold_result['test_results']['species_summary'])
            
            # Group by species and calculate averages
            species_groups = {}
            for species_info in all_species_results:
                species_name = species_info['species_name']
                if species_name not in species_groups:
                    species_groups[species_name] = []
                species_groups[species_name].append(species_info)
            
            # Calculate species statistics
            species_stats = []
            for species_name, species_folds in species_groups.items():
                avg_f1 = np.mean([s['f1_score'] for s in species_folds])
                avg_acc = np.mean([s['accuracy'] for s in species_folds])
                test_samples = species_folds[0]['test_samples']
                
                species_stats.append({
                    'name': species_name,
                    'f1': avg_f1,
                    'accuracy': avg_acc,
                    'samples': test_samples
                })
            
            # Sort by F1 score
            species_stats.sort(key=lambda x: x['f1'], reverse=True)
            
            # REMOVED: Performance by size category section
            
            # Top and bottom performers
            f.write(f"  Top 10 Performing Species:\n")
            for i, species in enumerate(species_stats[:10]):
                f.write(f"    {i+1:2d}. {species['name'][:50]:<50} F1={species['f1']:.3f} (n={species['samples']})\n")
            
            f.write(f"\n  Bottom 10 Performing Species (F1 > 0):\n")
            bottom_species = [s for s in species_stats if s['f1'] > 0][-10:]
            for i, species in enumerate(bottom_species):
                f.write(f"    {i+1:2d}. {species['name'][:50]:<50} F1={species['f1']:.3f} (n={species['samples']})\n")
            
            # Performance statistics
            all_f1s = [s['f1'] for s in species_stats if s['f1'] > 0]  # Exclude zero F1 scores
            f.write(f"\n  Overall Species Performance Statistics:\n")
            f.write(f"    Species with F1 > 0: {len(all_f1s)}/{len(species_stats)}\n")
            f.write(f"    Mean F1: {np.mean(all_f1s):.3f}\n")
            f.write(f"    Median F1: {np.median(all_f1s):.3f}\n")
            f.write(f"    Std F1: {np.std(all_f1s):.3f}\n")
            f.write(f"    F1 >= 0.8: {sum(1 for f1 in all_f1s if f1 >= 0.8)}\n")
            f.write(f"    F1 >= 0.5: {sum(1 for f1 in all_f1s if f1 >= 0.5)}\n")
            f.write(f"    F1 < 0.1: {sum(1 for f1 in all_f1s if f1 < 0.1)}\n")
        
        # Performance by sample size analysis (REPLACEMENT for size categories)
        f.write("\nPerformance by Sample Size Analysis:\n")
        f.write("-" * 45 + "\n")
        
        for exp in all_experiments:
            exp_name = exp['experiment_name']
            f.write(f"\n{exp_name}:\n")
            
            # Collect all species data
            all_species_results = []
            for fold_result in exp['fold_results']:
                all_species_results.extend(fold_result['test_results']['species_summary'])
            
            # Group by species and get average performance
            species_performance = {}
            for species_info in all_species_results:
                species_name = species_info['species_name']
                if species_name not in species_performance:
                    species_performance[species_name] = {
                        'f1_scores': [],
                        'test_samples': species_info['test_samples']
                    }
                species_performance[species_name]['f1_scores'].append(species_info['f1_score'])
            
            # Calculate average F1 per species
            sample_size_performance = []
            for species_name, data in species_performance.items():
                avg_f1 = np.mean(data['f1_scores'])
                sample_size_performance.append((data['test_samples'], avg_f1))
            
            # Group by sample size ranges
            size_ranges = {
                'Very Low (1-2)': [],
                'Low (3-5)': [],
                'Medium (6-10)': [],
                'High (11-20)': [],
                'Very High (21+)': []
            }
            
            for samples, f1 in sample_size_performance:
                if samples <= 2:
                    size_ranges['Very Low (1-2)'].append(f1)
                elif samples <= 5:
                    size_ranges['Low (3-5)'].append(f1)
                elif samples <= 10:
                    size_ranges['Medium (6-10)'].append(f1)
                elif samples <= 20:
                    size_ranges['High (11-20)'].append(f1)
                else:
                    size_ranges['Very High (21+)'].append(f1)
            
            # Report performance by sample size
            for size_range, f1_scores in size_ranges.items():
                if f1_scores:
                    avg_f1 = np.mean(f1_scores)
                    count = len(f1_scores)
                    f.write(f"    {size_range}: {count} species, Avg F1 = {avg_f1:.3f}\n")
        
        f.write(f"\n4. SAMPLE SIZE RECOMMENDATIONS:\n")
        f.write(f"   - Species with <3 samples show significantly lower performance\n")
        f.write(f"   - Optimal performance achieved with 6+ samples per species\n")
        f.write(f"   - Consider data augmentation for species with <5 samples\n")
        f.write(f"   - Focus data collection efforts on underrepresented species\n")
    
    print(f"✅ Enhanced report saved to {report_path}")
    print(f"📋 Report includes:")
    print(f"├── Dataset statistics")
    print(f"├── Cross-validation results")
    print(f"├── Statistical significance analysis")
    print(f"├── Species-level performance analysis")
    print(f"├── Performance by sample size analysis")
    print(f"└── Final recommendations")
                
# Enhanced training function with comprehensive metrics tracking
def train_model_with_metrics(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                           warmup_scheduler, device, num_epochs, experiment_name, fold_num,
                           gradient_clip_norm=1.0):
    
    model.to(device)
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    # Enhanced tracking dictionaries
    training_history = {
        'train_losses': [],
        'val_losses': [],
        'train_accs': [],
        'val_accs': [],
        'learning_rates': [],
        # New metrics to track
        'val_balanced_accs': [],
        'val_macro_f1s': [],
        'val_micro_f1s': [],
        'val_minority_f1s': [],
        'val_kappas': []
    }

    print(f"\n=== Training {experiment_name} - Fold {fold_num} for {num_epochs} epochs ===")

    for epoch in range(num_epochs):
        # Warmup phase
        if warmup_scheduler and epoch < warmup_scheduler.warmup_epochs:
            warmup_scheduler.step()
        
        # Training phase
        model.train()
        running_loss, running_corrects, total = 0.0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        epoch_train_loss = running_loss / total
        epoch_train_acc = running_corrects / total
        training_history['train_losses'].append(epoch_train_loss)
        training_history['train_accs'].append(epoch_train_acc)

        # Enhanced validation phase with comprehensive metrics
        model.eval()
        val_running_loss, val_running_corrects, val_total = 0.0, 0, 0
        all_val_preds, all_val_labels = [], []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels).item()
                val_total += labels.size(0)
                
                # Collect predictions and labels for comprehensive metrics
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_running_loss / val_total
        epoch_val_acc = val_running_corrects / val_total
        training_history['val_losses'].append(epoch_val_loss)
        training_history['val_accs'].append(epoch_val_acc)
        
        # Calculate comprehensive validation metrics
        val_balanced_acc = balanced_accuracy_score(all_val_labels, all_val_preds)
        val_macro_f1 = f1_score(all_val_labels, all_val_preds, average='macro', zero_division=0)
        val_micro_f1 = f1_score(all_val_labels, all_val_preds, average='micro', zero_division=0)
        val_kappa = cohen_kappa_score(all_val_labels, all_val_preds)
        
        # Calculate minority F1 (classes with < 5% of total samples)
        from collections import Counter
        label_counts = Counter(all_val_labels)
        total_samples = len(all_val_labels)
        minority_threshold = total_samples * 0.05
        minority_classes = [cls for cls, count in label_counts.items() if count < minority_threshold]
        
        if minority_classes:
            minority_mask = [label in minority_classes for label in all_val_labels]
            minority_labels = [all_val_labels[i] for i, is_min in enumerate(minority_mask) if is_min]
            minority_preds = [all_val_preds[i] for i, is_min in enumerate(minority_mask) if is_min]
            
            if minority_labels:
                val_minority_f1 = f1_score(minority_labels, minority_preds, average='macro', zero_division=0)
            else:
                val_minority_f1 = 0.0
        else:
            val_minority_f1 = 0.0
        
        # Store comprehensive metrics
        training_history['val_balanced_accs'].append(val_balanced_acc)
        training_history['val_macro_f1s'].append(val_macro_f1)
        training_history['val_micro_f1s'].append(val_micro_f1)
        training_history['val_minority_f1s'].append(val_minority_f1)
        training_history['val_kappas'].append(val_kappa)
        
        # Record learning rate
        current_lr = optimizer.param_groups[0]['lr']
        training_history['learning_rates'].append(current_lr)
        
        # Step scheduler (after warmup)
        if epoch >= (warmup_scheduler.warmup_epochs if warmup_scheduler else 0):
            if hasattr(scheduler, 'step'):
                if 'ReduceLROnPlateau' in str(type(scheduler)):
                    scheduler.step(epoch_val_loss)
                else:
                    scheduler.step()

        # Save best model (can use different metric for best model selection)
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = model.state_dict()

        # Enhanced logging
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train: Loss={epoch_train_loss:.4f}, Acc={epoch_train_acc:.4f}")
        print(f"  Val: Loss={epoch_val_loss:.4f}, Acc={epoch_val_acc:.4f}, Bal_Acc={val_balanced_acc:.4f}")
        print(f"  Val: Macro_F1={val_macro_f1:.4f}, Micro_F1={val_micro_f1:.4f}, Min_F1={val_minority_f1:.4f}, Kappa={val_kappa:.4f}")
        print(f"  LR: {current_lr:.6f}")
        print("-" * 80)

    model.load_state_dict(best_model_wts)
    print(f"Best Val Accuracy for Fold {fold_num}: {best_acc:.4f}")

    return {
        'model': model,
        'best_val_acc': best_acc,
        'training_history': training_history
    }


# Enhanced plotting function for comprehensive metrics
def plot_comprehensive_training_curves(all_experiments, output_dir):
    """Plot comprehensive training curves including all metrics"""
    
    for exp in all_experiments:
        exp_name = exp['experiment_name']
        fold_results = exp['fold_results']
        epochs = exp['epochs']
        
        # Create larger figure for more subplots
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        colors = plt.cm.Set3(np.linspace(0, 1, len(fold_results)))
        epochs_range = range(1, epochs + 1)
        
        # Collect all metrics data
        metrics_data = {}
        metric_names = ['train_losses', 'val_losses', 'train_accs', 'val_accs', 
                       'val_balanced_accs', 'val_macro_f1s', 'val_micro_f1s', 
                       'val_minority_f1s', 'val_kappas']
        
        for metric in metric_names:
            metrics_data[metric] = []
        
        # Plot individual fold curves
        for fold_idx, fold_result in enumerate(fold_results):
            history = fold_result['training_history']
            
            for metric_idx, metric in enumerate(metric_names):
                if metric in history:
                    row = metric_idx // 3
                    col = metric_idx % 3
                    
                    if row < 3 and col < 3:  # Ensure we don't exceed subplot limits
                        axes[row, col].plot(epochs_range, history[metric], 
                                          label=f'Fold {fold_result["fold"]}', 
                                          color=colors[fold_idx], linewidth=1.5, alpha=0.7)
                        
                        # Store data for averaging
                        metrics_data[metric].append(history[metric])
        
        # Plot average curves and set titles
        for metric_idx, metric in enumerate(metric_names):
            if metric_idx >= 9:  # Only plot first 9 metrics
                break
                
            row = metric_idx // 3
            col = metric_idx % 3
            
            if metrics_data[metric]:
                avg_values = np.mean(metrics_data[metric], axis=0)
                axes[row, col].plot(epochs_range, avg_values, 'k--', linewidth=3, label='Average')
            
            # Set titles and labels
            metric_display_names = {
                'train_losses': 'Training Loss',
                'val_losses': 'Validation Loss',
                'train_accs': 'Training Accuracy',
                'val_accs': 'Validation Accuracy',
                'val_balanced_accs': 'Validation Balanced Accuracy',
                'val_macro_f1s': 'Validation Macro F1',
                'val_micro_f1s': 'Validation Micro F1',
                'val_minority_f1s': 'Validation Minority F1',
                'val_kappas': 'Validation Cohen\'s Kappa'
            }
            
            axes[row, col].set_title(metric_display_names.get(metric, metric))
            axes[row, col].set_xlabel('Epoch')
            axes[row, col].set_ylabel(metric_display_names.get(metric, metric))
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
            
            # Special handling for loss plots (log scale might be helpful)
            if 'loss' in metric.lower():
                axes[row, col].set_yscale('log')
        
        plt.suptitle(f'{exp_name} - Comprehensive Training Curves (5-Fold CV)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{exp_name.lower().replace(" ", "_")}_comprehensive_training_curves.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Create separate figure for metrics comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Final epoch metrics comparison across folds
        final_metrics = ['val_accs', 'val_balanced_accs', 'val_macro_f1s', 
                        'val_micro_f1s', 'val_minority_f1s', 'val_kappas']
        final_metric_names = ['Validation Accuracy', 'Balanced Accuracy', 'Macro F1', 
                             'Micro F1', 'Minority F1', 'Cohen\'s Kappa']
        
        for idx, (metric, metric_name) in enumerate(zip(final_metrics, final_metric_names)):
            if idx >= 6:
                break
                
            row = idx // 3
            col = idx % 3
            
            # Get final values from each fold
            final_values = []
            for fold_result in fold_results:
                history = fold_result['training_history']
                if metric in history and len(history[metric]) > 0:
                    final_values.append(history[metric][-1])  # Last epoch value
            
            if final_values:
                fold_numbers = [fold_result['fold'] for fold_result in fold_results[:len(final_values)]]
                
                bars = axes[row, col].bar(fold_numbers, final_values, alpha=0.7, 
                                        color=colors[:len(final_values)])
                axes[row, col].set_title(f'Final {metric_name} by Fold')
                axes[row, col].set_xlabel('Fold')
                axes[row, col].set_ylabel(metric_name)
                axes[row, col].grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, final_values):
                    height = bar.get_height()
                    axes[row, col].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                                       f'{value:.3f}', ha='center', va='bottom', fontsize=10)
                
                # Add mean line
                mean_value = np.mean(final_values)
                axes[row, col].axhline(y=mean_value, color='red', linestyle='--', 
                                     label=f'Mean: {mean_value:.3f}')
                axes[row, col].legend()
        
        plt.suptitle(f'{exp_name} - Final Metrics Comparison Across Folds', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{exp_name.lower().replace(" ", "_")}_final_metrics_comparison.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()


# Enhanced summary statistics function
def calculate_comprehensive_cv_statistics(test_results, training_histories):
    """Calculate comprehensive cross-validation statistics including training curves"""
    
    # Existing test metrics
    test_metrics = ['test_acc', 'balanced_acc', 'macro_f1', 'weighted_f1', 'micro_f1', 
                   'minority_f1', 'kappa']
    
    cv_stats = {}
    
    # Test metrics statistics
    for metric in test_metrics:
        values = [result[metric] for result in test_results]
        cv_stats[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'values': values
        }
    
    # Training curve metrics statistics (final epoch values)
    training_metrics = ['val_balanced_accs', 'val_macro_f1s', 'val_micro_f1s', 
                       'val_minority_f1s', 'val_kappas']
    
    for metric in training_metrics:
        values = []
        for history in training_histories:
            if metric in history and len(history[metric]) > 0:
                values.append(history[metric][-1])  # Final epoch value
        
        if values:
            cv_stats[f'final_{metric}'] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
    
    # Convergence analysis
    convergence_stats = {}
    for metric in ['val_losses', 'val_accs', 'val_macro_f1s']:
        convergence_epochs = []
        for history in training_histories:
            if metric in history:
                values = history[metric]
                # Find epoch where improvement < 0.001 for 3 consecutive epochs
                convergence_epoch = len(values)  # Default to last epoch
                for i in range(3, len(values)):
                    if metric == 'val_losses':
                        # For loss, look for when it stops decreasing
                        recent_values = values[i-3:i]
                        if all(abs(recent_values[j] - recent_values[j+1]) < 0.001 for j in range(len(recent_values)-1)):
                            convergence_epoch = i
                            break
                    else:
                        # For other metrics, look for when it stops increasing
                        recent_values = values[i-3:i]
                        if all(abs(recent_values[j+1] - recent_values[j]) < 0.001 for j in range(len(recent_values)-1)):
                            convergence_epoch = i
                            break
                convergence_epochs.append(convergence_epoch)
        
        if convergence_epochs:
            convergence_stats[f'{metric}_convergence'] = {
                'mean_epoch': np.mean(convergence_epochs),
                'std_epoch': np.std(convergence_epochs),
                'min_epoch': np.min(convergence_epochs),
                'max_epoch': np.max(convergence_epochs)
            }
    
    cv_stats['convergence_analysis'] = convergence_stats
    
    return cv_stats

def run_cross_validation_experiment_enhanced(dataset, device, output_dir, experiment_config, 
                                            experiment_name, n_folds=5, epochs=20):
    
    print(f"\n{'='*80}")
    print(f"ENHANCED CROSS-VALIDATION EXPERIMENT: {experiment_name}")
    print(f"Configuration: {experiment_config}")
    print(f"Folds: {n_folds}, Epochs: {epochs}")
    print(f"{'='*80}")
    
    transforms_dict = get_transforms()
    
    # Get all labels for stratified CV
    labels = [sample[1] for sample in dataset.samples]
    indices = list(range(len(dataset)))
    
    # First split: separate test set (10%)
    train_val_indices, test_indices, train_val_labels, _ = train_test_split(
        indices, labels, test_size=0.1, stratify=labels, random_state=42
    )
    
    # Create test set
    test_dataset = Subset(dataset, test_indices)
    test_dataset.dataset.transform = transforms_dict['val']
    test_loader = DataLoader(test_dataset, experiment_config['batch_size'], shuffle=False, num_workers=2)
    
    # 5-fold CV on remaining 90% of data
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    all_test_results = []
    all_training_histories = []  # New: store training histories
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_indices, train_val_labels)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        
        # Map back to original indices
        fold_train_indices = [train_val_indices[i] for i in train_idx]
        fold_val_indices = [train_val_indices[i] for i in val_idx]
        
        # Create datasets for this fold
        train_dataset = Subset(dataset, fold_train_indices)
        val_dataset = Subset(dataset, fold_val_indices)
        
        train_dataset.dataset.transform = transforms_dict['train']
        val_dataset.dataset.transform = transforms_dict['val']
        
        print(f"Fold {fold + 1} - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=experiment_config['batch_size'], 
                                shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, experiment_config['batch_size'], shuffle=False, num_workers=2)
        
        # Model setup
        model = create_model(
            len(dataset.classes),
            experiment_config['dropout_rate']
        )
        
        # Optimizer setup
        if experiment_config['optimizer'] == 'adamw':
            optimizer = optim.AdamW(model.parameters(), 
                                  lr=experiment_config['learning_rate'],
                                  weight_decay=experiment_config['weight_decay'])
        else:
            optimizer = optim.Adam(model.parameters(), 
                                 lr=experiment_config['learning_rate'],
                                 weight_decay=experiment_config['weight_decay'])
        
        # Loss function setup
        if experiment_config['loss_type'] == 'focal_weighted':
            class_weights = calculate_class_weights(train_dataset, method=experiment_config['weight_method']).to(device)
            criterion = FocalLoss(
                alpha=class_weights,  # Use class weights
                gamma=experiment_config['focal_gamma'],
                label_smoothing=experiment_config['label_smoothing']
            )
        else:
            # Baseline case
            if experiment_config.get('label_smoothing', 0.0) > 0:
                criterion = nn.CrossEntropyLoss(label_smoothing=experiment_config['label_smoothing'])
            else:
                criterion = nn.CrossEntropyLoss()

        
        # Scheduler setup
        if experiment_config['scheduler_type'] == 'reduce_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', 
                patience=experiment_config['scheduler_patience'],
                factor=experiment_config['scheduler_factor']
            )
        # else:
        #     scheduler = optim.lr_scheduler.StepLR(
        #         optimizer,
        #         step_size=experiment_config['step_size'],
        #         gamma=experiment_config['gamma_scheduler']
        #     )
        
        # Warmup scheduler
        warmup_scheduler = WarmupScheduler(
            optimizer, 
            experiment_config['warmup_epochs'], 
            experiment_config['learning_rate']
        ) if experiment_config['warmup_epochs'] > 0 else None
        
        # Train model with enhanced metrics tracking
        train_result = train_model_with_metrics(  # Using new function
            model, train_loader, val_loader, criterion, optimizer, scheduler, 
            warmup_scheduler, device, epochs, experiment_name, fold + 1,
            experiment_config['gradient_clip_norm']
        )
        
        # Evaluate on test set
        test_result = evaluate_model(
            train_result['model'], test_loader, criterion, device, 
            dataset.classes, experiment_name, fold + 1
        )
        
        # Store results
        fold_result = {
            'fold': fold + 1,
            'best_val_acc': train_result['best_val_acc'],
            'training_history': train_result['training_history'],  # Enhanced training history
            'test_results': test_result
        }
        
        fold_results.append(fold_result)
        all_test_results.append(test_result)
        all_training_histories.append(train_result['training_history'])  # Store for analysis
    
    # Calculate enhanced cross-validation statistics
    cv_stats = calculate_comprehensive_cv_statistics(all_test_results, all_training_histories)
    
    return {
        'experiment_name': experiment_name,
        'experiment_config': experiment_config,
        'fold_results': fold_results,
        'cv_statistics': cv_stats,
        'n_folds': n_folds,
        'epochs': epochs
    }


# Enhanced results saving with training curve data
def save_enhanced_cv_results(all_experiments, output_dir):
    """Save enhanced cross-validation results including training curves"""
    
    # Original CSV files (keeping existing functionality)
    save_cv_results(all_experiments, output_dir)
    
    # Additional training curves data
    training_curves_data = []
    
    for exp in all_experiments:
        exp_name = exp['experiment_name']
        
        for fold_result in exp['fold_results']:
            fold_num = fold_result['fold']
            history = fold_result['training_history']
            
            # Save epoch-by-epoch data
            for epoch in range(len(history['train_losses'])):
                epoch_data = {
                    'Experiment': exp_name,
                    'Fold': fold_num,
                    'Epoch': epoch + 1,
                    'Train_Loss': history['train_losses'][epoch],
                    'Val_Loss': history['val_losses'][epoch],
                    'Train_Acc': history['train_accs'][epoch],
                    'Val_Acc': history['val_accs'][epoch],
                    'Learning_Rate': history['learning_rates'][epoch]
                }
                
                # Add enhanced metrics if available
                if 'val_balanced_accs' in history and epoch < len(history['val_balanced_accs']):
                    epoch_data.update({
                        'Val_Balanced_Acc': history['val_balanced_accs'][epoch],
                        'Val_Macro_F1': history['val_macro_f1s'][epoch],
                        'Val_Micro_F1': history['val_micro_f1s'][epoch],
                        'Val_Minority_F1': history['val_minority_f1s'][epoch],
                        'Val_Kappa': history['val_kappas'][epoch]
                    })
                
                training_curves_data.append(epoch_data)
    
    # Save training curves data
    training_curves_df = pd.DataFrame(training_curves_data)
    training_curves_df.to_csv(os.path.join(output_dir, 'cv_training_curves_detailed.csv'), index=False)
    
    # Training curves summary (final epoch values)
    final_curves_data = []
    for exp in all_experiments:
        exp_name = exp['experiment_name']
        
        for fold_result in exp['fold_results']:
            fold_num = fold_result['fold']
            history = fold_result['training_history']
            
            final_data = {
                'Experiment': exp_name,
                'Fold': fold_num,
                'Final_Train_Loss': history['train_losses'][-1],
                'Final_Val_Loss': history['val_losses'][-1],
                'Final_Train_Acc': history['train_accs'][-1],
                'Final_Val_Acc': history['val_accs'][-1],
                'Final_Learning_Rate': history['learning_rates'][-1],
                'Best_Val_Acc': fold_result['best_val_acc']
            }
            
            # Add enhanced final metrics
            if 'val_balanced_accs' in history:
                final_data.update({
                    'Final_Val_Balanced_Acc': history['val_balanced_accs'][-1],
                    'Final_Val_Macro_F1': history['val_macro_f1s'][-1],
                    'Final_Val_Micro_F1': history['val_micro_f1s'][-1],
                    'Final_Val_Minority_F1': history['val_minority_f1s'][-1],
                    'Final_Val_Kappa': history['val_kappas'][-1]
                })
            
            final_curves_data.append(final_data)
    
    final_curves_df = pd.DataFrame(final_curves_data)
    final_curves_df.to_csv(os.path.join(output_dir, 'cv_training_final_metrics.csv'), index=False)
    
    # Convergence analysis
    convergence_data = []
    for exp in all_experiments:
        exp_name = exp['experiment_name']
        convergence_stats = exp['cv_statistics'].get('convergence_analysis', {})
        
        for metric, stats in convergence_stats.items():
            convergence_data.append({
                'Experiment': exp_name,
                'Metric': metric,
                'Mean_Convergence_Epoch': stats['mean_epoch'],
                'Std_Convergence_Epoch': stats['std_epoch'],
                'Min_Convergence_Epoch': stats['min_epoch'],
                'Max_Convergence_Epoch': stats['max_epoch']
            })
    
    if convergence_data:
        convergence_df = pd.DataFrame(convergence_data)
        convergence_df.to_csv(os.path.join(output_dir, 'cv_convergence_analysis.csv'), index=False)
    
    print(f"\n✅ Enhanced results saved to {output_dir}")
    print(f"   📊 NEW FILES:")
    print(f"   ├── cv_training_curves_detailed.csv (epoch-by-epoch metrics)")
    print(f"   ├── cv_training_final_metrics.csv (final epoch metrics)")
    print(f"   ├── cv_convergence_analysis.csv (convergence analysis)")
    print(f"   └── [experiment]_comprehensive_training_curves.png")

def generate_enhanced_training_report(all_experiments, output_dir):
    """Generate enhanced report focusing on training dynamics"""
    
    report_path = os.path.join(output_dir, 'cv_enhanced_training_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("Enhanced Training Dynamics Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Training curve analysis
        f.write("Training Curve Analysis:\n")
        f.write("-" * 30 + "\n")
        
        for exp in all_experiments:
            exp_name = exp['experiment_name']
            f.write(f"\n{exp_name}:\n")
            
            # Collect final validation metrics across folds
            final_metrics = {
                'val_acc': [],
                'val_balanced_acc': [],
                'val_macro_f1': [],
                'val_micro_f1': [],
                'val_minority_f1': [],
                'val_kappa': []
            }
            
            # Training stability analysis
            training_stability = {
                'val_loss_std': [],
                'val_acc_std': [],
                'early_stopping_epochs': []
            }
            
            for fold_result in exp['fold_results']:
                history = fold_result['training_history']
                
                # Final validation metrics
                final_metrics['val_acc'].append(history['val_accs'][-1])
                if 'val_balanced_accs' in history:
                    final_metrics['val_balanced_acc'].append(history['val_balanced_accs'][-1])
                    final_metrics['val_macro_f1'].append(history['val_macro_f1s'][-1])
                    final_metrics['val_micro_f1'].append(history['val_micro_f1s'][-1])
                    final_metrics['val_minority_f1'].append(history['val_minority_f1s'][-1])
                    final_metrics['val_kappa'].append(history['val_kappas'][-1])
                
                # Training stability
                val_loss_std = np.std(history['val_losses'][-5:])  # Std of last 5 epochs
                val_acc_std = np.std(history['val_accs'][-5:])
                training_stability['val_loss_std'].append(val_loss_std)
                training_stability['val_acc_std'].append(val_acc_std)
                
                # Estimate early stopping point (when val loss stops improving)
                val_losses = history['val_losses']
                early_stop_epoch = len(val_losses)
                for i in range(5, len(val_losses)):
                    recent_losses = val_losses[i-5:i]
                    if all(recent_losses[j] <= recent_losses[j+1] for j in range(len(recent_losses)-1)):
                        early_stop_epoch = i
                        break
                training_stability['early_stopping_epochs'].append(early_stop_epoch)
            
            # Final metrics summary
            f.write(f"  Final Validation Metrics (Mean ± Std):\n")
            for metric, values in final_metrics.items():
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    f.write(f"    {metric.replace('_', ' ').title()}: {mean_val:.4f} ± {std_val:.4f}\n")
            
            # Training stability
            f.write(f"\n  Training Stability Analysis:\n")
            f.write(f"    Val Loss Stability (last 5 epochs std): {np.mean(training_stability['val_loss_std']):.4f} ± {np.std(training_stability['val_loss_std']):.4f}\n")
            f.write(f"    Val Acc Stability (last 5 epochs std): {np.mean(training_stability['val_acc_std']):.4f} ± {np.std(training_stability['val_acc_std']):.4f}\n")
            f.write(f"    Estimated Early Stop Epoch: {np.mean(training_stability['early_stopping_epochs']):.1f} ± {np.std(training_stability['early_stopping_epochs']):.1f}\n")
            
            # Learning curve characteristics
            f.write(f"\n  Learning Curve Characteristics:\n")
            
            # Average learning curves across folds
            all_train_losses = []
            all_val_losses = []
            all_val_accs = []
            
            for fold_result in exp['fold_results']:
                history = fold_result['training_history']
                all_train_losses.append(history['train_losses'])
                all_val_losses.append(history['val_losses'])
                all_val_accs.append(history['val_accs'])
            
            # Calculate average curves
            min_epochs = min(len(losses) for losses in all_train_losses)
            avg_train_losses = np.mean([losses[:min_epochs] for losses in all_train_losses], axis=0)
            avg_val_losses = np.mean([losses[:min_epochs] for losses in all_val_losses], axis=0)
            avg_val_accs = np.mean([accs[:min_epochs] for accs in all_val_accs], axis=0)
            
            # Training characteristics
            initial_train_loss = avg_train_losses[0]
            final_train_loss = avg_train_losses[-1]
            initial_val_loss = avg_val_losses[0]
            final_val_loss = avg_val_losses[-1]
            initial_val_acc = avg_val_accs[0]
            final_val_acc = avg_val_accs[-1]
            
            f.write(f"    Initial Train Loss: {initial_train_loss:.4f} → Final: {final_train_loss:.4f} (Δ: {initial_train_loss - final_train_loss:.4f})\n")
            f.write(f"    Initial Val Loss: {initial_val_loss:.4f} → Final: {final_val_loss:.4f} (Δ: {initial_val_loss - final_val_loss:.4f})\n")
            f.write(f"    Initial Val Acc: {initial_val_acc:.4f} → Final: {final_val_acc:.4f} (Δ: {final_val_acc - initial_val_acc:.4f})\n")
            
            # Overfitting analysis
            train_val_gap = final_train_loss - final_val_loss
            f.write(f"    Train-Val Loss Gap (final): {train_val_gap:.4f}")
            if train_val_gap < -0.1:
                f.write(" (Potential overfitting)")
            elif train_val_gap > 0.1:
                f.write(" (Potential underfitting)")
            else:
                f.write(" (Good generalization)")
            f.write("\n")
            
            # Learning rate analysis
            all_lrs = []
            for fold_result in exp['fold_results']:
                all_lrs.append(fold_result['training_history']['learning_rates'])
            
            avg_lrs = np.mean([lrs[:min_epochs] for lrs in all_lrs], axis=0)
            lr_reductions = sum(1 for i in range(1, len(avg_lrs)) if avg_lrs[i] < avg_lrs[i-1] * 0.9)
            f.write(f"    Learning Rate Reductions: {lr_reductions} times\n")
            f.write(f"    Initial LR: {avg_lrs[0]:.6f} → Final LR: {avg_lrs[-1]:.6f}\n")
        
        # Cross-experiment comparison
        if len(all_experiments) > 1:
            f.write(f"\nCross-Experiment Training Comparison:\n")
            f.write("-" * 40 + "\n")
            
            comparison_metrics = ['final_val_acc', 'final_val_macro_f1', 'final_val_minority_f1']
            
            f.write(f"{'Metric':<25} {'Experiment':<20} {'Mean':<10} {'Std':<10} {'Winner':<10}\n")
            f.write("-" * 75 + "\n")
            
            for metric in comparison_metrics:
                best_exp = None
                best_mean = -1
                
                for exp in all_experiments:
                    if f'final_val_{metric.split("_", 2)[-1]}s' in exp['cv_statistics']:
                        stats = exp['cv_statistics'][f'final_val_{metric.split("_", 2)[-1]}s']
                        mean_val = stats['mean']
                        std_val = stats['std']
                        
                        winner = ""
                        if mean_val > best_mean:
                            best_mean = mean_val
                            best_exp = exp['experiment_name']
                            winner = "★"
                        
                        f.write(f"{metric:<25} {exp['experiment_name']:<20} {mean_val:<10.4f} {std_val:<10.4f} {winner:<10}\n")
                f.write("\n")
        
        # Training recommendations
        f.write("Training Optimization Recommendations:\n")
        f.write("-" * 45 + "\n")
        
        for exp in all_experiments:
            exp_name = exp['experiment_name']
            f.write(f"\n{exp_name}:\n")
            
            # Calculate average convergence epoch
            convergence_stats = exp['cv_statistics'].get('convergence_analysis', {})
            
            if 'val_losses_convergence' in convergence_stats:
                avg_convergence = convergence_stats['val_losses_convergence']['mean_epoch']
                f.write(f"  • Optimal training epochs: ~{int(avg_convergence + 5)} (convergence at {avg_convergence:.1f})\n")
            
            # Learning rate recommendations
            config = exp['experiment_config']
            current_lr = config['learning_rate']
            
            # Analyze if learning rate was reduced
            lr_reductions = 0
            for fold_result in exp['fold_results']:
                lrs = fold_result['training_history']['learning_rates']
                lr_reductions += sum(1 for i in range(1, len(lrs)) if lrs[i] < lrs[i-1] * 0.9)
            
            avg_lr_reductions = lr_reductions / len(exp['fold_results'])
            
            if avg_lr_reductions > 3:
                f.write(f"  • Consider starting with lower learning rate (current: {current_lr:.6f})\n")
            elif avg_lr_reductions < 1:
                f.write(f"  • Could start with higher learning rate for faster convergence\n")
            else:
                f.write(f"  • Learning rate schedule is appropriate\n")
            
            # Batch size recommendations
            current_batch_size = config['batch_size']
            f.write(f"  • Current batch size ({current_batch_size}) appears suitable\n")
            
            # Early stopping recommendations
            if 'val_losses_convergence' in convergence_stats:
                patience_needed = int(convergence_stats['val_losses_convergence']['std_epoch']) + 3
                f.write(f"  • Recommended early stopping patience: {patience_needed} epochs\n")
        
        f.write(f"\nGeneral Training Recommendations:\n")
        f.write(f"  • Monitor validation metrics during training for better model selection\n")
        f.write(f"  • Use early stopping based on validation loss plateau\n")
        f.write(f"  • Consider ensemble of best folds for final deployment\n")
        f.write(f"  • Fine-tune hyperparameters based on convergence analysis\n")

def analyze_metrics_trends(all_experiments, output_dir):
    """Analyze and visualize metrics trends during training"""
    
    for exp in all_experiments:
        exp_name = exp['experiment_name']
        
        # Create trend analysis figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Collect data across all folds
        epochs_data = []
        metrics_data = {
            'val_macro_f1s': [],
            'val_minority_f1s': [],
            'val_balanced_accs': [],
            'val_kappas': []
        }
        
        for fold_result in exp['fold_results']:
            history = fold_result['training_history']
            epochs_data.append(list(range(1, len(history['val_accs']) + 1)))
            
            for metric in metrics_data.keys():
                if metric in history:
                    metrics_data[metric].append(history[metric])
        
        # Plot trends
        metric_names = ['Macro F1', 'Minority F1', 'Balanced Accuracy', 'Cohen\'s Kappa']
        colors = plt.cm.Set1(np.linspace(0, 1, len(exp['fold_results'])))
        
        for idx, (metric_key, metric_name) in enumerate(zip(metrics_data.keys(), metric_names)):
            ax = axes[idx // 2, idx % 2]
            
            if metrics_data[metric_key]:
                # Plot individual folds
                for fold_idx, (epochs, values) in enumerate(zip(epochs_data, metrics_data[metric_key])):
                    if values:
                        ax.plot(epochs, values, color=colors[fold_idx], alpha=0.6, 
                               linewidth=1, label=f'Fold {fold_idx + 1}')
                
                # Plot average trend
                min_epochs = min(len(values) for values in metrics_data[metric_key] if values)
                if min_epochs > 0:
                    avg_values = np.mean([values[:min_epochs] for values in metrics_data[metric_key] if values], axis=0)
                    std_values = np.std([values[:min_epochs] for values in metrics_data[metric_key] if values], axis=0)
                    
                    epochs_range = range(1, min_epochs + 1)
                    ax.plot(epochs_range, avg_values, 'k-', linewidth=3, label='Average')
                    ax.fill_between(epochs_range, 
                                  np.array(avg_values) - np.array(std_values),
                                  np.array(avg_values) + np.array(std_values),
                                  alpha=0.2, color='black')
            
            ax.set_title(f'Validation {metric_name} Trend')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{exp_name} - Validation Metrics Trends', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{exp_name.lower().replace(" ", "_")}_metrics_trends.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

            
# Main function
def main():
    start_time = time.time()
    
    # Load dataset
    full_dataset = HerbariumDataset(args.data_dir)
    class_counts = analyze_dataset_distribution(full_dataset)
    
    print(f"\nFound {len(full_dataset.classes)} species:")
    for i, species in enumerate(full_dataset.classes[:10]):
        print(f"  {i+1}. {species}")
    if len(full_dataset.classes) > 10:
        print(f"  ... and {len(full_dataset.classes) - 10} more species")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define experiment configurations
    experiments_to_run = []
    
    # Baseline configuration
    baseline_config = {
        'loss_type': 'baseline',
        'label_smoothing': 0.1,
        'optimizer': 'adamw',
        'learning_rate': 0.0001,
        'weight_decay': 0.0001,
        'scheduler_type': 'reduce_plateau',
        'scheduler_patience': 3,
        'scheduler_factor': 0.5,
        'dropout_rate': 0.3,
        'batch_size': 24,
        'gradient_clip_norm': 1.0,
        'warmup_epochs': 1
    }


    # Focal Weighted loss configuration
    focal_weighted_config = {
        'loss_type': 'focal_weighted',
        'focal_gamma': 2,
        'label_smoothing': 0.1,
        'optimizer': 'adamw',
        'learning_rate': 0.0002,
        'weight_decay': 0.0001,
        'scheduler_type': 'reduce_plateau',
        'scheduler_patience': 3,
        'scheduler_factor': 0.5,
        'dropout_rate': 0.3,
        'batch_size': 24,
        'gradient_clip_norm': 1.0,
        'warmup_epochs': 1,
        'weight_method': 'effective_num',
    }


    # Determine which experiments to run
    if args.run_baseline or args.run_both:
        experiments_to_run.append(('Baseline', baseline_config))
    
    if args.run_focal_weighted or args.run_both:
        experiments_to_run.append(('Focal + Weighted Loss', focal_weighted_config))
    
    print(f"\nRunning {len(experiments_to_run)} experiments:")
    for name, _ in experiments_to_run:
        print(f"  - {name}")
    
    print(f"Cross-validation: {args.n_folds} folds")
    print(f"Training epochs: {args.epochs}")
    print(f"Total training runs: {len(experiments_to_run) * args.n_folds}")
    
    # Run experiments
    all_experiments = []
    
    for exp_name, exp_config in experiments_to_run:
        print(f"\n{'='*80}")
        print(f"Starting {exp_name} experiment...")
        print(f"{'='*80}")
        
        experiment_result = run_cross_validation_experiment_enhanced(  # Use enhanced function
            full_dataset, device, args.output_dir, exp_config, 
            exp_name, args.n_folds, args.epochs
        )
        
        all_experiments.append(experiment_result)
    
    # Generate results and visualizations
    print(f"\n{'='*80}")
    print("GENERATING RESULTS AND VISUALIZATIONS")
    print(f"{'='*80}")
    
    # Save results
    # save_cv_results(all_experiments, args.output_dir)
    save_enhanced_cv_results(all_experiments, args.output_dir)
    
    # Generate visualizations
    plot_cv_results(all_experiments, args.output_dir)
    plot_comprehensive_training_curves(all_experiments, args.output_dir)  # New comprehensive plots
    # plot_training_curves_cv(all_experiments, args.output_dir)
    
    # Generate report
    generate_cv_report(all_experiments, full_dataset, args.output_dir)
    generate_enhanced_training_report(all_experiments, args.output_dir)
    analyze_metrics_trends(all_experiments, args.output_dir)

    print(f"\n✅ Enhanced training analysis complete!")
    print(f"📊 Additional files generated:")
    print(f"├── cv_enhanced_training_report.txt")
    print(f"├── cv_training_curves_detailed.csv")
    print(f"├── cv_training_final_metrics.csv")
    print(f"├── cv_convergence_analysis.csv")
    print(f"└── [experiment]_metrics_trends.png")
    
    # Print final summary
    print(f"\n{'='*80}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    
    for exp in all_experiments:
        exp_name = exp['experiment_name']
        cv_stats = exp['cv_statistics']
        
        print(f"\n🏆 {exp_name}:")
        print(f"  📊 Test Accuracy: {cv_stats['test_acc']['mean']:.4f} ± {cv_stats['test_acc']['std']:.4f}")
        print(f"  📈 Macro F1: {cv_stats['macro_f1']['mean']:.4f} ± {cv_stats['macro_f1']['std']:.4f}")
        print(f"  🔴 Minority F1: {cv_stats['minority_f1']['mean']:.4f} ± {cv_stats['minority_f1']['std']:.4f}")
        print(f"  🤝 Cohen's Kappa: {cv_stats['kappa']['mean']:.4f} ± {cv_stats['kappa']['std']:.4f}")
    
    # Best overall recommendation
    if len(all_experiments) > 1:
        best_exp = max(all_experiments, 
                      key=lambda x: (x['cv_statistics']['macro_f1']['mean'] + 
                                   x['cv_statistics']['test_acc']['mean']) / 2)
        
        print(f"\n🎖️ RECOMMENDED CONFIGURATION: {best_exp['experiment_name']}")
        print(f"   Expected Macro F1: {best_exp['cv_statistics']['macro_f1']['mean']:.4f} ± {best_exp['cv_statistics']['macro_f1']['std']:.4f}")
        print(f"   Expected Test Accuracy: {best_exp['cv_statistics']['test_acc']['mean']:.4f} ± {best_exp['cv_statistics']['test_acc']['std']:.4f}")
    
    total_duration = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*80}")
    print(f"Total execution time: {total_duration/60:.1f} minutes")
    print(f"Results saved to: {args.output_dir}")
    
    print(f"\nGenerated files:")
    print(f"├── 📊 RESULTS:")
    print(f"│   ├── cv_summary_results.csv")
    print(f"│   ├── cv_detailed_results.csv")
    print(f"│   └── cv_complete_results.json")
    print(f"├── 📈 VISUALIZATIONS:")
    print(f"│   ├── cv_results_comparison.png")
    print(f"│   └── [experiment]_cv_training_curves.png")
    print(f"└── 📋 REPORT:")
    print(f"    └── cv_final_report.txt")
    
    print(f"\n✅ Final training complete! Check {args.output_dir}/cv_final_report.txt for detailed results.")

if __name__ == '__main__':
    main()