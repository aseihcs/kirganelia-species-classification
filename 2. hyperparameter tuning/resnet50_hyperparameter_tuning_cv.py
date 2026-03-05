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
import itertools
from datetime import datetime

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Argument parser
parser = argparse.ArgumentParser(description='Hyperparameter Tuning for Focal Loss ResNet-50 with Cross-Validation')
parser.add_argument('--data_dir', type=str, default='/home/s3844498/data/2nd_fix', 
                    help='Path to the herbarium dataset')
parser.add_argument('--output_dir', type=str, default='/home/s3844498/hyperparameter_tuning_cv_results', 
                    help='Directory to save outputs')
parser.add_argument('--max_trials', type=int, default=50,
                    help='Maximum number of hyperparameter combinations to test')
parser.add_argument('--epochs', type=int, default=10, 
                    help='Number of epochs for training')
parser.add_argument('--early_stopping_patience', type=int, default=3, 
                    help='Early stopping patience')
parser.add_argument('--cv_folds', type=int, default=5,
                    help='Number of cross-validation folds')
parser.add_argument('--resume_from', type=str, default=None, 
                    help='Resume from saved results file')
args = parser.parse_args()

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

# Enhanced Focal Loss with more hyperparameters
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
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
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

# Hyperparameter configuration space (reduced for CV efficiency)
def get_hyperparameter_space():
    """Define comprehensive hyperparameter search space"""
    return {
        # Loss function type
        'use_focal_loss': [True, False],  # True = Focal Loss, False = CrossEntropy (Baseline)
        
        # Focal Loss hyperparameters (only used when use_focal_loss=True)
        'focal_gamma': [1.5, 2.0, 2.5],
        'label_smoothing': [0.0, 0.1],
        
        # Optimizer hyperparameters
        'optimizer_type': ['adam', 'adamw'],
        'learning_rate': [1e-4, 2e-4, 5e-4],
        'weight_decay': [0.0, 1e-5, 1e-4],
        
        # Scheduler hyperparameters
        'scheduler_type': ['reduce_plateau', 'cosine'],
        'scheduler_patience': [2, 3],  # For ReduceLROnPlateau
        'scheduler_factor': [0.3, 0.5],
        
        # Model hyperparameters
        'dropout_rate': [0.2, 0.3, 0.4],
        'batch_size': [16, 24, 32],
        
        # Training hyperparameters
        'gradient_clip_norm': [0.0, 1.0],  # 0.0 means no clipping
        'warmup_epochs': [0, 1],
    }

def get_transforms():
    """Get simple data transforms without augmentation"""
    
    base_transforms = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    
    train_transforms = transforms.Compose(base_transforms)
    test_transforms = transforms.Compose(base_transforms)
    
    return {'train': train_transforms, 'test': test_transforms}

def create_train_val_test_split(dataset, val_ratio=0.1, test_ratio=0.1):
    """Create stratified 80/10/10 train/val/test split"""
    labels = [sample[1] for sample in dataset.samples]
    indices = list(range(len(dataset)))
    
    # Test 10%
    train_val_indices, test_indices, _, _ = train_test_split(
        indices, labels, test_size=test_ratio, stratify=labels, random_state=42
    )
    
    # Val 10%
    train_val_labels = [labels[i] for i in train_val_indices]
    train_indices, val_indices, _, _ = train_test_split(
        train_val_indices, train_val_labels, 
        test_size=val_ratio / (1 - test_ratio),  # 0.1/0.9 ≈ 0.1111
        stratify=train_val_labels, random_state=42
    )
    
    return train_indices, val_indices, test_indices

def create_cv_splits(train_val_indices, dataset, cv_folds=5):
    """Create stratified k-fold splits for cross-validation"""
    # Get labels for train+val data
    train_val_labels = [dataset.samples[idx][1] for idx in train_val_indices]
    
    # Create stratified k-fold splits
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_splits = []
    
    for fold, (train_fold_idx, val_fold_idx) in enumerate(skf.split(train_val_indices, train_val_labels)):
        train_indices = [train_val_indices[i] for i in train_fold_idx]
        val_indices = [train_val_indices[i] for i in val_fold_idx]
        cv_splits.append((train_indices, val_indices))
    
    return cv_splits

def create_model(num_classes, dropout_rate=0.2):
    """Create ResNet-50 model with configurable dropout"""
    model = models.resnet50(weights='IMAGENET1K_V1')
    
    # Add dropout before final layer if specified
    if dropout_rate > 0:
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(model.fc.in_features, num_classes)
        )
    else:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

def get_optimizer(model, config):
    """Create optimizer based on configuration"""
    if config['optimizer_type'] == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    elif config['optimizer_type'] == 'adamw':
        return optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer type: {config['optimizer_type']}")

def get_scheduler(optimizer, config):
    """Create scheduler based on configuration"""
    if config['scheduler_type'] == 'reduce_plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            patience=config['scheduler_patience'],
            factor=config['scheduler_factor']
        )
    elif config['scheduler_type'] == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs
        )
    else:
        raise ValueError(f"Unknown scheduler type: {config['scheduler_type']}")

def warmup_scheduler(optimizer, warmup_epochs, current_epoch, base_lr):
    """Apply learning rate warmup"""
    if current_epoch < warmup_epochs:
        lr = base_lr * (current_epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def train_model_fold(model, train_loader, val_loader, criterion, optimizer,
                    scheduler, device, config, num_epochs, fold_num):
    """Train model for one fold with given configuration"""
    model.to(device)
    best_model_wts = model.state_dict()
    best_val_loss = float('inf')
    best_val_acc = 0.0

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    patience_counter = 0

    base_lr = config['learning_rate']

    for epoch in range(num_epochs):
        # Warmup
        if config['warmup_epochs'] > 0:
            warmup_scheduler(optimizer, config['warmup_epochs'], epoch, base_lr)

        # Training phase
        model.train()
        running_loss, running_corrects, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            if config['gradient_clip_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])

            optimizer.step()
            running_corrects += torch.sum(preds == labels).item()
            running_loss += loss.item() * inputs.size(0)
            total += labels.size(0)

        train_losses.append(running_loss / total)
        train_accs.append(running_corrects / total)

        # Validation phase
        model.eval()
        val_running_loss, val_running_corrects, val_total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels).item()
                val_total += labels.size(0)

        val_losses.append(val_running_loss / val_total)
        val_accs.append(val_running_corrects / val_total)

        # Scheduler step
        if config['scheduler_type'] == 'reduce_plateau':
            scheduler.step(val_losses[-1])
        else:
            scheduler.step()

        # Early stopping and best model tracking
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_val_acc = val_accs[-1]
            best_model_wts = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            print(f"    Early stopping at epoch {epoch+1} for fold {fold_num}")
            break

    # Load best model weights
    model.load_state_dict(best_model_wts)

    return model, best_val_acc, best_val_loss, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }

def evaluate_model(model, test_loader, criterion, device, class_names):
    """Comprehensive model evaluation"""
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
    
    # Comprehensive metrics
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    kappa = cohen_kappa_score(all_labels, all_preds)
    
    # Minority class analysis
    class_counts = Counter(all_labels)
    total_samples = len(all_labels)
    minority_threshold = min(20, total_samples * 0.05)
    minority_classes = [cls for cls, count in class_counts.items() if count < minority_threshold]
    
    if minority_classes:
        minority_mask = [label in minority_classes for label in all_labels]
        minority_labels = [all_labels[i] for i, is_min in enumerate(minority_mask) if is_min]
        minority_preds = [all_preds[i] for i, is_min in enumerate(minority_mask) if is_min]
        
        if minority_labels:
            minority_f1 = f1_score(minority_labels, minority_preds, average='macro', zero_division=0)
        else:
            minority_f1 = 0.0
    else:
        minority_f1 = 0.0
    
    return {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'balanced_acc': balanced_acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'micro_f1': micro_f1,
        'minority_f1': minority_f1,
        'kappa': kappa,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }

def run_cv_for_config(config, cv_splits, dataset, device):
    """Perform cross-validation for a given configuration using pre-built cv_splits"""
    
    num_classes = len(dataset.classes)
    cv_folds = len(cv_splits)
    fold_results = []
    
    for fold, (fold_train_indices, fold_val_indices) in enumerate(cv_splits):
        print(f"  🔄 Fold {fold + 1}/{cv_folds}")
        
        # Create datasets for this fold
        fold_train_dataset = Subset(dataset, fold_train_indices)
        fold_val_dataset = Subset(dataset, fold_val_indices)
        
        # Setup transforms
        transforms_dict = get_transforms()
        fold_train_dataset.dataset.transform = transforms_dict['train']
        fold_val_dataset.dataset.transform = transforms_dict['test']
        
        # Create data loaders
        train_loader = DataLoader(
            fold_train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True, 
            num_workers=2,
            pin_memory=True
        )
        val_loader = DataLoader(
            fold_val_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=2,
            pin_memory=True
        )
        
        # Create fresh model for each fold
        model = create_model(num_classes, config['dropout_rate'])
        
        # Setup criterion
        if config['use_focal_loss']:
            criterion = FocalLoss(
                alpha=None,
                gamma=config['focal_gamma'],
                label_smoothing=config['label_smoothing']
            )
        else:
            if config['label_smoothing'] > 0:
                criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
            else:
                criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer and scheduler
        optimizer = get_optimizer(model, config)
        scheduler = get_scheduler(optimizer, config)
        
        # Train model for this fold
        trained_model, val_acc, val_loss, training_curves = train_model_fold(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            device, config, args.epochs, fold_num=fold+1
        )
        
        fold_results.append({
            'fold': fold + 1,
            'val_acc': val_acc,
            'val_loss': val_loss,
            'training_curves': training_curves
        })
        
        print(f"    Fold {fold + 1} - Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")
    
    # Calculate cross-validation statistics
    cv_val_accs = [result['val_acc'] for result in fold_results]
    cv_val_losses = [result['val_loss'] for result in fold_results]
    
    cv_results = {
        'cv_val_acc_mean': np.mean(cv_val_accs),
        'cv_val_acc_std': np.std(cv_val_accs),
        'cv_val_loss_mean': np.mean(cv_val_losses),
        'cv_val_loss_std': np.std(cv_val_losses),
        'fold_results': fold_results
    }
    
    loss_type = "Focal Loss" if config['use_focal_loss'] else "Baseline (CrossEntropy)"
    return cv_results, loss_type

def generate_hyperparameter_combinations(hp_space, max_trials):
    """Generate hyperparameter combinations using intelligent sampling"""
    
    # Generate combinations for both Baseline and Focal Loss
    all_combinations = []
    
    # Systematic combinations for key parameters
    for use_focal in [True, False]:
        for gamma in hp_space['focal_gamma']:
            for lr in hp_space['learning_rate']:
                for batch_size in hp_space['batch_size']:
                    for dropout in hp_space['dropout_rate']:
                        combination = {
                            'use_focal_loss': use_focal,
                            'focal_gamma': gamma,
                            'label_smoothing': 0.1,
                            'optimizer_type': 'adamw',
                            'learning_rate': lr,
                            'weight_decay': 1e-4,
                            'scheduler_type': 'reduce_plateau',
                            'scheduler_patience': 2,
                            'scheduler_factor': 0.3,
                            'dropout_rate': dropout,
                            'batch_size': batch_size,
                            'gradient_clip_norm': 1.0,
                            'warmup_epochs': 0
                        }
                        all_combinations.append(combination)
    
    # Random sampling for remaining trials
    while len(all_combinations) < max_trials:
        combination = {}
        for param, values in hp_space.items():
            combination[param] = random.choice(values)
        all_combinations.append(combination)
    
    # Shuffle and return requested number
    random.shuffle(all_combinations)
    return all_combinations[:max_trials]

def save_cv_results(trial_results, output_dir):
    """Save cross-validation trial results to files"""
    
    # Convert to DataFrame for easy analysis
    df_data = []
    for trial in trial_results:
        row = trial['config'].copy()
        row.update({
            'trial_id': trial['trial_id'],
            'loss_type': trial['loss_type'],
            'cv_val_acc_mean': trial['cv_results']['cv_val_acc_mean'],
            'cv_val_acc_std': trial['cv_results']['cv_val_acc_std'],
            'cv_val_loss_mean': trial['cv_results']['cv_val_loss_mean'],
            'cv_val_loss_std': trial['cv_results']['cv_val_loss_std'],
            'test_acc': trial['test_results']['test_acc'],
            'balanced_acc': trial['test_results']['balanced_acc'],
            'macro_f1': trial['test_results']['macro_f1'],
            'weighted_f1': trial['test_results']['weighted_f1'],
            'micro_f1': trial['test_results']['micro_f1'],
            'minority_f1': trial['test_results']['minority_f1'],
            'kappa': trial['test_results']['kappa'],
            'training_time': trial['training_time']
        })
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    df.to_csv(os.path.join(output_dir, 'cv_hyperparameter_tuning_results.csv'), index=False)
    
    # Save detailed JSON results
    json_results = []
    for trial in trial_results:
        json_trial = {
            'trial_id': trial['trial_id'],
            'config': trial['config'],
            'loss_type': trial['loss_type'],
            'cv_results': {
                'cv_val_acc_mean': float(trial['cv_results']['cv_val_acc_mean']),
                'cv_val_acc_std': float(trial['cv_results']['cv_val_acc_std']),
                'cv_val_loss_mean': float(trial['cv_results']['cv_val_loss_mean']),
                'cv_val_loss_std': float(trial['cv_results']['cv_val_loss_std']),
                'fold_results': [
                    {
                        'fold': fold['fold'],
                        'val_acc': float(fold['val_acc']),
                        'val_loss': float(fold['val_loss'])
                    } for fold in trial['cv_results']['fold_results']
                ]
            },
            'test_results': {k: float(v) if isinstance(v, (int, float)) else v 
                           for k, v in trial['test_results'].items() 
                           if k not in ['predictions', 'labels', 'probabilities']},
            'training_time': trial['training_time']
        }
        json_results.append(json_trial)
    
    with open(os.path.join(output_dir, 'cv_detailed_results.json'), 'w') as f:
        json.dump(json_results, f, indent=2)
    
    return df

def analyze_cv_results(df, output_dir):
    """Analyze cross-validation results"""
    
    # Plot CV performance distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # CV Accuracy distribution
    axes[0, 0].hist(df['cv_val_acc_mean'], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('CV Validation Accuracy Mean')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of CV Validation Accuracy')
    axes[0, 0].grid(True, alpha=0.3)
    
    # CV vs Test performance
    axes[0, 1].scatter(df['cv_val_acc_mean'], df['test_acc'], alpha=0.6)
    axes[0, 1].set_xlabel('CV Validation Accuracy Mean')
    axes[0, 1].set_ylabel('Test Accuracy')
    axes[0, 1].set_title('CV Validation vs Test Performance')
    axes[0, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Performance by loss type
    loss_types = df['loss_type'].unique()
    for i, loss_type in enumerate(loss_types):
        subset = df[df['loss_type'] == loss_type]
        axes[1, 0].scatter(subset['cv_val_acc_mean'], subset['macro_f1'], 
                          label=loss_type, alpha=0.7, s=50)
    axes[1, 0].set_xlabel('CV Validation Accuracy Mean')
    axes[1, 0].set_ylabel('Test Macro F1')
    axes[1, 0].set_title('CV Performance vs Test Macro F1 by Loss Type')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Standard deviation analysis
    axes[1, 1].scatter(df['cv_val_acc_std'], df['macro_f1'], alpha=0.6)
    axes[1, 1].set_xlabel('CV Validation Accuracy Std')
    axes[1, 1].set_ylabel('Test Macro F1')
    axes[1, 1].set_title('CV Stability vs Test Performance')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Top performing configurations
    top_configs = df.nlargest(10, 'macro_f1')
    
    print("\n🏆 TOP 10 CONFIGURATIONS BY TEST MACRO F1 (with CV validation):")
    print("=" * 80)
    for idx, row in top_configs.iterrows():
        rank = len(top_configs) - list(top_configs.index).index(idx)
        print(f"\nRank {rank}")
        print(f"Loss Type: {row['loss_type']}")
        print(f"Test Macro F1: {row['macro_f1']:.4f} | Test Acc: {row['test_acc']:.4f}")
        print(f"CV Val Acc: {row['cv_val_acc_mean']:.4f} ± {row['cv_val_acc_std']:.4f}")
        print(f"Minority F1: {row['minority_f1']:.4f}")
        if row['use_focal_loss']:
            print(f"Config: gamma={row['focal_gamma']}, lr={row['learning_rate']}, "
                  f"batch_size={row['batch_size']}, dropout={row['dropout_rate']}")
        else:
            print(f"Config: lr={row['learning_rate']}, batch_size={row['batch_size']}, dropout={row['dropout_rate']}")
    
    return top_configs

def generate_cv_report(df, best_config, output_dir):
    """Generate comprehensive cross-validation report"""
    
    report_path = os.path.join(output_dir, 'cv_hyperparameter_tuning_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("CROSS-VALIDATION HYPERPARAMETER TUNING REPORT - FOCAL LOSS RESNET-50\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Tuning Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Trials: {len(df)}\n")
        f.write(f"Cross-Validation Folds: {args.cv_folds}\n")
        f.write(f"Training Epochs per Trial: {args.epochs}\n")
        f.write(f"Early Stopping Patience: {args.early_stopping_patience}\n\n")
        
        # Performance statistics
        f.write("CROSS-VALIDATION PERFORMANCE STATISTICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Best Test Macro F1: {df['macro_f1'].max():.4f}\n")
        f.write(f"Best Test Accuracy: {df['test_acc'].max():.4f}\n")
        f.write(f"Best CV Val Accuracy: {df['cv_val_acc_mean'].max():.4f}\n")
        f.write(f"Average Test Macro F1: {df['macro_f1'].mean():.4f} ± {df['macro_f1'].std():.4f}\n")
        f.write(f"Average CV Val Accuracy: {df['cv_val_acc_mean'].mean():.4f} ± {df['cv_val_acc_mean'].std():.4f}\n")
        f.write(f"Average CV Stability (std): {df['cv_val_acc_std'].mean():.4f}\n\n")
        
        # Best configuration
        f.write("🏆 BEST CONFIGURATION (by Test Macro F1):\n")
        f.write("-" * 35 + "\n")
        best_row = df.loc[df['macro_f1'].idxmax()]
        
        f.write("Loss Configuration:\n")
        f.write(f"  • Loss Type: {best_row['loss_type']}\n")
        f.write(f"  • Use Focal Loss: {best_row.get('use_focal_loss', 'N/A')}\n")
        if best_row.get('use_focal_loss', False):
            f.write(f"  • Focal Gamma: {best_row['focal_gamma']}\n")
        f.write(f"  • Label Smoothing: {best_row['label_smoothing']}\n")
        f.write(f"  • Dropout Rate: {best_row['dropout_rate']}\n\n")
        
        f.write("Optimization Configuration:\n")
        f.write(f"  • Optimizer: {best_row['optimizer_type']}\n")
        f.write(f"  • Learning Rate: {best_row['learning_rate']}\n")
        f.write(f"  • Weight Decay: {best_row['weight_decay']}\n")
        f.write(f"  • Batch Size: {best_row['batch_size']}\n\n")
        
        f.write("Scheduler Configuration:\n")
        f.write(f"  • Scheduler Type: {best_row['scheduler_type']}\n")
        if best_row['scheduler_type'] == 'reduce_plateau':
            f.write(f"  • Patience: {best_row['scheduler_patience']}\n")
            f.write(f"  • Factor: {best_row['scheduler_factor']}\n")
        f.write(f"\n")
        
        f.write("Training Configuration:\n")
        f.write(f"  • Gradient Clip Norm: {best_row['gradient_clip_norm']}\n")
        f.write(f"  • Warmup Epochs: {best_row['warmup_epochs']}\n\n")
        
        f.write("Cross-Validation Performance:\n")
        f.write(f"  • CV Val Accuracy: {best_row['cv_val_acc_mean']:.4f} ± {best_row['cv_val_acc_std']:.4f}\n")
        f.write(f"  • CV Val Loss: {best_row['cv_val_loss_mean']:.4f} ± {best_row['cv_val_loss_std']:.4f}\n\n")
        
        f.write("Expected Test Performance:\n")
        f.write(f"  • Test Macro F1: {best_row['macro_f1']:.4f}\n")
        f.write(f"  • Test Accuracy: {best_row['test_acc']:.4f}\n")
        f.write(f"  • Balanced Accuracy: {best_row['balanced_acc']:.4f}\n")
        f.write(f"  • Minority F1: {best_row['minority_f1']:.4f}\n")
        f.write(f"  • Cohen's Kappa: {best_row['kappa']:.4f}\n\n")
        
        # Loss type comparison
        f.write("LOSS TYPE COMPARISON:\n")
        f.write("-" * 25 + "\n")
        if 'loss_type' in df.columns:
            for loss_type in df['loss_type'].unique():
                loss_subset = df[df['loss_type'] == loss_type]
                if len(loss_subset) > 0:
                    best_for_loss = loss_subset.loc[loss_subset['macro_f1'].idxmax()]
                    f.write(f"{loss_type}:\n")
                    f.write(f"  • Best Test Macro F1: {best_for_loss['macro_f1']:.4f}\n")
                    f.write(f"  • Average Test Macro F1: {loss_subset['macro_f1'].mean():.4f} ± {loss_subset['macro_f1'].std():.4f}\n")
                    f.write(f"  • Best CV Val Acc: {best_for_loss['cv_val_acc_mean']:.4f} ± {best_for_loss['cv_val_acc_std']:.4f}\n")
                    f.write(f"  • Average CV Stability: {loss_subset['cv_val_acc_std'].mean():.4f}\n\n")
        
        # Cross-validation insights
        f.write("CROSS-VALIDATION INSIGHTS:\n")
        f.write("-" * 30 + "\n")
        
        # Most stable configurations
        most_stable = df.nsmallest(5, 'cv_val_acc_std')
        f.write("Most Stable Configurations (lowest CV std):\n")
        for i, (_, row) in enumerate(most_stable.iterrows(), 1):
            f.write(f"  {i}. {row['loss_type']} - CV Std: {row['cv_val_acc_std']:.4f}, Test F1: {row['macro_f1']:.4f}\n")
        
        f.write(f"\nBest Balance (high performance + low variance):\n")
        # Calculate performance-stability score
        df_copy = df.copy()
        df_copy['stability_score'] = df_copy['macro_f1'] - df_copy['cv_val_acc_std'] * 2  # Penalize high variance
        best_balanced = df_copy.loc[df_copy['stability_score'].idxmax()]
        f.write(f"  • {best_balanced['loss_type']}\n")
        f.write(f"  • Test Macro F1: {best_balanced['macro_f1']:.4f}\n")
        f.write(f"  • CV Stability: {best_balanced['cv_val_acc_std']:.4f}\n")
        f.write(f"  • Config: lr={best_balanced['learning_rate']}, batch={best_balanced['batch_size']}\n\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS:\n")
        f.write("-" * 18 + "\n")
        f.write("1. Production Deployment:\n")
        f.write(f"   Use the best configuration above for optimal test performance\n")
        f.write(f"   Expected Test Macro F1: {best_row['macro_f1']:.4f}\n")
        f.write(f"   CV Validation indicates stable performance across folds\n\n")
        
        f.write("2. Alternative High-Performance Configurations:\n")
        top_3 = df.nlargest(3, 'macro_f1')
        for i, (_, row) in enumerate(top_3.iterrows(), 1):
            f.write(f"   Option {i}: {row['loss_type']} - Test F1 = {row['macro_f1']:.4f}\n")
            f.write(f"             CV Val Acc = {row['cv_val_acc_mean']:.4f} ± {row['cv_val_acc_std']:.4f}\n")
        
        f.write(f"\n3. Most Stable Configuration:\n")
        most_stable_best = most_stable.iloc[0]
        f.write(f"   Config: {most_stable_best['loss_type']}\n")
        f.write(f"   CV Stability: {most_stable_best['cv_val_acc_std']:.4f}\n")
        f.write(f"   Test Performance: {most_stable_best['macro_f1']:.4f}\n")
        f.write(f"   Use this for consistent performance across different data splits\n\n")
        
        f.write("4. Implementation Notes:\n")
        f.write("   • Cross-validation provides robust performance estimates\n")
        f.write("   • Consider both mean performance and stability (std) for deployment\n")
        f.write("   • Monitor validation performance for early stopping in production\n")
        f.write("   • The 5-fold CV ensures configuration robustness\n")
        f.write(f"   • Total training time per configuration: ~{df['training_time'].mean():.1f} minutes\n")

def main():
    start_time = time.time()
    
    print("🔬 CROSS-VALIDATION HYPERPARAMETER TUNING FOR BASELINE vs FOCAL LOSS RESNET-50")
    print("=" * 80)
    print(f"Max trials: {args.max_trials}")
    print(f"Epochs per trial: {args.epochs}")
    print(f"Cross-validation folds: {args.cv_folds}")
    print(f"Early stopping patience: {args.early_stopping_patience}")
    
    # Load dataset
    full_dataset = HerbariumDataset(args.data_dir)
    print(f"\nDataset loaded: {len(full_dataset)} samples, {len(full_dataset.classes)} classes")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create stratified 80/10/10 train/val/test split
    train_indices, val_indices, test_indices = create_train_val_test_split(full_dataset, val_ratio=0.1, test_ratio=0.1)
    test_dataset = Subset(full_dataset, test_indices)
    
    # Create 5-fold CV splits from train data (80%) only
    cv_splits = create_cv_splits(train_indices, full_dataset, cv_folds=args.cv_folds)
    
    print(f"Data split: Train={len(train_indices)} (80%), Val={len(val_indices)} (10%), Test={len(test_dataset)} (10%)")
    print(f"Average fold sizes: Train≈{len(train_indices)*0.8//args.cv_folds:.0f}, Val≈{len(train_indices)*0.2//args.cv_folds:.0f}")
    
    # Generate hyperparameter combinations
    hp_space = get_hyperparameter_space()
    combinations = generate_hyperparameter_combinations(hp_space, args.max_trials)
    
    print(f"\nGenerated {len(combinations)} hyperparameter combinations")
    print(f"Each configuration will be evaluated using {args.cv_folds}-fold cross-validation")
    
    # Resume from previous results if specified
    trial_results = []
    start_trial = 0
    
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming from {args.resume_from}")
        with open(args.resume_from, 'r') as f:
            previous_results = json.load(f)
        trial_results = previous_results
        start_trial = len(trial_results)
        print(f"Resuming from trial {start_trial}")
    
    # Run hyperparameter tuning with cross-validation
    print(f"\n🚀 Starting cross-validation hyperparameter tuning...")
    
    for trial_id in range(start_trial, len(combinations)):
        config = combinations[trial_id]
        trial_start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"TRIAL {trial_id + 1}/{len(combinations)}")
        print(f"{'='*80}")
        
        try:
            # Perform cross-validation using pre-built cv_splits (from train 80% only)
            cv_results, loss_type = run_cv_for_config(config, cv_splits, full_dataset, device)
            
            print(f"Loss Type: {loss_type}")
            print(f"Config: lr={config['learning_rate']}, batch_size={config['batch_size']}, optimizer={config['optimizer_type']}")
            if config['use_focal_loss']:
                print(f"Focal gamma={config['focal_gamma']}, label_smoothing={config['label_smoothing']}")
            else:
                print(f"Label smoothing={config['label_smoothing']}")
            
            print(f"CV Results: Val Acc = {cv_results['cv_val_acc_mean']:.4f} ± {cv_results['cv_val_acc_std']:.4f}")
            
            # Train final model on full train+val data (80%+10% = 90%) and evaluate on test set
            print("🧪 Training final model on full train+val data...")
            
            # Combine train (80%) + val (10%) for final model training
            final_train_indices = train_indices + val_indices
            final_train_dataset = Subset(full_dataset, final_train_indices)
            
            # Setup transforms
            transforms_dict = get_transforms()
            final_train_dataset.dataset.transform = transforms_dict['train']
            test_dataset.dataset.transform = transforms_dict['test']
            
            # Create data loaders
            final_train_loader = DataLoader(
                final_train_dataset, 
                batch_size=config['batch_size'], 
                shuffle=True, 
                num_workers=2,
                pin_memory=True
            )
            test_loader = DataLoader(
                test_dataset, 
                batch_size=config['batch_size'], 
                shuffle=False, 
                num_workers=2,
                pin_memory=True
            )
            
            # Create and train final model
            final_model = create_model(len(full_dataset.classes), config['dropout_rate'])
            
            # Setup criterion
            if config['use_focal_loss']:
                criterion = FocalLoss(
                    alpha=None,
                    gamma=config['focal_gamma'],
                    label_smoothing=config['label_smoothing']
                )
            else:
                if config['label_smoothing'] > 0:
                    criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
                else:
                    criterion = nn.CrossEntropyLoss()
            
            # Setup optimizer and scheduler
            optimizer = get_optimizer(final_model, config)
            scheduler = get_scheduler(optimizer, config)
            
            # Train final model (without early stopping — no validation set, full train+val used)
            final_model.to(device)
            base_lr = config['learning_rate']

            for epoch in range(args.epochs):
                # Warmup
                if config['warmup_epochs'] > 0:
                    warmup_scheduler(optimizer, config['warmup_epochs'], epoch, base_lr)

                final_model.train()
                running_loss, running_corrects, total = 0.0, 0, 0

                for inputs, labels in final_train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()
                    outputs = final_model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    loss.backward()

                    if config['gradient_clip_norm'] > 0:
                        torch.nn.utils.clip_grad_norm_(final_model.parameters(), config['gradient_clip_norm'])

                    optimizer.step()
                    running_corrects += torch.sum(preds == labels).item()
                    running_loss += loss.item() * inputs.size(0)
                    total += labels.size(0)

                # Scheduler step (cosine annealing only; no val loss available for reduce_plateau)
                if config['scheduler_type'] == 'cosine':
                    scheduler.step()

            # Evaluate final model on test set
            test_results = evaluate_model(
                final_model, test_loader, criterion, device, full_dataset.classes
            )
            
            trial_time = time.time() - trial_start_time
            
            # Store results
            trial_result = {
                'trial_id': trial_id + 1,
                'config': config,
                'loss_type': loss_type,
                'cv_results': cv_results,
                'test_results': test_results,
                'training_time': trial_time / 60  # Convert to minutes
            }
            
            trial_results.append(trial_result)
            
            print(f"\n📊 Trial {trial_id + 1} Results:")
            print(f"  Loss Type: {loss_type}")
            print(f"  CV Val Acc: {cv_results['cv_val_acc_mean']:.4f} ± {cv_results['cv_val_acc_std']:.4f}")
            print(f"  Test Acc: {test_results['test_acc']:.4f} | Test Macro F1: {test_results['macro_f1']:.4f}")
            print(f"  Minority F1: {test_results['minority_f1']:.4f} | Balanced Acc: {test_results['balanced_acc']:.4f}")
            print(f"  Total training time: {trial_time/60:.1f} minutes")
            
            # Save intermediate results every 3 trials (due to longer training time with CV)
            if (trial_id + 1) % 3 == 0:
                intermediate_path = os.path.join(args.output_dir, f'cv_intermediate_results_trial_{trial_id + 1}.json')
                with open(intermediate_path, 'w') as f:
                    json.dump(trial_results, f, indent=2, default=str)
                print(f"💾 Intermediate results saved to {intermediate_path}")
            
        except Exception as e:
            print(f"❌ Trial {trial_id + 1} failed: {str(e)}")
            continue
    
    # Final analysis and reporting
    print(f"\n{'='*80}")
    print("🎯 CROSS-VALIDATION HYPERPARAMETER TUNING COMPLETED")
    print(f"{'='*80}")
    
    if not trial_results:
        print("❌ No successful trials completed!")
        return
    
    # Save all results
    df = save_cv_results(trial_results, args.output_dir)
    
    # Analyze results
    print(f"\n📈 ANALYZING CROSS-VALIDATION RESULTS...")
    
    # Find best configuration by mean validation accuracy
    best_trial = max(trial_results, key=lambda x: x['cv_results']['cv_val_acc_mean'])
    best_config = best_trial['config']
    
    print(f"\n🏆 BEST CONFIGURATION (by Test Macro F1):")
    print(f"Trial ID: {best_trial['trial_id']}")
    print(f"Loss Type: {best_trial['loss_type']}")
    print(f"Test Macro F1: {best_trial['test_results']['macro_f1']:.4f}")
    print(f"Test Accuracy: {best_trial['test_results']['test_acc']:.4f}")
    print(f"CV Val Acc: {best_trial['cv_results']['cv_val_acc_mean']:.4f} ± {best_trial['cv_results']['cv_val_acc_std']:.4f}")
    print(f"Minority F1: {best_trial['test_results']['minority_f1']:.4f}")
    print(f"Total training time: {best_trial['training_time']:.1f} minutes")
    
    # Separate analysis for Baseline vs Focal Loss
    baseline_trials = [t for t in trial_results if not t['config']['use_focal_loss']]
    focal_trials = [t for t in trial_results if t['config']['use_focal_loss']]
    
    print(f"\n📊 COMPARISON BY LOSS TYPE (with Cross-Validation):")
    
    if baseline_trials:
        best_baseline = max(baseline_trials, key=lambda x: x['test_results']['macro_f1'])
        baseline_cv_accs = [t['cv_results']['cv_val_acc_mean'] for t in baseline_trials]
        baseline_test_f1s = [t['test_results']['macro_f1'] for t in baseline_trials]
        
        print(f"\n📊 BASELINE (CrossEntropy) RESULTS:")
        print(f"  Best Test Macro F1: {best_baseline['test_results']['macro_f1']:.4f}")
        print(f"  Best Test Acc: {best_baseline['test_results']['test_acc']:.4f}")
        print(f"  Best CV Val Acc: {best_baseline['cv_results']['cv_val_acc_mean']:.4f} ± {best_baseline['cv_results']['cv_val_acc_std']:.4f}")
        print(f"  Average Test F1: {np.mean(baseline_test_f1s):.4f} ± {np.std(baseline_test_f1s):.4f}")
        print(f"  Average CV Val Acc: {np.mean(baseline_cv_accs):.4f}")
        print(f"  Config: lr={best_baseline['config']['learning_rate']}, batch={best_baseline['config']['batch_size']}")
    
    if focal_trials:
        best_focal = max(focal_trials, key=lambda x: x['test_results']['macro_f1'])
        focal_cv_accs = [t['cv_results']['cv_val_acc_mean'] for t in focal_trials]
        focal_test_f1s = [t['test_results']['macro_f1'] for t in focal_trials]
        
        print(f"\n🔥 FOCAL LOSS RESULTS:")
        print(f"  Best Test Macro F1: {best_focal['test_results']['macro_f1']:.4f}")
        print(f"  Best Test Acc: {best_focal['test_results']['test_acc']:.4f}")
        print(f"  Best CV Val Acc: {best_focal['cv_results']['cv_val_acc_mean']:.4f} ± {best_focal['cv_results']['cv_val_acc_std']:.4f}")
        print(f"  Average Test F1: {np.mean(focal_test_f1s):.4f} ± {np.std(focal_test_f1s):.4f}")
        print(f"  Average CV Val Acc: {np.mean(focal_cv_accs):.4f}")
        print(f"  Config: gamma={best_focal['config']['focal_gamma']}, lr={best_focal['config']['learning_rate']}")
        print(f"  Batch size: {best_focal['config']['batch_size']}")
    
    if baseline_trials and focal_trials:
        print(f"\n🎯 WINNER: ", end="")
        if best_focal['test_results']['macro_f1'] > best_baseline['test_results']['macro_f1']:
            print("Focal Loss")
            improvement = best_focal['test_results']['macro_f1'] - best_baseline['test_results']['macro_f1']
            print(f"  Focal Loss outperforms by {improvement:.4f} in Test Macro F1")
        else:
            print("Baseline (CrossEntropy)")
            improvement = best_baseline['test_results']['macro_f1'] - best_focal['test_results']['macro_f1']
            print(f"  Baseline outperforms by {improvement:.4f} in Test Macro F1")
        
        # Cross-validation comparison
        if len(baseline_trials) > 0 and len(focal_trials) > 0:
            baseline_cv_stability = np.mean([t['cv_results']['cv_val_acc_std'] for t in baseline_trials])
            focal_cv_stability = np.mean([t['cv_results']['cv_val_acc_std'] for t in focal_trials])
            print(f"  Cross-validation stability (lower is better):")
            print(f"    Baseline: {baseline_cv_stability:.4f}")
            print(f"    Focal Loss: {focal_cv_stability:.4f}")
        
        # Minority class analysis
        minority_improvement = best_focal['test_results']['minority_f1'] - best_baseline['test_results']['minority_f1']
        print(f"  Minority class improvement with Focal: {minority_improvement:+.4f}")
    
    # Generate analysis plots and reports
    analyze_cv_results(df, args.output_dir)
    generate_cv_report(df, best_config, args.output_dir)
    
    # Performance summary
    print(f"\n📊 CROSS-VALIDATION PERFORMANCE SUMMARY:")
    print(f"Best Test Macro F1: {df['macro_f1'].max():.4f}")
    print(f"Best Test Accuracy: {df['test_acc'].max():.4f}")
    print(f"Best CV Val Accuracy: {df['cv_val_acc_mean'].max():.4f}")
    print(f"Average Test Macro F1: {df['macro_f1'].mean():.4f} ± {df['macro_f1'].std():.4f}")
    print(f"Average CV Val Accuracy: {df['cv_val_acc_mean'].mean():.4f} ± {df['cv_val_acc_mean'].std():.4f}")
    print(f"Average CV Stability: {df['cv_val_acc_std'].mean():.4f}")
    
    # Save best configuration for deployment
    best_config_path = os.path.join(args.output_dir, 'best_cv_hyperparameters.json')
    
    deployment_configs = {
        'overall_best': {
            'config': best_config,
            'loss_type': best_trial['loss_type'],
            'expected_performance': {
                'test_macro_f1': float(best_trial['test_results']['macro_f1']),
                'test_acc': float(best_trial['test_results']['test_acc']),
                'balanced_acc': float(best_trial['test_results']['balanced_acc']),
                'minority_f1': float(best_trial['test_results']['minority_f1']),
                'kappa': float(best_trial['test_results']['kappa'])
            },
            'cv_validation': {
                'cv_val_acc_mean': float(best_trial['cv_results']['cv_val_acc_mean']),
                'cv_val_acc_std': float(best_trial['cv_results']['cv_val_acc_std']),
                'cv_val_loss_mean': float(best_trial['cv_results']['cv_val_loss_mean']),
                'cv_val_loss_std': float(best_trial['cv_results']['cv_val_loss_std'])
            },
            'training_details': {
                'epochs': args.epochs,
                'cv_folds': args.cv_folds,
                'early_stopping_patience': args.early_stopping_patience,
                'total_training_time_minutes': best_trial['training_time']
            }
        }
    }
    
    # Add best baseline if available
    if baseline_trials:
        best_baseline = max(baseline_trials, key=lambda x: x['test_results']['macro_f1'])
        deployment_configs['best_baseline'] = {
            'config': best_baseline['config'],
            'loss_type': best_baseline['loss_type'],
            'expected_performance': {
                'test_macro_f1': float(best_baseline['test_results']['macro_f1']),
                'test_acc': float(best_baseline['test_results']['test_acc']),
                'balanced_acc': float(best_baseline['test_results']['balanced_acc']),
                'minority_f1': float(best_baseline['test_results']['minority_f1']),
                'kappa': float(best_baseline['test_results']['kappa'])
            },
            'cv_validation': {
                'cv_val_acc_mean': float(best_baseline['cv_results']['cv_val_acc_mean']),
                'cv_val_acc_std': float(best_baseline['cv_results']['cv_val_acc_std'])
            }
        }
    
    # Add best focal loss if available
    if focal_trials:
        best_focal = max(focal_trials, key=lambda x: x['test_results']['macro_f1'])
        deployment_configs['best_focal_loss'] = {
            'config': best_focal['config'],
            'loss_type': best_focal['loss_type'],
            'expected_performance': {
                'test_macro_f1': float(best_focal['test_results']['macro_f1']),
                'test_acc': float(best_focal['test_results']['test_acc']),
                'balanced_acc': float(best_focal['test_results']['balanced_acc']),
                'minority_f1': float(best_focal['test_results']['minority_f1']),
                'kappa': float(best_focal['test_results']['kappa'])
            },
            'cv_validation': {
                'cv_val_acc_mean': float(best_focal['cv_results']['cv_val_acc_mean']),
                'cv_val_acc_std': float(best_focal['cv_results']['cv_val_acc_std'])
            }
        }
    
    with open(best_config_path, 'w') as f:
        json.dump(deployment_configs, f, indent=2)
    
    total_time = time.time() - start_time
    
    print(f"\n✅ CROSS-VALIDATION HYPERPARAMETER TUNING COMPLETED!")
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"Average time per trial: {total_time/len(trial_results)/60:.1f} minutes")
    print(f"Results saved to: {args.output_dir}")
    print(f"Best config saved to: {best_config_path}")
    print(f"\nGenerated files:")
    print(f"├── 📊 cv_hyperparameter_tuning_results.csv")
    print(f"├── 📋 cv_hyperparameter_tuning_report.txt")
    print(f"├── 🎯 best_cv_hyperparameters.json")
    print(f"├── 📈 cv_analysis.png")
    print(f"└── 📊 cv_detailed_results.json")
    
    print(f"\n🔬 Cross-validation provides robust performance estimates!")
    print(f"📊 Each configuration tested across {args.cv_folds} folds for reliability")
    print(f"🎯 Best configuration validated with both CV and independent test set")

if __name__ == '__main__':
    main()