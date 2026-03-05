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
parser = argparse.ArgumentParser(description='5-Fold CV Hyperparameter Tuning for Baseline vs Focal+Weighted EfficientNet-V2-S')
parser.add_argument('--data_dir', type=str, default='/home/s3844498/data/2nd_fix', 
                    help='Path to the herbarium dataset')
parser.add_argument('--output_dir', type=str, default='/home/s3844498/efficientnetv2s_cv_hyperparameter_tuning_results', 
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

# Hyperparameter configuration space (reduced for faster CV)
def get_hyperparameter_space():
    """Define focused hyperparameter search space for cross-validation"""
    return {
        # Loss function type
        'use_focal_weighted': [True, False],  # True = Focal+Weighted, False = Baseline (CrossEntropy)

        # Weighted Loss method (only used when use_focal_weighted=True)
        'weight_method': ['effective_num', 'inverse'],  # Methods for calculating class weights
        
        # Focal Loss hyperparameters (only used when use_focal_weighted=True)
        'focal_gamma': [1.5, 2.0, 2.5],
        'label_smoothing': [0.0, 0.1],
        
        # Optimizer hyperparameters
        'optimizer_type': ['adam', 'adamw'],
        'learning_rate': [1e-4, 2e-4, 5e-4],
        'weight_decay': [0.0, 1e-4, 1e-3],
        
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

def get_transforms():
    """Get data transforms for EfficientNet-V2-S (224x224 input)"""
    
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
    """Train model for one fold with given configuration - FIXED pattern like MobileNet/ResNet50"""
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
        
        print(f"    Fold {fold_num}, Epoch {epoch+1}/{num_epochs}: Training...", end="", flush=True)
        
        batch_count = 0
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
            
            batch_count += 1
            if batch_count % 100 == 0:
                print(".", end="", flush=True)
        
        train_loss = running_loss / total
        train_acc = running_corrects / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        print(" Validating...", end="", flush=True)
        
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
        
        val_loss = val_running_loss / val_total
        val_acc = val_running_corrects / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f" Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}, Val Loss: {val_loss:.3f}")
        
        # Scheduler step
        if config['scheduler_type'] == 'reduce_plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Early stopping and best model tracking
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_model_wts = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            print(f"    Early stopping triggered at epoch {epoch+1} for fold {fold_num}")
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
    
    print("    Evaluating on test set", end="", flush=True)
    
    with torch.no_grad():
        batch_count = 0
        for inputs, labels in test_loader:
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
            
            batch_count += 1
            # Print progress every 50 batches
            if batch_count % 50 == 0:
                print(".", end="", flush=True)
    
    test_loss = running_loss / total
    test_acc = running_corrects / total
    
    print(f" Complete!")
    
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

def run_cv_for_config(config, cv_splits, full_dataset, device):
    """Run 5-fold cross-validation for a given configuration - FIXED pattern like MobileNet/ResNet50"""
    fold_results = []
    
    # Setup transforms
    transforms_dict = get_transforms()
    
    loss_type = "Focal + Weighted" if config['use_focal_weighted'] else "Baseline (CrossEntropy)"
    
    for fold, (train_indices, val_indices) in enumerate(cv_splits):
        print(f"    Fold {fold + 1}/{len(cv_splits)}")
        
        # Create datasets for this fold
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        
        # Apply transforms
        train_dataset.dataset.transform = transforms_dict['train']
        val_dataset.dataset.transform = transforms_dict['test']
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True, 
            num_workers=2,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=2,
            pin_memory=True
        )
        
        # Create model
        model = create_model(len(full_dataset.classes), config['dropout_rate'])
        
        # Setup criterion
        if config['use_focal_weighted']:
            # Focal Loss + Weighted for imbalance handling
            class_weights = calculate_class_weights(train_dataset, method=config['weight_method']).to(device)
            criterion = FocalLoss(
                alpha=class_weights,
                gamma=config['focal_gamma'],
                label_smoothing=config['label_smoothing']
            )
        else:
            # Baseline CrossEntropy Loss
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
            device, config, args.epochs, fold + 1
        )
        
        fold_results.append({
            'fold': fold + 1,
            'val_acc': val_acc,
            'val_loss': val_loss,
            'training_curves': training_curves
        })
    
    # Calculate CV statistics
    cv_val_accs = [result['val_acc'] for result in fold_results]
    cv_val_losses = [result['val_loss'] for result in fold_results]
    
    cv_stats = {
        'mean_val_acc': np.mean(cv_val_accs),
        'std_val_acc': np.std(cv_val_accs),
        'mean_val_loss': np.mean(cv_val_losses),
        'std_val_loss': np.std(cv_val_losses),
        'fold_results': fold_results
    }
    
    return cv_stats, loss_type

def generate_hyperparameter_combinations(hp_space, max_trials):
    """Generate hyperparameter combinations using intelligent sampling"""
    
    # Generate combinations for both methods
    all_combinations = []
    
    # Important combinations
    for use_focal_weighted in [True, False]:
        for weight_method in ['effective_num', 'inverse']:
            for lr in [1e-4, 2e-4]:
                for batch_size in [16, 24]:
                    for dropout in [0.2, 0.3]:
                        combination = {
                            'use_focal_weighted': use_focal_weighted,
                            'weight_method': weight_method,
                            'focal_gamma': 2.0,
                            'label_smoothing': 0.1,
                            'optimizer_type': 'adamw',
                            'learning_rate': lr,
                            'weight_decay': 1e-4,
                            'scheduler_type': 'reduce_plateau',
                            'scheduler_patience': 3,
                            'scheduler_factor': 0.5,
                            'dropout_rate': dropout,
                            'batch_size': batch_size,
                            'gradient_clip_norm': 1.0,
                            'warmup_epochs': 1
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
    """Save cross-validation results to files"""
    
    # Convert to DataFrame for easy analysis
    df_data = []
    for trial in trial_results:
        row = trial['config'].copy()
        row.update({
            'trial_id': trial['trial_id'],
            'loss_type': trial['loss_type'],
            'mean_val_acc': trial['cv_stats']['mean_val_acc'],
            'std_val_acc': trial['cv_stats']['std_val_acc'],
            'mean_val_loss': trial['cv_stats']['mean_val_loss'],
            'std_val_loss': trial['cv_stats']['std_val_loss'],
            'test_acc': trial['test_results']['test_acc'],
            'balanced_acc': trial['test_results']['balanced_acc'],
            'macro_f1': trial['test_results']['macro_f1'],
            'weighted_f1': trial['test_results']['weighted_f1'],
            'micro_f1': trial['test_results']['micro_f1'],
            'minority_f1': trial['test_results']['minority_f1'],
            'kappa': trial['test_results']['kappa'],
            'total_training_time': trial['total_training_time']
        })
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    df.to_csv(os.path.join(output_dir, 'efficientnetv2s_cv_hyperparameter_tuning_results.csv'), index=False)
    
    # Save detailed JSON results
    json_results = []
    for trial in trial_results:
        json_trial = {
            'trial_id': trial['trial_id'],
            'config': trial['config'],
            'loss_type': trial['loss_type'],
            'cv_stats': {
                'mean_val_acc': float(trial['cv_stats']['mean_val_acc']),
                'std_val_acc': float(trial['cv_stats']['std_val_acc']),
                'mean_val_loss': float(trial['cv_stats']['mean_val_loss']),
                'std_val_loss': float(trial['cv_stats']['std_val_loss']),
                'fold_results': [
                    {
                        'fold': fold['fold'],
                        'val_acc': float(fold['val_acc']),
                        'val_loss': float(fold['val_loss'])
                    }
                    for fold in trial['cv_stats']['fold_results']
                ]
            },
            'test_results': {k: float(v) if isinstance(v, (int, float)) else v 
                           for k, v in trial['test_results'].items() 
                           if k not in ['predictions', 'labels', 'probabilities']},
            'total_training_time': trial['total_training_time']
        }
        json_results.append(json_trial)
    
    with open(os.path.join(output_dir, 'detailed_cv_results.json'), 'w') as f:
        json.dump(json_results, f, indent=2)
    
    return df

def analyze_cv_results(df, output_dir):
    """Analyze cross-validation results"""
    
    # CV performance vs test performance correlation
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(df['mean_val_acc'], df['test_acc'], alpha=0.6)
    plt.xlabel('CV Mean Validation Accuracy')
    plt.ylabel('Test Accuracy')
    plt.title('CV Validation vs Test Accuracy')
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.5)
    
    plt.subplot(1, 2, 2)
    plt.scatter(df['mean_val_acc'], df['macro_f1'], alpha=0.6)
    plt.xlabel('CV Mean Validation Accuracy')
    plt.ylabel('Test Macro F1')
    plt.title('CV Validation Accuracy vs Test Macro F1')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_vs_test_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # CV stability analysis
    plt.figure(figsize=(10, 6))
    plt.scatter(df['mean_val_acc'], df['std_val_acc'], alpha=0.6, s=50)
    plt.xlabel('CV Mean Validation Accuracy')
    plt.ylabel('CV Std Validation Accuracy')
    plt.title('Cross-Validation Stability Analysis')
    plt.grid(True, alpha=0.3)
    
    # Highlight stable and high-performing configs
    stable_threshold = df['std_val_acc'].quantile(0.25)  # Bottom 25% std = most stable
    high_perf_threshold = df['mean_val_acc'].quantile(0.75)  # Top 25% mean = highest performing
    
    stable_high_perf = df[(df['std_val_acc'] <= stable_threshold) & 
                          (df['mean_val_acc'] >= high_perf_threshold)]
    
    if len(stable_high_perf) > 0:
        plt.scatter(stable_high_perf['mean_val_acc'], stable_high_perf['std_val_acc'], 
                   c='red', s=100, label='Stable & High-Performing', alpha=0.8)
        plt.legend()
    
    plt.savefig(os.path.join(output_dir, 'cv_stability_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Top performing configurations analysis
    top_configs = df.nlargest(10, 'macro_f1')
    
    print("\n🏆 TOP 10 CONFIGURATIONS BY TEST MACRO F1 (5-FOLD CV):")
    print("=" * 80)
    for idx, row in top_configs.iterrows():
        rank = len(top_configs) - list(top_configs.index).index(idx)
        print(f"\nRank {rank}")
        print(f"Loss Type: {row['loss_type']}")
        print(f"CV Val Acc: {row['mean_val_acc']:.4f} ± {row['std_val_acc']:.4f}")
        print(f"Test Macro F1: {row['macro_f1']:.4f} | Test Acc: {row['test_acc']:.4f} | Minority F1: {row['minority_f1']:.4f}")
        if row['use_focal_weighted']:
            print(f"Config: weight_method={row['weight_method']}, lr={row['learning_rate']}, "
                  f"batch_size={row['batch_size']}, dropout={row['dropout_rate']}")
        else:
            print(f"Config: lr={row['learning_rate']}, batch_size={row['batch_size']}, dropout={row['dropout_rate']}")
        print(f"CV Stability: {row['std_val_acc']:.4f} (lower is more stable)")
    
    return top_configs

def generate_cv_report(df, best_config, output_dir):
    """Generate comprehensive cross-validation report"""
    
    report_path = os.path.join(output_dir, 'efficientnetv2s_cv_hyperparameter_tuning_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("5-FOLD CROSS-VALIDATION HYPERPARAMETER TUNING REPORT - EfficientNet-V2-S\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Tuning Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: EfficientNet-V2-S\n")
        f.write(f"Input Size: 224x224\n")
        f.write(f"Total Trials: {len(df)}\n")
        f.write(f"Cross-Validation Folds: {args.cv_folds}\n")
        f.write(f"Training Epochs per Fold: {args.epochs}\n")
        f.write(f"Early Stopping Patience: {args.early_stopping_patience}\n")
        f.write(f"Data Split: 80% train, 10% val (within CV), 10% test\n\n")
        
        # Performance statistics
        f.write("CROSS-VALIDATION PERFORMANCE STATISTICS:\n")
        f.write("-" * 45 + "\n")
        f.write(f"Best Macro F1 (Test): {df['macro_f1'].max():.4f}\n")
        f.write(f"Best Test Accuracy: {df['test_acc'].max():.4f}\n")
        f.write(f"Best CV Val Accuracy: {df['mean_val_acc'].max():.4f}\n")
        f.write(f"Average Macro F1: {df['macro_f1'].mean():.4f} ± {df['macro_f1'].std():.4f}\n")
        f.write(f"Average CV Val Accuracy: {df['mean_val_acc'].mean():.4f} ± {df['mean_val_acc'].std():.4f}\n")
        f.write(f"Average CV Stability (Val Acc Std): {df['std_val_acc'].mean():.4f}\n\n")
        
        # Best configuration
        f.write("🏆 BEST CONFIGURATION (by Test Macro F1):\n")
        f.write("-" * 45 + "\n")
        best_row = df.loc[df['macro_f1'].idxmax()]
        
        f.write("Model Configuration:\n")
        f.write(f"  • Model: EfficientNet-V2-S\n")
        f.write(f"  • Input Size: 224x224\n")
        f.write(f"  • Loss Type: {best_row['loss_type']}\n")
        f.write(f"  • Use Focal+Weighted: {best_row.get('use_focal_weighted', 'N/A')}\n")
        if best_row.get('use_focal_weighted', False):
            f.write(f"  • Focal Gamma: {best_row['focal_gamma']}\n")
            f.write(f"  • Weight Method: {best_row['weight_method']}\n")
        f.write(f"  • Label Smoothing: {best_row['label_smoothing']}\n")
        f.write(f"  • Dropout Rate: {best_row['dropout_rate']}\n\n")
        
        f.write("Optimization Configuration:\n")
        f.write(f"  • Optimizer: {best_row['optimizer_type']}\n")
        f.write(f"  • Learning Rate: {best_row['learning_rate']}\n")
        f.write(f"  • Weight Decay: {best_row['weight_decay']}\n")
        f.write(f"  • Batch Size: {best_row['batch_size']}\n\n")
        
        f.write("Training Configuration:\n")
        f.write(f"  • Scheduler: {best_row['scheduler_type']}\n")
        f.write(f"  • Gradient Clip Norm: {best_row['gradient_clip_norm']}\n")
        f.write(f"  • Warmup Epochs: {best_row['warmup_epochs']}\n")
        f.write(f"  • CV Folds: {args.cv_folds}\n")
        f.write(f"  • Early Stopping Patience: {args.early_stopping_patience}\n\n")
        
        f.write("Cross-Validation Performance:\n")
        f.write(f"  • CV Val Accuracy: {best_row['mean_val_acc']:.4f} ± {best_row['std_val_acc']:.4f}\n")
        f.write(f"  • CV Val Loss: {best_row['mean_val_loss']:.4f} ± {best_row['std_val_loss']:.4f}\n")
        f.write(f"  • CV Stability Score: {1 - best_row['std_val_acc']:.4f}\n\n")
        
        f.write("Final Test Performance:\n")
        f.write(f"  • Test Accuracy: {best_row['test_acc']:.4f}\n")
        f.write(f"  • Macro F1: {best_row['macro_f1']:.4f}\n")
        f.write(f"  • Balanced Accuracy: {best_row['balanced_acc']:.4f}\n")
        f.write(f"  • Minority F1: {best_row['minority_f1']:.4f}\n")
        f.write(f"  • Cohen's Kappa: {best_row['kappa']:.4f}\n\n")
        
        # Cross-validation insights
        f.write("CROSS-VALIDATION INSIGHTS:\n")
        f.write("-" * 30 + "\n")
        
        # Loss type comparison
        if 'loss_type' in df.columns:
            f.write("1. Loss Type Performance Comparison:\n")
            for loss_type in df['loss_type'].unique():
                subset = df[df['loss_type'] == loss_type]
                if len(subset) > 0:
                    f.write(f"   {loss_type}:\n")
                    f.write(f"     • Avg Macro F1: {subset['macro_f1'].mean():.4f} ± {subset['macro_f1'].std():.4f}\n")
                    f.write(f"     • Best Macro F1: {subset['macro_f1'].max():.4f}\n")
                    f.write(f"     • Avg CV Stability: {1 - subset['std_val_acc'].mean():.4f}\n")
                    f.write(f"     • Number of trials: {len(subset)}\n")
            f.write("\n")
        
        # Hyperparameter insights
        f.write("2. Key Hyperparameter Insights:\n")
        
        # Learning rate analysis
        lr_performance = df.groupby('learning_rate')['macro_f1'].agg(['mean', 'std', 'max'])
        best_lr = lr_performance['mean'].idxmax()
        f.write(f"   • Optimal Learning Rate: {best_lr} (avg F1: {lr_performance.loc[best_lr, 'mean']:.4f})\n")
        
        # Batch size analysis
        batch_performance = df.groupby('batch_size')['macro_f1'].agg(['mean', 'std', 'max'])
        best_batch = batch_performance['mean'].idxmax()
        f.write(f"   • Optimal Batch Size: {best_batch} (avg F1: {batch_performance.loc[best_batch, 'mean']:.4f})\n")
        
        # Dropout analysis
        dropout_performance = df.groupby('dropout_rate')['macro_f1'].agg(['mean', 'std', 'max'])
        best_dropout = dropout_performance['mean'].idxmax()
        f.write(f"   • Optimal Dropout Rate: {best_dropout} (avg F1: {dropout_performance.loc[best_dropout, 'mean']:.4f})\n")
        
        # Optimizer analysis
        opt_performance = df.groupby('optimizer_type')['macro_f1'].agg(['mean', 'std', 'max'])
        best_opt = opt_performance['mean'].idxmax()
        f.write(f"   • Best Optimizer: {best_opt} (avg F1: {opt_performance.loc[best_opt, 'mean']:.4f})\n\n")
        
        # Stability analysis
        f.write("3. Model Stability Analysis:\n")
        stable_configs = df[df['std_val_acc'] <= df['std_val_acc'].quantile(0.25)]
        if len(stable_configs) > 0:
            best_stable = stable_configs.loc[stable_configs['macro_f1'].idxmax()]
            f.write(f"   • Most stable high-performing config:\n")
            f.write(f"     Macro F1: {best_stable['macro_f1']:.4f}\n")
            f.write(f"     CV Stability: {1 - best_stable['std_val_acc']:.4f}\n")
            f.write(f"     Loss Type: {best_stable['loss_type']}\n")
            f.write(f"     Config: lr={best_stable['learning_rate']}, batch={best_stable['batch_size']}\n\n")
        
        # Recommendations
        f.write("DEPLOYMENT RECOMMENDATIONS:\n")
        f.write("-" * 30 + "\n")
        f.write("1. Primary Recommendation:\n")
        f.write(f"   Use the best configuration above for production deployment\n")
        f.write(f"   Expected performance: Macro F1 = {best_row['macro_f1']:.4f}\n")
        f.write(f"   Cross-validation confidence: {best_row['mean_val_acc']:.4f} ± {best_row['std_val_acc']:.4f}\n\n")
        
        f.write("2. Alternative Configurations:\n")
        top_3 = df.nlargest(3, 'macro_f1')
        for i, (_, row) in enumerate(top_3.iterrows(), 1):
            if i <= 3:
                f.write(f"   Option {i}: {row['loss_type']}\n")
                f.write(f"             Macro F1: {row['macro_f1']:.4f}\n")
                f.write(f"             CV Acc: {row['mean_val_acc']:.4f} ± {row['std_val_acc']:.4f}\n")
                if row.get('use_focal_weighted', False):
                    f.write(f"             gamma={row['focal_gamma']}, Weight method={row['weight_method']},  lr={row['learning_rate']}\n")
                else:
                    f.write(f"             lr={row['learning_rate']}\n")
        
        f.write(f"\n3. Training Efficiency:\n")
        efficient_trials = df[df['macro_f1'] >= df['macro_f1'].quantile(0.8)]
        if len(efficient_trials) > 0:
            fastest_good = efficient_trials.loc[efficient_trials['total_training_time'].idxmin()]
            f.write(f"   Fastest high-performing config:\n")
            f.write(f"   Training time: {fastest_good['total_training_time']:.1f} min (5-fold CV)\n")
            f.write(f"   Performance: Macro F1 = {fastest_good['macro_f1']:.4f}\n")
            f.write(f"   Config: {fastest_good['loss_type']}, lr={fastest_good['learning_rate']}\n\n")
        
        f.write("4. Cross-Validation Benefits:\n")
        f.write("   • Robust performance estimation through 5-fold validation\n")
        f.write("   • Reduced overfitting risk with multiple train/val splits\n")
        f.write("   • Confidence intervals for performance metrics\n")
        f.write("   • Better generalization assessment\n\n")
        
        f.write("5. Implementation Notes:\n")
        f.write("   • Use stratified sampling to maintain class distribution\n")
        f.write("   • Monitor both validation accuracy and stability (low std)\n")
        f.write("   • Early stopping prevents overfitting in individual folds\n")
        f.write("   • Final model should be trained on full training set\n")
        f.write("   • Validate on held-out test set before production\n")

def main():
    start_time = time.time()
    
    print("🔬 5-FOLD CROSS-VALIDATION HYPERPARAMETER TUNING")
    print("EfficientNet-V2-S: Baseline vs Focal+Weighted")
    print("=" * 70)
    print(f"Model: EfficientNet-V2-S")
    print(f"Input Size: 224x224")
    print(f"Cross-validation: {args.cv_folds}-fold stratified")
    print(f"Max trials: {args.max_trials}")
    print(f"Epochs per fold: {args.epochs}")
    print(f"Early stopping patience: {args.early_stopping_patience}")
    print(f"Data split: 90% for CV (80% train, 10% val), 10% test")
    
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
    print(f"Total CV experiments: {len(combinations)} configs × {args.cv_folds} folds = {len(combinations) * args.cv_folds}")
    
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
    
    # Run cross-validation hyperparameter tuning
    print(f"\n🚀 Starting 5-fold cross-validation hyperparameter tuning...")
    
    for trial_id in range(start_trial, len(combinations)):
        config = combinations[trial_id]
        trial_start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"TRIAL {trial_id + 1}/{len(combinations)}")
        print(f"{'='*70}")
        
        try:
            print(f"Model: EfficientNet-V2-S")
            print(f"Loss Type: {'Focal+Weighted' if config['use_focal_weighted'] else 'Baseline (CrossEntropy)'}")
            print(f"Config: lr={config['learning_rate']}, batch_size={config['batch_size']}")
            if config['use_focal_weighted']:
                print(f"Weight method={config['weight_method']}, focal gamma={config['focal_gamma']}, label_smoothing={config['label_smoothing']}")
            print(f"Running {args.cv_folds}-fold cross-validation...")
            
            # Perform cross-validation
            cv_stats, loss_type = run_cv_for_config(config, cv_splits, full_dataset, device)
            
            print(f"\n📊 Cross-Validation Results:")
            print(f"  Mean Val Acc: {cv_stats['mean_val_acc']:.4f} ± {cv_stats['std_val_acc']:.4f}")
            print(f"  Mean Val Loss: {cv_stats['mean_val_loss']:.4f} ± {cv_stats['std_val_loss']:.4f}")
            print(f"  CV Stability: {1 - cv_stats['std_val_acc']:.4f}")
            
            # Train final model on full train+val data (80%+10% = 90%) and evaluate on test set
            print(f"  🧪 Training final model on full train+val data...")
            
            # Combine train (80%) + val (10%) for final model training
            final_train_indices = train_indices + val_indices
            train_val_dataset = Subset(full_dataset, final_train_indices)
            
            # Setup transforms
            transforms_dict = get_transforms()
            train_val_dataset.dataset.transform = transforms_dict['train']
            test_dataset.dataset.transform = transforms_dict['test']
            
            # Create data loaders
            train_val_loader = DataLoader(
                train_val_dataset, 
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
            
            # Create final model
            final_model = create_model(len(full_dataset.classes), config['dropout_rate'])
            
            # Setup criterion for final training
            if config['use_focal_weighted']:
                class_weights = calculate_class_weights(train_val_dataset, method=config['weight_method']).to(device)
                final_criterion = FocalLoss(
                    alpha=class_weights,
                    gamma=config['focal_gamma'],
                    label_smoothing=config['label_smoothing']
                )
            else:
                if config['label_smoothing'] > 0:
                    final_criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
                else:
                    final_criterion = nn.CrossEntropyLoss()
            
            # Setup optimizer and scheduler for final training
            final_optimizer = get_optimizer(final_model, config)
            final_scheduler = get_scheduler(final_optimizer, config)
            
            # Train final model (simplified training loop)
            final_model.to(device)
            final_model.train()
            
            print("  Training final model", end="", flush=True)
            for epoch in range(args.epochs):
                if config['warmup_epochs'] > 0:
                    warmup_scheduler(final_optimizer, config['warmup_epochs'], epoch, config['learning_rate'])
                
                batch_count = 0
                for inputs, labels in train_val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    final_optimizer.zero_grad()
                    outputs = final_model(inputs)
                    loss = final_criterion(outputs, labels)
                    loss.backward()
                    
                    if config['gradient_clip_norm'] > 0:
                        torch.nn.utils.clip_grad_norm_(final_model.parameters(), config['gradient_clip_norm'])
                    
                    final_optimizer.step()
                    
                    batch_count += 1
                    if batch_count % 100 == 0:
                        print(".", end="", flush=True)
                
                if config['scheduler_type'] != 'reduce_plateau':
                    final_scheduler.step()
                
                print(f" Epoch {epoch+1}/{args.epochs}", end="", flush=True)
            
            print(" Complete!")
            
            # Evaluate on test set
            test_results = evaluate_model(
                final_model, test_loader, final_criterion, device, full_dataset.classes
            )
            
            trial_time = time.time() - trial_start_time
            
            # Store results
            trial_result = {
                'trial_id': trial_id + 1,
                'config': config,
                'loss_type': loss_type,
                'cv_stats': cv_stats,
                'test_results': test_results,
                'total_training_time': trial_time / 60  # Convert to minutes
            }
            
            trial_results.append(trial_result)
            
            print(f"\n📊 Final Trial {trial_id + 1} Results:")
            print(f"  CV: {cv_stats['mean_val_acc']:.4f} ± {cv_stats['std_val_acc']:.4f}")
            print(f"  Test Acc: {test_results['test_acc']:.4f} | Macro F1: {test_results['macro_f1']:.4f}")
            print(f"  Minority F1: {test_results['minority_f1']:.4f}")
            print(f"  Total time: {trial_time/60:.1f} minutes")
            
            # Save intermediate results every 3 trials
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
    print("🎯 5-FOLD CROSS-VALIDATION HYPERPARAMETER TUNING COMPLETED")
    print(f"{'='*80}")
    
    if not trial_results:
        print("❌ No successful trials completed!")
        return
    
    # Save all results
    df = save_cv_results(trial_results, args.output_dir)
    
    # Analyze results
    print(f"\n📈 ANALYZING CROSS-VALIDATION RESULTS...")
    
    # Find best configuration
    best_trial = max(trial_results, key=lambda x: x['cv_stats']['mean_val_acc'])
    best_config = best_trial['config']
    
    print(f"\n🏆 BEST CONFIGURATION:")
    print(f"Model: EfficientNet-V2-S")
    print(f"Trial ID: {best_trial['trial_id']}")
    print(f"Loss Type: {best_trial['loss_type']}")
    print(f"Test Macro F1: {best_trial['test_results']['macro_f1']:.4f}")
    print(f"Test Accuracy: {best_trial['test_results']['test_acc']:.4f}")
    print(f"CV Val Acc: {best_trial['cv_stats']['mean_val_acc']:.4f} ± {best_trial['cv_stats']['std_val_acc']:.4f}")
    print(f"CV Stability: {1 - best_trial['cv_stats']['std_val_acc']:.4f}")
    print(f"Total training time: {best_trial['total_training_time']:.1f} minutes")
    
    # Separate analysis for Baseline vs Focal+Weighted
    baseline_trials = [t for t in trial_results if not t['config']['use_focal_weighted']]
    focal_weighted_trials = [t for t in trial_results if t['config']['use_focal_weighted']]
    
    print(f"\n📊 CROSS-VALIDATION COMPARISON BY LOSS TYPE:")
    
    if baseline_trials:
        best_baseline = max(baseline_trials, key=lambda x: x['test_results']['macro_f1'])
        print(f"\n📊 BEST BASELINE (CrossEntropy):")
        print(f"  Test Macro F1: {best_baseline['test_results']['macro_f1']:.4f}")
        print(f"  Test Acc: {best_baseline['test_results']['test_acc']:.4f}")
        print(f"  CV Val Acc: {best_baseline['cv_stats']['mean_val_acc']:.4f} ± {best_baseline['cv_stats']['std_val_acc']:.4f}")
        print(f"  Config: lr={best_baseline['config']['learning_rate']}, batch={best_baseline['config']['batch_size']}")
    
    if focal_weighted_trials:
        best_focal_weighted = max(focal_weighted_trials, key=lambda x: x['test_results']['macro_f1'])
        print(f"\n🔥 BEST FOCAL + WEIGHTED:")
        print(f"  Test Macro F1: {best_focal_weighted['test_results']['macro_f1']:.4f}")
        print(f"  Test Acc: {best_focal_weighted['test_results']['test_acc']:.4f}")
        print(f"  CV Val Acc: {best_focal_weighted['cv_stats']['mean_val_acc']:.4f} ± {best_focal_weighted['cv_stats']['std_val_acc']:.4f}")
        print(f"  Config: gamma={best_focal_weighted['config']['focal_gamma']}, lr={best_focal_weighted['config']['learning_rate']}")
    
    if baseline_trials and focal_weighted_trials:
        print(f"\n🎯 WINNER: ", end="")
        if best_focal_weighted['test_results']['macro_f1'] > best_baseline['test_results']['macro_f1']:
            print("Focal + Weighted")
            improvement = best_focal_weighted['test_results']['macro_f1'] - best_baseline['test_results']['macro_f1']
            print(f"  Improvement: +{improvement:.4f} in Test Macro F1")
        else:
            print("Baseline (CrossEntropy)")
            improvement = best_baseline['test_results']['macro_f1'] - best_focal_weighted['test_results']['macro_f1']
            print(f"  Improvement: +{improvement:.4f} in Test Macro F1")
        
        # Stability comparison
        baseline_stability = 1 - best_baseline['cv_stats']['std_val_acc']
        focal_stability = 1 - best_focal_weighted['cv_stats']['std_val_acc']
        print(f"  Stability comparison: Baseline={baseline_stability:.4f}, Focal+Weighted={focal_stability:.4f}")
    
    # Generate analysis plots
    analyze_cv_results(df, args.output_dir)
    
    # Generate final report
    generate_cv_report(df, best_config, args.output_dir)
    
    # Performance summary
    print(f"\n📊 CROSS-VALIDATION PERFORMANCE SUMMARY:")
    print(f"Best Test Macro F1: {df['macro_f1'].max():.4f}")
    print(f"Best Test Accuracy: {df['test_acc'].max():.4f}")
    print(f"Best CV Val Accuracy: {df['mean_val_acc'].max():.4f}")
    print(f"Average Test Macro F1: {df['macro_f1'].mean():.4f} ± {df['macro_f1'].std():.4f}")
    print(f"Average CV Stability: {(1 - df['std_val_acc']).mean():.4f}")
    
    # Save best configuration for deployment
    best_config_path = os.path.join(args.output_dir, 'best_cv_hyperparameters_efficientnetv2s.json')
    
    deployment_configs = {
        'overall_best': {
            'model': 'EfficientNet-V2-S',
            'input_size': '224x224',
            'config': best_config,
            'loss_type': best_trial['loss_type'],
            'cv_performance': {
                'mean_val_acc': float(best_trial['cv_stats']['mean_val_acc']),
                'std_val_acc': float(best_trial['cv_stats']['std_val_acc']),
                'stability_score': float(1 - best_trial['cv_stats']['std_val_acc'])
            },
            'test_performance': {
                'macro_f1': float(best_trial['test_results']['macro_f1']),
                'test_acc': float(best_trial['test_results']['test_acc']),
                'balanced_acc': float(best_trial['test_results']['balanced_acc']),
                'minority_f1': float(best_trial['test_results']['minority_f1']),
                'kappa': float(best_trial['test_results']['kappa'])
            },
            'training_details': {
                'cv_folds': args.cv_folds,
                'epochs_per_fold': args.epochs,
                'early_stopping_patience': args.early_stopping_patience,
                'total_training_time_minutes': best_trial['total_training_time']
            }
        }
    }
    
    # Add best baseline and focal+weighted configs
    if baseline_trials:
        best_baseline = max(baseline_trials, key=lambda x: x['test_results']['macro_f1'])
        deployment_configs['best_baseline'] = {
            'model': 'EfficientNet-V2-S',
            'config': best_baseline['config'],
            'loss_type': best_baseline['loss_type'],
            'cv_performance': {
                'mean_val_acc': float(best_baseline['cv_stats']['mean_val_acc']),
                'std_val_acc': float(best_baseline['cv_stats']['std_val_acc']),
            },
            'test_performance': {
                'macro_f1': float(best_baseline['test_results']['macro_f1']),
                'test_acc': float(best_baseline['test_results']['test_acc']),
                'minority_f1': float(best_baseline['test_results']['minority_f1']),
                'kappa': float(best_baseline['test_results']['kappa'])
            }
        }
    
    if focal_weighted_trials:
        best_focal_weighted = max(focal_weighted_trials, key=lambda x: x['test_results']['macro_f1'])
        deployment_configs['best_focal_weighted'] = {
            'model': 'EfficientNet-V2-S',
            'config': best_focal_weighted['config'],
            'loss_type': best_focal_weighted['loss_type'],
            'cv_performance': {
                'mean_val_acc': float(best_focal_weighted['cv_stats']['mean_val_acc']),
                'std_val_acc': float(best_focal_weighted['cv_stats']['std_val_acc']),
                'stability_score': float(1 - best_focal_weighted['cv_stats']['std_val_acc'])
            },
            'test_performance': {
                'macro_f1': float(best_focal_weighted['test_results']['macro_f1']),
                'test_acc': float(best_focal_weighted['test_results']['test_acc']),
                'minority_f1': float(best_focal_weighted['test_results']['minority_f1']),
                'kappa': float(best_focal_weighted['test_results']['kappa'])
            }
        }
    
    with open(best_config_path, 'w') as f:
        json.dump(deployment_configs, f, indent=2)
    
    total_time = time.time() - start_time
    
    print(f"\n✅ 5-FOLD CROSS-VALIDATION HYPERPARAMETER TUNING COMPLETED!")
    print(f"Model: EfficientNet-V2-S")
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"Results saved to: {args.output_dir}")
    print(f"Best config saved to: {best_config_path}")
    print(f"\nGenerated files:")
    print(f"├── 📊 efficientnetv2s_cv_hyperparameter_tuning_results.csv")
    print(f"├── 📋 efficientnetv2s_cv_hyperparameter_tuning_report.txt")
    print(f"├── 🎯 best_cv_hyperparameters_efficientnetv2s.json")
    print(f"├── 📈 cv_vs_test_performance.png")
    print(f"├── 📉 cv_stability_analysis.png")
    print(f"└── 🔄 detailed_cv_results.json")
    
    print(f"\n🎯 KEY FINDINGS:")
    print(f"• Best approach: {best_trial['loss_type']}")
    print(f"• Test performance: {best_trial['test_results']['macro_f1']:.4f} Macro F1")
    print(f"• CV confidence: {best_trial['cv_stats']['mean_val_acc']:.4f} ± {best_trial['cv_stats']['std_val_acc']:.4f}")
    print(f"• Model stability: {1 - best_trial['cv_stats']['std_val_acc']:.4f}")
    
    if baseline_trials and focal_weighted_trials:
        print(f"• Method comparison: {'Focal+Weighted wins' if best_focal_weighted['test_results']['macro_f1'] > best_baseline['test_results']['macro_f1'] else 'Baseline wins'}")

if __name__ == '__main__':
    main()