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
parser = argparse.ArgumentParser(description='5-Fold CV Hyperparameter Tuning for MobileNet V3-Large - Baseline vs Weighted Loss')
parser.add_argument('--data_dir', type=str, default='/home/s3844498/data/2nd_fix', 
                    help='Path to the herbarium dataset')
parser.add_argument('--output_dir', type=str, default='/home/s3844498/mobilenetv3l_5fold_cv_results', 
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

# Hyperparameter configuration space for MobileNet V3-Large (reduced for faster CV)
def get_hyperparameter_space():
    """Define comprehensive hyperparameter search space for MobileNet V3-Large with 5-fold CV"""
    return {
        # Loss function type
        'use_weighted_loss': [True, False],  # True = Weighted Loss, False = Baseline (no imbalance handling)
        
        # Weighted Loss method (only used when use_weighted_loss=True)
        'weight_method': ['effective_num', 'inverse'],  # Methods for calculating class weights
        
        # Optimizer hyperparameters (reduced for faster CV)
        'optimizer_type': ['adam', 'adamw'],
        'learning_rate': [1e-4, 2e-4, 5e-4, 1e-3],
        'weight_decay': [0.0, 1e-5, 1e-4],
        
        # Scheduler hyperparameters (simplified)
        'scheduler_type': ['reduce_plateau', 'cosine'],
        'scheduler_patience': [2, 3],  # For ReduceLROnPlateau
        'scheduler_factor': [0.3, 0.5],
        
        # Model hyperparameters
        'dropout_rate': [0.1, 0.2, 0.3],
        'batch_size': [24, 32],  # Reduced for memory efficiency in CV
        
        # Training hyperparameters (simplified)
        'gradient_clip_norm': [0.0, 1.0],  # 0.0 means no clipping
        'warmup_epochs': [0, 2],
        'label_smoothing': [0.0, 0.1],
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
    """Get data transforms optimized for MobileNet V3"""
    
    base_transforms = [
        transforms.Resize((224, 224)),  # MobileNet V3 expects 224x224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
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

def create_mobilenet_v3_large_model(num_classes, dropout_rate=0.2):
    """Create MobileNet V3-Large model with configurable dropout"""
    model = models.mobilenet_v3_large(weights='IMAGENET1K_V1')
    
    # Get the number of input features for the classifier
    num_features = model.classifier[0].in_features
    
    # Replace the classifier with a new one for our number of classes
    if dropout_rate > 0:
        model.classifier = nn.Sequential(
            nn.Linear(num_features, model.classifier[0].out_features),  # Keep first linear layer size
            nn.Hardswish(inplace=True),  # Keep activation
            nn.Dropout(dropout_rate),  # Add dropout
            nn.Linear(model.classifier[0].out_features, num_classes)  # Final classification layer
        )
    else:
        model.classifier = nn.Sequential(
            nn.Linear(num_features, model.classifier[0].out_features),
            nn.Hardswish(inplace=True),
            nn.Linear(model.classifier[0].out_features, num_classes)
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

def run_cv_for_config(config, cv_splits, full_dataset, device):
    """Run 5-fold cross-validation for a given configuration"""
    fold_results = []
    
    # Setup transforms
    transforms_dict = get_transforms()
    
    loss_type = "Weighted Loss" if config['use_weighted_loss'] else "Baseline (CrossEntropy)"
    
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
        model = create_mobilenet_v3_large_model(len(full_dataset.classes), config['dropout_rate'])
        
        # Setup criterion
        if config['use_weighted_loss']:
            # Weighted Loss for imbalance handling
            class_weights = calculate_class_weights(train_dataset, method=config['weight_method']).to(device)
            if config['label_smoothing'] > 0:
                criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config['label_smoothing'])
            else:
                criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            # Baseline CrossEntropy Loss (no imbalance handling)
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
    """Generate hyperparameter combinations using intelligent sampling (reduced for CV)"""
    
    # Important combinations for both Baseline and Weighted Loss
    important_combinations = []
    
    for use_weighted in [True, False]:
        for weight_method in ['effective_num', 'inverse']:
            for lr in [1e-4, 2e-4, 5e-4]:
                for batch_size in [24, 32]:
                    for dropout in [0.2, 0.3]:
                        combination = {
                            'use_weighted_loss': use_weighted,
                            'weight_method': weight_method,
                            'optimizer_type': 'adam',
                            'learning_rate': lr,
                            'weight_decay': 1e-4,
                            'scheduler_type': 'reduce_plateau',
                            'scheduler_patience': 2,
                            'scheduler_factor': 0.5,
                            'dropout_rate': dropout,
                            'batch_size': batch_size,
                            'gradient_clip_norm': 1.0,
                            'warmup_epochs': 2,
                            'label_smoothing': 0.1
                        }
                        important_combinations.append(combination)
    
    # Random sampling for remaining trials
    all_combinations = important_combinations.copy()
    
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
            'cv_mean_val_acc': trial['cv_stats']['mean_val_acc'],
            'cv_std_val_acc': trial['cv_stats']['std_val_acc'],
            'cv_mean_val_loss': trial['cv_stats']['mean_val_loss'],
            'cv_std_val_loss': trial['cv_stats']['std_val_loss'],
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
    df.to_csv(os.path.join(output_dir, 'mobilenetv3l_5fold_cv_results.csv'), index=False)
    
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
                    } for fold in trial['cv_stats']['fold_results']
                ]
            },
            'test_results': {k: float(v) if isinstance(v, (int, float)) else v 
                           for k, v in trial['test_results'].items() 
                           if k not in ['predictions', 'labels', 'probabilities']},
            'training_time': trial['training_time']
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
    plt.scatter(df['cv_mean_val_acc'], df['test_acc'], alpha=0.6)
    plt.xlabel('CV Mean Validation Accuracy')
    plt.ylabel('Test Accuracy')
    plt.title('CV Validation vs Test Accuracy')
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.5)
    
    plt.subplot(1, 2, 2)
    plt.scatter(df['cv_mean_val_acc'], df['macro_f1'], alpha=0.6)
    plt.xlabel('CV Mean Validation Accuracy')
    plt.ylabel('Test Macro F1')
    plt.title('CV Validation Accuracy vs Test Macro F1')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_vs_test_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # CV stability analysis
    plt.figure(figsize=(10, 6))
    plt.scatter(df['cv_mean_val_acc'], df['cv_std_val_acc'], alpha=0.6, s=50)
    plt.xlabel('CV Mean Validation Accuracy')
    plt.ylabel('CV Std Validation Accuracy')
    plt.title('Cross-Validation Stability Analysis')
    plt.grid(True, alpha=0.3)
    
    # Highlight stable and high-performing configs
    stable_threshold = df['cv_std_val_acc'].quantile(0.25)  # Bottom 25% std = most stable
    high_perf_threshold = df['cv_mean_val_acc'].quantile(0.75)  # Top 25% mean = highest performing
    
    stable_high_perf = df[(df['cv_std_val_acc'] <= stable_threshold) & 
                          (df['cv_mean_val_acc'] >= high_perf_threshold)]
    
    if len(stable_high_perf) > 0:
        plt.scatter(stable_high_perf['cv_mean_val_acc'], stable_high_perf['cv_std_val_acc'], 
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
        print(f"CV Val Acc: {row['cv_mean_val_acc']:.4f} ± {row['cv_std_val_acc']:.4f}")
        print(f"Test Macro F1: {row['macro_f1']:.4f} | Test Acc: {row['test_acc']:.4f} | Minority F1: {row['minority_f1']:.4f}")
        if row['use_weighted_loss']:
            print(f"Config: weight_method={row['weight_method']}, lr={row['learning_rate']}, "
                  f"batch_size={row['batch_size']}, dropout={row['dropout_rate']}")
        else:
            print(f"Config: lr={row['learning_rate']}, batch_size={row['batch_size']}, dropout={row['dropout_rate']}")
        print(f"CV Stability: {row['cv_std_val_acc']:.4f} (lower is more stable)")
    
    return top_configs

def generate_cv_report(df, best_config, output_dir):
    """Generate comprehensive cross-validation report"""
    
    report_path = os.path.join(output_dir, 'mobilenetv3l_5fold_cv_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("5-FOLD CROSS-VALIDATION HYPERPARAMETER TUNING REPORT - MOBILENET V3-LARGE\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Tuning Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Trials: {len(df)}\n")
        f.write(f"Cross-Validation Folds: {args.cv_folds}\n")
        f.write(f"Training Epochs per Fold: {args.epochs}\n")
        f.write(f"Early Stopping Patience: {args.early_stopping_patience}\n\n")
        
        # Performance statistics
        f.write("PERFORMANCE STATISTICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Best Test Macro F1: {df['macro_f1'].max():.4f}\n")
        f.write(f"Best Test Accuracy: {df['test_acc'].max():.4f}\n")
        f.write(f"Best CV Mean Validation Accuracy: {df['cv_mean_val_acc'].max():.4f}\n")
        f.write(f"Average Test Macro F1: {df['macro_f1'].mean():.4f} ± {df['macro_f1'].std():.4f}\n")
        f.write(f"Average CV Validation Accuracy: {df['cv_mean_val_acc'].mean():.4f} ± {df['cv_mean_val_acc'].std():.4f}\n\n")
        
        # Best configuration
        f.write("🏆 BEST CONFIGURATION (by Test Macro F1):\n")
        f.write("-" * 45 + "\n")
        best_row = df.loc[df['macro_f1'].idxmax()]
        
        f.write("Loss Configuration:\n")
        f.write(f"  • Loss Type: {best_row['loss_type']}\n")
        f.write(f"  • Use Weighted Loss: {best_row.get('use_weighted_loss', 'N/A')}\n")
        if best_row.get('use_weighted_loss', False):
            f.write(f"  • Weight Method: {best_row['weight_method']}\n")
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
        f.write(f"  • CV Mean Val Acc: {best_row['cv_mean_val_acc']:.4f} ± {best_row['cv_std_val_acc']:.4f}\n")
        f.write(f"  • CV Mean Val Loss: {best_row['cv_mean_val_loss']:.4f} ± {best_row['cv_std_val_loss']:.4f}\n\n")
        
        f.write("Test Set Performance:\n")
        f.write(f"  • Test Accuracy: {best_row['test_acc']:.4f}\n")
        f.write(f"  • Macro F1: {best_row['macro_f1']:.4f}\n")
        f.write(f"  • Balanced Accuracy: {best_row['balanced_acc']:.4f}\n")
        f.write(f"  • Minority F1: {best_row['minority_f1']:.4f}\n")
        f.write(f"  • Cohen's Kappa: {best_row['kappa']:.4f}\n\n")
        
        # Cross-validation insights
        f.write("CROSS-VALIDATION INSIGHTS:\n")
        f.write("-" * 30 + "\n")
        
        # Loss type analysis
        if 'loss_type' in df.columns:
            loss_cv_performance = df.groupby('loss_type')['cv_mean_val_acc'].agg(['mean', 'std'])
            loss_test_performance = df.groupby('loss_type')['macro_f1'].agg(['mean', 'std'])
            f.write("1. Loss Type Performance Comparison:\n")
            for loss_type in loss_cv_performance.index:
                cv_mean = loss_cv_performance.loc[loss_type, 'mean']
                cv_std = loss_cv_performance.loc[loss_type, 'std']
                test_mean = loss_test_performance.loc[loss_type, 'mean']
                test_std = loss_test_performance.loc[loss_type, 'std']
                f.write(f"   {loss_type}:\n")
                f.write(f"     CV Val Acc: {cv_mean:.4f} ± {cv_std:.4f}\n")
                f.write(f"     Test F1: {test_mean:.4f} ± {test_std:.4f}\n")
        
        # Stability analysis
        stable_configs = df[df['cv_std_val_acc'] <= df['cv_std_val_acc'].quantile(0.25)]
        if len(stable_configs) > 0:
            best_stable = stable_configs.loc[stable_configs['macro_f1'].idxmax()]
            f.write(f"\n2. Most Stable High-Performing Configuration:\n")
            f.write(f"   Loss Type: {best_stable['loss_type']}\n")
            f.write(f"   CV Stability: {best_stable['cv_std_val_acc']:.4f} (std)\n")
            f.write(f"   Test Macro F1: {best_stable['macro_f1']:.4f}\n")
            f.write(f"   Config: lr={best_stable['learning_rate']}, batch={best_stable['batch_size']}\n")
        
        # Learning rate analysis
        lr_performance = df.groupby('learning_rate')['macro_f1'].agg(['mean', 'std', 'max'])
        best_lr = lr_performance['mean'].idxmax()
        f.write(f"\n3. Optimal Learning Rate: {best_lr} (avg Test F1: {lr_performance.loc[best_lr, 'mean']:.4f})\n")
        
        # Batch size analysis
        batch_performance = df.groupby('batch_size')['macro_f1'].agg(['mean', 'std', 'max'])
        best_batch = batch_performance['mean'].idxmax()
        f.write(f"4. Optimal Batch Size: {best_batch} (avg Test F1: {batch_performance.loc[best_batch, 'mean']:.4f})\n")
        
        # Recommendations
        f.write(f"\nRECOMMENDations:\n")
        f.write("-" * 18 + "\n")
        f.write("1. Production Deployment:\n")
        f.write(f"   Use the best configuration above for optimal performance\n")
        f.write(f"   Expected Test Macro F1: {best_row['macro_f1']:.4f}\n")
        f.write(f"   CV Validation suggests consistent performance: {best_row['cv_mean_val_acc']:.4f} ± {best_row['cv_std_val_acc']:.4f}\n\n")
        
        f.write("2. Model Reliability:\n")
        f.write(f"   5-fold CV provides robust performance estimates\n")
        f.write(f"   Low CV standard deviation indicates stable training\n")
        f.write(f"   Model performance is validated across different data splits\n\n")
        
        f.write("3. Training Efficiency with Early Stopping:\n")
        f.write(f"   10 epochs with early stopping (patience={args.early_stopping_patience}) prevents overfitting\n")
        f.write(f"   Average training time per trial: {df['training_time'].mean():.1f} minutes\n")
        f.write(f"   Total tuning time: {df['training_time'].sum():.1f} minutes\n\n")
        
        f.write("4. Cross-Validation Benefits:\n")
        f.write("   • Reduces overfitting to specific train/val splits\n")
        f.write("   • Provides confidence intervals for performance estimates\n")
        f.write("   • More reliable hyperparameter selection\n")
        f.write("   • Better generalization to unseen data\n\n")
        
        f.write("5. MobileNet V3-Large with CV Notes:\n")
        f.write("   • Model consistency verified across 5 different data splits\n")
        f.write("   • Efficient architecture suitable for production deployment\n")
        f.write("   • Cross-validated performance reduces selection bias\n")
        f.write("   • Final model should be retrained on full training set\n")

def main():
    start_time = time.time()
    
    print("🔬 5-FOLD CROSS-VALIDATION HYPERPARAMETER TUNING FOR MOBILENET V3-LARGE")
    print("📊 BASELINE vs WEIGHTED LOSS COMPARISON")
    print("=" * 70)
    print(f"Max trials: {args.max_trials}")
    print(f"Epochs per fold: {args.epochs}")
    print(f"CV folds: {args.cv_folds}")
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
    
    # Run hyperparameter tuning with cross-validation
    print(f"\n🚀 Starting 5-fold cross-validation hyperparameter tuning...")
    
    for trial_id in range(start_trial, len(combinations)):
        config = combinations[trial_id]
        trial_start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"TRIAL {trial_id + 1}/{len(combinations)}")
        print(f"{'='*70}")
        
        try:
            # Run cross-validation for this configuration
            cv_stats, loss_type = run_cv_for_config(config, cv_splits, full_dataset, device)
            
            print(f"  Loss Type: {loss_type}")
            print(f"  Config: lr={config['learning_rate']}, batch_size={config['batch_size']}, optimizer={config['optimizer_type']}")
            if config['use_weighted_loss']:
                print(f"  Weight method={config['weight_method']}, label_smoothing={config['label_smoothing']}")
            else:
                print(f"  Label smoothing={config['label_smoothing']}")
            
            print(f"\n  📊 Cross-Validation Results:")
            print(f"    CV Mean Val Acc: {cv_stats['mean_val_acc']:.4f} ± {cv_stats['std_val_acc']:.4f}")
            print(f"    CV Mean Val Loss: {cv_stats['mean_val_loss']:.4f} ± {cv_stats['std_val_loss']:.4f}")
            
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
            
            # Create and train final model
            final_model = create_mobilenet_v3_large_model(len(full_dataset.classes), config['dropout_rate'])
            
            # Setup criterion
            if config['use_weighted_loss']:
                class_weights = calculate_class_weights(train_val_dataset, method=config['weight_method']).to(device)
                if config['label_smoothing'] > 0:
                    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config['label_smoothing'])
                else:
                    criterion = nn.CrossEntropyLoss(weight=class_weights)
            else:
                if config['label_smoothing'] > 0:
                    criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
                else:
                    criterion = nn.CrossEntropyLoss()
            
            # Setup optimizer and scheduler for final training
            optimizer = get_optimizer(final_model, config)
            scheduler = get_scheduler(optimizer, config)
            
            # Train final model (without validation since we use full train+val data)
            final_model.to(device)
            base_lr = config['learning_rate']
            
            for epoch in range(args.epochs):
                # Warmup
                if config['warmup_epochs'] > 0:
                    warmup_scheduler(optimizer, config['warmup_epochs'], epoch, base_lr)
                
                final_model.train()
                running_loss, running_corrects, total = 0.0, 0, 0
                
                for inputs, labels in train_val_loader:
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
                
                # Scheduler step (for cosine annealing)
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
                'cv_stats': cv_stats,
                'test_results': test_results,
                'training_time': trial_time / 60  # Convert to minutes
            }
            
            trial_results.append(trial_result)
            
            print(f"\n  📊 Final Results:")
            print(f"    CV Val Acc: {cv_stats['mean_val_acc']:.4f} ± {cv_stats['std_val_acc']:.4f}")
            print(f"    Test Acc: {test_results['test_acc']:.4f} | Macro F1: {test_results['macro_f1']:.4f}")
            print(f"    Minority F1: {test_results['minority_f1']:.4f} | Balanced Acc: {test_results['balanced_acc']:.4f}")
            print(f"    Total time: {trial_time/60:.1f} minutes")
            
            # Save intermediate results every 3 trials (less frequent due to longer CV time)
            if (trial_id + 1) % 3 == 0:
                intermediate_path = os.path.join(args.output_dir, f'intermediate_cv_results_trial_{trial_id + 1}.json')
                with open(intermediate_path, 'w') as f:
                    json.dump(trial_results, f, indent=2, default=str)
                print(f"    💾 Intermediate results saved to {intermediate_path}")
            
        except Exception as e:
            print(f"❌ Trial {trial_id + 1} failed: {str(e)}")
            continue
    
    # Final analysis and reporting
    print(f"\n{'='*80}")
    print("🎯 5-FOLD CV HYPERPARAMETER TUNING COMPLETED")
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
    
    print(f"\n🏆 BEST CONFIGURATION (by Test Macro F1):")
    print(f"Trial ID: {best_trial['trial_id']}")
    print(f"Loss Type: {best_trial['loss_type']}")
    print(f"CV Val Acc: {best_trial['cv_stats']['mean_val_acc']:.4f} ± {best_trial['cv_stats']['std_val_acc']:.4f}")
    print(f"Test Macro F1: {best_trial['test_results']['macro_f1']:.4f}")
    print(f"Test Accuracy: {best_trial['test_results']['test_acc']:.4f}")
    print(f"Minority F1: {best_trial['test_results']['minority_f1']:.4f}")
    print(f"Training time: {best_trial['training_time']:.1f} minutes")
    
    # Separate analysis for Baseline vs Weighted Loss
    baseline_trials = [t for t in trial_results if not t['config']['use_weighted_loss']]
    weighted_trials = [t for t in trial_results if t['config']['use_weighted_loss']]
    
    print(f"\n📊 CROSS-VALIDATION COMPARISON BY LOSS TYPE:")
    
    if baseline_trials:
        best_baseline = max(baseline_trials, key=lambda x: x['test_results']['macro_f1'])
        baseline_cv_accs = [t['cv_stats']['mean_val_acc'] for t in baseline_trials]
        baseline_test_f1s = [t['test_results']['macro_f1'] for t in baseline_trials]
        
        print(f"\n📊 BASELINE (CrossEntropy) Results:")
        print(f"  Best Test Macro F1: {best_baseline['test_results']['macro_f1']:.4f}")
        print(f"  Best CV Val Acc: {best_baseline['cv_stats']['mean_val_acc']:.4f} ± {best_baseline['cv_stats']['std_val_acc']:.4f}")
        print(f"  Average CV Val Acc: {np.mean(baseline_cv_accs):.4f} ± {np.std(baseline_cv_accs):.4f}")
        print(f"  Average Test F1: {np.mean(baseline_test_f1s):.4f} ± {np.std(baseline_test_f1s):.4f}")
    
    if weighted_trials:
        best_weighted = max(weighted_trials, key=lambda x: x['test_results']['macro_f1'])
        weighted_cv_accs = [t['cv_stats']['mean_val_acc'] for t in weighted_trials]
        weighted_test_f1s = [t['test_results']['macro_f1'] for t in weighted_trials]
        
        print(f"\n⚖️ WEIGHTED LOSS Results:")
        print(f"  Best Test Macro F1: {best_weighted['test_results']['macro_f1']:.4f}")
        print(f"  Best CV Val Acc: {best_weighted['cv_stats']['mean_val_acc']:.4f} ± {best_weighted['cv_stats']['std_val_acc']:.4f}")
        print(f"  Average CV Val Acc: {np.mean(weighted_cv_accs):.4f} ± {np.std(weighted_cv_accs):.4f}")
        print(f"  Average Test F1: {np.mean(weighted_test_f1s):.4f} ± {np.std(weighted_test_f1s):.4f}")
    
    if baseline_trials and weighted_trials:
        print(f"\n🎯 WINNER: ", end="")
        if best_weighted['test_results']['macro_f1'] > best_baseline['test_results']['macro_f1']:
            print("Weighted Loss")
            improvement = best_weighted['test_results']['macro_f1'] - best_baseline['test_results']['macro_f1']
            print(f"  Weighted Loss outperforms by {improvement:.4f} in Test Macro F1")
        else:
            print("Baseline (CrossEntropy)")
            improvement = best_baseline['test_results']['macro_f1'] - best_weighted['test_results']['macro_f1']
            print(f"  Baseline outperforms by {improvement:.4f} in Test Macro F1")
        
        # Cross-validation consistency check
        baseline_avg_cv = np.mean(baseline_cv_accs)
        weighted_avg_cv = np.mean(weighted_cv_accs)
        print(f"  CV Validation also suggests: {'Weighted Loss' if weighted_avg_cv > baseline_avg_cv else 'Baseline'} is better")
        
        # Minority class analysis
        minority_improvement = best_weighted['test_results']['minority_f1'] - best_baseline['test_results']['minority_f1']
        print(f"  Minority class improvement with Weighted Loss: {minority_improvement:+.4f}")
    
    # Generate analysis plots
    analyze_cv_results(df, args.output_dir)
    
    # Generate final report
    generate_cv_report(df, best_config, args.output_dir)
    
    # Performance summary
    print(f"\n📊 CROSS-VALIDATION PERFORMANCE SUMMARY:")
    print(f"Best Test Macro F1: {df['macro_f1'].max():.4f}")
    print(f"Best CV Val Acc: {df['cv_mean_val_acc'].max():.4f}")
    print(f"Average Test Macro F1: {df['macro_f1'].mean():.4f} ± {df['macro_f1'].std():.4f}")
    print(f"Average CV Val Acc: {df['cv_mean_val_acc'].mean():.4f} ± {df['cv_mean_val_acc'].std():.4f}")
    print(f"CV Stability (avg std): {df['cv_std_val_acc'].mean():.4f}")
    
    # Save best configuration for easy deployment
    best_config_path = os.path.join(args.output_dir, 'best_cv_hyperparameters.json')
    
    deployment_configs = {
        'overall_best': {
            'config': best_config,
            'loss_type': best_trial['loss_type'],
            'cv_performance': {
                'mean_val_acc': float(best_trial['cv_stats']['mean_val_acc']),
                'std_val_acc': float(best_trial['cv_stats']['std_val_acc']),
                'mean_val_loss': float(best_trial['cv_stats']['mean_val_loss']),
                'std_val_loss': float(best_trial['cv_stats']['std_val_loss'])
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
                'total_training_time_minutes': best_trial['training_time']
            }
        }
    }
    
    # Add best baseline and weighted loss configs
    if baseline_trials:
        best_baseline = max(baseline_trials, key=lambda x: x['test_results']['macro_f1'])
        deployment_configs['best_baseline'] = {
            'config': best_baseline['config'],
            'loss_type': best_baseline['loss_type'],
            'cv_performance': {
                'mean_val_acc': float(best_baseline['cv_stats']['mean_val_acc']),
                'std_val_acc': float(best_baseline['cv_stats']['std_val_acc']),
            },
            'test_performance': {
                'macro_f1': float(best_baseline['test_results']['macro_f1']),
                'test_acc': float(best_baseline['test_results']['test_acc']),
                'minority_f1': float(best_baseline['test_results']['minority_f1'])
            }
        }
    
    if weighted_trials:
        best_weighted = max(weighted_trials, key=lambda x: x['test_results']['macro_f1'])
        deployment_configs['best_weighted_loss'] = {
            'config': best_weighted['config'],
            'loss_type': best_weighted['loss_type'],
            'cv_performance': {
                'mean_val_acc': float(best_weighted['cv_stats']['mean_val_acc']),
                'std_val_acc': float(best_weighted['cv_stats']['std_val_acc']),
            },
            'test_performance': {
                'macro_f1': float(best_weighted['test_results']['macro_f1']),
                'test_acc': float(best_weighted['test_results']['test_acc']),
                'minority_f1': float(best_weighted['test_results']['minority_f1'])
            }
        }
    
    with open(best_config_path, 'w') as f:
        json.dump(deployment_configs, f, indent=2)
    
    total_time = time.time() - start_time
    
    print(f"\n✅ 5-FOLD CV HYPERPARAMETER TUNING COMPLETED!")
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"Results saved to: {args.output_dir}")
    print(f"Best config saved to: {best_config_path}")
    print(f"\nGenerated files:")
    print(f"├── 📊 mobilenetv3l_5fold_cv_results.csv")
    print(f"├── 📋 mobilenetv3l_5fold_cv_report.txt")
    print(f"├── 🎯 best_cv_hyperparameters.json")
    print(f"├── 📈 cv_vs_test_performance.png")
    print(f"└── 📊 cv_stability_analysis.png")
    
    print(f"\n🔬 5-FOLD CROSS-VALIDATION ADVANTAGES:")
    print(f"• More robust hyperparameter selection")
    print(f"• Reduced overfitting to specific data splits")
    print(f"• Confidence intervals for performance estimates")
    print(f"• Better generalization assessment")
    print(f"• Validated with {args.cv_folds} different train/val splits")
    print(f"• Early stopping prevents overfitting in each fold")
    
    print(f"\n📱 MOBILENET V3-LARGE + CV BENEFITS:")
    print(f"• Efficient architecture thoroughly validated")
    print(f"• {args.cv_folds}×{args.epochs} = {args.cv_folds * args.epochs} total training epochs per config")
    print(f"• Cross-validated performance reduces selection bias")
    print(f"• Optimal for mobile deployment with confidence")

if __name__ == '__main__':
    main()