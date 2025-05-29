import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm
import logging
from datetime import datetime
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from test_dino import vit_small
from visualize_slide_and_annotations import extract_tissue_patches
import openslide
import xml.etree.ElementTree as ET


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TumorClassifier(nn.Module):
    """Tumor classifier using DINO ViT as foundation model with fine-tuning head"""
    def __init__(self, num_classes=2, freeze_backbone=True, dropout_rate=0.5):
        super().__init__()
        self.backbone = vit_small(pretrained=True, progress=False, key="DINO_p16", patch_size=16)
        self.feature_dim = self.backbone.embed_dim
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        self._init_classifier()
    
    def _init_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    
    def extract_features(self, x):
        with torch.no_grad():
            return self.backbone(x)


class PatchDataset(Dataset):
    """Dataset for loading extracted patches and labels"""
    def __init__(self, patches, labels, transform=None):
        self.patches = patches
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx]
        label = self.labels[idx]
        
        if self.transform:
            patch = self.transform(patch)
        
        return patch, label


def extract_patches_from_slides(slide_paths, tissue_paths, tumor_paths, patch_size=224, 
                               samples_per_slide=500, cache_dir="patch_cache"):
    """Extract patches from slides with tumor/non-tumor labels"""
    os.makedirs(cache_dir, exist_ok=True)
    
    all_patches = []
    all_labels = []
    all_metadata = []
    
    for idx, (slide_path, tissue_path, tumor_path) in enumerate(tqdm(
            zip(slide_paths, tissue_paths, tumor_paths), 
            desc="Processing slides", total=len(slide_paths))):
        
        slide_name = Path(slide_path).stem
        cache_file = Path(cache_dir) / f"{slide_name}_patches.npz"
        
        if cache_file.exists():
            logger.info(f"Loading cached patches for {slide_name}")
            data = np.load(cache_file)
            patches = data['patches']
            labels = data['labels']
            coords = data['coords']
        else:
            try:
                tumor_patches, tumor_coords = extract_tissue_patches(
                    slide_path, tumor_path, patch_size=patch_size, level=0
                )
                
                tissue_patches, tissue_coords = extract_tissue_patches(
                    slide_path, tissue_path, patch_size=patch_size, level=0
                )
                
                tumor_coords_set = {tuple(coord) for coord in tumor_coords}
                
                non_tumor_patches = []
                non_tumor_coords = []
                for i, coord in enumerate(tissue_coords):
                    if tuple(coord) not in tumor_coords_set:
                        non_tumor_patches.append(tissue_patches[i])
                        non_tumor_coords.append(coord)
                
                n_tumor = min(samples_per_slide // 2, len(tumor_patches))
                n_non_tumor = min(samples_per_slide // 2, len(non_tumor_patches))
                
                if n_tumor > 0 and n_non_tumor > 0:
                    tumor_indices = np.random.choice(len(tumor_patches), n_tumor, replace=False)
                    non_tumor_indices = np.random.choice(len(non_tumor_patches), n_non_tumor, replace=False)
                    
                    patches = np.concatenate([
                        tumor_patches[tumor_indices],
                        np.array(non_tumor_patches)[non_tumor_indices]
                    ])
                    
                    labels = np.concatenate([
                        np.ones(n_tumor, dtype=np.int64),
                        np.zeros(n_non_tumor, dtype=np.int64)
                    ])
                    
                    coords = np.concatenate([
                        tumor_coords[tumor_indices],
                        np.array(non_tumor_coords)[non_tumor_indices]
                    ])
                    
                    np.savez_compressed(cache_file, patches=patches, labels=labels, coords=coords)
                else:
                    logger.warning(f"Skipping slide {slide_name}: insufficient patches")
                    continue
                    
            except Exception as e:
                logger.error(f"Error processing slide {slide_name}: {str(e)}")
                continue
        
        all_patches.extend(patches)
        all_labels.extend(labels)
        all_metadata.extend([{'slide': slide_name, 'coord': coord} for coord in coords])
    
    return np.array(all_patches), np.array(all_labels), all_metadata


def get_data_transforms():
    """Get data augmentation transforms"""
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=10, min_delta=0.0001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif (self.mode == 'max' and score < self.best_score - self.min_delta) or \
             (self.mode == 'min' and score > self.best_score + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (self.mode == 'max' and score > self.best_score) or \
               (self.mode == 'min' and score < self.best_score):
                self.best_score = score
            self.counter = 0
        
        return self.early_stop


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch_idx, (patches, labels) in enumerate(tqdm(dataloader, desc="Training")):
        patches, labels = patches.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(patches)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for patches, labels in tqdm(dataloader, desc="Validating"):
            patches, labels = patches.to(device), labels.to(device)
            
            outputs = model(patches)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm.tolist()
    }
    
    return metrics


def main():
    data_dir = Path("moffit_data")
    slides = list((data_dir / "Failed_Slides_DNN").glob("*.svs"))
    slides += list((data_dir / "Pass_Slides_DNN").glob("*.svs"))

    tumor_annotations = list((data_dir / "Failed_tumor_annotations").glob("*.xml"))
    tumor_annotations += list((data_dir / "Pass_tumor_annotations").glob("*.xml"))
    tumor_names = [x.stem for x in tumor_annotations]

    usable_slides, usable_tissue_annotations, usable_tumor_annotations = [], [], []
    for s in slides:
        if s.stem in tumor_names:
            usable_slides.append(str(s))
            tissue_xml = str(s).replace("_DNN", "_annotations").replace(".svs", ".xml")
            tumor_xml = next((str(t) for t in tumor_annotations if t.stem == s.stem), None)
            if tissue_xml and tumor_xml and Path(tissue_xml).exists():
                usable_tissue_annotations.append(tissue_xml)
                usable_tumor_annotations.append(tumor_xml)
    
    output_dir = Path("trained_models")
    output_dir.mkdir(exist_ok=True)
    
    experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(exist_ok=True)
    
    config = {
        'patch_size': 224,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_epochs': 100,
        'early_stopping_patience': 15,
        'samples_per_slide': 1000,
        'freeze_backbone': True,
        'dropout_rate': 0.5,
        'val_split': 0.2,
        'test_slides': 1,
        'seed': 42
    }
    
    with open(experiment_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    test_slides = usable_slides[-config['test_slides']:]
    test_tissue = usable_tissue_annotations[-config['test_slides']:]
    test_tumor = usable_tumor_annotations[-config['test_slides']:]
    
    train_val_slides = usable_slides[:-config['test_slides']]
    train_val_tissue = usable_tissue_annotations[:-config['test_slides']]
    train_val_tumor = usable_tumor_annotations[:-config['test_slides']]
    
    n_val = int(len(train_val_slides) * config['val_split'])
    train_slides = train_val_slides[:-n_val] if n_val > 0 else train_val_slides
    train_tissue = train_val_tissue[:-n_val] if n_val > 0 else train_val_tissue
    train_tumor = train_val_tumor[:-n_val] if n_val > 0 else train_val_tumor
    
    val_slides = train_val_slides[-n_val:] if n_val > 0 else []
    val_tissue = train_val_tissue[-n_val:] if n_val > 0 else []
    val_tumor = train_val_tumor[-n_val:] if n_val > 0 else []
    
    logger.info(f"Train slides: {len(train_slides)}, Val slides: {len(val_slides)}, Test slides: {len(test_slides)}")
    
    logger.info("Extracting training patches...")
    train_patches, train_labels, train_metadata = extract_patches_from_slides(
        train_slides, train_tissue, train_tumor, 
        patch_size=config['patch_size'],
        samples_per_slide=config['samples_per_slide']
    )
    
    if len(val_slides) > 0:
        logger.info("Extracting validation patches...")
        val_patches, val_labels, val_metadata = extract_patches_from_slides(
            val_slides, val_tissue, val_tumor,
            patch_size=config['patch_size'],
            samples_per_slide=config['samples_per_slide']
        )
    else:
        val_split_idx = int(len(train_patches) * 0.2)
        val_patches = train_patches[:val_split_idx]
        val_labels = train_labels[:val_split_idx]
        train_patches = train_patches[val_split_idx:]
        train_labels = train_labels[val_split_idx:]
    
    logger.info(f"Train patches: {len(train_patches)}, Val patches: {len(val_patches)}")
    
    train_transform, val_transform = get_data_transforms()
    
    train_dataset = PatchDataset(train_patches, train_labels, transform=train_transform)
    val_dataset = PatchDataset(val_patches, val_labels, transform=val_transform)
    
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], 
        sampler=sampler, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], 
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    model = TumorClassifier(
        num_classes=2, 
        freeze_backbone=config['freeze_backbone'],
        dropout_rate=config['dropout_rate']
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).float().to(device))
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['num_epochs'], eta_min=1e-6
    )
    
    early_stopping = EarlyStopping(
        patience=config['early_stopping_patience'], 
        mode='max'
    )
    
    best_val_acc = 0
    best_model_path = experiment_dir / 'best_model.pth'
    training_history = defaultdict(list)
    
    logger.info("Starting training...")
    for epoch in range(config['num_epochs']):
        logger.info(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        scheduler.step()
        
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        for key, value in val_metrics.items():
            if key != 'confusion_matrix':
                training_history[f'val_{key}'].append(value)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        logger.info(f"Val Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
        logger.info(f"Val F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, best_model_path)
            logger.info(f"Saved best model with val accuracy: {best_val_acc:.4f}")
        
        if early_stopping(val_metrics['accuracy']):
            logger.info("Early stopping triggered")
            break
    
    with open(experiment_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    logger.info("\nLoading best model for testing...")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info("Extracting test patches...")
    test_patches, test_labels, test_metadata = extract_patches_from_slides(
        test_slides, test_tissue, test_tumor,
        patch_size=config['patch_size'],
        samples_per_slide=config['samples_per_slide'] * 2
    )
    
    test_dataset = PatchDataset(test_patches, test_labels, transform=val_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    logger.info("Evaluating on test set...")
    test_metrics = validate_epoch(model, test_loader, criterion, device)
    
    logger.info("\n=== Test Results ===")
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
    logger.info(f"Test F1: {test_metrics['f1']:.4f}")
    logger.info(f"Test AUC: {test_metrics['auc']:.4f}")
    logger.info(f"Confusion Matrix:\n{np.array(test_metrics['confusion_matrix'])}")
    
    with open(experiment_dir / 'test_results.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    final_model_path = experiment_dir / 'final_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'test_metrics': test_metrics
    }, final_model_path)
    
    logger.info(f"\nTraining complete! Results saved to {experiment_dir}")


if __name__ == "__main__":
    main()