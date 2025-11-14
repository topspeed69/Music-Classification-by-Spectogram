"""
Main Training Script for Self-Supervised Music Encoder
Trains CNN encoder with contrastive learning on spectrograms
"""

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from CNN.models import build_model
from CNN.augmentation import get_augmentation_pipeline
from CNN.data import create_dataloaders
from CNN.training import get_contrastive_loss
from CNN.utils.metrics import AverageMeter


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    losses = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (view1, view2) in enumerate(pbar):
        view1 = view1.to(device)
        view2 = view2.to(device)
        
        # Forward pass for both views
        z1 = model(view1)
        z2 = model(view2)
        
        # Compute loss
        loss = criterion(z1, z2)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        losses.update(loss.item(), view1.size(0))
        
        # Update progress bar
        pbar.set_postfix({'loss': losses.avg})
    
    return losses.avg


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    losses = AverageMeter()
    
    with torch.no_grad():
        for view1, view2 in tqdm(val_loader, desc="Validation"):
            view1 = view1.to(device)
            view2 = view2.to(device)
            
            # Forward pass
            z1 = model(view1)
            z2 = model(view2)
            
            # Compute loss
            loss = criterion(z1, z2)
            losses.update(loss.item(), view1.size(0))
    
    return losses.avg


def main(args):
    """Main training function"""
    
    # Load configurations
    with open(args.model_config) as f:
        model_config = yaml.safe_load(f)
    
    with open(args.training_config) as f:
        training_config = yaml.safe_load(f)
    
    with open(args.data_config) as f:
        data_config = yaml.safe_load(f)
    
    # Combine configs
    config = {**model_config, **training_config, **data_config}
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Build model
    print("Building model...")
    model = build_model(config)
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create augmentation pipelines
    train_transform = get_augmentation_pipeline(config, training=True)
    val_transform = get_augmentation_pipeline(config, training=False)
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config, train_transform, val_transform)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create loss function
    contrastive_config = training_config['training']['contrastive']
    criterion = get_contrastive_loss(
        loss_type=contrastive_config['loss_type'],
        temperature=contrastive_config['temperature'],
        use_cosine_similarity=contrastive_config['use_cosine_similarity']
    )
    
    # Create optimizer
    optimizer_config = training_config['training']['optimizer']
    optimizer = optim.Adam(
        model.parameters(),
        lr=optimizer_config['learning_rate'],
        weight_decay=optimizer_config['weight_decay']
    )
    
    # Create learning rate scheduler
    scheduler_config = training_config['training']['scheduler']
    if scheduler_config['type'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_config['training']['epochs'],
            eta_min=scheduler_config['min_lr']
        )
    
    # TensorBoard writer
    log_dir = Path(training_config['paths']['tensorboard_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # Checkpoint directory
    checkpoint_dir = Path(training_config['paths']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    epochs = training_config['training']['epochs']
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            print(f"Saved best model (val_loss: {val_loss:.4f})")
        
        # Save periodic checkpoint
        if epoch % model_config['checkpoint']['save_frequency'] == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
    
    print("\nTraining completed!")
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Self-Supervised Music Encoder')
    parser.add_argument('--model_config', type=str, default='../configs/model_config.yaml',
                       help='Path to model configuration')
    parser.add_argument('--training_config', type=str, default='../configs/training_config.yaml',
                       help='Path to training configuration')
    parser.add_argument('--data_config', type=str, default='../configs/data_config.yaml',
                       help='Path to data configuration')
    
    args = parser.parse_args()
    main(args)
