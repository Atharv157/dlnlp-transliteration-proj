import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import math
import numpy as np
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, model, train_loader, val_loader, roman_vocab, devanagari_vocab, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.roman_vocab = roman_vocab
        self.devanagari_vocab = devanagari_vocab
        self.config = config
        self.device = device
        
        # Training components
        self.optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))
        
        # ✅ DEBUG: Check ALL scheduler parameters
        print(f"🔍 DEBUG SCHEDULER PARAMS:")
        print(f"   min_lr: {config.get('min_lr')}, type: {type(config.get('min_lr'))}")
        print(f"   lr_decay_factor: {config.get('lr_decay_factor')}, type: {type(config.get('lr_decay_factor'))}")
        print(f"   lr_patience: {config.get('lr_patience')}, type: {type(config.get('lr_patience'))}")
        
        # Convert to proper types
        min_lr = float(config.get('min_lr', 1e-6))
        lr_decay_factor = float(config.get('lr_decay_factor', 0.5))
        lr_patience = int(config.get('lr_patience', 3))
        
        print(f"🔍 AFTER CONVERSION:")
        print(f"   min_lr: {min_lr}, type: {type(min_lr)}")
        print(f"   lr_decay_factor: {lr_decay_factor}, type: {type(lr_decay_factor)}")

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            factor=lr_decay_factor,
            patience=lr_patience,
            min_lr=min_lr
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index
        
        # Teacher forcing scheduling
        self.initial_teacher_forcing = config.get('teacher_forcing_ratio', 0.9)
        self.final_teacher_forcing = config.get('final_teacher_forcing', 0.3)
        self.teacher_forcing_decay = config.get('teacher_forcing_decay', 0.95)
        self.current_teacher_forcing = self.initial_teacher_forcing
        
        # Training state
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.teacher_forcing_ratios = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.epoch = 0
        
        # Create checkpoint directory
        os.makedirs('checkpoints', exist_ok=True)
    
    def update_teacher_forcing(self):
        """Update teacher forcing ratio with exponential decay"""
        self.current_teacher_forcing = max(
            self.final_teacher_forcing,
            self.current_teacher_forcing * self.teacher_forcing_decay
        )
        self.teacher_forcing_ratios.append(self.current_teacher_forcing)
        return self.current_teacher_forcing
    
    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch:02d} [Train]')
        
        # Get current teacher forcing ratio for this epoch
        teacher_forcing_ratio = self.current_teacher_forcing
        current_lr = self.optimizer.param_groups[0]['lr']
        
        for batch_idx, (src, trg, src_lengths, trg_lengths) in enumerate(progress_bar):
            src, trg, src_lengths = src.to(self.device), trg.to(self.device), src_lengths.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with current teacher forcing ratio
            output = self.model(src, trg, teacher_forcing_ratio=teacher_forcing_ratio, src_lengths=src_lengths)
            
            # Calculate loss (ignore <sos> token for loss calculation)
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            loss = self.criterion(output, trg)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('grad_clip', 1.0))
            
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
            # Update progress bar with teacher forcing and LR info
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{epoch_loss/(batch_idx+1):.4f}',
                'TF Ratio': f'{teacher_forcing_ratio:.3f}',
                'LR': f'{current_lr:.2e}'
            })
        
        return epoch_loss / len(self.train_loader)
    
    def validate_epoch(self, epoch):
        self.model.eval()
        epoch_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc=f'Epoch {epoch:02d} [Val]')
            
            for batch_idx,  (src, trg, src_lengths, trg_lengths) in enumerate(progress_bar):
                src, trg, src_lengths = src.to(self.device), trg.to(self.device), src_lengths.to(self.device)
                
                # Forward pass without teacher forcing during validation
                output = self.model(src, trg, teacher_forcing_ratio=0.0, src_lengths=src_lengths)
                
                # Calculate loss
                output_dim = output.shape[-1]
                output = output[:, 1:].reshape(-1, output_dim)
                trg = trg[:, 1:].reshape(-1)
                
                loss = self.criterion(output, trg)
                epoch_loss += loss.item()
                
                # Store predictions for metrics
                predictions = output.argmax(1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(trg.cpu().numpy())
                
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{epoch_loss/(batch_idx+1):.4f}'
                })
        
        avg_loss = epoch_loss / len(self.val_loader)
        
        # Calculate accuracy
        accuracy = self.calculate_accuracy(all_predictions, all_targets)
        
        return avg_loss, accuracy
    
    def calculate_accuracy(self, predictions, targets):
        # Filter out padding tokens (index 0)
        non_padding_mask = np.array(targets) != 0
        if non_padding_mask.sum() == 0:
            return 0.0
        
        correct = (np.array(predictions)[non_padding_mask] == np.array(targets)[non_padding_mask]).sum()
        total = non_padding_mask.sum()
        return correct / total
    
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'roman_vocab': self.roman_vocab,
            'devanagari_vocab': self.devanagari_vocab,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'teacher_forcing_ratios': self.teacher_forcing_ratios,
            'current_teacher_forcing': self.current_teacher_forcing,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        save_dir = self.config.get('save_dir')
        torch.save(checkpoint, f'{save_dir}/checkpoint_epoch_{epoch}.pth')
        
        if is_best:
            torch.save(checkpoint, f'{save_dir}/best_model.pth')
            print(f'New best model saved with val_loss: {self.best_val_loss:.4f}')
    
    def load_checkpoint(self, checkpoint_path, load_optimizer=True, load_scheduler=True):
        """Load checkpoint with options to resume training or just use for inference"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load training state (optional for inference)
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if load_scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load vocabularies and training history
        self.roman_vocab = checkpoint['roman_vocab']
        self.devanagari_vocab = checkpoint['devanagari_vocab']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.learning_rates = checkpoint.get('learning_rates', [])
        self.teacher_forcing_ratios = checkpoint['teacher_forcing_ratios']
        self.current_teacher_forcing = checkpoint['current_teacher_forcing']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return checkpoint['epoch']
    
    def train(self):
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Initial Teacher Forcing: {self.initial_teacher_forcing}")
        print(f"Final Teacher Forcing: {self.final_teacher_forcing}")
        print(f"Teacher Forcing Decay: {self.teacher_forcing_decay}")
        print(f"Initial Learning Rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"Config: {self.config}")
        
        start_epoch = 1
        num_epochs = self.config.get('num_epochs', 50)
        patience = self.config.get('patience', 7)
        
        for epoch in range(start_epoch, num_epochs + 1):
            self.epoch = epoch
            start_time = time.time()
            
            # Update teacher forcing for this epoch
            current_tf = self.update_teacher_forcing()
            
            # Training phase
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validation phase
            val_loss, val_accuracy = self.validate_epoch(epoch)
            self.val_losses.append(val_loss)
            
            # Update learning rate scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            epoch_time = time.time() - start_time
            
            # Print epoch summary
            print(f'Epoch: {epoch:02d} | Time: {epoch_time:.2f}s')
            print(f'\tTrain Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
            print(f'\tVal Accuracy: {val_accuracy:.4f}')
            print(f'\tTeacher Forcing: {current_tf:.3f}')
            print(f'\tLearning Rate: {current_lr:.2e}')
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
                self.save_checkpoint(epoch)
            
            # Early stopping
            if self.patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break
            
            print('-' * 60)
        
        print("Training completed!")
        return self.train_losses, self.val_losses, self.teacher_forcing_ratios, self.learning_rates