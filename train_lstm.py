import torch
import yaml
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append('src')

from src.data.preprocessor import DataPreprocessor
from src.models.lstm_seq2seq import initialize_lstm_model
from src.training.trainer import Trainer

def setup_environment():
    """Setup environment for Colab or local execution"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create necessary directories
    Path('checkpoints').mkdir(exist_ok=True)
    Path('configs').mkdir(exist_ok=True)
    
    return device

def load_config(config_path='configs/training_config.yaml'):
    """Load configuration with fallbacks for Colab"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        print(f"⚠️  Config file not found at {config_path}, using defaults")
        # Return default config
        return {
            'data_dir': 'data/raw/aksharantar',
            'train_data_path': 'aksharantar/train/hin_train.jsonl',
            'val_data_path': 'aksharantar/valid/hin_valid.jsonl',  # New
            'test_data_path': 'aksharantar/test/hin_test.jsonl',
            'sample_size': 100000,
            'batch_size': 64,
            'embedding_dim': 256,
            'hidden_dim': 512,
            'num_layers': 2,
            'dropout': 0.3,
            'learning_rate': 0.001,
            'num_epochs': 50,
            'teacher_forcing_ratio': 0.9,
            'seed': 42
        }

def main():
    print("🚀 Hindi Transliteration Model Training")
    print("=" * 50)
    
    # Setup environment
    device = setup_environment()
    
    # Load configuration
    config = load_config('/content/dlnlp-transliteration-cleaned/configs/training_config.yml')
    
    # Set random seeds for reproducibility
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Prepare data
    print("\n" + "="*60)
    print("📊 PREPARING DATA")
    print("="*60)
    
    try:
        preprocessor = DataPreprocessor(config)
        train_loader, val_loader, test_loader, roman_vocab, devanagari_vocab = preprocessor.prepare_data()
        print("✅ Data preparation successful!")
    except Exception as e:
        print(f"❌ Error preparing data: {e}")
        print("💡 Make sure your data is in the correct location:")
        print(f"   Training: {config.get('data_dir')}/{config.get('train_data_path')}")
        print(f"   Validation: {config.get('data_dir')}/{config.get('val_data_path')}")
        print(f"   Test: {config.get('data_dir')}/{config.get('test_data_path')}")
        return
    
    # Initialize model
    print("\n" + "="*60)
    print("🤖 INITIALIZING MODEL")
    print("="*60)
    
    try:
        model = initialize_lstm_model(
            roman_vocab_size=len(roman_vocab),
            devanagari_vocab_size=len(devanagari_vocab),
            device=device,
            embedding_dim=config.get('embedding_dim', 256),
            hidden_dim=config.get('hidden_dim', 512),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.3)
        )
        print("✅ Model initialization successful!")
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        return
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📈 Model Architecture:")
    print(f"   • Encoder: 2-layer BiLSTM, hidden_dim={config.get('hidden_dim', 512)}")
    print(f"   • Decoder: 2-layer LSTM with Attention, hidden_dim={config.get('hidden_dim', 512)}")
    print(f"   • Embedding dim: {config.get('embedding_dim', 256)}")
    print(f"   • Total parameters: {total_params:,}")
    print(f"   • Trainable parameters: {trainable_params:,}")
    print(f"   • Roman vocabulary size: {len(roman_vocab)}")
    print(f"   • Devanagari vocabulary size: {len(devanagari_vocab)}")
    
    # Initialize trainer
    print("\n" + "="*60)
    print("🎯 INITIALIZING TRAINER")
    print("="*60)
    
    try:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            roman_vocab=roman_vocab,
            devanagari_vocab=devanagari_vocab,
            config=config,
            device=device
        )
        print("✅ Trainer initialization successful!")
    except Exception as e:
        print(f"❌ Error initializing trainer: {e}")
        return
    
    # Start training
    print("\n" + "="*60)
    print("🔥 STARTING TRAINING")
    print("="*60)
    
    try:
        train_losses, val_losses, teacher_forcing_ratios, learning_rates = trainer.train()
        
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETED!")
        print("="*60)
        
        # Load best model for final evaluation
        print("\n📥 Loading best model for final evaluation...")
        trainer.load_checkpoint('checkpoints/best_model.pth', load_optimizer=False, load_scheduler=False)
        
        print(f"🏆 Best validation loss: {trainer.best_val_loss:.4f}")
        print(f"✅ Training completed successfully!")
        
        # Save final training summary
        print(f"\n💾 Checkpoints saved in: checkpoints/")
        print(f"   • best_model.pth - Best model for inference")
        print(f"   • checkpoint_epoch_*.pth - Training checkpoints")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    return trainer

if __name__ == "__main__":
    trainer = main()