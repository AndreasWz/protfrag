# train.py
import argparse
import yaml
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import torch

# Adapted imports for src/ layout
from src.data import FragmentDataModule
from src.model import FragmentDetector

# (load_config function from A2 is good) ...
def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train(config: dict):
    pl.seed_everything(config.get('seed', 42), workers=True)
    
    # 1. Initialize data module
    datamodule = FragmentDataModule(
        metadata_path=config['data']['metadata_path'],
        embedding_dir=config['data']['embeddings_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data'].get('num_workers', 4),
        embedding_type=config['data'].get('embedding_type', 'mean_pooled'),
        pin_memory=config['data'].get('pin_memory', True)
        # Note: A2's augmentations are removed for simplicity, add back if needed
    )
    
    # 2. Setup datamodule to get class weights
    datamodule.setup('fit')
    class_weights = datamodule.get_class_weights()
    print("\n=== Class Weights ===")
    print(f"Binary: {class_weights['binary']}")
    print(f"Multilabel: {class_weights['multilabel']}")
    
    # 3. Initialize model
    # We pass model hparams manually
    model = FragmentDetector(
        embedding_dim=config['model']['embedding_dim'],
        hidden_dims=config['model']['hidden_dims'],
        dropout=config['model']['dropout'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['model'].get('weight_decay', 1e-4), # Get from model or training
        binary_loss_weight=config['model'].get('binary_loss_weight', 1.0),
        multilabel_loss_weight=config['model'].get('multilabel_loss_weight', 1.0),
        use_class_weights=config['model'].get('use_class_weights', True),
        binary_class_weights=class_weights['binary'],
        multilabel_class_weights=class_weights['multilabel'],
        scheduler=config['model'].get('scheduler', 'cosine'),
        warmup_epochs=config['model'].get('warmup_epochs', 5),
        max_epochs=config['training']['max_epochs'] # For scheduler
    )
    
    # 4. Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['training']['checkpoint_dir'],
        filename='fragment-detector-{epoch:02d}-{val/binary_mcc:.3f}',
        monitor='val/binary_mcc',
        mode='max',
        save_top_k=3,
    )
    
    early_stopping = EarlyStopping(
        monitor='val/binary_mcc',
        patience=config['training'].get('early_stopping_patience', 15),
        mode='max',
    )
    callbacks = [checkpoint_callback, early_stopping, LearningRateMonitor(logging_interval='epoch')]
    
    # 5. Loggers
    tb_logger = TensorBoardLogger(save_dir=config['training']['log_dir'], name='fragment_detector')
    csv_logger = CSVLogger(save_dir=config['training']['log_dir'], name='fragment_detector')
    
    # 6. Trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator=config['training'].get('accelerator', 'auto'),
        devices=config['training'].get('devices', 1),
        precision=config['training'].get('precision', 32),
        callbacks=callbacks,
        logger=[tb_logger, csv_logger],
        gradient_clip_val=config['training'].get('gradient_clip_val', 1.0),
        accumulate_grad_batches=config['training'].get('accumulate_grad_batches', 1),
        deterministic=config.get('deterministic', True),
        log_every_n_steps=config['training'].get('log_every_n_steps', 10),
    )
    
    # 7. Train
    print("\n=== Starting Training ===")
    trainer.fit(model, datamodule=datamodule)
    
    # 8. Test
    print("\n=== Testing Best Model ===")
    trainer.test(datamodule=datamodule, ckpt_path='best')
    print(f"Best model path: {checkpoint_callback.best_model_path}")

# (main function from A2 to handle config overrides is good)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train protein fragment predictor')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument(
        '--override', type=str, nargs='*', 
        help='Override config params (e.g., model.dropout=0.5)'
    )
    args = parser.parse_args()
    config = load_config(args.config)
    
    if args.override:
        print("--- Overriding Configs ---")
        for override in args.override:
            keys, value = override.split('=')
            keys_list = keys.split('.')
            temp_cfg = config
            for k in keys_list[:-1]: temp_cfg = temp_cfg[k]
            try: value = eval(value)
            except: pass
            temp_cfg[keys_list[-1]] = value
            print(f"{keys}: {value}")
        print("--------------------------")
            
    train(config)