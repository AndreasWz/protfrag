import argparse
import yaml
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, CSVLogger # <--- HIER GEÄNDERT
import torch

# Angepasste Imports für src/ layout
from src.data import FragmentDataModule
from src.model import FragmentDetector

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train(config: dict):
    pl.seed_everything(config.get('seed', 42), workers=True)
    
    # 1. DataModule initialisieren
    datamodule = FragmentDataModule(
        metadata_path=config['data']['metadata_path'],
        embedding_dir=config['data']['embeddings_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data'].get('num_workers', 4),
        embedding_type=config['data'].get('embedding_type', 'mean_pooled'),
        pin_memory=config['data'].get('pin_memory', True),
        emb_dim=config['model']['embedding_dim'] # Wichtig für Dataset-Fallback
    )
    
    # 2. Datamodule einrichten, um Klassen-Gewichte zu erhalten
    datamodule.setup('fit')
    class_weights = datamodule.get_class_weights()
    print("\n=== Class Weights ===")
    print(f"Binary: {class_weights['binary']}")
    print(f"Multilabel: {class_weights['multilabel']}")
    
    # 3. Modell initialisieren
    model = FragmentDetector(
        embedding_dim=config['model']['embedding_dim'],
        hidden_dims=config['model']['hidden_dims'],
        dropout=config['model']['dropout'],
        learning_rate=config['model']['learning_rate'],
        weight_decay=config['model'].get('weight_decay', 1e-4),
        binary_loss_weight=config['model'].get('binary_loss_weight', 1.0),
        multilabel_loss_weight=config['model'].get('multilabel_loss_weight', 1.0),
        use_class_weights=config['model'].get('use_class_weights', True),
        binary_class_weights=class_weights['binary'],
        multilabel_class_weights=class_weights['multilabel'],
        scheduler=config['model'].get('scheduler', 'cosine'),
        warmup_epochs=config['model'].get('warmup_epochs', 5)
    )
    
    # 4. Callbacks
    
    # Callback 1: Speichert die Top-3-Modelle basierend auf der ZIELMETRIK (MCC)
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['training']['checkpoint_dir'],
        filename='fragment-detector-BEST_MCC-{epoch:02d}-{val/binary_mcc:.3f}', # Geänderter Name
        monitor='val/binary_mcc', # Beobachtet die Performance-Metrik
        mode='max',
        save_top_k=3,
        save_last=True # Speichert zusätzlich den letzten Checkpoint (gut für Debugging)
    )
    
    # --- NEUER CALLBACK ---
    # Callback 2: Speichert die Top-3-Modelle basierend auf dem STABILSTEN Loss
    loss_checkpoint_callback = ModelCheckpoint(
        dirpath=config['training']['checkpoint_dir'],
        filename='fragment-detector-MIN_LOSS-{epoch:02d}-{val/loss_total:.3f}', # Anderer Dateiname
        monitor='val/loss_total',  # Beobachtet den Loss
        mode='min',                # Will den *minimalen* Loss
        save_top_k=3               # Speichert die Top 3
    )
    # --- ENDE NEUER CALLBACK ---

    # EarlyStopping stoppt das Training, wenn der Loss (Stabilitäts-Indikator) 
    # sich nicht mehr verbessert (oder steigt).
    early_stopping = EarlyStopping(
        monitor='val/loss_total', # <--- BEOBACHTET DEN LOSS
        patience=config['training'].get('early_stopping_patience', 15), # 15 Epochen Geduld sind hier gut
        mode='min', # Wir wollen den Loss minimieren
        verbose=True
    )
    
    # --- NEUE CALLBACK-LISTE ---
    callbacks = [checkpoint_callback, loss_checkpoint_callback, early_stopping, LearningRateMonitor(logging_interval='epoch')]
    
    # 5. Loggers
    # --- HIER GEÄNDERT: TensorBoard durch WandbLogger ersetzt ---
    wandb_logger = WandbLogger(
        project="protfrag-tum", # Der Name deines W&B-Projekts
        save_dir=config['training']['log_dir'],
        log_model="all" # Lädt deine besten Checkpoints automatisch hoch
    )
    csv_logger = CSVLogger(save_dir=config['training']['log_dir'], name='fragment_detector')
    
    # 6. Trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator=config['training'].get('accelerator', 'auto'),
        devices=config['training'].get('devices', 1),
        precision=config['training'].get('precision', 32),
        callbacks=callbacks,
        logger=[wandb_logger, csv_logger], # <--- HIER GEÄNDERT
        gradient_clip_val=config['training'].get('gradient_clip_val', 1.0),
        accumulate_grad_batches=config['training'].get('accumulate_grad_batches', 1),
        deterministic=config.get('deterministic', True),
        log_every_n_steps=config['training'].get('log_every_n_steps', 10),
    )
    
    # 7. Trainieren
    print("\n=== Starting Training ===")
    trainer.fit(model, datamodule=datamodule)
    
    # 8. Testen
    print("\n=== Testing Best Model ===")
    # Wichtig: 'ckpt_path="best"' lädt automatisch den Checkpoint, der vom
    # *ersten* ModelCheckpoint-Callback (also 'checkpoint_callback') 
    # als "bester" (höchster MCC) markiert wurde.
    trainer.test(datamodule=datamodule, ckpt_path='best')
    print(f"Best model path (based on MCC): {checkpoint_callback.best_model_path}")

# (main function zum Laden/Überschreiben der Config)
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