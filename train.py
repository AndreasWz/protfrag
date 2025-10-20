import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from data.datamodule import ProtFragDataModule
from models.fragment_detector import FragmentDetector
import numpy as np

def infer_emb_dim(emb_dir, sample_id=None):
    import os
    import numpy as np
    if sample_id:
        arr = np.load(os.path.join(emb_dir, f"{sample_id}.npy"))
    else:
        files = [f for f in os.listdir(emb_dir) if f.endswith('.npy')]
        if not files:
            raise RuntimeError('No .npy found in emb_dir')
        arr = np.load(os.path.join(emb_dir, files[0]))
    return arr.shape[1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-csv", required=True)
    p.add_argument("--emb-dir", required=True)
    p.add_argument("--batch-size", default=64, type=int)
    p.add_argument("--max-epochs", default=20, type=int)
    p.add_argument("--gpus", default=1, type=int)
    p.add_argument("--pool", default="mean")
    p.add_argument("--hidden-dim", default=512, type=int)
    p.add_argument("--lr", default=1e-4, type=float)
    args = p.parse_args()

    emb_dim = infer_emb_dim(args.emb_dir)
    if args.pool == 'mean+max':
        emb_dim = emb_dim * 2

    dm = ProtFragDataModule(csv_path=args.data_csv, emb_dir=args.emb_dir, batch_size=args.batch_size, pool=args.pool)
    dm.setup()

    model = FragmentDetector(emb_dim=emb_dim, hidden_dim=args.hidden_dim, lr=args.lr)

    ckpt_cb = ModelCheckpoint(monitor='val/loss', mode='min')
    es = EarlyStopping(monitor='val/loss', patience=5, mode='min')

    if args.gpus and args.gpus > 0:
        accelerator = 'gpu'
        devices = args.gpus
    else:
        accelerator = 'cpu'
        devices = 1

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[ckpt_cb, es],
        accelerator=accelerator,
        devices=devices
    )
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    main()