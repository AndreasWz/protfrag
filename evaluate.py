"""
Evaluation script for protein fragment predictor.
Generates plots and a predictions.csv file.
"""
import argparse
import yaml
from pathlib import Path
import pytorch_lightning as pl
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# --- CORRECTED IMPORTS ---
from src.data import FragmentDataModule
from src.model import FragmentDetector
# ---

def evaluate(config: dict, checkpoint_path: str, output_dir: str):
    """Evaluate model and generate detailed results."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Initialize data module
    datamodule = FragmentDataModule(
        metadata_path=config['data']['metadata_path'],
        embedding_dir=config['data']['embeddings_dir'], # <-- Corrected: was embeddings_dir
        batch_size=config['data']['batch_size'],
        num_workers=config['data'].get('num_workers', 4),
        embedding_type=config['data'].get('embedding_type', 'mean_pooled')
    )
    
    # 2. Load model from checkpoint
    print(f"Loading model from {checkpoint_path}...")
    model = FragmentDetector.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # 3. Setup trainer
    trainer = pl.Trainer(
        accelerator=config['training'].get('accelerator', 'auto'),
        devices=1,
        logger=False
    )
    
    # 4. Get predictions
    # We use trainer.predict() to get the raw outputs
    print("\n=== Generating Predictions ===")
    datamodule.setup('test')
    predictions = trainer.predict(model, datamodule=datamodule)
    
    # 5. Combine prediction batches
    all_entries = []
    all_binary_probs = []
    all_binary_preds = []
    all_multilabel_probs = []
    all_multilabel_preds = []
    
    print("Processing batches...")
    for batch_pred in predictions:
        all_entries.extend(batch_pred['entry'])
        all_binary_probs.append(batch_pred['binary_probs'])
        all_binary_preds.append(batch_pred['binary_preds'])
        all_multilabel_probs.append(batch_pred['multilabel_probs'])
        all_multilabel_preds.append(batch_pred['multilabel_preds'])
    
    all_binary_probs = torch.cat(all_binary_probs).cpu().numpy()
    all_binary_preds = torch.cat(all_binary_preds).cpu().numpy()
    all_multilabel_probs = torch.cat(all_multilabel_probs).cpu().numpy()
    all_multilabel_preds = torch.cat(all_multilabel_preds).cpu().numpy()
    
    # 6. Load ground truth
    print("Loading ground truth...")
    test_df = pd.read_csv(config['data']['metadata_path'])
    test_df = test_df[test_df['split'] == 'test'].set_index('entry')
    
    # Align ground truth with predictions
    # This can fail if predictions are not a complete set, use reindex
    try:
        test_df = test_df.loc[all_entries]
    except KeyError:
        print("Warning: Prediction entries do not perfectly match test set. Using reindex.")
        test_df = test_df.reindex(all_entries)

    
    # 7. Create results dataframe
    results_df = pd.DataFrame({
        'entry': all_entries,
        'true_is_fragment': test_df['is_fragment'].values,
        'pred_is_fragment': all_binary_preds,
        'prob_is_fragment': all_binary_probs,
        'true_n_terminal': test_df['n_terminal'].values,
        'pred_n_terminal': all_multilabel_preds[:, 0],
        'prob_n_terminal': all_multilabel_probs[:, 0],
        'true_c_terminal': test_df['c_terminal'].values,
        'pred_c_terminal': all_multilabel_preds[:, 1],
        'prob_c_terminal': all_multilabel_probs[:, 1],
        'true_internal': test_df['internal'].values,
        'pred_internal': all_multilabel_preds[:, 2],
        'prob_internal': all_multilabel_probs[:, 2],
        'sequence_length': test_df['sequence_length'].values
    })
    
    # 8. Save predictions CSV
    results_df.to_csv(output_dir / 'predictions.csv', index=False)
    print(f"Saved predictions to {output_dir / 'predictions.csv'}")
    
    # 9. Binary classification report
    print("\n=== Binary Classification Report (Test Set) ===")
    print(classification_report(
        results_df['true_is_fragment'],
        results_df['pred_is_fragment'],
        target_names=['Complete', 'Fragment']
    ))
    
    # 10. Confusion matrix
    cm = confusion_matrix(results_df['true_is_fragment'], results_df['pred_is_fragment'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Complete', 'Fragment'],
                yticklabels=['Complete', 'Fragment'])
    plt.title('Binary Classification Confusion Matrix (Test Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_dir / 'binary_confusion_matrix.png', dpi=300)
    print(f"Saved confusion matrix to {output_dir / 'binary_confusion_matrix.png'}")
    
    # 11. Fragment type classification (only for true fragments)
    fragment_results = results_df[results_df['true_is_fragment'] == 1]
    
    if len(fragment_results) > 0:
        print("\n=== Fragment Type Classification Report (Test Set, Fragments Only) ===")
        report_str = ""
        for fragment_type in ['n_terminal', 'c_terminal', 'internal']:
            report = classification_report(
                fragment_results[f'true_{fragment_type}'],
                fragment_results[f'pred_{fragment_type}'],
                target_names=['Absent', 'Present'],
                zero_division=0,
                output_dict=True
            )
            report_str += f"\n--- {fragment_type.upper()} ---\n"
            report_str += f"  Precision (Present): {report['Present']['precision']:.3f}\n"
            report_str += f"  Recall (Present):    {report['Present']['recall']:.3f}\n"
            report_str += f"  F1-Score (Present):  {report['Present']['f1-score']:.3f}\n"
        print(report_str)
    
    # 12. Probability distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Binary probability distribution
    sns.histplot(data=results_df, x='prob_is_fragment', hue='true_is_fragment',
                 bins=50, alpha=0.6, ax=axes[0, 0], multiple="stack",
                 palette={0: 'blue', 1: 'red'})
    axes[0, 0].set_title('Binary Classification Probability Distribution')
    axes[0, 0].axvline(x=0.5, color='black', linestyle='--', linewidth=1)
    
    # Fragment type probabilities (only for fragments)
    if len(fragment_results) > 0:
        for idx, frag_type in enumerate(['n_terminal', 'c_terminal', 'internal']):
            ax = axes.flat[idx + 1]
            true_col = f'true_{frag_type}'
            prob_col = f'prob_{frag_type}'
            
            sns.histplot(data=fragment_results, x=prob_col, hue=true_col,
                         bins=30, alpha=0.6, ax=ax, multiple="stack",
                         palette={0: 'blue', 1: 'red'})
            ax.set_title(f'{frag_type.replace("_", " ").title()} Probability Distribution')
            ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'probability_distributions.png', dpi=300)
    print(f"Saved probability distributions to {output_dir / 'probability_distributions.png'}")
    
    print(f"\n=== Evaluation Complete ===")
    print(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate fragment predictor')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/evaluation',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Evaluate
    evaluate(config, args.checkpoint, args.output_dir)


if __name__ == '__main__':
    main()

