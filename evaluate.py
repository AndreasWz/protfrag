# evaluate.py
"""
Evaluation script for protein fragment predictor.
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

from data.datamodule import ProteinFragmentDataModule
from models.fragment_predictor import FragmentPredictor


def evaluate(config: dict, checkpoint_path: str, output_dir: str):
    """Evaluate model and generate detailed results."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize data module
    datamodule = ProteinFragmentDataModule(
        metadata_path=config['data']['metadata_path'],
        embeddings_dir=config['data']['embeddings_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data'].get('num_workers', 4),
        embedding_type=config['data'].get('embedding_type', 'mean_pooled')
    )
    
    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = FragmentPredictor.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # Test
    trainer = pl.Trainer(
        accelerator=config['training'].get('accelerator', 'auto'),
        devices=1,
        logger=False
    )
    
    print("\n=== Testing Model ===")
    test_results = trainer.test(model, datamodule=datamodule)
    
    # Get predictions
    print("\n=== Generating Predictions ===")
    datamodule.setup('test')
    predictions = trainer.predict(model, datamodule=datamodule)
    
    # Combine predictions
    all_entries = []
    all_binary_probs = []
    all_binary_preds = []
    all_multilabel_probs = []
    all_multilabel_preds = []
    
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
    
    # Load ground truth
    test_df = pd.read_csv(config['data']['metadata_path'])
    test_df = test_df[test_df['split'] == 'test'].reset_index(drop=True)
    
    # Create results dataframe
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
    
    # Save predictions
    results_df.to_csv(output_dir / 'predictions.csv', index=False)
    print(f"Saved predictions to {output_dir / 'predictions.csv'}")
    
    # Binary classification report
    print("\n=== Binary Classification Report ===")
    print(classification_report(
        results_df['true_is_fragment'],
        results_df['pred_is_fragment'],
        target_names=['Complete', 'Fragment']
    ))
    
    # Confusion matrix
    cm = confusion_matrix(results_df['true_is_fragment'], results_df['pred_is_fragment'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Complete', 'Fragment'],
                yticklabels=['Complete', 'Fragment'])
    plt.title('Binary Classification Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_dir / 'binary_confusion_matrix.png', dpi=300)
    print(f"Saved confusion matrix to {output_dir / 'binary_confusion_matrix.png'}")
    
    # Fragment type classification (only for true fragments)
    fragment_results = results_df[results_df['true_is_fragment'] == 1]
    
    if len(fragment_results) > 0:
        print("\n=== Fragment Type Classification Report ===")
        for fragment_type in ['n_terminal', 'c_terminal', 'internal']:
            print(f"\n{fragment_type.upper()}:")
            print(classification_report(
                fragment_results[f'true_{fragment_type}'],
                fragment_results[f'pred_{fragment_type}'],
                target_names=['Absent', 'Present'],
                zero_division=0
            ))
    
    # Probability distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Binary probability distribution
    axes[0, 0].hist(results_df[results_df['true_is_fragment'] == 0]['prob_is_fragment'],
                    bins=50, alpha=0.5, label='Complete', color='blue')
    axes[0, 0].hist(results_df[results_df['true_is_fragment'] == 1]['prob_is_fragment'],
                    bins=50, alpha=0.5, label='Fragment', color='red')
    axes[0, 0].set_xlabel('Predicted Probability (Fragment)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Binary Classification Probability Distribution')
    axes[0, 0].legend()
    axes[0, 0].axvline(x=0.5, color='black', linestyle='--', linewidth=1)
    
    # Fragment type probabilities (only for fragments)
    if len(fragment_results) > 0:
        for idx, frag_type in enumerate(['n_terminal', 'c_terminal', 'internal']):
            ax = axes.flat[idx + 1]
            true_col = f'true_{frag_type}'
            prob_col = f'prob_{frag_type}'
            
            ax.hist(fragment_results[fragment_results[true_col] == 0][prob_col],
                   bins=30, alpha=0.5, label='Absent', color='blue')
            ax.hist(fragment_results[fragment_results[true_col] == 1][prob_col],
                   bins=30, alpha=0.5, label='Present', color='red')
            ax.set_xlabel(f'Predicted Probability ({frag_type.replace("_", " ").title()})')
            ax.set_ylabel('Count')
            ax.set_title(f'{frag_type.replace("_", " ").title()} Probability Distribution')
            ax.legend()
            ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'probability_distributions.png', dpi=300)
    print(f"Saved probability distributions to {output_dir / 'probability_distributions.png'}")
    
    # Performance by sequence length
    results_df['length_bin'] = pd.cut(results_df['sequence_length'], bins=5)
    length_performance = results_df.groupby('length_bin').apply(
        lambda x: (x['pred_is_fragment'] == x['true_is_fragment']).mean()
    )
    
    plt.figure(figsize=(10, 6))
    length_performance.plot(kind='bar')
    plt.title('Binary Classification Accuracy by Sequence Length')
    plt.xlabel('Sequence Length Bin')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_by_length.png', dpi=300)
    print(f"Saved length analysis to {output_dir / 'accuracy_by_length.png'}")
    
    # Save test metrics
    with open(output_dir / 'test_metrics.txt', 'w') as f:
        f.write("=== Test Metrics ===\n\n")
        for key, value in test_results[0].items():
            f.write(f"{key}: {value:.4f}\n")
    
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