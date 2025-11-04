"""
Evaluation script for protein fragment predictor.
Generates plots, a metrics JSON, and a predictions.csv file.
---
Includes baseline model comparison (Logistic Regression, Decision Tree).
Includes detailed error analysis.
"""
import argparse
import yaml
import json
from pathlib import Path
import pytorch_lightning as pl
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, 
    precision_recall_curve, average_precision_score,
    matthews_corrcoef
)
# --- NEU: Imports für Baseline-Modelle ---
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# --- KORRIGIERTE IMPORTE ---
from src.data import FragmentDataModule
from src.model import FragmentDetector
# ---

# --- NEU: Helper-Funktion zum Laden von Daten für Baselines ---
def load_data_for_baselines(datamodule: FragmentDataModule, config: dict):
    """
    Loads training and test data (embeddings + lengths) for scikit-learn models.
    Handles missing embeddings by returning zero-tensors.
    """
    embedding_dir = Path(config['data']['embeddings_dir'])
    embedding_type = config['data'].get('embedding_type', 'mean_pooled')
    emb_dim = config['model']['embedding_dim']

    print("Loading data for baseline models...")
    
    # Sicherstellen, dass die DFs geladen sind
    datamodule.setup('fit')
    datamodule.setup('test')

    def _load_data_from_df(df):
        embeddings_list = []
        lengths_list = []
        labels_list = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading embeddings"):
            entry_id = row['entry']
            emb_path = embedding_dir / f"{entry_id}.pt"
            
            try:
                emb_dict = torch.load(emb_path, map_location='cpu')
                embedding = emb_dict[embedding_type].numpy()
            except (FileNotFoundError, KeyError):
                embedding = np.zeros(emb_dim) # Gleiches Verhalten wie DataLoader
            
            embeddings_list.append(embedding)
            lengths_list.append(row['sequence_length'])
            labels_list.append(row['is_fragment'])

        return (
            np.array(embeddings_list), 
            np.array(lengths_list).reshape(-1, 1), 
            np.array(labels_list)
        )

    print("Loading training data...")
    X_train_embed, X_train_len, y_train = _load_data_from_df(datamodule.train_df)
    
    print("Loading test data...")
    X_test_embed, X_test_len, y_test = _load_data_from_df(datamodule.test_df)

    # Skalieren der Längen-Daten
    scaler_len = StandardScaler()
    X_train_len_scaled = scaler_len.fit_transform(X_train_len)
    X_test_len_scaled = scaler_len.transform(X_test_len)
    
    # Skalieren der Embedding-Daten
    scaler_embed = StandardScaler()
    X_train_embed_scaled = scaler_embed.fit_transform(X_train_embed)
    X_test_embed_scaled = scaler_embed.transform(X_test_embed)

    # --- GEÄNDERTE RÜCKGABE (FIX) ---
    # Gibt skalierte Embeddings, skalierte Längen, unskalierte Längen und Labels zurück
    return (
        (X_train_embed_scaled, X_train_len_scaled, X_train_len, y_train),
        (X_test_embed_scaled, X_test_len_scaled, X_test_len, y_test),
    )

# --- NEU: Helper-Funktion zum Trainieren und Auswerten der Baselines ---
def run_baseline_comparison(
    baseline_data: tuple, 
    dl_results_df: pd.DataFrame, 
    output_dir: Path
):
    # --- GEÄNDERTES UNPACKING (FIX) ---
    (X_train_embed, X_train_len_scaled, X_train_len, y_train), \
    (X_test_embed, X_test_len_scaled, X_test_len, y_test) = baseline_data
    
    print("\n=== Training und Evaluierung der Baseline-Modelle ===")
    
    models = {}
    
    # 1. Decision Tree (Längen-Cutoff) - Verwendet unskalierte Längen
    print("Training Decision Tree (Length Cutoff)...")
    dt_model = DecisionTreeClassifier(max_depth=1, random_state=42)
    dt_model.fit(X_train_len, y_train)
    models['DT (Length Cutoff)'] = dt_model.predict(X_test_len)
    
    # 2. Logistic Regression (Länge) - Verwendet skalierte Längen (FIX)
    print("Training Logistic Regression (Length)...")
    lr_len_model = LogisticRegression(random_state=42, class_weight='balanced')
    lr_len_model.fit(X_train_len_scaled, y_train)
    models['LR (Length)'] = lr_len_model.predict(X_test_len_scaled)
    
    # 3. Logistic Regression (Embeddings) - Verwendet skalierte Embeddings
    print("Training Logistic Regression (Embeddings)...")
    lr_embed_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    lr_embed_model.fit(X_train_embed, y_train)
    models['LR (Embeddings)'] = lr_embed_model.predict(X_test_embed)
    
    # 4. Deep Learning Modell (zum Vergleich)
    models['Deep Model (FragmentDetector)'] = dl_results_df['pred_is_fragment'].values
    
    # --- Ergebnisse berechnen und ausgeben ---
    print("\n--- Baseline-Vergleich (Test Set) ---")
    
    results_summary = {}
    
    print(f"{'Modell':<30} | {'MCC':<10} | {'F1 (Fragment)':<15}")
    print("-" * 60)
    
    for name, y_pred in models.items():
        mcc = matthews_corrcoef(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        f1_fragment = report.get('1', {}).get('f1-score', 0.0) # '1' ist die "Fragment"-Klasse
        
        results_summary[name] = {
            'mcc': mcc,
            'f1_fragment': f1_fragment,
            'precision_fragment': report.get('1', {}).get('precision', 0.0),
            'recall_fragment': report.get('1', {}).get('recall', 0.0)
        }
        
        print(f"{name:<30} | {mcc:<10.3f} | {f1_fragment:<15.3f}")
        
    # Baseline-Metriken speichern
    baseline_metrics_path = output_dir / 'baseline_metrics.json'
    with open(baseline_metrics_path, 'w') as f:
        json.dump(results_summary, f, indent=4)
    print(f"\nBaseline-Metriken gespeichert in {baseline_metrics_path}")
    
    # --- NEUER PLOT: Baseline-Vergleich ---
    print("Erstelle Baseline-Vergleichsplot...")
    results_df_plot = pd.DataFrame.from_dict(results_summary, orient='index').reset_index().rename(columns={'index': 'Model'})
    
    # Daten für das Plotten "schmelzen" (von wide zu long format)
    results_df_melted = results_df_plot.melt(
        id_vars='Model', 
        value_vars=['mcc', 'f1_fragment'], 
        var_name='Metric', 
        value_name='Score'
    )
    
    plt.figure(figsize=(12, 7))
    barplot = sns.barplot(
        data=results_df_melted, 
        x='Model', 
        y='Score', 
        hue='Metric',
        palette={'mcc': 'royalblue', 'f1_fragment': 'mediumseagreen'}
    )
    plt.title('Baseline-Modell-Vergleich (Test Set)', fontsize=16)
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Modell', fontsize=12)
    plt.xticks(rotation=15, ha='right')
    
    # Labels zu den Balken hinzufügen
    for p in barplot.patches:
        barplot.annotate(
            format(p.get_height(), '.3f'), 
            (p.get_x() + p.get_width() / 2., p.get_height()), 
            ha = 'center', 
            va = 'center', 
            xytext = (0, 9), 
            textcoords = 'offset points'
        )
        
    plt.legend(title='Metrik')
    plt.tight_layout()
    plt.savefig(output_dir / 'baseline_comparison.png', dpi=300)
    print(f"Baseline-Vergleichsplot gespeichert in {output_dir / 'baseline_comparison.png'}")
    
    return results_summary

# --- NEU: Helper-Funktion für qualitative Fehleranalyse ---
def perform_error_analysis(results_df: pd.DataFrame, output_dir: Path):
    """
    Findet die schlimmsten False Positives und False Negatives und speichert sie.
    """
    print("\n=== Detaillierte Fehleranalyse (Deep Learning Model) ===")
    
    # 1. False Positives (FP) finden
    # Wahrheit = Complete (0), Vorhersage = Fragment (1)
    # Sortiert nach 'prob_is_fragment' (absteigend) -> Die, bei denen das Modell am sichersten FALSCH lag.
    fp_df = results_df[
        (results_df['true_is_fragment'] == 0) & (results_df['pred_is_fragment'] == 1)
    ].sort_values(by='prob_is_fragment', ascending=False)
    
    # 2. False Negatives (FN) finden
    # Wahrheit = Fragment (1), Vorhersage = Complete (0)
    # Sortiert nach 'prob_is_fragment' (aufsteigend) -> Die, bei denen das Modell am sichersten FALSCH lag.
    fn_df = results_df[
        (results_df['true_is_fragment'] == 1) & (results_df['pred_is_fragment'] == 0)
    ].sort_values(by='prob_is_fragment', ascending=True)

    # 3. Ergebnisse in Datei speichern
    error_file_path = output_dir / 'error_analysis.txt'
    with open(error_file_path, 'w') as f:
        f.write("=== Detaillierte Fehleranalyse (Deep Learning Model) ===\n\n")
        
        f.write(f"--- Top 10 'Schlimmste' False Positives (Komplett -> Fragment) ---\n")
        f.write("Modell war sich am sichersten, dass dies FRAGMENTE sind, aber sie waren KOMPLETT.\n\n")
        f.write(fp_df.head(10).to_string(columns=['entry', 'prob_is_fragment', 'sequence_length']))
        f.write("\n\n" + "="*30 + "\n\n")
        
        f.write(f"--- Top 10 'Schlimmste' False Negatives (Fragment -> Komplett) ---\n")
        f.write("Modell war sich am sichersten, dass dies KOMPLETT ist, aber sie waren FRAGMENTE.\n\n")
        f.write(fn_df.head(10).to_string(columns=['entry', 'prob_is_fragment', 'sequence_length', 'true_n_terminal', 'true_c_terminal', 'true_internal']))
        f.write("\n")

    print(f"Fehleranalyse gespeichert in {error_file_path}")
    
    # Auch in der Konsole ausgeben
    print(f"--- Top 5 'Schlimmste' False Positives (Komplett -> Fragment) ---")
    print(fp_df.head(5)[['entry', 'prob_is_fragment', 'sequence_length']])
    print("\n")
    print(f"--- Top 5 'Schlimmste' False Negatives (Fragment -> Komplett) ---")
    print(fn_df.head(5)[['entry', 'prob_is_fragment', 'sequence_length', 'true_n_terminal', 'true_c_terminal', 'true_internal']])


# --- Haupt-Evaluierungsfunktion (leicht angepasst) ---
def evaluate(config: dict, checkpoint_path: str, output_dir: str):
    """Evaluate model and generate detailed results."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. DataModule initialisieren
    datamodule = FragmentDataModule(
        metadata_path=config['data']['metadata_path'],
        embedding_dir=config['data']['embeddings_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data'].get('num_workers', 4),
        embedding_type=config['data'].get('embedding_type', 'mean_pooled')
    )
    
    # 2. Modell laden
    print(f"Loading model from {checkpoint_path}...")
    model = FragmentDetector.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # 3. Trainer einrichten
    trainer = pl.Trainer(
        accelerator=config['training'].get('accelerator', 'auto'),
        devices=1,
        logger=False
    )
    
    # 4. Test-Loop für DL-Modell-Metriken ausführen
    print("\n=== Running Test Step (Deep Learning Model) ===")
    test_results = trainer.test(model, datamodule=datamodule, verbose=False)
    
    metrics_path = output_dir / 'test_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(test_results[0], f, indent=4)
    print(f"Saved test metrics to {metrics_path}")
    print(json.dumps(test_results[0], indent=4))

    
    # 5. Roh-Vorhersagen für Plots erhalten
    print("\n=== Generating Raw Predictions (Deep Learning Model) ===")
    datamodule.setup('test') # Sicherstellen, dass test_df geladen ist
    predictions = trainer.predict(model, datamodule=datamodule)
    
    # 6. Batches kombinieren
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
    
    # 7. Ground Truth laden
    print("Loading ground truth...")
    test_df = pd.read_csv(config['data']['metadata_path'])
    test_df = test_df[test_df['split'] == 'test'].set_index('entry')
    
    try:
        test_df = test_df.loc[all_entries]
    except KeyError:
        print("Warning: Prediction entries do not perfectly match test set. Using reindex.")
        test_df = test_df.reindex(all_entries).dropna(subset=['is_fragment'])
        all_entries = test_df.index.tolist()
        test_df = test_df.loc[all_entries]


    
    # 8. Results-DataFrame (vom DL-Modell) erstellen
    results_df = pd.DataFrame({
        'entry': all_entries,
        'true_is_fragment': test_df['is_fragment'].values,
        'pred_is_fragment': all_binary_preds, # DL-Modell-Vorhersage
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
    
    # 9. Predictions CSV speichern
    results_df.to_csv(output_dir / 'predictions.csv', index=False)
    print(f"Saved predictions to {output_dir / 'predictions.csv'}")
    
    # 10. Classification Report (DL-Modell)
    print("\n=== Binary Classification Report (Deep Learning Model) ===")
    print(classification_report(
        results_df['true_is_fragment'],
        results_df['pred_is_fragment'],
        target_names=['Complete', 'Fragment']
    ))
    
    # --- AB HIER: Alle Plots für das DL-MODELL ---
    
    # 11. Confusion Matrix
    cm = confusion_matrix(results_df['true_is_fragment'], results_df['pred_is_fragment'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Complete', 'Fragment'],
                yticklabels=['Complete', 'Fragment'])
    plt.title('Deep Model Classification Confusion Matrix (Test Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_dir / 'binary_confusion_matrix.png', dpi=300)
    print(f"Saved confusion matrix to {output_dir / 'binary_confusion_matrix.png'}")

    # 12. ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(results_df['true_is_fragment'], results_df['prob_is_fragment'])
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Deep Model Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / 'binary_roc_curve.png', dpi=300)
    print(f"Saved ROC curve to {output_dir / 'binary_roc_curve.png'}")
    
    # 13. Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(results_df['true_is_fragment'], results_df['prob_is_fragment'])
    pr_auc = average_precision_score(results_df['true_is_fragment'], results_df['prob_is_fragment'])
    
    plt.plot(recall, precision, color='darkgreen', lw=2, label=f'PR curve (AP = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Deep Model Binary Classification Precision-Recall (PR) Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_dir / 'binary_pr_curve.png', dpi=300)
    print(f"Saved PR curve to {output_dir / 'binary_pr_curve.png'}")

    # 14. Fragment-Typ-Report
    fragment_results = results_df[results_df['true_is_fragment'] == 1]
    
    if len(fragment_results) > 0:
        print("\n=== Fragment Type Classification Report (Deep Learning Model) ===")
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
    
    # 15. Wahrscheinlichkeitsverteilungen
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    sns.histplot(data=results_df, x='prob_is_fragment', hue='true_is_fragment',
                 bins=50, alpha=0.6, ax=axes[0, 0], multiple="stack",
                 palette={0: 'blue', 1: 'red'})
    axes[0, 0].set_title('Deep Model Binary Classification Probability')
    axes[0, 0].axvline(x=0.5, color='black', linestyle='--', linewidth=1)
    
    if len(fragment_results) > 0:
        for idx, frag_type in enumerate(['n_terminal', 'c_terminal', 'internal']):
            ax = axes.flat[idx + 1]
            true_col = f'true_{frag_type}'
            prob_col = f'prob_{fragment_type}'
            
            sns.histplot(data=fragment_results, x=prob_col, hue=true_col,
                         bins=30, alpha=0.6, ax=ax, multiple="stack",
                         palette={0: 'blue', 1: 'red'})
            ax.set_title(f'Deep Model {frag_type.replace("_", " ").title()} Probability')
            ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'probability_distributions.png', dpi=300)
    print(f"Saved probability distributions to {output_dir / 'probability_distributions.png'}")
    
    # 16. MCC nach Sequenzlänge
    plt.figure(figsize=(10, 6))
    results_df['length_bin'] = pd.qcut(results_df['sequence_length'], q=5, duplicates='drop')
    
    mcc_scores = []
    bin_labels = []
    for bin_val, group in results_df.groupby('length_bin', observed=True):
        # Sicherstellen, dass beide Klassen im Bin vorhanden sind, sonst ist MCC 0
        if len(group['true_is_fragment'].unique()) < 2:
            mcc = 0.0
        else:
            mcc = matthews_corrcoef(group['true_is_fragment'], group['pred_is_fragment'])
        
        mcc_scores.append(mcc)
        bin_labels.append(str(bin_val))

    sns.barplot(x=bin_labels, y=mcc_scores, hue=bin_labels, palette="viridis", legend=False)
    plt.title('Deep Model Binary MCC by Sequence Length Bin (Test Set)')
    plt.xlabel('Sequence Length Bin')
    plt.ylabel('Matthews Correlation Coefficient (MCC)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'mcc_by_length.png', dpi=300)
    print(f"Saved MCC by length plot to {output_dir / 'mcc_by_length.png'}")
    
    # --- NEU: Detaillierte Fehleranalyse ---
    try:
        perform_error_analysis(results_df, output_dir)
    except Exception as e:
        print(f"\n--- Konnte Fehleranalyse nicht durchführen ---")
        print(f"Fehler: {e}")

    # --- Baseline-Modelle ausführen ---
    try:
        baseline_data = load_data_for_baselines(datamodule, config)
        run_baseline_comparison(baseline_data, results_df, output_dir)
    except Exception as e:
        print(f"\n--- Konnte Baseline-Modelle nicht ausführen ---")
        print(f"Fehler: {e}")
        print("Stellen Sie sicher, dass genügend Speicher vorhanden ist, um alle Embeddings zu laden.")

    
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
    
    # Config laden
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Evaluieren
    evaluate(config, args.checkpoint, args.output_dir)


if __name__ == '__main__':
    main()

