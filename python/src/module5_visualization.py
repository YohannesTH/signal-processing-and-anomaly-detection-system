import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import os

sns.set_theme(style="whitegrid")

def main():
    print("=== MODULE 5: Visualization & Insights ===")
    
    # Paths setup
    preds_path = '../../data/outputs/predictions.csv'
    raw_sig_path = '../../data/raw/signals.csv'
    
    if not os.path.exists(preds_path) or not os.path.exists(raw_sig_path):
        print("Missing required artifacts. Please execute prior modules sequentially.")
        return
        
    df_preds = pd.read_csv(preds_path)
    df_raw = pd.read_csv(raw_sig_path)
    
    out_dir = '../../data/outputs/'
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Plot complete time series highlighting strictly true vs false anomalous segments
    plt.figure(figsize=(16, 6))
    plt.plot(df_raw['time'], df_raw['value'], label='Signal Architecture', color='mediumblue', alpha=0.75, linewidth=1)
    
    true_anomalies = df_raw[df_raw['label'] == 1]
    plt.scatter(true_anomalies['time'], true_anomalies['value'], color='crimson', label='True Anomalies', s=15, zorder=5)
    
    plt.title('Complete Time Series Data Tracing Ground Truth Anomalies', fontsize=14, pad=15)
    plt.xlabel('Time (Seconds)', fontsize=12)
    plt.ylabel('Amplitude Deviation', fontsize=12)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'timeseries_anomalies.png'), dpi=300)
    print("- Saved time series anomalous plot")
    
    # 2. PCA Cluster Separation Layout
    features = ['mean', 'variance', 'energy', 'dominant_frequency', 'entropy']
    X = df_preds[features].values
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    df_preds['pca1'] = X_pca[:, 0]
    df_preds['pca2'] = X_pca[:, 1]
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    custom_palette = {0: 'steelblue', 1: 'darkred'}
    
    sns.scatterplot(data=df_preds, x='pca1', y='pca2', hue='label', ax=axes[0], palette=custom_palette, alpha=0.8)
    axes[0].set_title('Ground Truth (True Labels)', fontsize=12)
    
    sns.scatterplot(data=df_preds, x='pca1', y='pca2', hue='pred_isolation_forest', ax=axes[1], palette=custom_palette, alpha=0.8)
    axes[1].set_title('Isolation Forest Estimates', fontsize=12)
    
    sns.scatterplot(data=df_preds, x='pca1', y='pca2', hue='pred_one_class_svm', ax=axes[2], palette=custom_palette, alpha=0.8)
    axes[2].set_title('One-Class SVM Estimates', fontsize=12)
    
    sns.scatterplot(data=df_preds, x='pca1', y='pca2', hue='pred_kmeans', ax=axes[3], palette=custom_palette, alpha=0.8)
    axes[3].set_title('KMeans Extraction Strategy', fontsize=12)
    
    fig.suptitle('2D PCA Feature Space Projections Comparing Models vs Reality', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pca_clusters_comparison.png'), dpi=300)
    print("- Saved PCA multidimensional comparison plot")
    
    # 3. Create Analytical Summary Insights
    print("\n--- Summary Insights Extract ---")
    
    # Best Model Performance Indexing
    models = ['pred_isolation_forest', 'pred_one_class_svm', 'pred_kmeans']
    scores = {m: f1_score(df_preds['label'], df_preds[m], zero_division=0) for m in models}
    best_model = max(scores, key=scores.get)
    clean_name = best_model.replace('pred_', '').replace('_', ' ').title()
    
    print(f"1. Highest Utility Model:")
    print(f"   The **{clean_name}** achieved the strongest performance recognizing anomalies globally with an F1 score of {scores[best_model]:.4f}.\n")
    
    # Random Forest diagnostic for Feature Importance Matrix
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(df_preds[features], df_preds['label'])
    
    importances = list(zip(features, rf.feature_importances_))
    importances.sort(key=lambda x: x[1], reverse=True)
    
    print(f"2. Extracted Feature Importance Diagnostic:")
    for feat, imp in importances:
        print(f"   - {feat.replace('_', ' ').title()}: {imp*100:.2f}% contribution")
        
    print(f"\nConclusion: The `{importances[0][0]}` attribute was predominantly most useful, signifying severe {importances[0][0]} distortions mapped best to our anomaly signatures.\n")

if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
