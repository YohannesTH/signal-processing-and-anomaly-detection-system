import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score
import os

def main():
    print("=== MODULE 4: Anomaly Detection Models ===")
    
    # 1. Load the pristine scaled features from Module 3
    data_path = '../../data/processed/processed_features.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run Module 3 first.")
        return
        
    df = pd.read_csv(data_path)
    features = ['mean', 'variance', 'energy', 'dominant_frequency', 'entropy']
    X = df[features].values
    y_true = df['label'].values 

    print(f"Loaded {len(df)} records. Training unsupervised anomaly detectors...")

    # --- A. Isolation Forest ---
    # Setup Isolation Forest. Typical normal distributions assume majority of data is normal.
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    # Fit & Predict: returns 1 for inliers, -1 for outliers
    preds_if_raw = iso_forest.fit_predict(X)
    # Convert to 0=normal, 1=anomaly to match our labels
    preds_if = np.where(preds_if_raw == -1, 1, 0)

    # --- B. One-Class SVM ---
    ocsvm = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
    preds_ocsvm_raw = ocsvm.fit_predict(X)
    preds_ocsvm = np.where(preds_ocsvm_raw == -1, 1, 0)

    # --- C. KMeans for Anomaly Detection ---
    # We cluster into 2 principal groups
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    preds_kmeans_raw = kmeans.fit_predict(X)
    
    # Identify which cluster represents anomalies based on volumetric expectation (fewer points = anomalies)
    cluster_0_count = np.sum(preds_kmeans_raw == 0)
    cluster_1_count = np.sum(preds_kmeans_raw == 1)
    if cluster_1_count < cluster_0_count:
        preds_kmeans = preds_kmeans_raw
    else:
        preds_kmeans = 1 - preds_kmeans_raw

    # 2. Compare Models rigorously
    print("\n--- Model Evaluation Results ---")
    def evaluate(name, y_true, y_pred):
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f = f1_score(y_true, y_pred, zero_division=0)
        print(f"{name+':':<20} Precision: {p:.4f} | Recall: {r:.4f} | F1-Score: {f:.4f}")
        return p, r, f

    evaluate("Isolation Forest", y_true, preds_if)
    evaluate("One-Class SVM", y_true, preds_ocsvm)
    evaluate("KMeans", y_true, preds_kmeans)

    # 3. Add prediction columns transparently to dataset
    df['pred_isolation_forest'] = preds_if
    df['pred_one_class_svm'] = preds_ocsvm
    df['pred_kmeans'] = preds_kmeans

    # 4. Export consolidated predictions
    out_dir = '../../data/outputs'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'predictions.csv')
    df.to_csv(out_path, index=False)
    print(f"\nModel predictions appended and successfully saved to:\n  -> {out_path}")

if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
