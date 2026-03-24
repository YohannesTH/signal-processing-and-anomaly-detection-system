# Hybrid Signal Processing and Anomaly Detection System

A robust, modular system combining the mathematical processing power of MATLAB with the advanced machine learning and visualization capabilities of Python to detect anomalies in synthetic time-series signals.

## Architecture & Workflow

### 1. MATLAB: Signal Generation & Feature Extraction
- **Module 1**: `matlab/src/module1_signal_generation.m`
  - Generates synthetic time-series data streams (sine waves + Gaussian noise).
  - Programmatically injects discrete anomaly signatures: Sudden spikes, gradual drift, and frequency shifts.
- **Module 2**: `matlab/src/module2_feature_engineering.m`
  - Reads the raw signal data.
  - Applies rolling window feature extraction utilizing Fast Fourier Transforms (FFT) for frequency domain analysis.
  - Utilizes Wavelet Transforms (Daubechies `db4`) to calculate spectral entropy in the time-frequency domain.

### 2. Python: Machine Learning & Visualization
- **Module 3**: `python/notebooks/01_Data_Pipeline_and_EDA.ipynb`
  - Ingests the MATLAB-extracted feature matrices.
  - Standardizes the data distribution and performs rigorous Exploratory Data Analysis (EDA) via correlation heatmaps and pairplots.
- **Module 4**: `python/src/module4_anomaly_detection.py`
  - Trains unsupervised clustering and anomaly algorithms natively: Isolation Forest, One-Class SVM, and KMeans.
  - Automatically compares precision, recall, and F1-Scores against the injected ground truth labels.
- **Module 5**: `python/src/module5_visualization.py`
  - Performs PCA for multidimensional cluster separation.
  - Synthesizes an interactive time-series overlay plotting the exact locations of classified anomalies.
  - Extracts the most powerful diagnostic features utilizing Random Forest feature importance.

## Directory Structure
- `/matlab/src/` - Core `.m` scripts defining structural architecture.
- `/python/src/` - Core Python deployment scripts.
- `/python/notebooks/` - Analytical Jupyter pipeline.
- `/data/` - Holds raw signals, features, and model output artifacts.

## Execution Requirements
Run the system chronologically from Module 1 to Module 5. MATLAB outputs are seamlessly linked to the Python ingestion layer via the `/data/` directory CSV interoperability.