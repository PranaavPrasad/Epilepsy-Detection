# Quick Setup Guide

## Prerequisites
- Python 3.12 installed
- Kaggle account with API credentials
- ~10 GB free disk space

## Installation Steps

### 1. Kaggle API Setup
1. Go to https://www.kaggle.com/settings
2. Scroll to "API" section
3. Click "Create New Token" → downloads `kaggle.json`
4. Place file in: `C:\Users\<YourUsername>\.kaggle\kaggle.json`

### 2. Environment Setup
```powershell
# Already done - virtual environment created
# Packages installed from requirements.txt
```

## Running the Project

### Phase 1: Exploratory Data Analysis
**Notebook**: `notebooks/01_exploratory_data_analysis.ipynb`

**What it does**:
- Downloads CHB-MIT dataset (~40 GB) OR analyzes existing data
- Explores dataset structure
- Analyzes subject distributions
- Selects 6 subjects for training

**Note**: Dataset download takes time. You can skip this and manually download from Kaggle.

### Phase 2: Preprocessing
**Notebook**: `notebooks/02_preprocessing.ipynb`

**What it does**:
- Filters EEG signals (0.5-50 Hz)
- Normalizes data
- Creates 4-second windows with 2-second overlap
- Labels seizure/normal periods
- Balances dataset
- Saves processed data to `data/processed/preprocessed_data.h5`

**Runtime**: ~30-60 minutes depending on data size

### Phase 3: Model Training
**Notebook**: `notebooks/03_model_training.ipynb`

**What it does**:
- Loads preprocessed data
- Trains 3 models:
  - CNN
  - Bi-LSTM
  - Hybrid CNN+Bi-LSTM
- Evaluates and compares models
- Saves best models to `models/` directory
- Generates performance visualizations

**Runtime**: ~20-40 minutes (with GPU) or 2-3 hours (CPU only)

## Expected Output

After completing all phases:

```
Epilepsy-Detection/
├── data/
│   ├── processed/
│   │   ├── preprocessed_data.h5      # Processed EEG data
│   │   ├── subject_summary.csv        # Subject metadata
│   │   ├── model_comparison.csv       # Results table
│   │   └── *.png                      # Visualizations
├── models/
│   ├── cnn_model.keras               # Trained CNN
│   ├── bilstm_model.keras            # Trained Bi-LSTM
│   └── hybrid_model.keras            # Trained Hybrid (best)
```

## Troubleshooting

### Issue: Dataset too large
**Solution**: The notebooks are configured to use only 5-6 subjects (~5-7 GB)

### Issue: Out of memory during training
**Solution**: Reduce `batch_size` in training notebooks (try 16 or 8)

### Issue: Slow training
**Solution**: 
- Ensure GPU is available (check with: `import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))`)
- Reduce number of epochs
- Process fewer files per subject in preprocessing

### Issue: Kaggle download fails
**Solution**: 
- Manually download from [Kaggle](https://www.kaggle.com/datasets/abhishekinnvonix/seizure-epilepcy-chb-mit-eeg-dataset-pediatric)
- Extract to `data/raw/` directory

## Progress Tracking

All notebooks include `tqdm` progress bars to show:
- Data processing progress
- Training epochs
- Evaluation steps

## Next Steps

1. Run all notebooks in sequence
2. Review results in `data/processed/model_comparison.csv`
3. Check visualizations
4. Use best model for predictions

## Support

Check `REPORT.md` for detailed technical information.
