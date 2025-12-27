# Keystroke Dynamics - MVP

A simple keystroke dynamics project that collects typing timing data from multiple users, trains a machine learning model, and predicts which user is typing.

## Features

- **Collect Mode**: Capture keystroke timing data (keydown/keyup events) from users typing a fixed prompt
- **Train Mode**: Train a RandomForestClassifier on collected data to identify users by typing patterns
- **Predict Mode**: Predict which user is typing based on their keystroke dynamics

## Installation

1. Install Python 3.10 or higher

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python app.py
```

2. **Collect Data** (Collect tab):
   - Select a user label (A, B, or C)
   - Type the prompt text: "the quick brown fox jumps over the lazy dog"
   - Click "Save Session" to save the typing data
   - Collect at least 10 sessions per user for best results

3. **Train Model** (Train tab):
   - Click "Train Model" to train the classifier
   - The model will be saved to `artifacts/model.joblib`
   - Training results (accuracy, user counts) will be displayed

4. **Predict** (Predict tab):
   - Type the same prompt text
   - Click "Predict" to see which user the model thinks is typing
   - Results show predicted user, confidence, and probabilities for all users

## Project Structure

```
keystyleid/
  app.py              # Main tkinter GUI application
  features.py         # Feature extraction (dwell times, flight times)
  train.py            # Training pipeline
  predict.py          # Prediction pipeline
  data/
    sessions.json     # Collected typing sessions
  artifacts/
    model.joblib      # Trained model (created after training)
    feature_order.joblib  # Feature order schema
  requirements.txt    # Python dependencies
  README.md          # This file
```

## How It Works

1. **Data Collection**: Captures keydown and keyup timestamps in milliseconds as users type
2. **Feature Extraction**: Computes 6 features per session:
   - Dwell time statistics (mean, std, median): time between keydown and keyup
   - Flight time statistics (mean, std, median): time between consecutive keydowns
3. **Training**: Uses RandomForestClassifier to learn typing patterns for each user
4. **Prediction**: Extracts features from new typing and predicts the user

## Requirements

- Python 3.10+
- scikit-learn
- joblib
- tkinter (usually included with Python)

## Notes

- The application validates that typed text matches the prompt (minimum 80% similarity)
- Requires at least 10 sessions per user before training
- Requires at least 2 different users to train a classifier
- All data is stored locally in JSON format

