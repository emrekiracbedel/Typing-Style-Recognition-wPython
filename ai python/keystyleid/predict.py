"""
Prediction pipeline for keystroke dynamics.
"""

import os
import joblib
from typing import List, Dict, Any
from features import extract_features


def predict_user(events: List[Dict[str, Any]], 
                 model_path: str = 'artifacts/model.joblib') -> Dict[str, Any]:
    """
    Predict which user typed based on keystroke events.
    
    Args:
        events: List of events with 'key', 'type' ('down'/'up'), and 't' (timestamp)
        model_path: Path to saved model
    
    Returns:
        Dictionary with prediction results
    """
    if not os.path.exists(model_path):
        return {
            'success': False,
            'error': 'Model not found. Please train the model first.'
        }
    
    # Load model
    model = joblib.load(model_path)
    
    # Extract features
    features = extract_features(events)
    
    # Convert to feature vector in same order as training
    feature_vector = [
        features['dwell_mean'],
        features['dwell_std'],
        features['dwell_median'],
        features['flight_mean'],
        features['flight_std'],
        features['flight_median']
    ]
    
    # Predict
    predicted_user = model.predict([feature_vector])[0]
    probabilities = model.predict_proba([feature_vector])[0]
    
    # Get confidence (max probability)
    confidence = max(probabilities)
    
    # Map probabilities to class names
    class_names = model.classes_
    prob_dict = {class_names[i]: float(prob) for i, prob in enumerate(probabilities)}
    
    return {
        'success': True,
        'predicted_user': predicted_user,
        'confidence': confidence,
        'probabilities': prob_dict
    }

