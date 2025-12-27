"""
Training pipeline for keystroke dynamics model.
"""

import json
import os
from typing import List, Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from features import extract_features


def load_sessions(data_path: str = 'data/sessions.json') -> List[Dict[str, Any]]:
    """Load sessions from JSON file."""
    if not os.path.exists(data_path):
        return []
    
    with open(data_path, 'r') as f:
        return json.load(f)


def check_minimum_samples(sessions: List[Dict[str, Any]], min_per_user: int = 10) -> tuple[bool, Dict[str, int]]:
    """
    Check if we have enough samples per user.
    
    Returns:
        (has_enough, user_counts)
    """
    user_counts = {}
    for session in sessions:
        user = session.get('user', '')
        user_counts[user] = user_counts.get(user, 0) + 1
    
    has_enough = all(count >= min_per_user for count in user_counts.values() if count > 0)
    return has_enough, user_counts


def prepare_features(sessions: List[Dict[str, Any]]) -> tuple[List[List[float]], List[str]]:
    """
    Extract features from all sessions.
    
    Returns:
        (features_list, labels_list)
    """
    features_list = []
    labels_list = []
    
    for session in sessions:
        events = session.get('events', [])
        features = extract_features(events)
        
        # Convert to list in fixed order
        feature_vector = [
            features['dwell_mean'],
            features['dwell_std'],
            features['dwell_median'],
            features['flight_mean'],
            features['flight_std'],
            features['flight_median']
        ]
        
        features_list.append(feature_vector)
        labels_list.append(session.get('user', ''))
    
    return features_list, labels_list


def train_model(data_path: str = 'data/sessions.json', 
                model_path: str = 'artifacts/model.joblib',
                min_per_user: int = 10,
                test_size: float = 0.2) -> Dict[str, Any]:
    """
    Train the keystroke dynamics model.
    
    Returns:
        Dictionary with training results (accuracy, user_counts, etc.)
    """
    # Load sessions
    sessions = load_sessions(data_path)
    
    if len(sessions) == 0:
        return {
            'success': False,
            'error': 'No sessions found. Please collect data first.'
        }
    
    # Check minimum samples
    has_enough, user_counts = check_minimum_samples(sessions, min_per_user)
    
    if not has_enough:
        return {
            'success': False,
            'error': f'Not enough samples. Need at least {min_per_user} per user.',
            'user_counts': user_counts
        }
    
    # Prepare features
    X, y = prepare_features(sessions)
    
    if len(set(y)) < 2:
        return {
            'success': False,
            'error': 'Need at least 2 different users to train a classifier.'
        }
    
    # Split data
    if len(sessions) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
    else:
        # Too few samples, use all for training
        X_train, X_test, y_train, y_test = X, X, y, y
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    
    # Save feature order (for consistency)
    feature_order = [
        'dwell_mean', 'dwell_std', 'dwell_median',
        'flight_mean', 'flight_std', 'flight_median'
    ]
    joblib.dump(feature_order, 'artifacts/feature_order.joblib')
    
    return {
        'success': True,
        'accuracy': accuracy,
        'user_counts': user_counts,
        'total_sessions': len(sessions),
        'train_size': len(X_train),
        'test_size': len(X_test)
    }


if __name__ == '__main__':
    result = train_model()
    if result['success']:
        print(f"Training successful!")
        print(f"Accuracy: {result['accuracy']:.2%}")
        print(f"User counts: {result['user_counts']}")
        print(f"Total sessions: {result['total_sessions']}")
    else:
        print(f"Training failed: {result['error']}")
        if 'user_counts' in result:
            print(f"Current user counts: {result['user_counts']}")

