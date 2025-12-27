"""
Feature extraction for keystroke dynamics.
Extracts dwell times and flight times from key events.
"""

import json
from typing import List, Dict, Any
from collections import defaultdict


def extract_features(events: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Extract features from key events.
    
    Args:
        events: List of events with 'key', 'type' ('down'/'up'), and 't' (timestamp)
    
    Returns:
        Dictionary with 6 features: dwell_mean, dwell_std, dwell_median,
        flight_mean, flight_std, flight_median
    """
    if len(events) < 2:
        # Not enough events, return zeros
        return {
            'dwell_mean': 0.0,
            'dwell_std': 0.0,
            'dwell_median': 0.0,
            'flight_mean': 0.0,
            'flight_std': 0.0,
            'flight_median': 0.0
        }
    
    # Separate keydown and keyup events
    keydowns = {}
    keyups = {}
    
    for event in events:
        key = event['key']
        event_type = event['type']
        timestamp = event['t']
        
        if event_type == 'down':
            if key not in keydowns:
                keydowns[key] = []
            keydowns[key].append(timestamp)
        elif event_type == 'up':
            if key not in keyups:
                keyups[key] = []
            keyups[key].append(timestamp)
    
    # Calculate dwell times (keyup - keydown for each key press)
    # Match keydown/keyup pairs in chronological order
    dwell_times = []
    
    # Create a list of all events with their type
    all_events = []
    for event in events:
        all_events.append((event['t'], event['type'], event['key']))
    
    all_events.sort(key=lambda x: x[0])  # Sort by timestamp
    
    # Match keydown with next keyup of same key
    pending_downs = {}  # key -> list of timestamps
    for timestamp, event_type, key in all_events:
        if event_type == 'down':
            if key not in pending_downs:
                pending_downs[key] = []
            pending_downs[key].append(timestamp)
        elif event_type == 'up':
            if key in pending_downs and len(pending_downs[key]) > 0:
                down_time = pending_downs[key].pop(0)
                dwell_time = timestamp - down_time
                if dwell_time > 0:  # Sanity check
                    dwell_times.append(dwell_time)
    
    # Calculate flight times (time between consecutive keydowns)
    # Get all keydown events in chronological order
    all_downs = []
    for key in keydowns:
        for timestamp in keydowns[key]:
            all_downs.append((timestamp, key))
    
    all_downs.sort(key=lambda x: x[0])
    
    flight_times = []
    for i in range(len(all_downs) - 1):
        flight_time = all_downs[i + 1][0] - all_downs[i][0]
        if flight_time > 0:  # Sanity check
            flight_times.append(flight_time)
    
    # Calculate statistics
    def calc_stats(values):
        if not values:
            return 0.0, 0.0, 0.0
        import statistics
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        median = statistics.median(values)
        return mean, std, median
    
    dwell_mean, dwell_std, dwell_median = calc_stats(dwell_times)
    flight_mean, flight_std, flight_median = calc_stats(flight_times)
    
    return {
        'dwell_mean': dwell_mean,
        'dwell_std': dwell_std,
        'dwell_median': dwell_median,
        'flight_mean': flight_mean,
        'flight_std': flight_std,
        'flight_median': flight_median
    }


def validate_typed_text(typed_text: str, prompt: str, min_similarity: float = 0.8) -> bool:
    """
    Validate that typed text is similar enough to the prompt.
    
    Args:
        typed_text: The text the user typed
        prompt: The expected prompt text
        min_similarity: Minimum similarity ratio (0-1)
    
    Returns:
        True if similar enough, False otherwise
    """
    typed_clean = typed_text.strip().lower()
    prompt_clean = prompt.strip().lower()
    
    if typed_clean == prompt_clean:
        return True
    
    # Simple character-based similarity
    if len(prompt_clean) == 0:
        return False
    
    # Count matching characters in order
    matches = 0
    prompt_idx = 0
    for char in typed_clean:
        if prompt_idx < len(prompt_clean) and char == prompt_clean[prompt_idx]:
            matches += 1
            prompt_idx += 1
    
    similarity = matches / len(prompt_clean)
    return similarity >= min_similarity

