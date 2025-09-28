#!/usr/bin/env python3
"""
Test script for personality calculation debugging
"""
import numpy as np
import sys
import os

# Add the booth-backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'booth-backend'))

from eeg_processor import EEGProcessor

def generate_realistic_eeg_data(duration_seconds=60, sampling_rate=250):
    """
    Generate realistic synthetic EEG data for testing
    """
    num_samples = duration_seconds * sampling_rate
    channels = []
    
    # Generate 8 channels of synthetic EEG data
    for ch in range(8):
        # Base noise
        data = np.random.normal(0, 2, num_samples)  # ~2µV noise
        
        # Add EEG-like frequency components
        time = np.linspace(0, duration_seconds, num_samples)
        
        # Alpha rhythm (8-13 Hz) - stronger in posterior channels
        alpha_strength = 3 if ch >= 4 else 1.5
        data += alpha_strength * np.sin(2 * np.pi * 10 * time + np.random.uniform(0, 2*np.pi))
        
        # Beta activity (13-30 Hz) - stronger in frontal channels
        beta_strength = 2 if ch < 4 else 1
        data += beta_strength * np.sin(2 * np.pi * 18 * time + np.random.uniform(0, 2*np.pi))
        
        # Theta (4-8 Hz) - variable across channels
        theta_strength = np.random.uniform(0.5, 2)
        data += theta_strength * np.sin(2 * np.pi * 6 * time + np.random.uniform(0, 2*np.pi))
        
        # Delta (1-4 Hz) - low frequency drift
        delta_strength = np.random.uniform(1, 3)
        data += delta_strength * np.sin(2 * np.pi * 2 * time + np.random.uniform(0, 2*np.pi))
        
        # Gamma (30-45 Hz) - small amount
        gamma_strength = np.random.uniform(0.2, 0.8)
        data += gamma_strength * np.sin(2 * np.pi * 35 * time + np.random.uniform(0, 2*np.pi))
        
        # Add some artifacts occasionally
        if np.random.random() < 0.1:  # 10% chance of artifacts
            artifact_start = np.random.randint(0, num_samples - 1000)
            data[artifact_start:artifact_start + 500] += np.random.uniform(-20, 20)
        
        channels.append(data.tolist())
    
    return channels

def test_different_personality_profiles():
    """
    Test personality calculation with different EEG profiles
    """
    processor = EEGProcessor()
    
    profiles = [
        "Normal mixed activity",
        "High alpha (relaxed)",
        "High beta (focused/anxious)",
        "High theta (drowsy/creative)",
        "High gamma (high cognition)"
    ]
    
    print("Testing Personality Calculation with Different EEG Profiles")
    print("=" * 60)
    
    for i, profile_name in enumerate(profiles):
        print(f"\n{i+1}. Testing Profile: {profile_name}")
        print("-" * 40)
        
        # Generate data with different characteristics
        if "high alpha" in profile_name.lower():
            # Simulate relaxed state - more alpha
            data = generate_realistic_eeg_data(120)  # 2 minutes
            # Boost alpha in posterior channels
            for ch in range(4, 8):
                time = np.linspace(0, 120, len(data[ch]))
                alpha_boost = 5 * np.sin(2 * np.pi * 10 * time)
                data[ch] = [x + y for x, y in zip(data[ch], alpha_boost)]
                
        elif "high beta" in profile_name.lower():
            # Simulate focused/anxious state - more beta
            data = generate_realistic_eeg_data(120)
            for ch in range(8):
                time = np.linspace(0, 120, len(data[ch]))
                beta_boost = 4 * np.sin(2 * np.pi * 20 * time)
                data[ch] = [x + y for x, y in zip(data[ch], beta_boost)]
                
        elif "high theta" in profile_name.lower():
            # Simulate drowsy/creative state - more theta
            data = generate_realistic_eeg_data(120)
            for ch in range(8):
                time = np.linspace(0, 120, len(data[ch]))
                theta_boost = 4 * np.sin(2 * np.pi * 6 * time)
                data[ch] = [x + y for x, y in zip(data[ch], theta_boost)]
                
        elif "high gamma" in profile_name.lower():
            # Simulate high cognition - more gamma
            data = generate_realistic_eeg_data(120)
            for ch in range(8):
                time = np.linspace(0, 120, len(data[ch]))
                gamma_boost = 2 * np.sin(2 * np.pi * 40 * time)
                data[ch] = [x + y for x, y in zip(data[ch], gamma_boost)]
                
        else:
            # Normal mixed activity
            data = generate_realistic_eeg_data(120)
        
        # Test the personality calculation
        result = processor.test_personality_calculation(data)
        
        if result:
            print(f"\nPersonality scores for {profile_name}:")
            for trait, score in result['scores'].items():
                print(f"  {trait}: {score}%")
            print(f"Confidence: {result.get('confidence', 'N/A')}%")
        else:
            print("Failed to calculate personality scores")
        
        print("\n" + "="*60)

if __name__ == "__main__":
    test_different_personality_profiles()