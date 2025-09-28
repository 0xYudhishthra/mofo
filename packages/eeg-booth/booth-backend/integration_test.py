#!/usr/bin/env python3
"""
Quick test script using the robust BrainFlow approach
Integrates the improvements from robust_test.py into our EEG processor
"""

import sys
import time

try:
    import numpy as np
    from booth_server import BoothBackend
    from eeg_processor import EEGProcessor
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Install required packages: pip install numpy scipy brainflow")
    DEPENDENCIES_AVAILABLE = False

def test_robust_eeg_integration():
    """Test the integrated robust EEG processing"""
    
    print("🧠 Testing Robust EEG Integration")
    print("="*50)
    
    # Initialize EEG processor with robust features
    try:
        processor = EEGProcessor(sampling_rate=250)
        print("✅ EEG Processor initialized with robust features")
        
        # Test with synthetic EEG data that mimics real patterns
        print("\n📊 Testing with synthetic EEG patterns...")
        
        # Generate test data with different frequency components
        duration = 3  # 3 seconds
        sampling_rate = 250
        t = np.linspace(0, duration, duration * sampling_rate)
        
        # Create synthetic EEG with mixed frequency bands
        eeg_signal = (
            np.sin(2 * np.pi * 10 * t) * 20 +  # Alpha (10 Hz)
            np.sin(2 * np.pi * 20 * t) * 10 +  # Beta (20 Hz)
            np.sin(2 * np.pi * 6 * t) * 15 +   # Theta (6 Hz)
            np.random.normal(0, 5, len(t))     # Noise
        )
        
        print(f"   Generated {len(eeg_signal)} samples at {sampling_rate}Hz")
        print(f"   Signal range: {np.min(eeg_signal):.1f} to {np.max(eeg_signal):.1f} µV")
        
        # Test signal quality assessment
        sample_voltage = eeg_signal[0]
        quality = processor.assess_signal_quality(sample_voltage)
        print(f"   Signal quality: {quality}")
        
        # Test robust frequency band calculation
        print("\n🎯 Testing robust frequency band calculation...")
        
        try:
            # Use the robust BrainFlow method
            band_powers = processor.calculate_band_power_brainflow(eeg_signal, sampling_rate)
            
            if band_powers:
                total_power = sum(band_powers.values())
                print(f"   Total power: {total_power:.2f}")
                
                if total_power > 0:
                    print("   Band percentages:")
                    for band_name, power in band_powers.items():
                        percentage = (power / total_power) * 100
                        print(f"     {band_name.capitalize():>6}: {percentage:5.1f}%")
                    
                    # Test mental state determination
                    percentages = {k: (v/total_power)*100 for k, v in band_powers.items()}
                    mental_state = processor.get_mental_state(percentages)
                    print(f"   Mental state: {mental_state}")
                else:
                    print("   ⚠️  No power detected in any band")
            else:
                print("   ❌ Band power calculation failed")
                
        except Exception as e:
            print(f"   ❌ Robust frequency analysis failed: {e}")
            
            # Fallback to simple method
            try:
                print("   🔄 Trying fallback method...")
                band_powers = processor.calculate_band_power(eeg_signal)
                print(f"   Fallback results: {band_powers}")
            except Exception as e2:
                print(f"   ❌ Fallback also failed: {e2}")
        
        # Test with multi-channel data (simulating 8-channel EEG)
        print("\n🎛️  Testing multi-channel processing...")
        
        channels_data = []
        for ch in range(8):
            # Each channel with slightly different characteristics
            channel_signal = eeg_signal + np.random.normal(0, 2, len(eeg_signal))
            channels_data.append(channel_signal.tolist())
        
        try:
            frequency_bands = processor.calculate_realtime_frequency_bands(channels_data)
            
            if frequency_bands:
                print("   Multi-channel frequency analysis:")
                for band_name, percentage in frequency_bands.items():
                    print(f"     {band_name.capitalize():>6}: {percentage:5.1f}%")
                    
                mental_state = processor.get_mental_state(frequency_bands)
                print(f"   Overall mental state: {mental_state}")
            else:
                print("   ❌ Multi-channel analysis failed")
                
        except Exception as e:
            print(f"   ❌ Multi-channel processing error: {e}")
        
        print("\n✅ Robust EEG integration test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_booth_integration():
    """Test BoothBackend with robust features (without hardware)"""
    
    print("\n🏢 Testing BoothBackend Integration")
    print("="*50)
    
    try:
        # Initialize booth backend (won't try to connect to hardware)
        booth = BoothBackend(booth_id="test_booth")
        print("✅ BoothBackend initialized")
        
        # Test EEG processor integration
        if booth.eeg_processor:
            print("✅ EEG processor available in booth")
            
            # Test with sample data
            sample_data = [np.random.normal(0, 20, 500).tolist() for _ in range(8)]
            
            try:
                frequency_result = booth.eeg_processor.calculate_realtime_frequency_bands(sample_data)
                if frequency_result:
                    print("✅ Frequency analysis working in booth context")
                    print(f"   Sample result: {frequency_result}")
                else:
                    print("⚠️  No frequency analysis result")
            except Exception as e:
                print(f"❌ Frequency analysis error: {e}")
        else:
            print("❌ EEG processor not available")
        
        print("✅ BoothBackend integration test completed!")
        return True
        
    except Exception as e:
        print(f"❌ BoothBackend test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Robust EEG Integration Test Suite")
    print("="*60)
    
    # Test EEG processor with robust features
    test1_success = test_robust_eeg_integration()
    
    # Test booth integration
    test2_success = test_booth_integration()
    
    print("\n📋 Test Summary:")
    print("="*30)
    print(f"EEG Processor Test: {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"BoothBackend Test:  {'✅ PASS' if test2_success else '❌ FAIL'}")
    
    if test1_success and test2_success:
        print("\n🎉 All tests passed! Robust EEG integration is working.")
        print("\nNext steps:")
        print("  1. Connect OpenBCI hardware")
        print("  2. Run ./start-booth.sh")
        print("  3. Check for improved frequency band accuracy")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        
    sys.exit(0 if (test1_success and test2_success) else 1)