#!/usr/bin/env python3

import os
import time
import numpy as np
from dotenv import load_dotenv
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
from brainflow.exit_codes import *

# Load environment variables
load_dotenv()

def test_openbci_voltage():
    """
    Test script to connect to OpenBCI and print live voltage readings
    """
    
    # Enable BrainFlow logging
    BoardShim.enable_dev_board_logger()
    
    # Set up board parameters for OpenBCI Cyton
    params = BrainFlowInputParams()
    params.serial_port = "/dev/cu.usbserial-DM01MV82"  # Your OpenBCI serial port
    board_id = BoardIds.CYTON_BOARD  # OpenBCI Cyton (8-channel)
    
    board_shim = None
    
    try:
        print("🔌 Connecting to OpenBCI Cyton...")
        board_shim = BoardShim(board_id, params)
        board_shim.prepare_session()
        
        print("✅ Connected! Starting data stream...")
        board_shim.start_stream()
        
        # Wait a moment for stream to stabilize
        time.sleep(2)
        
        # Get channel info
        eeg_channels = BoardShim.get_eeg_channels(board_id)
        sampling_rate = BoardShim.get_sampling_rate(board_id)
        
        # Check if stream is active
        print(f"🔍 Stream active: {board_shim.get_board_data_count()}")
        
        print(f"📊 Sampling Rate: {sampling_rate} Hz")
        print(f"🧠 EEG Channels: {eeg_channels}")
        print("\n" + "="*80)
        print("LIVE EEG VOLTAGE READINGS (µV)")
        print("="*80)
        
        packet_count = 0
        no_data_count = 0
        
        # Read data for 30 seconds
        start_time = time.time()
        while time.time() - start_time < 30:
            
            # Get available data
            data = board_shim.get_current_board_data(250)  # Get last 1 second
            
            # Debug: Print data shape info
            if time.time() - start_time < 5:  # Only for first 5 seconds
                print(f"Debug: Data shape: {data.shape}, Samples: {data.shape[1] if len(data.shape) > 1 else 0}")
            
            if data.shape[1] > 0:
                packet_count += data.shape[1]
                
                # Extract EEG channels (convert to µV)
                eeg_data = data[eeg_channels]
                
                # Get the most recent sample
                latest_sample = eeg_data[:, -1]
                
                # Print voltage readings every 250 samples (1 second)
                if packet_count % 250 == 0:
                    print(f"\n⏰ Time: {time.time() - start_time:.1f}s | Packet: #{packet_count}")
                    
                    for i, voltage in enumerate(latest_sample):
                        channel_name = f"Ch{i+1}"
                        
                        # Color coding based on voltage range
                        if abs(voltage) > 100:
                            status = "🔴 HIGH"
                        elif abs(voltage) > 50:
                            status = "🟡 MED"
                        else:
                            status = "🟢 OK"
                            
                        print(f"  {channel_name}: {voltage:+8.2f} µV [{status}]")
                    
                    # Calculate and show RMS values for signal quality
                    if data.shape[1] >= 250:  # If we have enough samples
                        rms_values = []
                        for ch_idx in range(len(eeg_channels)):
                            ch_data = eeg_data[ch_idx, -250:]  # Last 1 second
                            rms = np.sqrt(np.mean(ch_data ** 2))
                            rms_values.append(rms)
                        
                        print(f"\n  📈 Signal Quality (RMS µV):")
                        for i, rms in enumerate(rms_values):
                            quality = "Good" if 5 < rms < 100 else "Check"
                            print(f"     Ch{i+1}: {rms:.1f} µV [{quality}]")
                        
                        # Calculate frequency bands using BrainFlow
                        try:
                            # Use first channel for frequency analysis
                            ch1_data = eeg_data[0, -250:].copy()  # Last 1 second of Ch1
                            
                            # Only proceed if we have enough data points
                            if len(ch1_data) < 64:
                                print(f"  ⚠️  Not enough data for frequency analysis ({len(ch1_data)} samples)")
                                continue
                            
                            # Apply basic filtering
                            DataFilter.perform_bandpass(ch1_data, sampling_rate, 1.0, 45.0, 4,
                                                      FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
                            
                            # Calculate PSD with safer parameters
                            # Use much smaller window to ensure nfft < data_len
                            data_len = len(ch1_data)
                            nfft = min(64, data_len // 4)  # Use 1/4 of data length or 64, whichever is smaller
                            overlap = nfft // 4  # Use smaller overlap
                            
                            if nfft < 16:  # Minimum sensible FFT size
                                print(f"  ⚠️  Data too short for reliable frequency analysis")
                                continue
                                
                            psd_data = DataFilter.get_psd_welch(ch1_data, nfft, overlap, sampling_rate, 0)
                            
                            # Extract frequency bands
                            delta_power = DataFilter.get_band_power(psd_data, 0.5, 4.0)
                            theta_power = DataFilter.get_band_power(psd_data, 4.0, 8.0)  
                            alpha_power = DataFilter.get_band_power(psd_data, 8.0, 13.0)
                            beta_power = DataFilter.get_band_power(psd_data, 13.0, 30.0)
                            gamma_power = DataFilter.get_band_power(psd_data, 30.0, 45.0)
                            
                            total_power = delta_power + theta_power + alpha_power + beta_power + gamma_power
                            
                            if total_power > 0:
                                print(f"\n  🧠 Frequency Bands (Ch1):")
                                print(f"     Delta (0.5-4Hz):  {(delta_power/total_power)*100:5.1f}%")
                                print(f"     Theta (4-8Hz):    {(theta_power/total_power)*100:5.1f}%")
                                print(f"     Alpha (8-13Hz):   {(alpha_power/total_power)*100:5.1f}%")
                                print(f"     Beta (13-30Hz):   {(beta_power/total_power)*100:5.1f}%")
                                print(f"     Gamma (30-45Hz):  {(gamma_power/total_power)*100:5.1f}%")
                                
                        except Exception as e:
                            print(f"  ⚠️  Frequency analysis error: {e}")
                            print(f"      Debug: data_len={len(ch1_data) if 'ch1_data' in locals() else 'N/A'}, nfft={nfft if 'nfft' in locals() else 'N/A'}")
                    
                    print("-" * 60)
            else:
                no_data_count += 1
                if no_data_count % 50 == 0:  # Print every 5 seconds (50 * 0.1s)
                    print(f"⚠️  No data received for {no_data_count * 0.1:.1f} seconds...")
            
            time.sleep(0.1)  # Small delay to prevent overwhelming output
            
    except BrainFlowError as e:
        print(f"❌ BrainFlow Error: {e}")
        return False
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping data acquisition...")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
        
    finally:
        if board_shim:
            try:
                if board_shim.is_prepared():
                    print("🔄 Stopping stream...")
                    board_shim.stop_stream()
                    board_shim.release_session()
                    print("✅ Disconnected safely")
            except:
                pass
    
    return True

def test_eeg_processor():
    """Test the EEG processor with sample data (fallback if no hardware)"""
    try:
        from eeg_processor import EEGProcessor
        
        processor = EEGProcessor(sampling_rate=250)
        
        # Generate some sample EEG-like data (8 channels, 2 seconds)
        samples = 500
        channels = 8
        
        sample_data = []
        for ch in range(channels):
            # Generate realistic EEG-like signal with noise
            t = np.linspace(0, 2, samples)
            
            # Alpha wave around 10 Hz
            alpha = 20 * np.sin(2 * np.pi * 10 * t)
            
            # Beta wave around 20 Hz  
            beta = 10 * np.sin(2 * np.pi * 20 * t)
            
            # Some noise
            noise = np.random.normal(0, 5, samples)
            
            signal = alpha + beta + noise
            sample_data.append(signal.tolist())
        
        # Test frequency analysis
        result = processor.calculate_realtime_frequency_bands(sample_data)
        print("Frequency bands:", result)
        
    except Exception as e:
        print("EEG processor test error:", e)

if __name__ == "__main__":
    print("🧠 OpenBCI Voltage Test Script")
    print("="*50)
    
    # Try to connect to real hardware first
    if not test_openbci_voltage():
        print("\n🔄 Hardware test failed, trying software test...")
        test_eeg_processor()
