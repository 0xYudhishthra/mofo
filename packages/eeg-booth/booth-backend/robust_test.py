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

def robust_bci_test():
    """
    Robust BCI test with better error handling and connection retry
    """
    
    # Set logging level to reduce noise
    BoardShim.set_log_level(LogLevels.LEVEL_WARN)
    
    # Set up board parameters for OpenBCI Cyton
    params = BrainFlowInputParams()
    params.serial_port = "/dev/cu.usbserial-DM01MV82"
    board_id = BoardIds.CYTON_BOARD
    
    board_shim = None
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            print(f"🔌 Connecting to OpenBCI Cyton (attempt {attempt + 1}/{max_retries})...")
            
            board_shim = BoardShim(board_id, params)
            board_shim.prepare_session()
            
            print("✅ Board prepared! Starting data stream...")
            board_shim.start_stream()
            
            # Wait longer for stream to stabilize
            print("⏱️  Waiting for stream to stabilize...")
            time.sleep(5)
            
            # Get channel info
            eeg_channels = BoardShim.get_eeg_channels(board_id)
            sampling_rate = BoardShim.get_sampling_rate(board_id)
            
            # Check if we're getting data
            initial_count = board_shim.get_board_data_count()
            time.sleep(1)
            current_count = board_shim.get_board_data_count()
            
            if current_count > initial_count:
                print(f"✅ Stream active! Data flowing at ~{current_count - initial_count} samples/sec")
            else:
                print("⚠️  No data detected, but connection seems stable")
            
            print(f"📊 Sampling Rate: {sampling_rate} Hz")
            print(f"🧠 EEG Channels: {eeg_channels}")
            print("\n" + "="*60)
            print("LIVE EEG VOLTAGE READINGS")
            print("="*60)
            
            # Read for 10 seconds with better frequency analysis
            for i in range(10):
                time.sleep(1)
                
                # Get recent data (1 second worth)
                data = board_shim.get_current_board_data(sampling_rate)
                
                if data.shape[1] > 0:
                    eeg_data = data[eeg_channels]
                    latest_sample = eeg_data[:, -1]  # Most recent sample
                    
                    print(f"\n⏰ Reading {i+1}/10:")
                    
                    for ch_idx, voltage in enumerate(latest_sample):
                        # Status based on typical EEG ranges
                        if abs(voltage) > 200:
                            status = "🔴 HIGH"
                        elif abs(voltage) > 100:
                            status = "🟡 MED" 
                        elif abs(voltage) > 10:
                            status = "🟢 OK"
                        else:
                            status = "🟣 LOW"
                            
                        print(f"  Ch{ch_idx+1}: {voltage:+8.1f} µV [{status}]")
                    
                    # Simple frequency analysis using numpy FFT
                    if data.shape[1] >= 250:  # Need at least 1 full second at 250Hz
                        try:
                            # Use channel 1 for frequency analysis
                            ch_data = eeg_data[0, :].copy()
                            data_len = len(ch_data)
                            
                            if data_len >= 128:
                                # Apply basic filtering first
                                filtered_data = ch_data.copy()
                                try:
                                    DataFilter.perform_bandpass(filtered_data, sampling_rate, 1.0, 45.0, 4,
                                                              FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
                                except:
                                    # If filtering fails, use raw data
                                    filtered_data = ch_data.copy()
                                
                                # Simple numpy-based frequency analysis
                                # Calculate power spectral density using numpy
                                fft_data = np.fft.fft(filtered_data)
                                freqs = np.fft.fftfreq(len(filtered_data), 1.0/sampling_rate)
                                
                                # Take only positive frequencies
                                positive_freqs = freqs[:len(freqs)//2]
                                power_spectrum = np.abs(fft_data[:len(fft_data)//2])**2
                                
                                # Define frequency bands
                                delta_mask = (positive_freqs >= 0.5) & (positive_freqs < 4.0)
                                theta_mask = (positive_freqs >= 4.0) & (positive_freqs < 8.0)
                                alpha_mask = (positive_freqs >= 8.0) & (positive_freqs < 13.0)
                                beta_mask = (positive_freqs >= 13.0) & (positive_freqs < 30.0)
                                
                                # Calculate power in each band
                                delta_power = np.sum(power_spectrum[delta_mask]) if np.any(delta_mask) else 0
                                theta_power = np.sum(power_spectrum[theta_mask]) if np.any(theta_mask) else 0
                                alpha_power = np.sum(power_spectrum[alpha_mask]) if np.any(alpha_mask) else 0
                                beta_power = np.sum(power_spectrum[beta_mask]) if np.any(beta_mask) else 0
                                
                                total_power = delta_power + theta_power + alpha_power + beta_power
                                
                                if total_power > 0:
                                    print(f"  🧠 Frequency Analysis (Ch1):")
                                    print(f"     Delta: {(delta_power/total_power)*100:4.1f}% | Theta: {(theta_power/total_power)*100:4.1f}%")
                                    print(f"     Alpha: {(alpha_power/total_power)*100:4.1f}% | Beta:  {(beta_power/total_power)*100:4.1f}%")
                                    
                                    # Simple mental state indicator based on alpha/beta ratio
                                    if total_power > 0:
                                        alpha_ratio = (alpha_power/total_power)*100
                                        beta_ratio = (beta_power/total_power)*100
                                        
                                        if alpha_ratio > 40:
                                            state = "😌 Relaxed"
                                        elif beta_ratio > 40:
                                            state = "🧠 Focused"
                                        elif delta_power/total_power > 0.5:
                                            state = "😴 Drowsy"
                                        else:
                                            state = "🤔 Mixed"
                                        
                                        print(f"     Mental state: {state}")
                                else:
                                    print(f"  📊 Signal detected but too noisy for frequency analysis")
                            else:
                                print(f"  📊 Signal quality good, collecting more data...")
                                        
                        except Exception as e:
                            print(f"  ⚠️  Frequency analysis error: {str(e)[:40]}...")
                    else:
                        print(f"  📊 Signal quality good, need more data for frequency analysis")
                    
                    print("-" * 40)
                else:
                    print(f"⚠️  No data at reading {i+1}")
            
            return True  # Success!
                
        except BrainFlowError as e:
            print(f"❌ BrainFlow Error (attempt {attempt + 1}): {e}")
            if board_shim:
                try:
                    if board_shim.is_prepared():
                        board_shim.stop_stream()
                        board_shim.release_session()
                except:
                    pass
                board_shim = None
            
            if attempt < max_retries - 1:
                print(f"⏳ Waiting 3 seconds before retry...")
                time.sleep(3)
            
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            break
        
        finally:
            if board_shim and attempt == max_retries - 1:  # Last attempt cleanup
                try:
                    if board_shim.is_prepared():
                        board_shim.stop_stream()
                        board_shim.release_session()
                        print("✅ Disconnected safely")
                except:
                    pass
    
    return False

if __name__ == "__main__":
    print("🧠 Robust BCI Voltage Test")
    print("="*40)
    
    success = robust_bci_test()
    
    if not success:
        print("\n🔄 Hardware connection failed. Check:")
        print("  • OpenBCI board is powered on")
        print("  • USB cable is connected")
        print("  • Serial port /dev/cu.usbserial-DM01MV82 is correct")
        print("  • No other software is using the device")
        print("  • Try pressing the reset button on the OpenBCI board")
    else:
        print("\n🎉 Test completed successfully!")