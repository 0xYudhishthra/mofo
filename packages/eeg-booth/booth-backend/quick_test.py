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

def quick_bci_test():
    """
    Quick BCI test - 10 seconds with clean output
    """
    
    # Disable verbose logging for cleaner output
    BoardShim.set_log_level(LogLevels.LEVEL_ERROR)
    
    # Set up board parameters for OpenBCI Cyton
    params = BrainFlowInputParams()
    params.serial_port = "/dev/cu.usbserial-DM01MV82"
    board_id = BoardIds.CYTON_BOARD
    
    board_shim = None
    
    try:
        print("🔌 Connecting to OpenBCI Cyton...")
        board_shim = BoardShim(board_id, params)
        board_shim.prepare_session()
        board_shim.start_stream()
        
        # Wait for stream to stabilize
        time.sleep(2)
        
        # Get channel info
        eeg_channels = BoardShim.get_eeg_channels(board_id)
        sampling_rate = BoardShim.get_sampling_rate(board_id)
        
        print(f"✅ Connected! Sampling at {sampling_rate} Hz")
        print(f"🧠 Reading {len(eeg_channels)} EEG channels")
        print("\n" + "="*60)
        print("LIVE EEG VOLTAGE READINGS")
        print("="*60)
        
        # Read for 10 seconds, print every 2 seconds
        for i in range(5):
            time.sleep(2)
            
            # Get recent data
            data = board_shim.get_current_board_data(500)  # 2 seconds worth
            
            if data.shape[1] > 0:
                eeg_data = data[eeg_channels]
                latest_sample = eeg_data[:, -1]  # Most recent sample
                
                print(f"\n⏰ Reading {i+1}/5 (t={2*(i+1)}s):")
                
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
                
                # Calculate signal quality (RMS over last 500 samples)
                if data.shape[1] >= 500:
                    rms_values = []
                    for ch_idx in range(len(eeg_channels)):
                        ch_data = eeg_data[ch_idx, -500:]
                        rms = np.sqrt(np.mean(ch_data ** 2))
                        rms_values.append(rms)
                    
                    avg_rms = np.mean(rms_values)
                    print(f"  📊 Avg Signal RMS: {avg_rms:.1f} µV")
                    
                    if avg_rms < 50:
                        print("  ✨ Good signal quality!")
                    elif avg_rms < 200:
                        print("  ⚠️  Moderate signal - check electrode contact")
                    else:
                        print("  🚨 High noise - check connections & environment")
                
                print("-" * 40)
            else:
                print(f"⚠️  No data at reading {i+1}")
                
    except BrainFlowError as e:
        print(f"❌ BrainFlow Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        if board_shim:
            try:
                if board_shim.is_prepared():
                    board_shim.stop_stream()
                    board_shim.release_session()
                    print("\n✅ Disconnected safely")
            except:
                pass
    
    return True

if __name__ == "__main__":
    print("🧠 Quick BCI Voltage Test")
    print("="*40)
    quick_bci_test()
    print("\n🎉 Test complete!")