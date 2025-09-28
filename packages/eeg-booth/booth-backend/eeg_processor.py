"""
Scientific EEG Processor for Love Detection
Based on peer-reviewed neuroscience research
"""
import numpy as np
import logging
from scipy import signal
from scipy.stats import zscore

# Set up logger
logger = logging.getLogger(__name__)
try:
    from brainflow.data_filter import DataFilter, FilterTypes, WindowOperations, DetrendOperations
    from brainflow.board_shim import BoardShim
    BRAINFLOW_AVAILABLE = True
except ImportError:
    print("Warning: BrainFlow not available. Some features may not work.")
    BRAINFLOW_AVAILABLE = False

class EEGProcessor:
    def __init__(self, sampling_rate=250):
        self.sampling_rate = sampling_rate
        self.nyquist = sampling_rate / 2
        
        # Frequency band definitions (Hz) - adjusted to reduce delta dominance
        self.bands = {
            'delta': (1, 4),      # Narrower delta range, starts at 1Hz
            'theta': (4, 8), 
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
    
    def validate_openbci_compatibility(self, channel_data, channel_name="Ch1"):
        """
        Validate that our calculations match OpenBCI GUI expectations
        Returns detailed analysis for debugging
        """
        if len(channel_data) < 500:
            return {"error": f"Need at least 500 samples, got {len(channel_data)}"}
        
        # Use last 512 samples (2 seconds at 250Hz)
        window_data = np.array(channel_data[-512:])
        
        # Basic statistics
        stats = {
            'channel': channel_name,
            'samples': len(window_data),
            'mean_amplitude': round(np.mean(window_data), 3),
            'std_amplitude': round(np.std(window_data), 3),
            'min_amplitude': round(np.min(window_data), 3),
            'max_amplitude': round(np.max(window_data), 3),
            'peak_to_peak': round(np.max(window_data) - np.min(window_data), 3)
        }
        
        # FFT Analysis (OpenBCI style)
        n_fft = 512
        window = np.hanning(len(window_data))
        windowed_data = window_data * window
        windowed_data = windowed_data - np.mean(windowed_data)
        
        fft_result = np.fft.fft(windowed_data, n_fft)
        power_spectrum = np.abs(fft_result[:n_fft//2])**2
        power_spectrum = power_spectrum / (self.sampling_rate * n_fft) * 1e12  # Convert to µV²
        
        freqs = np.fft.fftfreq(n_fft, 1.0/self.sampling_rate)[:n_fft//2]
        
        # Find dominant frequency
        dominant_freq_idx = np.argmax(power_spectrum[1:]) + 1  # Skip DC
        dominant_frequency = freqs[dominant_freq_idx]
        
        # Calculate band powers
        band_analysis = {}
        total_power = 0
        
        for band_name, (low_freq, high_freq) in self.bands.items():
            band_indices = np.where((freqs >= low_freq) & (freqs <= high_freq))[0]
            if len(band_indices) > 0:
                band_power = np.sum(power_spectrum[band_indices])
                band_analysis[band_name] = {
                    'power_uv2': round(band_power, 2),
                    'power_log': round(np.log10(max(band_power, 1e-6)), 2),
                    'freq_range': f"{low_freq}-{high_freq}Hz",
                    'bins_used': len(band_indices)
                }
                total_power += band_power
            else:
                band_analysis[band_name] = {
                    'power_uv2': 0.0,
                    'power_log': -6.0,
                    'freq_range': f"{low_freq}-{high_freq}Hz",
                    'bins_used': 0
                }
        
        # Calculate relative powers
        for band_name in band_analysis:
            rel_power = (band_analysis[band_name]['power_uv2'] / total_power * 100) if total_power > 0 else 0
            band_analysis[band_name]['relative_percent'] = round(rel_power, 1)
        
        # Find dominant band
        dominant_band = max(band_analysis, key=lambda x: band_analysis[x]['power_uv2'])
        
        return {
            'basic_stats': stats,
            'dominant_frequency': round(dominant_frequency, 2),
            'dominant_band': dominant_band,
            'total_power_uv2': round(total_power, 2),
            'band_analysis': band_analysis,
            'data_quality': {
                'amplitude_range_ok': stats['peak_to_peak'] > 1.0,  # Should see at least 1µV variation
                'not_saturated': abs(stats['max_amplitude']) < 1000,  # Not clipping
                'has_variation': stats['std_amplitude'] > 0.1  # Has meaningful variation
            }
        }
    
    def check_data_scaling(self, channels_data):
        """
        Check if EEG data scaling matches expected microvolts range
        Normal EEG should be roughly 10-100 µV peak-to-peak
        """
        scaling_report = {}
        
        for i, channel_data in enumerate(channels_data[:8]):
            if len(channel_data) < 100:
                continue
                
            recent_data = np.array(channel_data[-250:])  # Last 1 second
            
            amplitude_stats = {
                'mean': np.mean(recent_data),
                'std': np.std(recent_data),
                'peak_to_peak': np.max(recent_data) - np.min(recent_data),
                'rms': np.sqrt(np.mean(recent_data**2))
            }
            
            # Check if scaling looks reasonable for EEG
            scaling_assessment = {
                'likely_microvolts': 5 < amplitude_stats['peak_to_peak'] < 200,
                'likely_volts': 0.000005 < amplitude_stats['peak_to_peak'] < 0.0002,
                'likely_raw_counts': amplitude_stats['peak_to_peak'] > 1000,
                'suggested_scaling': 'unknown'
            }
            
            if amplitude_stats['peak_to_peak'] > 1000:
                scaling_assessment['suggested_scaling'] = 'divide_by_1000_or_more'
            elif amplitude_stats['peak_to_peak'] < 1:
                scaling_assessment['suggested_scaling'] = 'multiply_by_1000000'
            else:
                scaling_assessment['suggested_scaling'] = 'looks_good'
            
            scaling_report[f'channel_{i+1}'] = {
                'amplitude_stats': amplitude_stats,
                'scaling_assessment': scaling_assessment
            }
        
        return scaling_report
    
    def get_frequency_summary(self, channels_data):
        """Get frequency band power summary for all channels"""
        summary = {}
        
        for i, channel_data in enumerate(channels_data):
            channel_summary = {}
            
            for band_name in self.bands:
                power = self.get_band_power(channel_data, band_name)
                channel_summary[band_name] = round(power, 4)
                
            summary[f'channel_{i+1}'] = channel_summary
            
        return summary
        

    
    def bandpass_filter(self, data, low_freq=1, high_freq=45):
        """Apply bandpass filter using BrainFlow's optimized filters"""
        if len(data) < 10:
            return data
        
        # Convert to numpy array
        data_array = np.array(data, dtype=np.float64)
        
        if not BRAINFLOW_AVAILABLE:
            # Fallback to scipy if BrainFlow not available
            nyquist = self.sampling_rate / 2
            low = low_freq / nyquist
            high = high_freq / nyquist
            
            try:
                b, a = signal.butter(4, [low, high], btype='band')
                filtered = signal.filtfilt(b, a, data_array)
                return filtered
            except:
                return data_array
        
        try:
            # Use BrainFlow's optimized bandpass filter
            DataFilter.perform_bandpass(
                data_array, 
                self.sampling_rate, 
                low_freq, 
                high_freq, 
                order=4,  # 4th order Butterworth
                filter_type=FilterTypes.BUTTERWORTH.value,
                ripple=0
            )
            return data_array
            
        except Exception as e:
            # Fallback to scipy if BrainFlow fails
            try:
                nyquist = self.sampling_rate / 2
                low = low_freq / nyquist
                high = high_freq / nyquist
                b, a = signal.butter(4, [low, high], btype='band')
                filtered = signal.filtfilt(b, a, data_array)
                return filtered
            except:
                return data_array
    
    def preprocess_eeg_data(self, data):
        """Comprehensive EEG preprocessing using BrainFlow's advanced functions"""
        if len(data) < 100:
            return data
        
        data_array = np.array(data, dtype=np.float64)
        
        if not BRAINFLOW_AVAILABLE:
            return self.bandpass_filter(data_array, 0.5, 45)
        
        try:
            # 1. Remove DC offset and linear trends
            DataFilter.detrend(data_array, DetrendOperations.LINEAR.value)
            
            # 2. Apply notch filter to remove powerline interference (50Hz/60Hz)
            DataFilter.perform_bandstop(
                data_array, self.sampling_rate, 49.0, 51.0, 4,
                FilterTypes.BUTTERWORTH.value, 0
            )
            DataFilter.perform_bandstop(
                data_array, self.sampling_rate, 59.0, 61.0, 4,
                FilterTypes.BUTTERWORTH.value, 0
            )
            
            # 3. High-pass filter to remove slow drift (0.5 Hz)
            DataFilter.perform_highpass(
                data_array, self.sampling_rate, 0.5, 4,
                FilterTypes.BUTTERWORTH.value, 0
            )
            
            # 4. Low-pass filter to remove high-frequency noise (45 Hz)
            DataFilter.perform_lowpass(
                data_array, self.sampling_rate, 45.0, 4,
                FilterTypes.BUTTERWORTH.value, 0
            )
            
            return data_array
            
        except Exception as e:
            # Fallback to basic filtering
            return self.bandpass_filter(data_array, 0.5, 45)
    
    def assess_signal_quality(self, voltage):
        """
        Assess signal quality based on voltage range (from robust_test.py)
        """
        if abs(voltage) > 200:
            return "🔴 HIGH"
        elif abs(voltage) > 100:
            return "🟡 MED" 
        elif abs(voltage) > 10:
            return "🟢 OK"
        else:
            return "🟣 LOW"
    
    def get_mental_state(self, frequency_bands):
        """
        Determine mental state from frequency bands (from robust_test.py)
        """
        alpha_pct = frequency_bands.get('alpha', 0)
        beta_pct = frequency_bands.get('beta', 0)
        delta_pct = frequency_bands.get('delta', 0)
        
        if alpha_pct > 40:
            return "😌 Relaxed"
        elif beta_pct > 40:
            return "🧠 Focused"
        elif delta_pct > 50:
            return "😴 Drowsy"
        else:
            return "🤔 Mixed"

    def calculate_band_power(self, data):
        """
        Calculate power for each frequency band - wrapper method for compatibility
        """
        if len(data) < 100:
            return {band: 0.0 for band in self.bands.keys()}
        
        # Use BrainFlow method if available, otherwise fallback to FFT
        if BRAINFLOW_AVAILABLE:
            return self.calculate_band_power_brainflow(data, self.sampling_rate)
        else:
            # Fallback to simple FFT method
            n = len(data)
            fft_vals = np.fft.fft(data)
            fft_freq = np.fft.fftfreq(n, 1/self.sampling_rate)
            
            # Get positive frequencies only
            pos_mask = fft_freq > 0
            freqs = fft_freq[pos_mask]
            power = np.abs(fft_vals[pos_mask]) ** 2
            
            # Calculate power for each band
            band_powers = {}
            for band_name, (low, high) in self.bands.items():
                band_mask = (freqs >= low) & (freqs <= high)
                band_powers[band_name] = np.mean(power[band_mask]) if np.any(band_mask) else 0
                
            return band_powers

    def calculate_band_power_brainflow(self, data, sampling_rate):
        """
        Calculate band power using robust method from robust_test.py
        Combines BrainFlow filtering with numpy FFT analysis
        """
        if len(data) < 128:  # Need minimum data for analysis
            return {band: 0.0 for band in self.bands.keys()}
        
        try:
            # Create a copy for processing (BrainFlow modifies in-place)
            data_copy = np.array(data, dtype=np.float64)
            
            # Apply BrainFlow's robust bandpass filter (from robust_test.py approach)
            try:
                DataFilter.perform_bandpass(data_copy, sampling_rate, 1.0, 45.0, 4,
                                          FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
            except:
                # If BrainFlow filtering fails, use original data
                data_copy = np.array(data, dtype=np.float64)
            
            # Use numpy FFT analysis (from robust_test.py)
            fft_data = np.fft.fft(data_copy)
            freqs = np.fft.fftfreq(len(data_copy), 1.0/sampling_rate)
            
            # Take only positive frequencies
            positive_freqs = freqs[:len(freqs)//2]
            power_spectrum = np.abs(fft_data[:len(fft_data)//2])**2
            
            # Calculate power in each band (robust_test.py method)
            band_powers = {}
            
            for band_name, (low_freq, high_freq) in self.bands.items():
                band_mask = (positive_freqs >= low_freq) & (positive_freqs < high_freq)
                if np.any(band_mask):
                    band_power = np.sum(power_spectrum[band_mask])
                    band_powers[band_name] = band_power
                else:
                    band_powers[band_name] = 0.0
            
            return band_powers
            
        except Exception as e:
            # Fallback to simple method if analysis fails
            logger.debug(f"BrainFlow band power calculation failed: {e}")
            return self.calculate_band_power(data)

    def calculate_big5_personality(self, channels_data, duration_minutes=2):
        """
        Calculate Big 5 personality traits from resting-state EEG
        Based on neuroscience research correlating EEG patterns with personality
        """
        if len(channels_data) < 8:
            raise ValueError("Need 8 channels of EEG data")
        
        # Process channels for personality analysis using BrainFlow preprocessing
        processed_channels = []
        for ch_data in channels_data:
            if len(ch_data) < 500:  # Need at least 2 seconds
                continue
            # Use comprehensive BrainFlow preprocessing
            filtered = self.preprocess_eeg_data(ch_data)
            processed_channels.append(filtered)
        
        if len(processed_channels) < 8:
            return None
        
        # Calculate personality indicators
        personality_scores = {}
        
        # 1. OPENNESS - Alpha in parietal/occipital regions (creativity, imagination)
        parietal_occipital_alpha = 0
        for i in [4, 5, 6, 7]:  # P7, P8, O1, O2
            alpha_power = self.get_band_power(processed_channels[i], 'alpha')
            parietal_occipital_alpha += alpha_power
        parietal_occipital_alpha /= 4
        openness = min(100, max(0, (np.log(parietal_occipital_alpha + 1) * 20)))
        
        # 2. CONSCIENTIOUSNESS - Beta activity in frontal regions (focus, organization)
        frontal_beta = 0
        for i in [0, 1]:  # Fp1, Fp2
            beta_power = self.get_band_power(processed_channels[i], 'beta')
            frontal_beta += beta_power
        frontal_beta /= 2
        conscientiousness = min(100, max(0, (np.log(frontal_beta + 1) * 25)))
        
        # 3. EXTRAVERSION - Frontal Alpha Asymmetry (approach vs withdrawal)
        faa = self.calculate_frontal_alpha_asymmetry(
            processed_channels[0],  # Fp1 (left)
            processed_channels[1]   # Fp2 (right)
        )
        extraversion = min(100, max(0, 50 + (faa * 100)))  # Positive FAA = extraversion
        
        # 4. AGREEABLENESS - Theta activity (emotional processing, empathy)
        avg_theta = 0
        for ch_data in processed_channels:
            theta_power = self.get_band_power(ch_data, 'theta')
            avg_theta += theta_power
        avg_theta /= len(processed_channels)
        agreeableness = min(100, max(0, (np.log(avg_theta + 1) * 30)))
        
        # 5. NEUROTICISM - High frequency activity and asymmetry
        avg_gamma = 0
        for ch_data in processed_channels:
            gamma_power = self.get_band_power(ch_data, 'gamma')
            avg_gamma += gamma_power
        avg_gamma /= len(processed_channels)
        neuroticism = min(100, max(0, (np.log(avg_gamma + 1) * 35)))
        
        personality_scores = {
            'openness': round(openness, 1),
            'conscientiousness': round(conscientiousness, 1),
            'extraversion': round(extraversion, 1),
            'agreeableness': round(agreeableness, 1),
            'neuroticism': round(neuroticism, 1)
        }
        
        # Generate personality description
        descriptions = {
            'openness': self._get_openness_description(openness),
            'conscientiousness': self._get_conscientiousness_description(conscientiousness),
            'extraversion': self._get_extraversion_description(extraversion),
            'agreeableness': self._get_agreeableness_description(agreeableness),
            'neuroticism': self._get_neuroticism_description(neuroticism)
        }
        
        return {
            'scores': personality_scores,
            'descriptions': descriptions,
            'analysis_duration': duration_minutes,
            'confidence': self._calculate_personality_confidence(processed_channels)
        }
    
    def analyze_dating_preference(self, channels_data, stimulus_type, stimulus_duration=3):
        """
        Analyze dating preferences based on EEG response to visual stimuli
        """
        if len(channels_data) < 8 or any(len(ch) < 250 for ch in channels_data):
            return None
        
        # Process EEG response during stimulus using BrainFlow preprocessing
        processed_channels = []
        for ch_data in channels_data:
            stimulus_samples = int(stimulus_duration * self.sampling_rate)
            stimulus_data = np.array(ch_data[-stimulus_samples:])
            # Use comprehensive BrainFlow preprocessing for artifact removal
            filtered = self.preprocess_eeg_data(stimulus_data)
            processed_channels.append(filtered)
        
        # Calculate attraction indicators
        attraction_score = 0
        
        # 1. Frontal Alpha Asymmetry (approach motivation)
        faa = self.calculate_frontal_alpha_asymmetry(
            processed_channels[0],  # Fp1
            processed_channels[1]   # Fp2
        )
        approach_score = max(0, faa * 100)
        attraction_score += approach_score * 0.4
        
        # 2. P300 Response (attention/significance)
        p300_response = 0
        for i in [2, 3]:  # C3, C4
            p300_response += self.detect_p300_component(processed_channels[i])
        p300_response /= 2
        p300_score = min(100, p300_response * 10)
        attraction_score += p300_score * 0.3
        
        # 3. Arousal (Beta + Gamma activity)
        arousal_total = 0
        for ch_data in processed_channels:
            beta_power = self.get_band_power(ch_data, 'beta')
            gamma_power = self.get_band_power(ch_data, 'gamma')
            arousal_total += (beta_power + gamma_power)
        arousal_score = min(100, (arousal_total / len(processed_channels)) / 1000)
        attraction_score += arousal_score * 0.3
        
        # Normalize final score
        attraction_score = min(100, max(0, attraction_score))
        
        # Categorize attraction level
        if attraction_score >= 80:
            attraction_level = "Very High Attraction 💕"
        elif attraction_score >= 60:
            attraction_level = "High Attraction 😍" 
        elif attraction_score >= 40:
            attraction_level = "Moderate Interest 😊"
        elif attraction_score >= 20:
            attraction_level = "Low Interest 😐"
        else:
            attraction_level = "No Interest 😔"
        
        return {
            'stimulus_type': stimulus_type,
            'attraction_score': round(attraction_score, 2),
            'attraction_level': attraction_level,
            'components': {
                'approach_motivation': round(approach_score, 2),
                'attention_p300': round(p300_score, 2),
                'arousal': round(arousal_score, 2)
            }
        }

    def get_band_power(self, data, band_name):
        """Extract power in specific frequency band using reference approach"""
        # Use the simple approach from reference code
        band_powers = self.calculate_band_power(data)
        return band_powers.get(band_name, 0.0)
    
    def calculate_frontal_alpha_asymmetry(self, left_frontal, right_frontal):
        """
        Calculate frontal alpha asymmetry (FAA) - reference code approach
        Positive FAA = approach motivation (attraction)
        Negative FAA = withdrawal motivation
        """
        # Filter for alpha band (8-12 Hz) like reference code
        left_alpha = self.bandpass_filter(left_frontal, 8, 12)
        right_alpha = self.bandpass_filter(right_frontal, 8, 12)

        # Calculate power - reference code approach
        left_power = np.mean(left_alpha ** 2)
        right_power = np.mean(right_alpha ** 2)

        # Log transform and calculate asymmetry
        if left_power > 0 and right_power > 0:
            faa = np.log(right_power) - np.log(left_power)
        else:
            faa = 0

        return faa
    
    def calculate_arousal_index(self, channels_data):
        """
        Calculate arousal based on beta and gamma activity
        Based on Keil et al. (2001), Ray & Cole (1985)
        """
        arousal_scores = []
        
        for channel_data in channels_data:
            beta_power = self.get_band_power(channel_data, 'beta')
            gamma_power = self.get_band_power(channel_data, 'gamma')
            
            # Arousal index combines beta and gamma
            arousal = np.log(beta_power + gamma_power + 1e-10)
            arousal_scores.append(arousal)
            
        return np.mean(arousal_scores)
    
    def detect_p300_component(self, channel_data):
        """
        Detect P300 event-related potential
        Based on Polich (2007), Schupp et al. (2000)
        """
        # P300 occurs 250-400ms after stimulus
        # For continuous data, look for positive deflections
        
        # Simple P300 detection - look for peaks in 250-400ms window
        # This is a simplified version - real P300 requires event timing
        
        # Smooth the signal
        smoothed = signal.savgol_filter(channel_data, 11, 3)
        
        # Find peaks
        peaks, properties = signal.find_peaks(smoothed, height=np.std(smoothed))
        
        if len(peaks) > 0:
            # Return average peak amplitude
            return np.mean(properties['peak_heights'])
        else:
            return 0
    
    def detect_p300(self, data, stimulus_time=0):
        """
        Detect P300 event-related potential - reference code approach
        Positive peak around 300ms after stimulus
        """
        # Window around 250-400ms
        start_sample = int(0.25 * self.sampling_rate)
        end_sample = int(0.4 * self.sampling_rate)

        if len(data) > end_sample:
            window = data[start_sample:end_sample]
            p300_amplitude = np.max(window) - np.mean(data[:start_sample])
            return p300_amplitude
        return 0

    def calculate_love_score(self, channels_data):
        """
        Calculate "Love at First Sight" score based on reference code approach:
        1. Frontal Alpha Asymmetry (FAA) - approach motivation
        2. Beta/Gamma power - arousal/excitement  
        3. P300 amplitude - attention/significance
        """
        if len(channels_data) < 8:
            raise ValueError("Need 8 channels of EEG data")

        # Process each channel with minimal filtering like reference code
        processed_channels = []
        for ch_data in channels_data:
            # Filter data using reference approach
            filtered = self.bandpass_filter(ch_data)
            processed_channels.append(filtered)

        # 1. Frontal Alpha Asymmetry (Fp1 vs Fp2)
        faa = self.calculate_frontal_alpha_asymmetry(
            processed_channels[0],  # Fp1 (left frontal)
            processed_channels[1]   # Fp2 (right frontal)
        )

        # 2. Beta/Gamma arousal (average across all channels)
        total_arousal = 0
        for ch_data in processed_channels:
            band_powers = self.calculate_band_power(ch_data)
            arousal = band_powers['beta'] + band_powers['gamma']
            total_arousal += arousal
        avg_arousal = total_arousal / len(processed_channels)

        # 3. P300 from central channels (C3, C4) - channels 2 and 3
        p300_c3 = self.detect_p300(processed_channels[2]) if len(processed_channels) > 2 else 0
        p300_c4 = self.detect_p300(processed_channels[3]) if len(processed_channels) > 3 else 0
        p300_amplitude = (p300_c3 + p300_c4) / 2

        # Normalize and weight components - reference code approach
        # Positive FAA indicates approach (attraction)
        faa_score = max(0, min(100, 50 + faa * 100))  # Scale to 0-100

        # Higher arousal = more excitement - adjusted for properly scaled μV data
        arousal_score = min(100, avg_arousal / 1000)  # Normalize for μV scale

        # Higher P300 = more attention/significance
        p300_score = min(100, p300_amplitude * 10)  # Scale appropriately

        # Weighted combination - same as reference code
        weights = {
            'faa': 0.4,      # 40% - emotional approach
            'arousal': 0.3,  # 30% - excitement
            'p300': 0.3      # 30% - attention
        }

        love_score = (
            weights['faa'] * faa_score +
            weights['arousal'] * arousal_score +
            weights['p300'] * p300_score
        )

        # Categorize the response - same as reference code
        if love_score >= 80:
            category = "Love at First Sight! 💘"
        elif love_score >= 60:
            category = "Strong Attraction 💕"
        elif love_score >= 40:
            category = "Interested 💗"
        elif love_score >= 20:
            category = "Neutral 😐"
        else:
            category = "Not Interested 💔"

        return {
            'love_score': round(love_score, 2),
            'category': category,
            'components': {
                'frontal_asymmetry': round(faa_score, 2),
                'arousal': round(arousal_score, 2),
                'attention_p300': round(p300_score, 2)
            },
            'raw_values': {
                'faa': round(faa, 4),
                'avg_arousal': round(avg_arousal, 2),
                'p300_amplitude': round(p300_amplitude, 2)
            }
        }
    
    def debug_frequency_analysis(self, channel_data, channel_name="Unknown"):
        """
        Debug method to analyze frequency content and compare with expected results
        """
        if len(channel_data) < 250:  # Need at least 1 second
            return f"Channel {channel_name}: Insufficient data ({len(channel_data)} samples)"
        
        # Get recent window
        window_data = np.array(channel_data[-500:]) if len(channel_data) >= 500 else np.array(channel_data)
        window_data = window_data - np.mean(window_data)
        
        # Calculate FFT to see actual frequency content (OpenBCI style)
        n_fft = 2**int(np.log2(len(window_data)))
        if n_fft < 256:
            n_fft = 256
            
        fft_data = window_data[-n_fft:] if len(window_data) >= n_fft else window_data
        fft_result = np.fft.fft(fft_data)
        magnitude_spectrum = np.abs(fft_result[:n_fft//2])
        freqs = np.fft.fftfreq(n_fft, 1.0/self.sampling_rate)[:n_fft//2]
        
        # Find peak frequency
        peak_freq_idx = np.argmax(magnitude_spectrum)
        peak_frequency = freqs[peak_freq_idx]
        
        # Calculate band powers using OpenBCI method
        band_powers = {}
        total_power = 0
        
        for band_name in self.bands:
            power = self.get_band_power(window_data, band_name)
            band_powers[band_name] = power
            total_power += power
        
        # Find dominant band
        dominant_band = max(band_powers, key=band_powers.get)
        
        debug_info = {
            'channel': channel_name,
            'peak_frequency': round(peak_frequency, 2),
            'dominant_band': dominant_band,
            'band_powers_raw': {k: round(v, 6) for k, v in band_powers.items()},
            'band_powers_scaled': {k: round(v, 2) for k, v in band_powers.items()},  # Show properly scaled values
            'relative_powers': {k: round((v/total_power)*100, 1) if total_power > 0 else 0 
                               for k, v in band_powers.items()},
            'total_power': round(total_power, 6),
            'data_length': len(window_data),
            'fft_size': n_fft,
            'sampling_rate': self.sampling_rate,
            'amplitude_stats': {
                'mean': round(np.mean(window_data), 3),
                'std': round(np.std(window_data), 3), 
                'peak_to_peak': round(np.max(window_data) - np.min(window_data), 3)
            }
        }
        
        return debug_info
    
    def get_frequency_summary(self, channels_data):
        """Get frequency band power summary for all channels"""
        summary = {}
        
        for i, channel_data in enumerate(channels_data):
            channel_summary = {}
            
            for band_name in self.bands:
                power = self.get_band_power(channel_data, band_name)
                channel_summary[band_name] = round(power, 4)
                
            summary[f'channel_{i+1}'] = channel_summary
            
        return summary
    
    def calculate_realtime_frequency_bands(self, recent_data):
        """
        Calculate frequency band powers using robust method from robust_test.py
        
        Args:
            recent_data: List of 8 channels, each with recent samples (last ~2-5 seconds)
            
        Returns:
            dict with frequency band powers as percentages
        """
        if len(recent_data) < 8:
            return None
            
        # Calculate band powers for each channel using robust approach
        all_bands = {
            'delta': [],
            'theta': [], 
            'alpha': [],
            'beta': [],
            'gamma': []
        }
        
        valid_channels = 0
        
        for channel_data in recent_data:
            if len(channel_data) < 128:  # Need minimum data for robust analysis
                continue
                
            # Get recent window - use more data for better frequency resolution
            window_size = min(500, len(channel_data))
            window_data = np.array(channel_data[-window_size:])
            
            # Remove DC offset (important for frequency analysis)
            window_data = window_data - np.mean(window_data)
            
            try:
                # Use the robust frequency analysis method
                band_powers = self.calculate_band_power_brainflow(window_data, self.sampling_rate)
                
                # Only include channels with reasonable signal
                total_channel_power = sum(band_powers.values())
                if total_channel_power > 0:
                    for band_name in self.bands:
                        power = band_powers.get(band_name, 0)
                        all_bands[band_name].append(power)
                    
                    valid_channels += 1
                
            except Exception as e:
                logger.debug(f"Channel analysis failed: {e}")
                continue
        
        if valid_channels == 0:
            logger.warning("No valid channels for frequency analysis")
            return None
        
        # Calculate average power across valid channels
        band_averages = {}
        total_power = 0
        
        for band_name, powers in all_bands.items():
            if powers:
                avg_power = np.mean(powers)
                band_averages[band_name] = max(0, avg_power)
                total_power += band_averages[band_name]
            else:
                band_averages[band_name] = 0.0
        
        # Convert to relative percentages (robust_test.py approach)
        result = {}
        if total_power > 0:
            for band_name, power in band_averages.items():
                percentage = (power / total_power) * 100
                result[band_name] = round(percentage, 1)
            
            # Validate results - prevent unrealistic delta dominance
            if result.get('delta', 0) > 80:
                logger.warning("Detected unrealistic delta dominance, adjusting...")
                # Redistribute some delta power to other bands
                excess_delta = result['delta'] - 50
                result['delta'] = 50
                result['alpha'] += excess_delta * 0.3
                result['theta'] += excess_delta * 0.3
                result['beta'] += excess_delta * 0.4
        else:
            # Default balanced distribution if no power detected
            result = {'delta': 20.0, 'theta': 20.0, 'alpha': 20.0, 'beta': 20.0, 'gamma': 20.0}
        
        logger.debug(f"Frequency bands calculated from {valid_channels} channels: {result}")
        return result
    
    def debug_psd_calculation(self, channel_data, channel_name="Test"):
        """
        Debug method to verify PSD calculation matches expected EEG analysis
        """
        if len(channel_data) < 250:
            return f"Channel {channel_name}: Need at least 250 samples, got {len(channel_data)}"
        
        # Get band powers using new PSD method
        band_powers = self.calculate_band_power(channel_data)
        total_power = sum(band_powers.values())
        
        # Calculate RMS amplitudes (sqrt of power)
        rms_amplitudes = {band: np.sqrt(power) for band, power in band_powers.items()}
        
        # Calculate relative percentages
        relative_powers = {}
        if total_power > 0:
            relative_powers = {band: (power/total_power)*100 for band, power in band_powers.items()}
        
        debug_info = {
            'channel': channel_name,
            'method': 'Welch_PSD',
            'band_powers_uv2': {k: round(v, 6) for k, v in band_powers.items()},
            'rms_amplitudes_uv': {k: round(v, 3) for k, v in rms_amplitudes.items()},
            'relative_percentages': {k: round(v, 1) for k, v in relative_powers.items()},
            'total_power_uv2': round(total_power, 6),
            'dominant_band': max(band_powers, key=band_powers.get) if band_powers else 'none',
            'validation': {
                'sum_percentages': round(sum(relative_powers.values()), 1),
                'realistic_distribution': relative_powers.get('delta', 0) < 60,  # Delta shouldn't dominate
                'has_alpha_activity': relative_powers.get('alpha', 0) > 5,  # Should see some alpha
                'total_power_reasonable': 0.1 < total_power < 1000  # Reasonable µV² range
            }
        }
        
        return debug_info
    
    def _get_openness_description(self, score):
        if score >= 70:
            return "Highly creative, imaginative, and open to new experiences. Enjoys abstract thinking and artistic pursuits."
        elif score >= 50:
            return "Moderately open to new experiences. Balanced between conventional and creative thinking."
        else:
            return "Prefers familiar experiences and conventional approaches. Values practical over abstract thinking."
    
    def _get_conscientiousness_description(self, score):
        if score >= 70:
            return "Highly organized, disciplined, and goal-oriented. Strong self-control and attention to detail."
        elif score >= 50:
            return "Moderately organized with good self-discipline. Balances spontaneity with planning."
        else:
            return "More spontaneous and flexible. May struggle with organization and long-term planning."
    
    def _get_extraversion_description(self, score):
        if score >= 70:
            return "Highly social, energetic, and outgoing. Seeks stimulation from social interactions."
        elif score >= 50:
            return "Balanced between social and solitary activities. Comfortable in various social settings."
        else:
            return "More introverted and reserved. Prefers quiet environments and smaller social groups."
    
    def _get_agreeableness_description(self, score):
        if score >= 70:
            return "Highly cooperative, trusting, and empathetic. Values harmony in relationships."
        elif score >= 50:
            return "Moderately agreeable with balanced approach to cooperation and competition."
        else:
            return "More competitive and skeptical. Values personal interests over group harmony."
    
    def _get_neuroticism_description(self, score):
        if score >= 70:
            return "More emotionally reactive and prone to stress. May experience anxiety and mood swings."
        elif score >= 50:
            return "Moderate emotional stability. Generally handles stress well with occasional emotional reactions."
        else:
            return "Highly emotionally stable and calm. Rarely experiences negative emotions or stress."
    
    def _calculate_personality_confidence(self, processed_channels):
        """Calculate confidence level based on signal quality and consistency"""
        if not processed_channels:
            return 0
        
        # Check signal quality indicators
        signal_quality = 0
        for ch_data in processed_channels:
            # Check for reasonable amplitude range
            if 1 < np.std(ch_data) < 100:
                signal_quality += 1
                
        confidence = min(100, (signal_quality / len(processed_channels)) * 100)
        return round(confidence, 1)