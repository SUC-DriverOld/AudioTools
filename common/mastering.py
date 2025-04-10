import numpy as np
import librosa
import time
import os
import json
import numba
import warnings
import pyloudnorm as pyln
import scipy.signal as signal
import soundfile as sf
from multiprocessing import Pool
from scipy import signal, interpolate
from statsmodels.nonparametric.smoothers_lowess import lowess
from suggestions import get_suggestions_for_genre

os.environ['NUMBA_DISABLE_INTEL_SVML'] = '1'
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILES_DIR = os.path.join(SCRIPT_DIR, "profiles")
REFERENCES_DIR = os.path.join(SCRIPT_DIR, "references")
CUSTOM_DIR = os.path.join(SCRIPT_DIR, "custom")

class Config:
    def __init__(self):
        self.threshold = 0.95
        self.knee_width = 0.1
        self.min_value = 1e-8
        self.max_piece_size = 44100 * 5  # 5 seconds
        self.internal_sample_rate = 44100
        self.lowess_frac = 0.15  # Increased for more smoothing
        self.lowess_it = 2  # Increased for more robustness
        self.lowess_delta = 0.1  # Increased for faster computation
        self.rms_correction_steps = 4
        self.limiter_threshold = 0.98
        self.limiter_knee_width = 0.3
        self.limiter_attack_ms = 1
        self.limiter_release_ms = 1500
        self.reference_file = None
        self.fft_size = 4096
        self.lin_log_oversampling = 4
        self.clipping_threshold = 0.99
        self.clipping_samples_threshold = 8
        self.high_shelf_freq = 8000
        self.high_shelf_gain_db_mid = -1.5  # Reduced from -2.5
        self.high_shelf_gain_db_side = -0.5  # Reduced from -0.8
        self.lowpass_cutoff = 18000  # Changed to 18kHz
        self.bypass_high_shelf = False
        self.compressor_threshold = -3
        self.compressor_ratio = 4
        self.compressor_knee_width = 6
        self.compressor_attack_ms = 5
        self.compressor_release_ms = 50
        self.limiter_thresholds = [0.95, 0.98]
        self.limiter_knee_widths = [0.1, 0.05]
        self.limiter_attack_times = [1, 0.5]
        self.limiter_release_times = [50, 25]
        self.limiter_mix = 0.95  # Reduced limited signal mix
        self.genre = None
        self.oversampling_factor = 4  # Increased from 4 to 8
        self.epsilon = 1e-8  # Small value to prevent division by zero
        self.bass_preservation_freq = 10
        self.bass_preservation_blend = 0.98
        self.apply_stereo_widening = True  # or False
        self.stereo_width_adjustment_factor = 0.2  # Adjusts 20% of the difference by default
        self.sample_rate = 44100  # Add this line
        self.loudness_option = "normal"  # Default to "normal"
        self.eq_style = "Neutral"  # Default to "Neutral"
        self.use_loudest_parts = True
        self.loudness_threshold = 0.4

def load_audio(file_path, config):
    print(f"Loading audio file: {file_path}")
    audio, sr = librosa.load(file_path, sr=None, mono=False)
    print(f"Loaded audio shape: {audio.shape}, max={np.max(np.abs(audio))}, min={np.min(np.abs(audio))}")
    return audio, sr

def save_audio(audio, file_path, sr):
    print(f"Saving audio to: {file_path}")
    print(f"Audio shape: {audio.shape}, max={np.max(np.abs(audio))}, min={np.min(np.abs(audio))}")
    print(f"Saving with sample rate: {sr}")
    sf.write(file_path, audio.T, sr, subtype='PCM_24')

def apply_dither(audio, bits=24):
    """
    Fast triangular dithering using optimized NumPy operations
    """
    # Single random operation instead of two separate ones
    noise = (2 * np.random.random(audio.shape) - 1) * (1.0 / (2**(bits-1)))
    return audio + noise

def lr_to_ms(array):
    mid = (array[0] + array[1]) * 0.5
    side = (array[0] - array[1]) * 0.5
    return mid, side

def ms_to_lr(mid, side):
    min_length = min(len(mid), len(side))
    mid = mid[:min_length]
    side = side[:min_length]
    
    left = mid + side
    right = mid - side
    return np.vstack((left, right))

def oversample(audio, factor):
    return signal.resample_poly(audio, factor, 1, axis=-1)

def downsample(audio, factor, sample_rate):
    filtered_audio = improved_anti_aliasing_filter(audio, sample_rate)
    return signal.resample_poly(filtered_audio, 1, factor, axis=-1)

def improved_anti_aliasing_filter(audio, sample_rate):
    nyquist = sample_rate / 2
    cutoff = 0.9 * nyquist
    sos = signal.butter(10, cutoff / nyquist, btype='low', output='sos')
    
    filtered_audio = np.zeros_like(audio)
    for channel in range(audio.shape[0]):
        filtered_audio[channel] = signal.sosfilt(sos, audio[channel])
    
    return filtered_audio

def apply_lowpass_filter(audio, config):
    nyquist = config.internal_sample_rate * config.oversampling_factor / 2
    sos = signal.butter(10, config.lowpass_cutoff / nyquist, btype='low', output='sos')
    
    filtered_audio = np.zeros_like(audio)
    for channel in range(audio.shape[0]):
        filtered_audio[channel] = signal.sosfilt(sos, audio[channel])
    
    return filtered_audio

def add_subtle_mid_channel_saturation(mid, config):
    # Define saturation parameters
    saturation_amount = 0.03  # Very subtle, adjust as needed
    blend_factor = 0.3  # Subtle blend, adjust as needed

    # Apply saturation to the entire mid channel
    saturated_mid = np.tanh(mid * (1 + saturation_amount)) / (1 + saturation_amount)

    # Blend the saturated mid with the original mid
    mid_enhanced = mid * (1 - blend_factor) + saturated_mid * blend_factor

    return mid_enhanced

def apply_peaking_filter(signal, freq, q, gain_db, sample_rate):
    w0 = 2 * np.pi * freq / sample_rate
    alpha = np.sin(w0) / (2 * q)
    A = 10 ** (gain_db / 40)

    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A

    b = [b0, b1, b2]
    a = [a0, a1, a2]

    return signal.filtfilt(b, a, signal, padlen=len(signal)-1)

def calculate_average_spectrum(audio, sample_rate, fft_size):
    _, _, specs = signal.stft(
        audio,
        sample_rate,
        window="hann",
        nperseg=fft_size,
        noverlap=fft_size // 2,
        boundary=None,
        padded=False,
    )
    return np.abs(specs).mean(axis=1)

def smooth_spectrum(spectrum, config):
    fft_size = (len(spectrum) - 1) * 2
    grid_linear = np.linspace(0, config.internal_sample_rate / 2, len(spectrum))
    grid_logarithmic = np.logspace(
        np.log10(4 * config.internal_sample_rate / fft_size),
        np.log10(config.internal_sample_rate / 2),
        (len(spectrum) - 1) * config.lin_log_oversampling + 1,
    )

    interpolator = interpolate.interp1d(grid_linear, spectrum, "cubic", bounds_error=False, fill_value="extrapolate")
    spectrum_log = interpolator(grid_logarithmic)

    spectrum_smoothed = lowess(
        spectrum_log,
        np.arange(len(spectrum_log)),
        frac=config.lowess_frac,
        it=config.lowess_it,
        delta=config.lowess_delta * len(spectrum_log),
    )[:, 1]

    interpolator = interpolate.interp1d(
        grid_logarithmic, spectrum_smoothed, "cubic", bounds_error=False, fill_value="extrapolate"
    )
    spectrum_filtered = interpolator(grid_linear)

    spectrum_filtered[0] = 0
    spectrum_filtered[1] = spectrum[1]

    return spectrum_filtered

def calculate_improved_rms(audio, sample_rate, config):
    def rms(x):
        return np.sqrt(np.mean(np.square(x)) + config.epsilon)

    # Define piece size (3 seconds)
    piece_size = 3 * sample_rate
    
    # Divide audio into pieces
    pieces = np.array_split(audio, max(1, len(audio) // piece_size))

    # Calculate RMS for each piece
    rms_values = np.array([rms(piece) for piece in pieces])

    # Calculate average RMS
    avg_rms = np.mean(rms_values)

    # Identify loudest pieces (above average RMS)
    loud_mask = rms_values >= avg_rms

    # Calculate final RMS using only the loudest pieces
    final_rms = rms(np.concatenate([pieces[i] for i in range(len(pieces)) if loud_mask[i]]))

    return final_rms

def match_rms_ms(target_mid, target_side, reference_mid, reference_side, sample_rate, config):
    def match_rms(target, reference):
        target_rms = calculate_improved_rms(target, sample_rate, config)
        reference_rms = calculate_improved_rms(reference, sample_rate, config)
        gain = reference_rms / target_rms
        return target * gain

    # Calculate the RMS of the target side channel
    target_side_rms = calculate_improved_rms(target_side, sample_rate, config)
    target_mid_rms = calculate_improved_rms(target_mid, sample_rate, config)

    # Define a threshold for considering the audio as "nearly mono"
    mono_threshold = 0.01  # Adjust this value as needed

    matched_mid = match_rms(target_mid, reference_mid)

    if target_side_rms / target_mid_rms < mono_threshold:
        print("Detected nearly mono audio. Skipping side channel RMS matching.")
        # Scale the side channel by the same factor as the mid channel
        mid_scale_factor = np.max(np.abs(matched_mid)) / np.max(np.abs(target_mid))
        matched_side = target_side * mid_scale_factor
    else:
        matched_side = match_rms(target_side, reference_side)

    return matched_mid, matched_side

def match_frequencies_ms(target_mid, target_side, reference_mid, reference_side, config):
    def calculate_average_fft(*args, sample_rate, fft_size, config):
        if len(args) == 1:
            # Single audio input
            audio = args[0]
            mid = side = audio
        elif len(args) == 2:
            # Separate mid and side inputs
            mid, side = args
        else:
            raise ValueError("Invalid number of arguments for calculate_average_fft")

        if config.use_loudest_parts:
            segment_length = sample_rate // 10  # 100ms segments
            num_segments = len(mid) // segment_length
            segments_mid = np.array_split(mid[:num_segments * segment_length], num_segments)
            
            # Calculate RMS based on mid channel only
            segment_rms = np.sqrt(np.mean(np.square(segments_mid), axis=1))
            
            loud_mask = segment_rms > (config.loudness_threshold * np.max(segment_rms))
            loud_segments_mid = [seg for seg, is_loud in zip(segments_mid, loud_mask) if is_loud]
            
            if len(loud_segments_mid) == 0:
                print("No segments above threshold, using entire audio.")
                return mid, side
            else:
                percentage_used = (len(loud_segments_mid) / len(segments_mid)) * 100
                print(f"Using {percentage_used:.2f}% of the audio (threshold: {config.loudness_threshold})")
            
            # Use the same mask for side channel
            segments_side = np.array_split(side[:num_segments * segment_length], num_segments)
            loud_segments_side = [seg for seg, is_loud in zip(segments_side, loud_mask) if is_loud]
            
            mid = np.concatenate(loud_segments_mid)
            side = np.concatenate(loud_segments_side)

        _, _, specs_mid = signal.stft(
            mid,
            sample_rate,
            window="hann",
            nperseg=fft_size,
            noverlap=fft_size // 2,
            boundary=None,
            padded=False,
        )
        _, _, specs_side = signal.stft(
            side,
            sample_rate,
            window="hann",
            nperseg=fft_size,
            noverlap=fft_size // 2,
            boundary=None,
            padded=False,
        )
        return np.abs(specs_mid).mean(axis=1), np.abs(specs_side).mean(axis=1)

    def smooth_spectrum(spectrum, config):
        fft_size = (len(spectrum) - 1) * 2
        grid_linear = np.linspace(0, config.internal_sample_rate / 2, len(spectrum))
        grid_logarithmic = np.logspace(
            np.log10(4 * config.internal_sample_rate / fft_size),
            np.log10(config.internal_sample_rate / 2),
            (len(spectrum) - 1) * config.lin_log_oversampling + 1,
        )

        interpolator = interpolate.interp1d(grid_linear, spectrum, "cubic", bounds_error=False, fill_value="extrapolate")
        spectrum_log = interpolator(grid_logarithmic)

        spectrum_smoothed = lowess(
            spectrum_log,
            np.arange(len(spectrum_log)),
            frac=config.lowess_frac,
            it=config.lowess_it,
            delta=config.lowess_delta * len(spectrum_log),
        )[:, 1]

        interpolator = interpolate.interp1d(
            grid_logarithmic, spectrum_smoothed, "cubic", bounds_error=False, fill_value="extrapolate"
        )
        spectrum_filtered = interpolator(grid_linear)

        spectrum_filtered[0] = 0
        spectrum_filtered[1] = spectrum[1]

        return spectrum_filtered

    def get_fir(target_mid, target_side, reference_mid, reference_side, config, is_side=False):
        target_fft_mid, target_fft_side = calculate_average_fft(
            target_mid, target_side, 
            sample_rate=config.internal_sample_rate * config.oversampling_factor, 
            fft_size=config.fft_size, 
            config=config
        )
        reference_fft_mid, reference_fft_side = calculate_average_fft(
            reference_mid, reference_side, 
            sample_rate=config.internal_sample_rate * config.oversampling_factor, 
            fft_size=config.fft_size, 
            config=config
        )
        
        target_fft = target_fft_side if is_side else target_fft_mid
        reference_fft = reference_fft_side if is_side else reference_fft_mid
        
        target_fft = np.maximum(target_fft, config.min_value)
        matching_fft = reference_fft / target_fft
        
        max_boost_db = 2 if is_side else 4  # Further reduced max boost
        matching_fft = np.clip(matching_fft, 10**(-max_boost_db/20), 10**(max_boost_db/20))
        
        matching_fft_filtered = smooth_spectrum(matching_fft, config)
        
        # Apply softer bass preservation
        bass_freq = config.bass_preservation_freq
        bass_blend = config.bass_preservation_blend
        
        # Create a gentler transition curve
        freqs = np.linspace(0, config.internal_sample_rate/2, len(matching_fft))
        bass_preservation = 1 - (1 - bass_blend) * (1 / (1 + (freqs / bass_freq)**2))
        
        # Apply the softer bass preservation
        matching_fft = 1 + (matching_fft - 1) * bass_preservation
        
        fir = np.fft.irfft(matching_fft_filtered)
        fir = np.fft.ifftshift(fir) * signal.windows.hann(len(fir))
        
        return fir

    mid_fir = get_fir(target_mid, target_side, reference_mid, reference_side, config, is_side=False)
    side_fir = get_fir(target_mid, target_side, reference_mid, reference_side, config, is_side=True)

    result_mid = signal.fftconvolve(target_mid, mid_fir, mode="same")
    result_side = signal.fftconvolve(target_side, side_fir, mode="same")

    def frequency_dependent_mix(freq, low_freq=10, high_freq=100000):
        return 0.99 * (1 - np.exp(-freq/low_freq)) * np.exp(-freq/high_freq)

    freqs = np.linspace(0, config.internal_sample_rate * config.oversampling_factor / 2, len(result_mid))
    mix = frequency_dependent_mix(freqs)
    result_mid = (1 - mix) * target_mid + mix * result_mid
    result_side = (1 - mix) * target_side + mix * result_side

    return result_mid, result_side

def gradual_level_correction(target_mid, target_side, reference_mid, reference_side, config):
    def apply_correction(target, reference):
        target_rms = np.sqrt(np.mean(target**2) + config.epsilon)
        reference_rms = np.sqrt(np.mean(reference**2) + config.epsilon)
        gain = np.clip(reference_rms / target_rms, 0.5, 2.0)
        return target * gain ** (1 / config.rms_correction_steps)

    for step in range(config.rms_correction_steps):
        target_mid = apply_correction(target_mid, reference_mid)
        target_side = apply_correction(target_side, reference_side)

    return target_mid, target_side

def rms(audio):
    return np.sqrt(np.mean(np.square(audio)))

def segment_audio(audio, config):
    segment_length = config.internal_sample_rate  # 1 second segments
    num_full_segments = len(audio) // segment_length
    segments = np.array_split(audio[:num_full_segments * segment_length], num_full_segments)
    return np.array(segments)

def analyze_stereo_width(mid, side):
    mid_energy = np.mean(np.square(mid))
    side_energy = np.mean(np.square(side))
    return side_energy / (mid_energy + side_energy + 1e-8)

def adjust_stereo_balance(mid, side, target_width, config):
    current_width = analyze_stereo_width(mid, side, config)
    
    width_difference = target_width - current_width
    if abs(width_difference) < 0.05:  # Less than 5% difference
        return mid, side
    
    adjustment_factor = 1 + config.stereo_width_adjustment_factor * width_difference
    
    # Only adjust the side channel
    adjusted_side = side * adjustment_factor
    
    # Ensure RMS remains constant
    original_rms = np.sqrt(np.mean(mid**2 + side**2))
    adjusted_rms = np.sqrt(np.mean(mid**2 + adjusted_side**2))
    rms_correction = original_rms / adjusted_rms
    
    return mid, adjusted_side * rms_correction

def finalize_stereo_image(target_mid, target_side, reference_mid, reference_side, config):
    print("Finalizing stereo image...")
    try:
        # Calculate stereo widths
        initial_width = analyze_stereo_width(target_mid, target_side)
        reference_width = analyze_stereo_width(reference_mid, reference_side)
        
        print(f"Initial stereo width: {initial_width:.4f}")
        print(f"Reference stereo width: {reference_width:.4f}")
        
        # Check for near-mono signal
        if initial_width < 0.02:
            print("Input signal is nearly mono. Skipping stereo adjustment.")
            return ms_to_lr(target_mid, target_side)
        
        # Calculate initial RMS
        initial_rms = np.sqrt(np.mean(target_mid**2 + target_side**2))
        
        # Calculate width difference and apply adjustment with upper limit
        width_difference = reference_width - initial_width
        max_adjustment = 0.15  # 15% maximum adjustment
        
        if width_difference > 0:
            adjustment_factor = 1 + min(width_difference / initial_width, max_adjustment)
            adjusted_side = target_side * adjustment_factor
            print(f"Applied {(adjustment_factor - 1) * 100:.1f}% stereo width increase.")
        else:
            print("No stereo width increase needed.")
            adjusted_side = target_side
        
        # Convert to left-right
        result = ms_to_lr(target_mid, adjusted_side)
        
        # RMS matching
        current_rms = np.sqrt(np.mean(result**2))
        rms_adjustment = initial_rms / current_rms
        result *= rms_adjustment
        
        final_mid, final_side = lr_to_ms(result)
        final_width = analyze_stereo_width(final_mid, final_side)
        
        print(f"Initial stereo width: {initial_width:.4f}")
        print(f"Final stereo width: {final_width:.4f}")
        print(f"Stereo image finalization complete. Output max amplitude: {np.max(np.abs(result)):.4f}")
        
        return result
    except Exception as e:
        print(f"Error during stereo image finalization: {str(e)}")
        return ms_to_lr(target_mid, target_side)

def process_band(args):
    mid, side, target_balance, config, band = args
    if band[0] == 0:
        sos = signal.butter(10, band[1], btype='lowpass', fs=config.internal_sample_rate, output='sos')
    else:
        sos = signal.butter(10, band, btype='bandpass', fs=config.internal_sample_rate, output='sos')
    
    band_mid = signal.sosfilt(sos, mid)
    band_side = signal.sosfilt(sos, side)
    
    return adjust_stereo_balance(band_mid, band_side, target_balance, config)

def frequency_band_stereo_adjust(mid, side, target_balance, config):
    bands = [0, 250, 8000, config.internal_sample_rate // 2]  # Reduced to 3 bands
    
    with Pool() as pool:
        results = pool.map(process_band, [(mid, side, target_balance, config, (bands[i], bands[i+1])) for i in range(len(bands) - 1)])
    
    adjusted_mid = sum(result[0] for result in results)
    adjusted_side = sum(result[1] for result in results)
    
    return adjusted_mid, adjusted_side

    
@numba.jit(nopython=True)
def process_chunk(chunk, threshold, knee_width, attack_coeff, release_coeff):
    x = np.abs(chunk)
    gain_reduction = np.maximum(x / threshold, 1.0)
    
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if threshold - knee_width / 2 < x[i, j] < threshold + knee_width / 2:
                gain_reduction[i, j] = 1.0 + ((x[i, j] - (threshold - knee_width / 2)) / knee_width) ** 2 * (x[i, j] / threshold - 1.0) / 2

    smoothed_gain = np.zeros_like(gain_reduction)
    smoothed_gain[:, 0] = gain_reduction[:, 0]

    for i in range(gain_reduction.shape[0]):
        for j in range(1, gain_reduction.shape[1]):
            if gain_reduction[i, j] > smoothed_gain[i, j-1]:
                smoothed_gain[i, j] = attack_coeff * smoothed_gain[i, j-1] + (1 - attack_coeff) * gain_reduction[i, j]
            else:
                smoothed_gain[i, j] = release_coeff * smoothed_gain[i, j-1] + (1 - release_coeff) * gain_reduction[i, j]

    return chunk / smoothed_gain

def soft_knee_compressor(audio, config):
    threshold = -6.0  # dB, slightly lower for more gentle compression
    ratio = 2.5  # Gentler ratio
    knee_width = 6.0
    attack_ms = 10.0
    release_ms = 500.0  # Longer release for smoother action

    threshold_linear = 10 ** (threshold / 20)
    
    attack_coeff = np.exp(-1 / (attack_ms * config.internal_sample_rate * config.oversampling_factor / 1000))
    release_coeff = np.exp(-1 / (release_ms * config.internal_sample_rate * config.oversampling_factor / 1000))
    
    chunk_size = 4096  # Matching the FFT size from the reference
    result = np.zeros_like(audio)
    
    for i in range(0, audio.shape[1], chunk_size):
        chunk = audio[:, i:i+chunk_size]
        compressed_chunk = process_chunk(chunk, threshold_linear, knee_width, attack_coeff, release_coeff)
        
        # Apply compression ratio
        compressed_chunk = np.sign(compressed_chunk) * (np.abs(compressed_chunk) ** (1/ratio))
        result[:, i:i+chunk_size] = compressed_chunk
    
    return result



@numba.jit(nopython=True)
def process_multi_stage_chunk(chunk, thresholds, knee_widths, attack_coeffs, release_coeffs):
    result = chunk.copy()
    for threshold, knee_width, attack_coeff, release_coeff in zip(thresholds, knee_widths, attack_coeffs, release_coeffs):
        result = process_chunk(result, threshold, knee_width, attack_coeff, release_coeff)
    return result


def envelope_follower(x, attack_samples, release_samples):
    env = np.zeros_like(x)
    for i in range(1, x.shape[1]):
        env[:, i] = np.maximum(x[:, i], env[:, i-1] + (x[:, i] - env[:, i-1]) * (1 - np.exp(-1 / release_samples)))
    return env

@numba.jit(nopython=True, parallel=True)
def process_limiter_stage(audio, threshold, knee_width, attack_ms, release_ms, sample_rate):
    attack_samples = int(attack_ms * sample_rate / 1000)
    release_samples = int(release_ms * sample_rate / 1000)
    
    # Pre-calculate exponential terms
    attack_coeff = 1 - np.exp(-1 / attack_samples)
    release_coeff = 1 - np.exp(-1 / release_samples)
    
    # Calculate gain reduction
    gain_reduction = np.maximum(1, np.abs(audio) / threshold)
    
    # Apply knee
    knee_range = knee_width / 2
    soft_knee = np.clip((gain_reduction - (1 - knee_range)) / knee_width, 0, 1)
    gain_reduction = 1 + soft_knee**2 * (gain_reduction - 1)
    
    # Calculate smoothed gain reduction using optimized envelope follower
    smoothed_gain_reduction = np.zeros_like(gain_reduction)
    
    for i in numba.prange(audio.shape[0]):
        env = 0
        for j in range(audio.shape[1]):
            if gain_reduction[i, j] > env:
                env += (gain_reduction[i, j] - env) * attack_coeff
            else:
                env += (gain_reduction[i, j] - env) * release_coeff
            smoothed_gain_reduction[i, j] = env
    
    # Apply gain reduction only where necessary, avoiding division by zero
    epsilon = 1e-10  # Small value to prevent division by zero
    result = np.where(smoothed_gain_reduction > 1, 
                      audio / np.maximum(smoothed_gain_reduction, epsilon), 
                      audio)
    
    return result

def process_limiter_stage_with_logging(audio, threshold, knee_width, attack_ms, release_ms, sample_rate):
    result = process_limiter_stage(audio, threshold, knee_width, attack_ms, release_ms, sample_rate)
    
    print(f"Limiter stage - Threshold: {threshold}, Max input: {np.max(np.abs(audio))}")
    print(f"Max gain reduction: {np.max(result / (audio + np.finfo(audio.dtype).eps))}")
    print(f"Max output: {np.max(np.abs(result))}")
    
    return result

def multi_stage_limiter(audio, config):
     # First stage: existing implementation
    threshold1 = 10 ** (-0.6 / 20)  # -0.6 dB 
    knee_width1 = 0.1
    attack_time1 = 1.0  # ms
    release_time1 = 200.0  # ms 
    
    # Second stage: slightly more aggressive
    threshold2 = 10 ** (-0.5 / 20)  # -0.5 dB
    knee_width2 = 0.1
    attack_time2 = 3.0  # ms
    release_time2 = 900.0  # ms 
    
    sample_rate = config.internal_sample_rate * config.oversampling_factor
    
    # First stage (your existing implementation)
    result = process_limiter_stage(
        audio,
        threshold1,
        knee_width1,
        attack_time1,
        release_time1,
        sample_rate
    )
    
    # Second stage
    result = process_limiter_stage(
        result,
        threshold2,
        knee_width2,
        attack_time2,
        release_time2,
        sample_rate
    )
    
    return result

def load_genre_profile(genre):
    profile_path = os.path.join(PROFILES_DIR, f"{genre}.json")
    if not os.path.exists(profile_path):
        profile_path = os.path.join(CUSTOM_DIR, genre, f"{genre}.json")
    assert os.path.exists(profile_path), f"Genre profile '{genre}.json' not found"
    
    with open(profile_path, "r") as f:
        profile = json.load(f)
    
    # If the profile has a 'features' key, return its contents
    # Otherwise, return the whole profile (for new flat structure)
    return profile.get('features', profile)

def calculate_lufs(audio, sr):
    # Ensure audio is in float32 format
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    # Ensure audio is in the range -1.0 to 1.0
    if audio.max() > 1.0 or audio.min() < -1.0:
        audio = audio / np.max(np.abs(audio))
    
    # Create BS.1770 meter
    meter = pyln.Meter(sr)
    
    # Ensure audio is in (samples, channels) shape
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    elif audio.shape[0] == 2 and audio.shape[1] > 2:
        audio = audio.T
    
    # Calculate integrated loudness
    loudness = meter.integrated_loudness(audio)
    return loudness

def process_audio(target, reference, step, config, genre_profile=None):
    start_time = time.time()
    print(f"Input target shape: {target.shape}, max={np.max(np.abs(target))}, min={np.min(np.abs(target))}")
    
    log_audio_metrics(target, "Target (Before Processing)", config)
    log_audio_metrics(reference, "Reference" if genre_profile is None else "Synthetic Reference", config)
    
    # Calculate and log initial LUFS
    target_lufs = calculate_lufs(target, config.internal_sample_rate)
    print(f"Target LUFS before processing: {target_lufs:.2f}")

    if genre_profile is None:
        reference_lufs = calculate_lufs(reference, config.internal_sample_rate)
        print(f"Reference LUFS: {reference_lufs:.2f}")
    else:
        synthetic_reference_lufs = calculate_lufs(reference, config.internal_sample_rate)
        print(f"Synthetic Reference LUFS: {synthetic_reference_lufs:.2f}")
        print(f"Genre profile LUFS: {genre_profile['lufs']:.2f}")
    
    def calculate_rms(audio):
        return np.sqrt(np.mean(np.square(audio)))

    # Calculate and log initial RMS values
    target_rms = calculate_rms(target)
    if genre_profile is None:
        reference_rms = calculate_rms(reference)
        print(f"Initial RMS - Reference: {reference_rms:.6f}, Target: {target_rms:.6f}")
    else:
        synthetic_reference_rms = calculate_rms(reference)
        print(f"Initial RMS - Synthetic Reference: {synthetic_reference_rms:.6f}, Target: {target_rms:.6f}")
        print(f"Genre Profile Initial RMS: {genre_profile['initial_rms']:.6f}")

    # Store the initial loudness ratio
    initial_loudness_ratio = 1.0
    if config.loudness_option == "dynamic":
        initial_loudness_ratio = 0.80
        print("Applying dynamic loudness: reducing volume by 20%")
    elif config.loudness_option == "soft":
        initial_loudness_ratio = 0.70
        print("Applying soft loudness: reducing volume by 30%")
    elif config.loudness_option == "loud":
        initial_loudness_ratio = 1.20
        print("Applying loud loudness: increasing volume by 20%")
    else:
        print("Applying normal loudness: no adjustment")

    # Apply initial loudness adjustment
    target *= initial_loudness_ratio

    # Recalculate RMS and LUFS after loudness adjustment
    if config.loudness_option != "normal":
        target_rms = calculate_rms(target)
        target_lufs = calculate_lufs(target, config.internal_sample_rate)
        print(f"After loudness adjustment - Target RMS: {target_rms:.6f}, Target LUFS: {target_lufs:.2f}")
    
    # Ensure input audio is in 64-bit float precision
    target = target.astype(np.float64)
    reference = reference.astype(np.float64)
    
    # Oversample
    target = oversample(target, config.oversampling_factor)
    reference = oversample(reference, config.oversampling_factor)
    
    oversampled_rate = config.internal_sample_rate * config.oversampling_factor
    print(f"After oversampling: target_max={np.max(np.abs(target))}")
    
    # Apply anti-aliasing filter
    target = improved_anti_aliasing_filter(target, oversampled_rate)
    reference = improved_anti_aliasing_filter(reference, oversampled_rate)
    print(f"After anti-aliasing: target_max={np.max(np.abs(target))}, reference_max={np.max(np.abs(reference))}")
    
    # Convert to mid-side
    target_mid, target_side = lr_to_ms(target)
    reference_mid, reference_side = lr_to_ms(reference)
    print(f"After mid-side conversion: target_mid_max={np.max(np.abs(target_mid))}, target_side_max={np.max(np.abs(target_side))}")
    print(f"reference_mid_max={np.max(np.abs(reference_mid))}, reference_side_max={np.max(np.abs(reference_side))}")
    
    # Apply processing steps
    if step >= 1:
        print("===== Step 1: RMS Matching and Saturation =====")
        if genre_profile is None:
            target_mid, target_side = match_rms_ms(target_mid, target_side, reference_mid, reference_side, oversampled_rate, config)
        else:
            target_mid, target_side = match_rms_ms(target_mid, target_side, reference_mid, reference_side, oversampled_rate, config)
        print(f"After RMS matching: target_mid_max={np.max(np.abs(target_mid))}, target_side_max={np.max(np.abs(target_side))}")
        
        # Calculate and log RMS after matching
        processed_mid_side = ms_to_lr(target_mid, target_side)
        processed_rms = calculate_improved_rms(processed_mid_side, oversampled_rate, config)
        reference_rms = calculate_improved_rms(reference, oversampled_rate, config)
        print(f"After RMS matching - Reference RMS: {reference_rms:.6f}, Processed RMS: {processed_rms:.6f}")
        
        print("After RMS Matching:")
        log_audio_metrics(ms_to_lr(target_mid, target_side), "Target", config)
    
    if step >= 2:
        print("===== Step 2: Frequency Matching and EQ Style =====")
        # Apply subtle saturation to mid channel
        target_mid = add_subtle_mid_channel_saturation(target_mid, config)
        print(f"After mid channel saturation: target_mid_max={np.max(np.abs(target_mid))}, target_side_max={np.max(np.abs(target_side))}")

        if genre_profile is None:
            target_mid, target_side = match_frequencies_ms(target_mid, target_side, reference_mid, reference_side, config)
        else:
            target_mid, target_side = match_frequencies_ms(target_mid, target_side, reference_mid, reference_side, config)
        print(f"After frequency matching: target_mid_max={np.max(np.abs(target_mid))}, target_side_max={np.max(np.abs(target_side))}")
        
        # Apply high-pass filter to side channel
        target_side = low_shelf_tighten(target_side, config.internal_sample_rate, cutoff_freq=100, gain=0.5, order=4)
        print(f"After side channel high-pass: target_side_max={np.max(np.abs(target_side))}")

        # Apply EQ style after frequency matching
        if config.eq_style != "Neutral":
            print(f"Applying {config.eq_style} EQ style")
            target_mid, target_side = apply_eq_style(target_mid, target_side, config.internal_sample_rate, config.eq_style)
        
        # Apply lowpass filter
        result = ms_to_lr(target_mid, target_side)
        result = apply_lowpass_filter(result, config)
        target_mid, target_side = lr_to_ms(result)
        print("After Lowpass Filter:")
        log_audio_metrics(result, "Target", config)
    
    if step >= 3:
        print("===== Step 3: Stereo Width Adjustment and Level Correction =====")
        if genre_profile is None:
            target_mid, target_side = gradual_level_correction(target_mid, target_side, reference_mid, reference_side, config)
        else:
            target_mid, target_side = gradual_level_correction(target_mid, target_side, reference_mid, reference_side, config)
        print(f"After gradual level correction: target_mid_max={np.max(np.abs(target_mid))}, target_side_max={np.max(np.abs(target_side))}")
        
        print("After Level Correction:")
        log_audio_metrics(ms_to_lr(target_mid, target_side), "Target", config)
    
    if step >= 4:
        print("===== Step 4: Stereo Balance and Frequency Band Adjustments =====")
        if genre_profile is None:
            result = finalize_stereo_image(target_mid, target_side, reference_mid, reference_side, config)
        else:
            result = finalize_stereo_image(target_mid, target_side, reference_mid, reference_side, config)
        print(f"After stereo finalization: result_max={np.max(np.abs(result))}")
        
        print("After Stereo Adjustment:")
        log_audio_metrics(result, "Target", config)
    else:
        result = ms_to_lr(target_mid, target_side)
    
    if step >= 5:
        print("===== Step 5: Final Mastering =====")
        print("Applying final mastering processes...")
        print(f"Before multi-stage limiting: result_max={np.max(np.abs(result)):.4f}")
        
        # Before final limiting, reapply the loudness ratio
        result *= initial_loudness_ratio
        
        result = multi_stage_limiter(result, config)
        print(f"After multi-stage limiting: result_max={np.max(np.abs(result)):.4f}")

    # Apply final hard limiting before downsampling
    result = np.clip(result, -0.95, 0.95)
    print(f"After final hard limiting (before downsampling): result_max={np.max(np.abs(result)):.4f}")

    # Downsample (now outside of step 5)
    result = downsample(result, config.oversampling_factor, oversampled_rate)
    print(f"After downsampling: result_max={np.max(np.abs(result)):.4f}")

    # Add normalization step
    target_peak_db = -0.5
    current_peak_db = 20 * np.log10(np.max(np.abs(result)))
    if current_peak_db < target_peak_db:
        gain_db = target_peak_db - current_peak_db
        gain_linear = 10 ** (gain_db / 20)
        result *= gain_linear
        print(f"Normalized to {target_peak_db} dB. Applied gain: {gain_db:.2f} dB")
    else:
        print(f"Current peak ({current_peak_db:.2f} dB) is already at or above target peak. No additional normalization applied.")

    print(f"Final output: result_max={np.max(np.abs(result)):.4f}")

    # Calculate and log initial LUFS before True Peak limiting
    initial_lufs = calculate_lufs(result, config.internal_sample_rate)
    print(f"LUFS before True Peak limiting: {initial_lufs:.2f}")
    
    # Apply simple dithering before final True Peak limiting
    result = apply_dither(result)
    
    # Calculate True Peak
    true_peak_db = calculate_true_peak(result, config.internal_sample_rate)
    print(f"Initial True Peak (dBTP): {true_peak_db:.2f}")

    # Apply True Peak limiting if necessary
    if true_peak_db > -0.4:
        gain_reduction_db = -0.4 - true_peak_db
        gain_factor = 10 ** (gain_reduction_db / 20)
        result *= gain_factor
        print(f"Applied True Peak limiting. Gain reduction: {gain_reduction_db:.2f} dB")
        
        # Recalculate True Peak after limiting
        true_peak_db = calculate_true_peak(result, config.internal_sample_rate)
        print(f"Final True Peak after limiting (dBTP): {true_peak_db:.2f}")
    else:
        print("True Peak is already below -0.4 dBTP. No additional limiting applied.")

    print(f"Final output: result_max={np.max(np.abs(result)):.4f}")

    # Genre-specific loudness adjustment
    if genre_profile and genre_profile['genre'] in ['Piano', 'Orchestral', 'Speech']:
        if config.loudness_option == "dynamic":
            # Reduce gain to simulate -0.6 dB true peak
            result *= 10 ** (-0.3 / 20)  # Additional -0.3 dB
        elif config.loudness_option == "soft":
            # Reduce gain to simulate -0.9 dB true peak
            result *= 10 ** (-0.6 / 20)  # Additional -0.6 dB

    # Calculate and log final LUFS after all processing
    final_lufs = calculate_lufs(result, config.internal_sample_rate)
    print(f"Final LUFS after all processing: {final_lufs:.2f}")

    print("After Final Processing:")
    log_audio_metrics(result, "Target", config)
    print("==============================")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total processing time: {total_time:.2f} seconds")
    
    return result

def calculate_true_peak(audio, sample_rate):
    # Upsample by a factor of 4 for true peak calculation
    upsampled = signal.resample_poly(audio, 4, 1, axis=-1)
    peak = np.max(np.abs(upsampled))
    true_peak_db = 20 * np.log10(peak)
    return true_peak_db

def log_audio_metrics(audio, name, config):
    print(f"--- {name} Metrics ---")
    print(f"Shape: {audio.shape}")
    print(f"Max amplitude: {np.max(np.abs(audio)):.4f}")
    lufs = calculate_lufs(audio, config.internal_sample_rate)
    print(f"LUFS: {lufs:.2f}")
    
    mid, side = lr_to_ms(audio)
    print(f"Mid RMS: {np.sqrt(np.mean(np.square(mid))):.4f}")
    print(f"Side RMS: {np.sqrt(np.mean(np.square(side))):.4f}")
    
    stereo_width = analyze_stereo_width(mid, side)
    print(f"Stereo Width: {stereo_width:.4f}")

def apply_guardrails(initial_value, suggested_value):
    max_increase = initial_value * 1.25  # 25% increase limit
    min_decrease = initial_value * 0.95  # 5% decrease limit
    return max(min(suggested_value, max_increase), min_decrease)

def create_reference_from_profile(genre_profile, config):
    genre = genre_profile['genre']
    secured_genre_file = os.path.join(REFERENCES_DIR, f"{genre}.mp3")
    if not os.path.exists(secured_genre_file):
        secured_genre_file = os.path.join(CUSTOM_DIR, genre, f"{genre}.mp3")
    assert os.path.exists(secured_genre_file), f"Reference file not found: {genre}.mp3"
    
    print(f"Loading reference file: {secured_genre_file}")
    audio, sr = librosa.load(secured_genre_file, sr=None, mono=False)
    
    # Ensure the audio is stereo
    if audio.ndim == 1:
        audio = np.tile(audio, (2, 1))
    elif audio.shape[0] > 2:
        audio = audio[:2]
    
    # Downsample by factor of 4
    target_sr = sr // 4
    audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr)
    
    # Get model suggestions
    model_suggestions = get_suggestions_for_genre(genre)
    print(f"Model suggestions for {genre}:")
    print(f"RMS Mid: {model_suggestions['rms_mid']:.4f}")
    print(f"RMS Side: {model_suggestions['rms_side']:.4f}")
    print(f"Stereo Width: {model_suggestions['stereo_width']:.4f}")
    
    # Calculate Initial RMS
    mid, side = lr_to_ms(audio)
    initial_rms_mid = np.sqrt(np.mean(mid**2))
    initial_rms_side = np.sqrt(np.mean(side**2))
    print(f"Initial RMS - Mid: {initial_rms_mid:.4f}, Side: {initial_rms_side:.4f}")
    
    # Apply guardrails
    suggested_rms_mid = apply_guardrails(initial_rms_mid, model_suggestions['rms_mid'])
    suggested_rms_side = apply_guardrails(initial_rms_side, model_suggestions['rms_side'])
    print(f"After guardrails - RMS Mid: {suggested_rms_mid:.4f}, RMS Side: {suggested_rms_side:.4f}")
    
    # Adjust RMS
    mid_factor = suggested_rms_mid / initial_rms_mid
    side_factor = suggested_rms_side / initial_rms_side
    
    adjusted_mid = mid * mid_factor
    adjusted_side = side * side_factor
    
    adjusted_audio = ms_to_lr(adjusted_mid, adjusted_side)
    
    # Calculate final RMS values
    final_mid, final_side = lr_to_ms(adjusted_audio)
    final_rms_mid = np.sqrt(np.mean(final_mid**2))
    final_rms_side = np.sqrt(np.mean(final_side**2))
    print(f"Final RMS - Mid: {final_rms_mid:.4f}, Side: {final_rms_side:.4f}")
    
    return adjusted_audio, target_sr

def boost_band(audio, sample_rate, low_cutoff, high_cutoff, gain, order=4):
    nyquist = 0.5 * sample_rate
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    sos = signal.butter(order, [low, high], btype='bandpass', output='sos')
    filtered_signal = signal.sosfilt(sos, audio)
    return audio + (filtered_signal * (gain - 1))

def high_shelf_boost(audio, sample_rate, cutoff_freq, gain, order=4):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    sos = signal.butter(order, normal_cutoff, btype='highpass', output='sos')
    filtered_signal = signal.sosfilt(sos, audio)
    return audio + (filtered_signal * (gain - 1))

def low_shelf_tighten(audio, sample_rate, cutoff_freq, gain, order=4):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    sos = signal.butter(order, normal_cutoff, btype='lowpass', output='sos')
    filtered_signal = signal.sosfilt(sos, audio)
    return audio * gain + filtered_signal * (1 - gain)

def apply_eq_style(mid, side, sample_rate, eq_style):
    print(f"Applying {eq_style} EQ style")
    if eq_style == "Warm":
        # Mid channel processing
        mid = boost_band(mid, sample_rate, low_cutoff=200, high_cutoff=300, gain=1.19, order=4)  # +1.5dB
        mid = boost_band(mid, sample_rate, low_cutoff=2000, high_cutoff=3000, gain=0.89, order=4)  # -1dB
        
        # Side channel processing
        side = boost_band(side, sample_rate, low_cutoff=3500, high_cutoff=4500, gain=0.92, order=4)  # -0.7dB
        side = boost_band(side, sample_rate, low_cutoff=150, high_cutoff=210, gain=1.06, order=4)  # +0.5dB

    elif eq_style == "Bright":
        # Mid channel processing
        mid = boost_band(mid, sample_rate, low_cutoff=2700, high_cutoff=3300, gain=1.19, order=4)  # +1.5dB
        mid = boost_band(mid, sample_rate, low_cutoff=500, high_cutoff=600, gain=1.08, order=4)  # +0.7dB
        
        # Side channel processing
        side = boost_band(side, sample_rate, low_cutoff=200, high_cutoff=300, gain=0.92, order=4)  # -0.7dB
        side = high_shelf_boost(side, sample_rate, cutoff_freq=8000, gain=1.19, order=4)  # +1.5dB

    elif eq_style == "Fusion":
        # Combination of both Warm and Bright
        # Mid channel processing
        mid = boost_band(mid, sample_rate, low_cutoff=200, high_cutoff=300, gain=1.10, order=4)  # Moderate low boost
        mid = boost_band(mid, sample_rate, low_cutoff=2700, high_cutoff=3300, gain=1.15, order=4)  # Boost similar to bright

        # Side channel processing
        side = boost_band(side, sample_rate, low_cutoff=200, high_cutoff=300, gain=0.97, order=4)  # Slight cut
        side = high_shelf_boost(side, sample_rate, cutoff_freq=8000, gain=1.12, order=4)  # Slight high-end boost

    print(f"After EQ - Mid max: {np.max(np.abs(mid)):.4f}, Side max: {np.max(np.abs(side)):.4f}")
    return mid, side

def master_audio(input_file, output_file, config, eq_style, is_preview=False):
    start_time = time.time()
    print(f"Master audio function called with: input_file={input_file}, output_file={output_file}, reference_file={config.reference_file}, eq_style={eq_style}, is_preview={is_preview}")
    config.eq_style = eq_style
    
    load_start = time.time()
    target, sr = load_audio(input_file, config)
    load_end = time.time()
    print(f"Audio loading time: {load_end - load_start:.2f} seconds")
    
    print(f"Original audio length: {len(target[0])} samples")
    print(f"Original audio duration: {len(target[0]) / sr:.2f} seconds")
    
    if is_preview:
        preview_start = time.time()
        preview_duration = 30  # seconds
        preview_samples = min(sr * preview_duration, target.shape[1])
        target = target[:, :preview_samples]
        preview_end = time.time()
        print(f"Preview creation time: {preview_end - preview_start:.2f} seconds")
        print(f"Processing preview: {preview_samples} samples")
        print(f"Preview duration: {preview_samples / sr:.2f} seconds")
    else:
        print(f"Processing full track: {len(target[0])} samples")

    if config.reference_file:
        print(f"Using reference file: {config.reference_file}")
        reference, _ = load_audio(config.reference_file, config)
        genre_profile = None
    elif config.genre:
        print(f"Using genre profile: {config.genre}")
        genre_profile = load_genre_profile(config.genre)
        reference, _ = create_reference_from_profile(genre_profile, config)
        log_audio_metrics(reference, "Reference from Genre", config)
    else:
        raise ValueError("Either genre or reference file must be specified")
    
    process_start = time.time()
    processed_audio = process_audio(target, reference, 5, config, genre_profile)
    process_end = time.time()
    print(f"Audio processing time: {process_end - process_start:.2f} seconds")
    
    save_start = time.time()
    save_audio(processed_audio, output_file, sr)    
    save_end = time.time()
    print(f"Audio saving time: {save_end - save_start:.2f} seconds")
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total Python processing time: {total_time:.2f} seconds")
    print("Mastering completed")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audio Mastering Tool")
    parser.add_argument("input_file", help="Path to the input audio file")
    parser.add_argument("-o", "--output_folder", help="Folder to save the output audio file, if not specified, the output will be saved in the same folder as the input file")
    parser.add_argument("-f", "--format", choices=["wav", "flac", "mp3"], default="wav", help="Output audio format")
    parser.add_argument("-r", "--reference", help="Path to a custom reference audio file")
    parser.add_argument("-g", "--genre", help="Genre profile to use for mastering (not including the extension), json file must be present in the genres folder. For example, 'Ambient'")
    parser.add_argument("-l", "--loudness", choices=['soft', 'dynamic', 'normal', 'loud'], default="normal", help="Loudness option")
    parser.add_argument("-eq", "--eq-profile", choices=["Neutral", "Warm", "Bright", "Fusion"], default="Neutral", help="EQ profile to use for mastering")
    parser.add_argument("-p", "--preview", action="store_true", help="Process only the preview, 30 seconds of the audio file")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    all_genres = [g.replace(".json", "") for g in os.listdir(PROFILES_DIR) if g.endswith(".json")]
    if os.path.exists(CUSTOM_DIR):
        all_genres += [f for f in os.listdir(CUSTOM_DIR) if os.path.isdir(os.path.join(CUSTOM_DIR, f))]

    if args.genre and args.genre not in all_genres:
        raise ValueError(f"Genre profile not found: {args.genre}")

    config = Config()
    config.loudness_option = args.loudness
    if args.reference:
        if not os.path.exists(args.reference):
            raise FileNotFoundError(f"Reference file not found: {args.reference}")
        config.reference_file = args.reference
        print(f"Using custom reference file: {config.reference_file}")
    elif args.genre:
        config.genre = args.genre
        print(f"Using genre profile: {config.genre}")
    else:
        raise ValueError("Either genre or reference file must be specified")
    
    if not args.output_folder:
        output_folder = os.path.dirname(args.input_file)
    else:
        output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)
    extension = os.path.splitext(os.path.basename(args.input_file))[-1]
    base_name = os.path.basename(args.input_file).replace(extension, "")
    is_preview = "_preview" if args.preview else ""
    output_file = os.path.join(output_folder, f"{base_name}_mastered_{config.genre}_{config.loudness_option}_{args.eq_profile}{is_preview}.{args.format}")

    print(f"Received parameters: input_file={args.input_file}, output_file={output_file}, reference_file={config.reference_file}, genre={config.genre}, loudness={config.loudness_option}, eq_profile={args.eq_profile}, preview={args.preview}")

    master_audio(args.input_file, output_file, config, args.eq_profile, is_preview=args.preview)