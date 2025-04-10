import librosa
import numpy as np
from mir_eval.separation import bss_eval_sources


def read_and_resample_audio(file_path, target_sr=44100):
    audio, _ = librosa.load(file_path, sr=target_sr, mono=False)
    if audio.ndim == 1:
        audio = np.vstack((audio, audio))
    elif audio.ndim == 2 and audio.shape[0] == 1:
        audio = np.vstack((audio[0], audio[0]))
    return audio


def match_length(ref_audio, est_audio):
    min_length = min(ref_audio.shape[1], est_audio.shape[1])
    ref_audio = ref_audio[:, :min_length]
    est_audio = est_audio[:, :min_length]
    return ref_audio, est_audio


def compute_sdr(reference, estimated):
    sdr, _, _, _ = bss_eval_sources(reference, estimated)
    return sdr


def process_audio(true_path, estimated_path):
    target_sr = 44100

    true_audio = read_and_resample_audio(
        true_path, target_sr)
    estimated_audio = read_and_resample_audio(
        estimated_path, target_sr)
    true_audio, estimated_audio = match_length(true_audio, estimated_audio)
    sdr = compute_sdr(true_audio, estimated_audio)
    print(f"SDR: {sdr}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('SDR measure for audio files')
    parser.add_argument('-t', '--true_audio', type=str,
                        required=True, help='Path to the true audio file')
    parser.add_argument('-e', '--estimated_audio', type=str,
                        required=True, help='Path to the estimated audio file')
    args = parser.parse_args()

    process_audio(args.true_audio, args.estimated_audio)
