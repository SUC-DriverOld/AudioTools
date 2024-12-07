"""requirements: praat-parselmouth librosa matplotlib tqdm"""

import os
import numpy as np
import matplotlib.pyplot as plt
import parselmouth
import librosa
from tqdm import tqdm

NOTE_ORDER = [
    "C0", "C#0", "D0", "D#0", "E0", "F0", "F#0", "G0", "G#0", "A0", "A#0", "B0",
    "C1", "C#1", "D1", "D#1", "E1", "F1", "F#1", "G1", "G#1", "A1", "A#1", "B1",
    "C2", "C#2", "D2", "D#2", "E2", "F2", "F#2", "G2", "G#2", "A2", "A#2", "B2",
    "C3", "C#3", "D3", "D#3", "E3", "F3", "F#3", "G3", "G#3", "A3", "A#3", "B3",
    "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4",
    "C5", "C#5", "D5", "D#5", "E5", "F5", "F#5", "G5", "G#5", "A5", "A#5", "B5",
    "C6", "C#6", "D6", "D#6", "E6", "F6", "F#6", "G6", "G#6", "A6", "A#6", "B6",
    "C7", "C#7", "D7", "D#7", "E7", "F7", "F#7", "G7", "G#7", "A7", "A#7", "B7"
]
F0_MIN = 40
F0_MAX = 2200

class PMF0Predictor():
    def __init__(self, hop_length=512, f0_min=F0_MIN, f0_max=F0_MAX, sampling_rate=44100):
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.sampling_rate = sampling_rate
        self.name = "pm"

    def interpolate_f0(self, f0):
        vuv_vector = np.zeros_like(f0, dtype=np.float32)
        vuv_vector[f0 > 0.0] = 1.0
        vuv_vector[f0 <= 0.0] = 0.0

        nzindex = np.nonzero(f0)[0]
        data = f0[nzindex]
        nzindex = nzindex.astype(np.float32)
        time_org = self.hop_length / self.sampling_rate * nzindex
        time_frame = np.arange(f0.shape[0]) * self.hop_length / self.sampling_rate

        if data.shape[0] <= 0:
            return np.zeros(f0.shape[0], dtype=np.float32), vuv_vector

        if data.shape[0] == 1:
            return np.ones(f0.shape[0], dtype=np.float32) * f0[0], vuv_vector

        f0 = np.interp(time_frame, time_org, data, left=data[0], right=data[-1])
        return f0, vuv_vector

    def compute_f0(self, wav, p_len=None):
        x = wav
        if p_len is None:
            p_len = x.shape[0] // self.hop_length
        else:
            assert abs(p_len - x.shape[0] // self.hop_length) < 4, "pad length error"
        time_step = self.hop_length / self.sampling_rate * 1000
        f0 = parselmouth.Sound(x, self.sampling_rate).to_pitch_ac(
            time_step=time_step / 1000, voicing_threshold=0.6,
            pitch_floor=self.f0_min, pitch_ceiling=self.f0_max).selected_array['frequency']

        pad_size = (p_len - len(f0) + 1) // 2
        if pad_size > 0 or p_len - len(f0) - pad_size > 0:
            f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode='constant')
        f0, uv = self.interpolate_f0(f0)
        return f0

    def process_folder(self, folder_path):
        pitch_count = {}
        total_frames = 0

        for file_name in tqdm(os.listdir(folder_path)):
            if file_name.lower().endswith('.wav'):
                file_path = os.path.join(folder_path, file_name)
                wav_data, _ = librosa.load(file_path, sr=self.sampling_rate)
                f0 = self.compute_f0(wav_data.astype(np.float32))

                for freq in f0:
                    note = frequency_to_note(freq)
                    if note:
                        pitch_count[note] = pitch_count.get(note, 0) + 1
                        total_frames += 1

        pitch_distribution = {note: (count / total_frames * 100) for note, count in pitch_count.items()}
        return pitch_distribution

def frequency_to_note(freq):
    if freq <= 0:
        return None
    A4 = 440.0
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    half_steps = int(np.round(12 * np.log2(freq / A4)))
    note = notes[((half_steps + 9) % 12)]
    octave = 4 + ((half_steps + 9) // 12)
    return f"{note}{octave}"

def plot_pitch_distribution(pitch_distribution):
    sorted_distribution = {
        note: pitch_distribution[note]
        for note in NOTE_ORDER
        if note in pitch_distribution
    }
    notes = list(sorted_distribution.keys())
    percentages = list(sorted_distribution.values())
    plt.figure(figsize=(12, 6))
    bars = plt.bar(notes, percentages)

    for bar, percentage in zip(bars, percentages):
        if percentage >= 0.1:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{percentage:.1f}%",
                ha='center', va='bottom', fontsize=8
            )

    plt.xlabel("Pitch (Notes)")
    plt.ylabel("Percentage (%)")
    plt.title("Pitch Distribution")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("pitch_distribution.png")
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Pitch Detection')
    parser.add_argument('input_path', type=str, help='Path to the folder containing audio files')
    args = parser.parse_args()
    input_path = args.input_path

    predictor = PMF0Predictor()
    pitch_distribution = predictor.process_folder(input_path)
    plot_pitch_distribution(pitch_distribution)
