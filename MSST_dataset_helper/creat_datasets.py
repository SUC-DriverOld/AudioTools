import torch
import torchaudio
import os
import random
from tqdm import tqdm

def process_audio(audio, mp3_audio, sr, output_folder):
    min_length = min(audio.size(1), mp3_audio.size(1))
    audio = audio[:, :min_length]
    mp3_audio = mp3_audio[:, :min_length]

    # 随机选择一个频率阈值
    freq_threshold = random.choice([16000, 17000, 18000, 19000, 20000])
    freqs = torch.fft.fftfreq(audio.size(1), 1/sr)

    # 计算低频和高频成分
    fft_audio_left = torch.fft.fft(audio[0])
    fft_audio_right = torch.fft.fft(audio[1])
    fft_mp3_left = torch.fft.fft(mp3_audio[0])
    fft_mp3_right = torch.fft.fft(mp3_audio[1])

    # 处理wav文件
    low_freq_audio_left = fft_audio_left.clone()
    low_freq_audio_right = fft_audio_right.clone()
    high_freq_audio_left = fft_audio_left.clone()
    high_freq_audio_right = fft_audio_right.clone()

    low_freq_audio_left[torch.abs(freqs) > freq_threshold] = 0
    low_freq_audio_right[torch.abs(freqs) > freq_threshold] = 0
    high_freq_audio_left[torch.abs(freqs) <= freq_threshold] = 0
    high_freq_audio_right[torch.abs(freqs) <= freq_threshold] = 0

    # 处理mp3文件
    low_freq_mp3_audio_left = fft_mp3_left.clone()
    low_freq_mp3_audio_right = fft_mp3_right.clone()
    high_freq_mp3_audio_left = fft_mp3_left.clone()
    high_freq_mp3_audio_right = fft_mp3_right.clone()

    low_freq_mp3_audio_left[torch.abs(freqs) > freq_threshold] = 0
    low_freq_mp3_audio_right[torch.abs(freqs) > freq_threshold] = 0
    high_freq_mp3_audio_left[torch.abs(freqs) <= freq_threshold] = 0
    high_freq_mp3_audio_right[torch.abs(freqs) <= freq_threshold] = 0

    # 合成音频
    low_freq_mp3_audio_left = torch.fft.ifft(low_freq_mp3_audio_left)
    low_freq_mp3_audio_right = torch.fft.ifft(low_freq_mp3_audio_right)
    origin_audio = torch.stack((low_freq_mp3_audio_left.real, low_freq_mp3_audio_right.real))
    addition_audio = origin_audio - audio

    # 保存文件
    # origin_file = os.path.join(output_folder, "mixture.wav")
    addition_file = os.path.join(output_folder, "addition.wav")
    restored_file = os.path.join(output_folder, "restored.wav")

    # torchaudio.save(origin_file, origin_audio, sr, format="wav", bits_per_sample=16)
    torchaudio.save(addition_file, addition_audio, sr, format="wav", bits_per_sample=16)
    torchaudio.save(restored_file, audio, sr, format="wav", bits_per_sample=16)

    return freq_threshold


def mp3(input_file):
    to_mp3 = f"ffmpeg -i \"{input_file}\" -ar 44100 -ac 2 -b:a 320k temp.mp3 -y"
    to_wav = f"ffmpeg -i temp.mp3 temp.wav -y"
    os.system(f"{to_mp3} > {os.devnull} 2>&1")
    os.system(f"{to_wav} > {os.devnull} 2>&1")
    mp3_audio, _ = torchaudio.load("temp.wav", normalize=True)
    os.remove("temp.mp3")
    os.remove("temp.wav")
    return mp3_audio


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", type=str, default="input")
    parser.add_argument("-o", "--output_folder", type=str, default="output")
    args = parser.parse_args()

    freqs = []
    for audio_name in tqdm(os.listdir(args.input_folder)):
        try:
            original_audio, original_sr = torchaudio.load(os.path.join(args.input_folder, audio_name), normalize=True)
            if original_audio.shape[0] != 2:
                print(f"Audio {audio_name} is not stereo.")
                continue
            if original_sr != 44100:
                resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=44100)
                audio_process = resampler(original_audio)
            else:
                audio_process = original_audio

            mp3_audio = mp3(os.path.join(args.input_folder, audio_name))

            base_name = os.path.splitext(os.path.basename(audio_name))[0]
            output_folder = os.path.join(args.output_folder, base_name)
            os.makedirs(output_folder, exist_ok=True)

            freq_threshold = process_audio(audio_process, mp3_audio, 44100, output_folder)
            freqs.append(freq_threshold)
        except Exception as e:
            print(f"Cound not process {audio_name}. Error: {str(e)}")
            continue

    print(f"16kHz: {freqs.count(16000)}, percentage: {freqs.count(16000)/len(freqs)}")
    print(f"17kHz: {freqs.count(17000)}, percentage: {freqs.count(17000)/len(freqs)}")
    print(f"18kHz: {freqs.count(18000)}, percentage: {freqs.count(18000)/len(freqs)}")
    print(f"19kHz: {freqs.count(19000)}, percentage: {freqs.count(19000)/len(freqs)}")
    print(f"20kHz: {freqs.count(20000)}, percentage: {freqs.count(20000)/len(freqs)}")
    print(f"Total files: {len(freqs)}, Averange: {sum(freqs)/len(freqs)}")
