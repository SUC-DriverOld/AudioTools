import os
import librosa
import soundfile as sf

input_1 = r"valid_raw\female"
input_2 = r"valid_raw\male"
output = r"valid"
os.makedirs(output, exist_ok=True)

input_1_list = os.listdir(input_1)
input_2_list = os.listdir(input_2)

def match_length(audio1, audio2):
    min_length = min(audio1.shape[1], audio2.shape[1])
    audio1_new = audio1[:, :min_length]
    audio2_new = audio2[:, :min_length]
    print(f"audio1.shape: {audio1.shape}, audio2.shape: {audio2.shape}, min_length: {min_length}")
    print(f"audio1_new.shape: {audio1_new.shape}, audio2_new.shape: {audio2_new.shape}")
    return audio1_new, audio2_new

index = 1
foldername = 1
for file1 in input_1_list:
    file1_path = os.path.join(input_1, file1)
    file1_base_name = os.path.splitext(file1)[0]
    audio1, sr1 = librosa.load(file1_path, sr=44100, mono=False)
    
    for file2 in input_2_list:
        if index in [
            1,3,5,7,9,
            12,14,16,18,20,
            21,23,25,27,29,
            32,34,36,38,40,
            41,43,45,47,49,
            52,54,56,58,60,
            61,63,65,67,69,
            72,74,76,78,80,
            81,83,85,87,89,
            92,94,96,98,100
        ]:
            file2_path = os.path.join(input_2, file2)
            file2_base_name = os.path.splitext(file2)[0]
            audio2, sr2 = librosa.load(file2_path, sr=44100, mono=False)

            audio1_new, audio2_new = match_length(audio1, audio2)
            os.makedirs(os.path.join(output, str(foldername)), exist_ok=True)

            sf.write(os.path.join(output, str(foldername), f"female.wav"), audio1_new.T, 44100)
            sf.write(os.path.join(output, str(foldername), f"male.wav"), audio2_new.T, 44100)
            mixture = audio1_new + audio2_new
            sf.write(os.path.join(output, str(foldername), f"mixture.wav"), mixture.T, 44100)
            foldername += 1
        index += 1