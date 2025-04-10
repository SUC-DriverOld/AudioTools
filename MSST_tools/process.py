import os
from pydub import AudioSegment
import numpy as np

# 反相位操作
def invert_phase(audio_path):
    # 加载音频文件
    audio = AudioSegment.from_wav(audio_path)
    
    # 将音频转为numpy数组
    samples = np.array(audio.get_array_of_samples())
    
    # 反相位：将所有样本值的符号反转
    inverted_samples = -samples
    
    # 创建反相位后的音频
    inverted_audio = audio._spawn(inverted_samples.tobytes())
    
    # 设置音频的帧宽与采样率一致
    inverted_audio = inverted_audio.set_frame_rate(audio.frame_rate)
    
    return inverted_audio

# 遍历文件夹并处理所有的addition.wav文件
def process_folder(folder_path):
    # 遍历文件夹及子文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower() == 'addition.wav':
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")
                
                # 反相位操作
                inverted_audio = invert_phase(file_path)
                
                # 保存反相位后的音频到原路径，覆盖原文件
                inverted_audio.export(file_path, format='wav')
                print(f"Saved inverted audio: {file_path}")

# 输入文件夹路径并开始处理
if __name__ == "__main__":
    folder_path = "train"
    process_folder(folder_path)
    print("Phase inversion completed for all 'addition.wav' files.")
