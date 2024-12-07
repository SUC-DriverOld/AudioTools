import os
from pydub import AudioSegment
from collections import defaultdict

input_folder = 'opencpop-psswd-Mmwjxhn2017\wavs'
output_folder = 'opencpop'

os.makedirs(output_folder, exist_ok=True)
song_segments = defaultdict(list)

for filename in os.listdir(input_folder):
    if filename.endswith('.wav'):
        song_id = filename[:4]
        song_segments[song_id].append(os.path.join(input_folder, filename))

for song_id, segments in song_segments.items():
    segments.sort(key=lambda x: int(os.path.splitext(x)[0][-4:]))
    combined_audio = AudioSegment.from_wav(segments[0])

    for segment_path in segments[1:]:
        audio_segment = AudioSegment.from_wav(segment_path)
        combined_audio += audio_segment

    output_path = os.path.join(output_folder, f'{song_id}.wav')
    combined_audio.export(output_path, format='wav')
    print(f'合并完成: {output_path}')
