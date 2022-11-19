import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text

# 虽然该文件只定义了prepare_align函数，
# 但是该函数也只是简单的将LJSpeech数据集中的音频数据和文本数据进行了处理并保存，
# 并没有提取对齐信息。


def prepare_align(config):
    in_dir = config["path"]["corpus_path"] # LJSpeech数据集存储路径 './Data/LJSpeech-1.1'
    out_dir = config["path"]["raw_path"] # 数据转化后的存储路径
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    speaker = "LJSpeech"
    with open(os.path.join(in_dir, "metadata.csv"), encoding="utf-8") as f:
        for line in tqdm(f):
            parts = line.strip().split("|") # 分割音每一行文本 获得一条语音的路径名和内容文本
            base_name = parts[0] # 该条音频的文件名
            text = parts[2]  # 音频对应的内容文本
            text = _clean_text(text, cleaners) # 使用text库提供的接口结合cleaner对文本进行调整

            wav_path = os.path.join(in_dir, "wavs", "{}.wav".format(base_name)) # 获取完整音频文件路径 './Data/LJSpeech-1.1\\wavs\\LJ001-0001.wav'
            if os.path.exists(wav_path):
                os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                wav, _ = librosa.load(wav_path, sr=sampling_rate)
                wav = wav / max(abs(wav)) * max_wav_value ## 不理解
                # 将处理之后的wav保存
                wavfile.write(
                    os.path.join(out_dir, speaker, "{}.wav".format(base_name)),
                    sampling_rate,
                    wav.astype(np.int16),
                )
                # 将调整后的文本序列保存
                with open(
                    os.path.join(out_dir, speaker, "{}.lab".format(base_name)),
                    "w",
                ) as f1:
                    f1.write(text)