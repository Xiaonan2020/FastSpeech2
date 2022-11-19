import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text


def prepare_align(config):
    in_dir = config["path"]["corpus_path"] # "/home/ming/Data/LibriTTS/train-clean-360"
    out_dir = config["path"]["raw_path"] # "./raw_data/LibriTTS"
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"] # 22050
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"] # 32768
    cleaners = config["preprocessing"]["text"]["text_cleaners"] # english_cleaners

    # os.listdir() 返回指定目录下的所有文件名和目录名
    for speaker in tqdm(os.listdir(in_dir)): # 文件名为speaker
        # os.path.join()将in_dir与speaker连接起来，返回目录下的所有文件名和目录名
        # in_dir/speaker/chapter
        for chapter in os.listdir(os.path.join(in_dir, speaker)):
            # 返回in_dir/speaker/chapter目录下的所有文件
            # 该目录下的文件包括三种：.normalized.txt  .wav  .original.txt
            # file_name文件名
            for file_name in os.listdir(os.path.join(in_dir, speaker, chapter)):
                if file_name[-4:] != ".wav":
                    continue
                # 100_121669_000001_000000.normalized.txt
                # 100_121669_000001_000000.original.txt
                # 100_121669_000001_000000.wav
                # 不是wav文件就跳过，取wav文件的文件名，不取后缀
                base_name = file_name[:-4]
                # .normalized.txt文件中存着一句英语句子，如Tom, the Piper's Son
                text_path = os.path.join(
                    in_dir, speaker, chapter, "{}.normalized.txt".format(base_name)
                )
                wav_path = os.path.join(
                    in_dir, speaker, chapter, "{}.wav".format(base_name)
                )
                # 读取文本内容，如text=Tom, the Piper's Son
                with open(text_path) as f:
                    text = f.readline().strip("\n")
                # 乱码处理、大小写处理、缩写展开、空格处理、数字处理
                text = _clean_text(text, cleaners)

                # 创建文件夹out_dir/speaker,且目录存在不会触发目录存在异常
                os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                # librosa音频信号处理库函数
                # load 从文件加载音频数据,而且可以通过参数设置是否保留双声道,采样率,重采样类型
                # 返回类型wav为numpy.ndarray  _为sampling_rate
                wav, _ = librosa.load(wav_path, sampling_rate)
                # wav = wav / (max(|wav|) * 32768)
                # 归一化，好处1，消除奇异样本数据的影响，好处2，cond
                wav = wav / max(abs(wav)) * max_wav_value
                # 将numpy格式的wav写入到指定文件中，out_dir/speaker/{base_name}.wav,sr,数值类型转换
                wavfile.write(
                    os.path.join(out_dir, speaker, "{}.wav".format(base_name)),
                    sampling_rate,
                    wav.astype(np.int16),# 设置改类型是由ndarray中的数值大小范围决定的，int16：-32768~32768
                )
                # 打开out_dir/speaker/{base_name}.lab,
                # 将从{base_name}.normalized.txt文件中读取出来，然后经过处理的text写入到{base_name}.lab文件中
                with open(
                    os.path.join(out_dir, speaker, "{}.lab".format(base_name)),
                    "w",
                ) as f1:
                    f1.write(text)