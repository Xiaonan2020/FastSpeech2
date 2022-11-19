import os
import random
import json

import tgt
import librosa
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio as Audio

# 该文件中才是从下载的TextGrid文件中提取每条音频对应的duration、pitch和energy信息；其中的config是通过config/LJSpeech/preprocess.yam文件加载而来。
# 主要是把语音数据，对应的textgrid数据和.lab 文本数据进行整合，提取出需要的energy, pitch, mel-scale spectrogram等信息

# 输出文件：
#     1.speakers.json: speaker信息
#     2.stats.json: pitch,energy的范围（max-min)
#     3.train.txt, val.txt: basename, speaker, phone transcription of wav files
#     4.pitch, duration,mel,energy 文件夹（里面是.npy文件）:wav files的energy, pitch, mel-scale spectrogram等信息
# 输出位置：out_dir


# 定义处理所有数据的处理类
class Preprocessor:
    def __init__(self, config):
        """加载configs，按照预设路径读入数据"""

        self.config = config
        self.in_dir = config["path"]["raw_path"] # prepare_align.py处理后的LJSpeech数据的路径
        self.out_dir = config["path"]["preprocessed_path"] # 数据存储后的路径
        self.val_size = config["preprocessing"]["val_size"] # 验证集的大小
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"] # 采样率
        self.hop_length = config["preprocessing"]["stft"]["hop_length"] #256

        assert config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        assert config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        # 是否进行pitch_phoneme_averaging
        self.pitch_phoneme_averaging = (
            config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )# True
        # 是否进行energy_phoneme_averaging
        self.energy_phoneme_averaging = (
            config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )# True
        # 是否进行正则化
        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"] # True
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"] # True
        # 初始化STFT模块
        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"], # 1024
            config["preprocessing"]["stft"]["hop_length"], # 256
            config["preprocessing"]["stft"]["win_length"], # 1024
            config["preprocessing"]["mel"]["n_mel_channels"], # 80
            config["preprocessing"]["audio"]["sampling_rate"], # 22050
            config["preprocessing"]["mel"]["mel_fmin"], # 0
            config["preprocessing"]["mel"]["mel_fmax"], # 8000
        )

    # 提出所需数据
    def build_from_path(self):
        """
        主要程序，主要作用是：
            1.加载从precess_utterance这个function里获得的信息
            2.对信息进行normalize,
            3.最后按照指定路径写入文件
            （speaker.json, stats.json, train.txt, val.txt)
        """
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)

        print("Processing Data ...")
        out = list()
        n_frames = 0
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        # Compute pitch, energy, duration, and mel-spectrogram
        speakers = {} # {'LJSpeech': 0}
        # 下面的一个speaker就是一个文件夹，同一speaker的音频放在同一路径下
        for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
            speakers[speaker] = i # speakers {'LJSpeech': 0}
            for wav_name in os.listdir(os.path.join(self.in_dir, speaker)):
                if ".wav" not in wav_name:
                    continue

                basename = wav_name.split(".")[0] # 'LJ001-0001.wav'
                # 基于音频文件的basename构建对应的对齐文件路径名
                tg_path = os.path.join(
                    self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
                ) # './preprocessed_data/LJSpeech\\TextGrid\\LJSpeech\\LJ001-0001.TextGrid'
                if os.path.exists(tg_path):
                    ret = self.process_utterance(speaker, basename) # 提取单个音频的mel、pitch、energy数据
                    if ret is None:
                        continue
                    else:
                        info, pitch, energy, n = ret # n是mel谱图序列的总帧数
                    out.append(info) # 记录info中文本相关的数据，是一个用“|”分割的字符串

                if len(pitch) > 0:
                    pitch_scaler.partial_fit(pitch.reshape((-1, 1))) # 在线计算X上的平均值和标准，以便以后缩放。
                if len(energy) > 0:
                    energy_scaler.partial_fit(energy.reshape((-1, 1)))

                n_frames += n

        print("Computing statistic quantities ...")
        # Perform normalization if necessary
        if self.pitch_normalization:
            pitch_mean = pitch_scaler.mean_[0]
            pitch_std = pitch_scaler.scale_[0]
        else:
            # A numerical trick to avoid normalization...
            pitch_mean = 0
            pitch_std = 1
        if self.energy_normalization:
            energy_mean = energy_scaler.mean_[0]
            energy_std = energy_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_std = 1

        pitch_min, pitch_max = self.normalize(
            os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
        )
        energy_min, energy_max = self.normalize(
            os.path.join(self.out_dir, "energy"), energy_mean, energy_std
        )

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
            }
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        random.shuffle(out)
        out = [r for r in out if r is not None]

        # Write metadata 划分训练集文本数据和验证集文本数据
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size :]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        return out

    # 基于文件路径提取音频文件的mel、pitch、energy、duration数据
    def process_utterance(self, speaker, basename):
        """
        被build_from_path这个function调用
        主要作用是
            1.通过get_alignment这个function获取textgrid files里的信息
            2.计算出wav files里的foundamental frequency/pitch
            3.通过stft（短时傅里叶变换）把声音文件转成mel频谱
            4.计算出wav files里的energy
            5.将获得的pitch, energy, mel频谱信息分别写入以.npy为后缀的文件
        """
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename)) # 音频文件路径  './raw_data/LJSpeech\\LJSpeech\\LJ001-0001.wav'
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename)) # 文本文件路径 './raw_data/LJSpeech\\LJSpeech\\LJ001-0001.lab'
        tg_path = os.path.join(
            self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
        ) # './preprocessed_data/LJSpeech\\TextGrid\\LJSpeech\\LJ001-0001.TextGrid'

        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path) # 读取textgrid对象
        # 数据提取。phone中是textgrid对象中文本转为音素的列表，duration中为音素列表中每个元素对应的mel帧数，即每个音素的持续时间，start为音频开始时间，end为结束时间
        phone, duration, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        )
        text = "{" + " ".join(phone) + "}" # 文本信息拼接成字符串方便存储
        if start >= end:
            return None

        # Read and trim wav files
        wav, _ = librosa.load(wav_path) # 加载音频
        wav = wav[
            int(self.sampling_rate * start) : int(self.sampling_rate * end)
        ].astype(np.float32) # 裁剪

        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n") # 音频对应文本

        # Compute fundamental frequency #  提取基频 # raw pitch extractor
        pitch, t = pw.dio(
            wav.astype(np.float64), #WORLD使用float类型
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate) # 修改基频 # pitch refinement

        pitch = pitch[: sum(duration)] # 与总的mel谱图帧数对齐
        if np.sum(pitch != 0) <= 1:
            return None

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT) # 计算mel谱图
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        energy = energy[: sum(duration)]

        if self.pitch_phoneme_averaging:
            # perform linear interpolation 线性插值，就是将pitch序列中为0的值赋值一个合理的数值
            nonzero_ids = np.where(pitch != 0)[0] # 获取pitch中不为值不为0的索引
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch))) # 插值后，pitch中为0的部分通过插值得到了补充

            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    pitch[i] = np.mean(pitch[pos : pos + d])
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[: len(duration)]

        if self.energy_phoneme_averaging:
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    energy[i] = np.mean(energy[pos : pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(duration)]

        # Save files
        dur_filename = "{}-duration-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "duration", dur_filename), duration) # 保存时序时间

        pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch) # 保存pitch

        energy_filename = "{}-energy-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy) # 保存energy

        mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        ) # 保存mel谱图

        return (
            "|".join([basename, speaker, text, raw_text]), # 存储文本形式的数据，字符串
            self.remove_outlier(pitch), # 去除离群值的pitch序列
            self.remove_outlier(energy), # 去除离群值的energy序列
            mel_spectrogram.shape[1], # 记录mel谱图序列帧数
        )

    def get_alignment(self, tier): # 提取对齐信息
        """
        被process_utterance这个function调用
        主要作用是提取textgrid files里的phone,duration,start_time, end_time等信息
        """
        sil_phones = ["sil", "sp", "spn"]
        # tier中存储的主要内容就是音频的持续时间，以及文中中每个音素对应的持续时间信息
        phones = [] # 音素
        durations = [] # 持续时间
        start_time = 0 # 区间开始时间
        end_time = 0 # 区间结束时间
        end_idx = 0
        for t in tier._objects:  # t的类型是Interval(0.0, 0.04, "P")，第一个开始时间，第二个是结束时间，第三个即为该段对应的文本区间
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones) # 记录已记录的音素的个数
            else:
                # For silent phones
                phones.append(p)
            # 记录持续时间，将时间单位秒转换为mel帧数
            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

    def remove_outlier(self, values): # 删除离群值，使用箱型图的逻辑
        """
        用来normalize data
        """
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        """
        用来normalize data
        """
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value
