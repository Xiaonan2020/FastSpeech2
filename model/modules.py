import os
import json
import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils.tools import get_mask_from_lengths, pad

# 本文件主要是定义Variance Adaptor，
# 其中主要包括Duration Predictor、Length Regulator、Pitch Predictor和Energy Predictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VarianceAdaptor(nn.Module):
    """Variance Adaptor"""

    def __init__(self, preprocess_config, model_config):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)
        # 设置pitch和energy的级别  'phoneme_level'
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ] # 'phoneme_level'
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ] # 'phoneme_level'
        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]
        # 设置pitch和energy的量化方式 "linear"
        pitch_quantization = model_config["variance_embedding"]["pitch_quantization"] # 'linear'
        energy_quantization = model_config["variance_embedding"]["energy_quantization"] # 'linear'
        n_bins = model_config["variance_embedding"]["n_bins"] #256
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]
        # 加载pitch和energy的正则化所需参数
        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats["pitch"][:2]
            energy_min, energy_max = stats["energy"][:2]

        if pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            ) #  Parameter 255
        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
            )
        # pitch和energy的嵌入层
        self.pitch_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"] # 256 256
        ) # Embedding(256, 256)
        self.energy_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )

    # 计算pitch嵌入层
    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask) # pitch预测器预测的数值 [b len]
        if target is not None: # target存在，训练过程，使用target计算embedding
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins)) # [b len 256]
        else: # target不存在，预测过程，使用prediction计算embedding
            prediction = prediction * control # control是用于控制的系数
            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
        return prediction, embedding # prediction用于训练过程计算损失，embedding与x相加进行后续计算

    # # 计算energy嵌入层
    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask) # energy预测器预测的数值
        if target is not None: # target存在，训练过程，使用target计算embedding
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
            # bucketize 返回input中每个元素的所属的桶的索引，桶的边界由boundaries设置。返回的是和input相同大小的新的Tensor
        else: # target不存在，预测过程，使用prediction计算embedding
            prediction = prediction * control # control是用于控制的系数
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding # prediction用于训练过程计算损失，embedding与x相加进行后续计算

    def forward(
        self,
        x,
        src_mask,
        mel_mask=None,
        max_len=None,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):

        log_duration_prediction = self.duration_predictor(x, src_mask)  # 对音素序列预测的持续时间 [b len]
        if self.pitch_feature_level == "phoneme_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, src_mask, p_control
            )
            x = x + pitch_embedding # 累加pitch嵌入层
        if self.energy_feature_level == "phoneme_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, src_mask, p_control
            )
            x = x + energy_embedding  # 累加energy嵌入层


        if duration_target is not None: # duration_target，训练过程，使用duration_target计算
            x, mel_len = self.length_regulator(x, duration_target, max_len)  # 使用duration_target调整x, 将x扩充到mel普的长度
            duration_rounded = duration_target
        else: # 预测过程
            # 基于log_duration_prediction构建duration_rounded，用于调整x
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            ) # torch.round 每个元素都舍入到最接近的整数
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)

        # 同上，与phoneme_level一致
        if self.pitch_feature_level == "frame_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, mel_mask, p_control
            )
            x = x + pitch_embedding
        if self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, mel_mask, p_control
            )
            x = x + energy_embedding

        return (
            x,
            pitch_prediction, # 此处三个prediction用于后续计算损失
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )

# 长度调节器
class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    # 对输入的音素序列x进行长度调正
    def LR(self, x, duration, max_len):
        """
        基于音素持续时间将音素序列长度与mel谱图长度对齐
        @param x: 经过FFT块转换后的音素序列，[batch_size, max_sequence_len, encoder_dim]
        @param duration: 音素持续时间矩阵，[batch_size, max_sequence_len]
        @param max_len: 音素谱图序列中最大长度
        @return: 长度经过调整后的音素序列，[batch_size, max_len, encoder_dim]
        """
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target) # 获得一个长度完整调整之后音素序列 第i个音素 [max_mel_len 256]
            output.append(expanded)
            mel_len.append(expanded.shape[0])  # 记录一个mel谱图长度大小，方便后续生成mask

        # 如果传入max_len就按其进行pad，如果没有就以output中最长序列大小进行pad
        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output) # [b max_mel_len 256]

        return output, torch.LongTensor(mel_len).to(device)

    def expand(self, batch, predicted):
        """
                将输入的一个音素序列的长度按其对应的持续时间调整
                @param batch:一个音频对应文本的音素序列，[max_sequence_len, encoder_dim]
                @param predicted:音素序列中每个音素对应的持续序列，长度为max_sequence_len
                @return:长度调整后的音素序列，长度与mel谱图长度一致
                """
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item() # i对应的音素对应持续时间，即需要重复的次数
            out.append(vec.expand(max(int(expand_size), 0), -1)) # 将i对应的音素的表征向量vec重复expand_size次
        out = torch.cat(out, 0) # 将整个音素序列cat起来

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len # [b max_mel_len 256]


class VariancePredictor(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"] # 输入尺寸 256
        self.filter_size = model_config["variance_predictor"]["filter_size"] # 输出尺寸 256
        self.kernel = model_config["variance_predictor"]["kernel_size"] # 卷积核大小 3
        self.conv_output_size = model_config["variance_predictor"]["filter_size"] # 256
        self.dropout = model_config["variance_predictor"]["dropout"] # 0.5
        # 定义一个包含激活函数和正则项的卷积序列，即[Con1D+Relu+LN+Dropout]+[Con1D+Relu+LN+Dropout]
        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output) # [Con1D+Relu+LN+Dropout]+[Con1D+Relu+LN+Dropout]
        out = self.linear_layer(out) # 最后输出前的线性层
        out = out.squeeze(-1)  # 因为线性层返回的是1，即输出的尺寸的最后一维是1，将其压缩掉

        if mask is not None:
            out = out.masked_fill(mask, 0.0) # 将mask对应地方设置为0

        return out


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x
