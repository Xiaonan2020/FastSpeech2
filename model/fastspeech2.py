import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths

# 本文件将Encoder, Decoder, PostNet和Variance Adaptor模块集成在一起，完成FastSpeech2模型搭建


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config) # Variance Adaptor之前网络，为编码器
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config) # Variance Adaptor
        self.decoder = Decoder(model_config) # Variance Adaptor之后网络，为解码器
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"], # 256
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"], # 80
        )
        self.postnet = PostNet()

        self.speaker_emb = None
        if model_config["multi_speaker"]: # 如果为多speaker
            # 加载speaker文件
            with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json"),"r",) as f:
                n_speaker = len(json.load(f))
            # 构建speaker嵌入层
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"], # 256
            )

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,  # 控制系数
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len) # 文本序列mask [batci_size, max_len]  pad 的部分设为true
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len) # mel谱图序列mask [batci_size, max_mel_len]  pad 的部分设为true
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks) # 编码

        if self.speaker_emb is not None: # 如果存在speaker嵌入层，将其和output相加
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )
        # 通过Variance Adaptor模块计算
        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks) # 解码
        output = self.mel_linear(output) # 线性转换

        postnet_output = self.postnet(output) + output # 后处理

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )