import torch
import torch.nn as nn

# FastSpeech2在训练时会对duration predictor、pitch predictor和energy predictor同时训练，
# 结合之前自回归模型均会对最后mel经过postnet处理的前后计算损失，
# 故训练过程中会计算五个损失。loss.py文件中就定义了损失类


class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature" # "phoneme_level"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature" # "phoneme_level"
        ]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, inputs, predictions):
        (
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
        ) = inputs[6:] # 目标，相当于label
        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
        ) = predictions # 模型的输出
        src_masks = ~src_masks # 取反 非pad部分设为True
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1) # 对目标持续时间取log
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets) # 解码器预测的mel谱图的损失
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets) # 解码器预测的mel谱图经过postnet处理后的损失

        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets) # pitch损失
        energy_loss = self.mse_loss(energy_predictions, energy_targets) # energy损失
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets) # duration损失

        total_loss = (
            mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss
        )

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
        )
