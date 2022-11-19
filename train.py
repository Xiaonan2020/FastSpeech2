import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset

from evaluate import evaluate

# 该文件是FastSpeech模型训练过程实现代码，整体流程与普通模型训练一样，
# 需要注意的一点就是数据划分过程中，是分成了一个大batch，其中包含数个real batch，
# 故训练过程在正常的两个for循环嵌套外是一个“while True”的训练，
# 其不是基于epoch来判断训练终止，而是当total_step达到了设置了训练步数才终止训练


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs # 加载预处理、模型和训练的配置文件

    # Get dataset
    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    ) # 加载训练数据集
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    # Prepare model
    model, optimizer = get_model(args, configs, device, train=True) # 加载模型和优化器
    model = nn.DataParallel(model)
    num_param = get_param_num(model) # 计算模型参数量
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device) # 定义损失函数
    print("Number of FastSpeech2 Parameters:", num_param)

    # Load vocoder
    vocoder = get_vocoder(model_config, device) # 加载声码器

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train") # './output/log/LJSpeech\\train'
    val_log_path = os.path.join(train_config["path"]["log_path"], "val") # './output/log/LJSpeech\\val'
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    # 使用tensorboard记录训练过程
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    step = args.restore_step + 1 # 当前步数
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"] # 梯度累步数值 1
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"] # 梯度剪裁的值 1.0
    total_step = train_config["step"]["total_step"] # 总的训练步数 900000
    log_step = train_config["step"]["log_step"] # 100
    save_step = train_config["step"]["save_step"] # 100000
    synth_step = train_config["step"]["synth_step"] # 1000
    val_step = train_config["step"]["val_step"] # 1000

    outer_bar = tqdm(total=total_step, desc="Training", position=0) # 显示所有步数的运行情况
    outer_bar.n = args.restore_step  # 加载之前已经训练完的步数
    outer_bar.update()

    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1) # 显示当前epoch内的训练步数情况
        for batchs in loader: # 根据前面的设置，一个batchs中是有group_size个batch的
            for batch in batchs:
                batch = to_device(batch, device)

                # Forward
                output = model(*(batch[2:])) # speakers,texts,text_lens,max(text_lens),mels,mel_lens,max(mel_lens),pitches,energies,durations,

                # Cal Loss
                losses = Loss(batch, output)
                total_loss = losses[0]

                # Backward
                total_loss = total_loss / grad_acc_step
                total_loss.backward()
                if step % grad_acc_step == 0: # 到了梯度累计释放的步数
                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh) # 梯度剪裁

                    # Update weights
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                if step % log_step == 0: # 到了记录的步数
                    losses = [l.item() for l in losses]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
                        *losses
                    )

                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2) # 将日志信息在进度掉的后面显示

                    log(train_logger, step, losses=losses) # 调用定义的日志函数在tensorboard中记录信息

                if step % synth_step == 0: # 到了合成音频的步数
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        batch,
                        output,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    log(
                        train_logger,
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                    )
                    sampling_rate = preprocess_config["preprocessing"]["audio"][
                        "sampling_rate"
                    ]
                    # 记录以target_mel谱图使用vocoder重构的音频
                    log(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    )
                    # 记录以生成的prediction_mel谱图使用vocoder重构的音频
                    log(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    )

                if step % val_step == 0: # 到了验证的步数
                    model.eval() # 先设置为验证模式
                    message = evaluate(model, step, configs, val_logger, vocoder)
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)

                    model.train() # 退出时设置回训练模式

                if step % save_step == 0: # 到了模型保存的步数
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer._optimizer.state_dict(),
                        },
                        os.path.join(
                            train_config["path"]["ckpt_path"],
                            "{}.pth.tar".format(step),
                        ),
                    )

                if step == total_step: # 如果到了设置的训练总步数，就停止训练
                    quit()
                step += 1
                outer_bar.update(1)  # 当前epoch每训练一个step也要在outer_bar中更新

            inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    # parser.add_argument("-p","--preprocess_config",type=str,required=True,help="path to preprocess.yaml",)
    # parser.add_argument("-m", "--model_config", type=str, required=True, help="path to model.yaml")
    # parser.add_argument("-t", "--train_config", type=str, required=True, help="path to train.yaml")

    parser.add_argument("-p","--preprocess_config", required=False, default="./config/LJSpeech/preprocess.yaml",
                        type=str,help="path to preprocess.yaml")
    parser.add_argument("-m", "--model_config", required=False, default="./config/LJSpeech/model.yaml",
                        type=str, help="path to model.yaml")
    parser.add_argument("-t", "--train_config", required=False, default="./config/LJSpeech/train.yaml",
                        type=str, help="path to train.yaml")


    args = parser.parse_args()


    # Read Config
    preprocess_config = yaml.load(open(args.preprocess_config, "r", encoding='utf-8'), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r", encoding='utf-8'), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r", encoding='utf-8'), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    main(args, configs)
