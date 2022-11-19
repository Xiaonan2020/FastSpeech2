import argparse
import yaml

from preprocessor import ljspeech, aishell3, libritts

# 该文件就是相当于一个接口，针对不同的数据集调用对应的文件函数进行数据准备，
# 主要就是调用数据集对应的prepare_align函数处理数据
# prepare_align是preprocessor/ljspeech.py中定义的函数

# config为配置文件中的内容，dataset为一个配置项，用以识别需要训练的数据集
def main(config):
    if "LJSpeech" in config["dataset"]:
        ljspeech.prepare_align(config) # preprocessor/ljspeech.py
    if "AISHELL3" in config["dataset"]:
        aishell3.prepare_align(config)
    if "LibriTTS" in config["dataset"]:
        libritts.prepare_align(config)


if __name__ == "__main__":
    # 运行的时候加上一个参数config config为preprocess.yaml的路径
    parser = argparse.ArgumentParser()
    # parser.add_argument("config", type=str, help="path to preprocess.yaml") # # 加载对应的yaml文件，便于后面添加相应参数
    parser.add_argument("--config", required=False,default="./config/LJSpeech/preprocess.yaml", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()
    print(args)

    # config为preprocess.yaml的内容，传入该文件的路径，读取该文件，yaml.FullLoader参数读取全部yaml语言
    # 禁止执行任意函数，这样 load() 函数也变得更加安全
    config = yaml.load(open(args.config, "r", encoding='utf-8'), Loader=yaml.FullLoader)
    main(config)
