import argparse

import yaml

from preprocessor.preprocessor import Preprocessor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("config", type=str, help="path to preprocess.yaml")
    parser.add_argument("--config", required=False, default="./config/LJSpeech/preprocess.yaml", type=str,
                        help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r", encoding='utf-8'), Loader=yaml.FullLoader)
    preprocessor = Preprocessor(config)
    preprocessor.build_from_path()
