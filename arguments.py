import argparse
import numpy as np
import torch
import random
import yaml
import re



class Namespace(object):
    def __init__(self, somedict):
        for key, value in somedict.items():
            assert isinstance(key, str) and re.match("[A-Za-z_-]", key)
            if isinstance(value, dict):
                self.__dict__[key] = Namespace(value)
            else:
                self.__dict__[key] = value

    def __getattr__(self, attribute):
        raise AttributeError(
            f"Can not find {attribute} in namespace. Please write {attribute} in your config file(xxx.yaml)!")


def set_deterministic(seed):
    # seed by default is None
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', type=str, required=True, help="xxx.yaml")
    parser.add_argument('--method', default='oritext_mlp2cprompt_oriimg',
                        choices=['mlp', 'resmlp', 'sprompt', 'cprompt', 'cprompt_oriprompt', 'cprompt_mlp', 'cprompt_resmlp', 'oriclip2cprompt_mlp', 'oritext_mlp2cprompt_oriimg'],
                        type=str, help='our methods')
    parser.add_argument('--il_setting', default='class-il', choices=['class-il', 'task-il'], type=str, help='Incremental setting')
    parser.add_argument('--visual', default='RN50', choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16'], type=str,
                        help='Visual backbone')
    parser.add_argument('--buffer_size', type=int, default=200, help='The size of the memory buffer.')

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--cifar10_dir', type=str, default="/home/liwujin/code_python/datasets/cifar10")
    parser.add_argument('--cifar100_dir', type=str, default="/home/liwujin/code_python/datasets/cifar100")
    parser.add_argument('--tinyimagenet_dir', type=str, default="/home/liwujin/code_python/datasets/tiny-imagenet-200")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        print(data)
        for key, value in Namespace(data).__dict__.items():
            vars(args)[key] = value

    set_deterministic(args.seed)

    return args