import torch
from general_tools.utils import get_root
ROOT = get_root("internn")
import sys
sys.path.append(ROOT)
from pytorch_utils import *
from internn_utils import read_config, recursive_default
from general_tools.utils import get_root
from easydict import EasyDict as edict

ROOT = get_root("internn")
DEFAULT_YAML = ROOT / "lm/configs/00_master.yaml"

def process_config(path="./lm_config.yaml"):
    defaults = read_config(DEFAULT_YAML)

    if not path:
        path = "./lm_config.yaml"
    config = read_config(path)
    config = edict(config)

    config.mask_id = config.vocab_size
    config.vocab_size_extended = config.vocab_size + 2 # additional BERT tokens or something -- not totally clear
    config.vocab_size = len(config.alphabet)

    if not config.experiment_description:
        config.experiment_description = config.experiment_type

    config.experiment = config[config.experiment_type]
    if config.device in ["gpu","cuda"]:
        config.device = torch.device('cuda:0')

    config = recursive_default(config, defaults)



    ## EMBEDDING
    if config.experiment_type == "embedding":

        # Never use an embedding layer with the embedding
        config.embedding_layer_with_logits = None


    ## LOGITS
    elif config.experiment_type == "logit":
        pass

    else:
        raise Exception(f"What is {config.experiment_type} supposed to be?")

    """
    sep_token (separator token, used when building a sequence from multiple sequences, e.g. two sequences for sequence classification or for a text and a question for question answering
    pad_token
    cls_token - class token
    mask_token
    """


    return config

if __name__ == "__main__":
    config = process_config()
    print(config)