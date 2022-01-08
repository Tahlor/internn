import torch
from general_tools.utils import get_root

import internn_utils

ROOT = get_root("internn")
import sys
sys.path.append(ROOT)
from pytorch_utils import *
from internn_utils import read_config, recursive_default, process_config
from general_tools.utils import get_root
from easydict import EasyDict as edict

ROOT = get_root("internn")
DEFAULT_YAML = ROOT / "lm/configs/00_master.yaml"

def bert_config(config):
    config.vocab_size = len(config.alphabet)
    config.mask_id = config.vocab_size
    config.vocab_size_extended = config.vocab_size + 2 # additional BERT tokens or something -- not totally clear
    if internn_utils.is_galois():
        config.batch_size = 192
    config.lr = config.lr * config.batch_size / 192

    if config.TESTING:
        config.workers = 0
        config.sen_loader_pt_file = 'train_test_sentenceDataset_SMALL.pt'
        config.epoch_length = config.batch_size * 2 # do 2 repeats per epoch
    else:
        config.sen_loader_pt_file = 'train_test_sentenceDataset.pt'

    if not config.experiment_description:
        config.experiment_description = config.experiment_type

    config.experiment = config[config.experiment_type]
    if config.device in ["gpu","cuda"]:
        config.device = torch.device('cuda:0')

    ## EMBEDDING
    if config.experiment_type == "vgg_embeddings":

        # Never use an embedding layer with the embedding
        config.embedding_layer_with_logits = None

    ## LOGITS
    elif config.experiment_type == "vgg_logits":
        pass

    else:
        raise Exception(f"What is {config.experiment_type} supposed to be?")
    config.vgg_embeddings.loader_key = "embedding"
    config.vgg_logits.loader_key = "vgg_logits"


    """
    sep_token (separator token, used when building a sequence from multiple sequences, e.g. two sequences for sequence classification or for a text and a question for question answering
    pad_token
    cls_token - class token
    mask_token
    """


    return config

if __name__ == "__main__":
    config = bert_config(process_config(DEFAULT_YAML))
    print(config)