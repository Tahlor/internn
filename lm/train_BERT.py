# -*- coding: utf-8 -*-
"""CharBERTV3_BASE

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sOlk2rIdrl2_LSFyi1WZUa7Sc9Kn94qs

### CALCULATE LANGUAGE MODEL CER -- only evaluate the letters predicted, need to return these indices I guess.
### IMAGES NOT SAVED IN LOADER -- ugh
### embedding vs logit version
## wandb......
## LR wrt batch size
## WHY isn't the language model that good?
## What is going on with do nt?
## Fix folder structure
## Get resume to work smoothly

# RANDOMCHAR is wrong shape
# Need to learn LOGIT version to preload
# Finish working on stats -- calculate CER only for the characters in question??? Do both!

"""
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

import numpy as np
from lm_utils import *
from error_measures import *
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import os
from sen_loader import *
from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import *
from CustomBert import *
from pytorch_utils import *
from general_tools.my_logging import *
import wandb
import random
from transformers import AdamW
import process_config_BERT
import matplotlib.pyplot as plt
import sys

sys.path.append("..")
import stats

logger = logging.getLogger("root."+__name__)

VISION_MODEL_ACTIVE = False

RESULTS = edict({})
def printR(key, value):
    print(key, value)
    RESULTS[key] = value

config_path = Path(sys.argv[1])
logger.info(("ARG: ", config_path))

config = process_config_BERT.bert_config(process_config(config_path))

printR("config",config)

if config.wandb:
    wandb.config = config
    wandb.init(project="TrainBERT")

LOADER_PATH = ROOT / config.folder_dependencies.embedding_dataset_folder
VGG_MODEL_PATH = ROOT / config.folder_dependencies.VGG_model_path


# folder management
# Get the config path relative to results/configs folder
configs_path = Path(get_max_root(["results","configs"],config_path))
config.folder_outputs = config.folder_outputs.replace("*EXPERIMENT*", str(config_path.resolve().relative_to(configs_path).parent))
exp = "RUN" if not config.TESTING else "RUN_TESTING"

# incrementer
config.folder_outputs = incrementer(config.folder_outputs, exp, make_new_folder=True)

## FIX - should write out new yaml
#shutil.copy(config_path, config.folder_outputs)
with (config.folder_outputs / config_path.name).open(mode='w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

MODEL_PATH = ROOT / "lm" / config.folder_outputs
text = config.alphabet
corpus = [char for char in text]
EXPERIMENT_NAME = config.experiment_prefix + config.experiment_type
train_dataset = SentenceDataset(PATH=LOADER_PATH / config.sen_loader_pt_file,
                                which='Embeddings',
                                train_mode=config.train_mode,
                                sentence_length=config.sentence_length,
                                sentence_filter="lowercase",
                                vocab_size=config.vocab_size,
                                normalize=config.embedding_norm,
                                alphabet=config.alphabet
                                )

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=config.batch_size,
                                           shuffle=True,
                                           collate_fn=collate_fn_embeddings,
                                           num_workers=config.workers)

#sample = next(iter(train_dataset))
# printR("Batch Data Shape", sample["data"].squeeze(0).shape)
# printR("Num Batch Labels", sample["gt_one_hot"].squeeze(0).shape)
# printR("Output shape", sample["vgg_logits"].shape)
# printR("Sen Lengths", sample["length"])
# printR("GT One Hot", get_text(sample["gt_one_hot"]))

if True:
    model = BertModelCustom(BertConfig(vocab_size=config.vocab_size_extended,
                                   hidden_size=config.embedding_dim,
                                   num_attention_heads=config.attention_heads)).to(config.device)
else:
    from transformers import AutoTokenizer, AutoModelWithLMHead
    tokenizer = AutoTokenizer.from_pretrained("google/reformer-enwik8")
    model = AutoModelWithLMHead.from_pretrained("google/reformer-enwik8")

objective = nn.CrossEntropyLoss()
#objective = nn.NLLLoss(ignore_index=0)

parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
printR(f"PARAMETERS", parameters)
embedding = nn.Linear(config.vocab_size, config.embedding_dim).to(config.device) if config.experiment.embedding_layer else None
all_parameters = [{'params': model.parameters()}]

if embedding:
    printR(f"Embedding parameters", sum(p.numel() for p in embedding.parameters() if p.requires_grad))
    all_parameters.append({'params': embedding.parameters()})


# 51258938
#  2083333

optimizer = AdamW(all_parameters, lr=config.lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=config.patience, factor=config.decay_factor)

latest = get_latest_file(config.folder_outputs, EXPERIMENT_NAME)
if latest:
    logger.info(f"Loading {latest}")
    old_state = load_model(latest, model, optimizer, scheduler)
    config.starting_epoch, loss = old_state["epoch"], old_state["loss"]

## TEST MODEL
input_ids, attention_mask, labels, mask_index = get_inputs("my name is sam", corpus=corpus, mask_id=config.mask_id)
output = model(input_ids.to(config.device), attention_mask.to(config.device))

prev_input = None
cur_input = None
prev_loss = 100
cur_loss = 100
prev_output = None
cur_output = None
val_loss = []
STEP_GLOBAL = 0

wtd_sum = 0
wt = 0
cer_list = []

lr = optimizer.param_groups[0]['lr']
printR("INITIAL LR", lr)

def run_one_batch_default(sample):
    """

    Args:
        sample:

    Returns:
        output: BATCH, SEQ, VOCAB - BERT + LinearLayer to match output vocabulary
        y_hat:  BATCH, SEQ, VOCAB - uses the MLM decoder - the decoder is just a linear layer -- we haven't been training this one
        y_truth
    """
    if VISION_MODEL_ACTIVE:
        x, y_truth, attention_mask = sample["image"], sample["masked_gt"].to(
        config.device), sample["attention_mask"].to(config.device)
        x = sample["image"].reshape([-1, *sample["image"].shape[2:]])
        x = x.to(config.device)
        if "embedding" in config.experiment.loader_key:
            x = vision_model.get_embedding(x)
        else:
            x = vision_model(x)
        x = x.reshape(*sample["image"].shape[:2], -1)

    else:
        x, y_truth, attention_mask = sample[config.experiment.loader_key].to(config.device), sample["masked_gt"].to(
        config.device), sample["attention_mask"].to(config.device)
    attention_mask = attention_mask.to(config.device)  # 1, sentence_length
    if config.experiment.embedding_layer:
        # Use an additional embedding layer when passing into BERT
        # You still can't use BERT embeddings, since these assume input is discrete
        #output, y_hat = model(input_ids=x, attention_mask=attention_mask)
        x = embedding(x)
    output, y_hat = model(inputs_embeds=x, attention_mask=attention_mask)

    # if EXCLUSIVE:
    #     output = output[sample["mask_idx"]]
    #     y_hat = y_hat[sample["mask_idx"]]
    #     y_truth = y_truth[sample["mask_idx"]]
    return output, y_hat, y_truth

def run_one_batch_lm_only(sample):
    """ Not fully implemented? Main difference is it probably uses a tokenizer etc., not the embedding we used in the vision system
    Args:
        sample:
        exclusive:

    Returns:

    """
    attention_mask = sample["attention_mask"].to(config.device)
    input_ids = y_truth = sample["gt_idxs"].to(config.device)
    output, y_hat = model(input_ids=input_ids, attention_mask=attention_mask)
    # if EXCLUSIVE:
    #     output = output[sample["mask_idx"]]
    #     y_hat = y_hat[sample["mask_idx"]]
    #     y_truth = y_truth[sample["mask_idx"]]
    return output, y_hat, y_truth

run_one = run_one_batch_lm_only if "language_only" in config.experiment_type else run_one_batch_default
losses = []

#### LOAD THE OLD MODELS
if "lm_model_path" in config.folder_dependencies:
    try:
        load_model(ROOT / config.folder_dependencies.lm_model_path, model, optimizer=optimizer, scheduler=scheduler)
    except Exception as e:
        logger.error(str(e))

if "finetuned_cnn_path" in config.folder_dependencies:
    vision_model.load_state_dict(torch.load(config.folder_dependencies.finetuned_cnn_path))

if "results_path" in config.folder_dependencies:
    RESULTS = np.load(config.folder_dependencies.finetuned_cnn_path, allow_pickle=True).item()
else:
    RESULTS["train_loss"] = losses
    RESULTS["train_CER"] = cer_list
    RESULTS["test_loss"] = []
    RESULTS["test_CER"] = []

config.folder_dependencies.lm_model_path = SAVE_PATH = incrementer(MODEL_PATH, EXPERIMENT_NAME + ".pt", incrementer=False)
config.folder_dependencies.finetuned_cnn_path = SAVE_PATH_VISION = incrementer(MODEL_PATH, EXPERIMENT_NAME + "_CNN.pt", incrementer=False)
config.folder_dependencies.finetuned_cnn_path = RESULTS_NPY_PATH = SAVE_PATH.with_suffix(".npy")

COUNTER = stats.Counter(instances_per_epoch=config.epoch_length)
l1 = stats.AutoStat(COUNTER, name="Loss1")
l2 = stats.AutoStat(COUNTER, name="Loss2")
STAT = l1
def run_epoch():
    global STAT
    global losses, STEP_GLOBAL
    logger.info(("epoch", epoch))
    train_dataset.set_train_mode()
    model.train()
    losses_10 = []
    start_time = datetime.datetime.now()
    COUNTER.epochs += 1
    for ii, sample in enumerate(train_loader):
        output, y_hat, y_truth = run_one(sample)
        optimizer.zero_grad()
        COUNTER.update(updates=1, instances=y_hat.shape[0], preds=sample["num_preds"])

        ### -100 is a special index that is IGNORED by the objective; so everything was already "exclusive"; y_truth INDICES of correct letters, -100 ignored
        loss = objective(output.view(-1, config.vocab_size_extended), y_truth.squeeze(0).view(-1).to(config.device)) # y_hat includes extra tokens?
        loss.backward()
        STAT.accumulate(loss.item(), weight=sample["num_preds"])
        losses_10.append(loss.item())
        optimizer.step()

        STEP_GLOBAL = STEP_GLOBAL + 1
        if torch.any(torch.isnan(loss)):
            logger.error("NaN in loss")
            return

        if STEP_GLOBAL % config.steps_per_lr_update == 0 or STEP_GLOBAL < 10:
            l1.reset_accumulator()
            l2.reset_accumulator()
            logger.info(("L1", STEP_GLOBAL, l1))
            logger.info(("L2", STEP_GLOBAL, l2))

            l = np.average(losses_10)
            losses.append(l)
            losses_10 = []
            logger.info(("STEP", STEP_GLOBAL, l))
            cer_list.append(cer_calculation(sample,output))
            logger.info(("CER", cer_list[-1]))
            scheduler.step(l)
            lr = optimizer.param_groups[0]['lr']
            if lr < config["lr"]:
                logger.info(("New LR:", lr))
                config["lr"] = lr
            if config.wandb:
                wandb.log({"loss": l, "epoch": epoch, "step": ii, "cer":cer_list[-1]}, step=STEP_GLOBAL, commit=False)

        # if (datetime.datetime.now() - start_time).seconds / 60 > config.update_freq_time:
        #     start_time = datetime.datetime.now()
        #     break
        # Define an epoch by some arbitrary number of updates
        if ii*config.batch_size > config.epoch_length:
            break

        ### CHANGE TRAINING MODE IF NEEDED
        old_mode = train_dataset.train_mode
        if config.train_mode2:
            ### YOU CAN'T CHANGE THE DATALOADER ONCE YOU'VE SAMPLED IT -- EVERYTHING IS LAGGED ONE UPDATE
            if epoch >= config.train_mode2_start and random.random() < config.train_mode2_probability:
                train_dataset.train_mode = config.train_mode2
                STAT = l1
            else:
                train_dataset.train_mode = config.train_mode
                STAT = l2
            if old_mode != train_dataset.train_mode:
                train_dataset.parse_train_mode()



    if epoch % config.save_freq_epoch == 0:
        save_model(SAVE_PATH, model, optimizer, epoch=epoch+1, loss=losses[-1] if losses else 0, scheduler=scheduler)
        if VISION_MODEL_ACTIVE:
            save_model(SAVE_PATH_VISION, vision_model, optimizer, epoch=epoch+1, loss=losses[-1] if losses else 0, scheduler=scheduler)
    # RUN TEST
    np.save(RESULTS_NPY_PATH, RESULTS, allow_pickle=True)
    plot(losses, MODEL_PATH / "losses.png")

def run_test_set():
    torch.cuda.empty_cache()
    cer = [] ; losses = []
    train_dataset.set_test_mode()
    start_time = datetime.datetime.now()
    model.eval()
    try:
        for sample in train_loader:
            with torch.no_grad():
                output, y_hat, y_truth = run_one(sample)
                loss = objective(output.view(-1, config.vocab_size_extended), y_truth.squeeze(0).view(-1).to(config.device)) # y_hat includes extra tokens?
            cer.append(cer_calculation(sample, output)); losses.append(loss.cpu().detach().item())

            # 1 minute of testing
            if (datetime.datetime.now() - start_time).seconds / 60 > 1:
                start_time = datetime.datetime.now()
                break

            if config.TESTING:
                break

    except Exception as e:
        logger.info((e))
    avg_loss = np.average(losses)
    avg_cer = np.average(cer)
    logger.info(f"TEST CER: {avg_cer:0.3f}")
    logger.info(f"TEST loss: {avg_loss:0.3f}")
    RESULTS["test_loss"].append(avg_loss)
    RESULTS["test_CER"].append(avg_cer)
    plot(RESULTS["test_CER"], MODEL_PATH / "TEST_CER.png")
    return avg_loss, avg_cer


def load_vision():
    sys.path.append(str(ROOT / "models"))
    import VGG
    vision_model = VGG.VGG_embedding().to(config.device)
    VGG.loadVGG(vision_model, path=VGG_MODEL_PATH)
    return vision_model
    #load_model(VGG_MODEL_PATH, vision_model, optimizer=optimizer, scheduler=scheduler)

def check_epoch():
    global VISION_MODEL_ACTIVE, train_loader
    if config.vision_fine_tuning and config.vision_fine_tuning_start == epoch:
        logger.info("ACTIVATING VISION MODEL")
        vision_model = load_vision()
        optimizer.add_param_group({'params': vision_model.parameters()})
        config.batch_size = int(config.batch_size/config.sentence_length)
        config.steps_per_lr_update *= config.sentence_length
        VISION_MODEL_ACTIVE = True
        train_dataset.which = "Both"
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=config.batch_size,
                                                   shuffle=True,
                                                   collate_fn=collate_fn_embeddings,
                                                   num_workers=config.workers)

        for g in optimizer.param_groups:
            g['lr'] = g['lr'] / config.sentence_length

        train_dataset.load_images()

for epoch in range(config.starting_epoch if config.starting_epoch else 0, config.epochs):
    check_epoch()
    run_epoch()
    test_loss, test_cer = run_test_set()
    if config.wandb:
        wandb.log({"test_loss": test_loss, "epoch": epoch, "test_cer": test_cer}, step=STEP_GLOBAL, commit=True)

total = 0
correct = 0
