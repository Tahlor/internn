# -*- coding: utf-8 -*-
"""CharBERTV3_BASE

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sOlk2rIdrl2_LSFyi1WZUa7Sc9Kn94qs

"""
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

print("ARG: ", sys.argv[1])
config = process_config_BERT.process_config(sys.argv[1])

print(config)

if config.wandb:
    wandb.config = config
    wandb.init(project="TrainBERT")

LOADER_PATH = ROOT / config.folder_dependencies.embedding_dataset_folder
VGG_MODEL_PATH = ROOT / config.folder_dependencies.VGG_model_folder
MODEL_PATH = ROOT / config.folder_outputs
text = config.alphabet
corpus = [char for char in text]
EXPERIMENT_NAME = config.experiment_prefix + config.experiment_type

train_dataset = SentenceDataset(PATH=LOADER_PATH / 'train_test_sentenceDataset.pt',
                                which='Embeddings',
                                train_mode=config.train_mode,
                                sentence_length=config.sentence_length,
                                sentence_filter="lowercase",
                                vocab_size=config.vocab_size,
                                normalize=config.embedding_norm,

                                )

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=config.batch_size,
                                           shuffle=True,
                                           collate_fn=collate_fn_embeddings,
                                           num_workers=config.workers)


sample = next(iter(train_dataset))
print("Batch Data Shape = ", sample["data"].squeeze(0).shape)
print("Num Batch Labels = ", sample["gt_one_hot"].squeeze(0).shape)
print("Output shape: ", sample["vgg_logits"].shape)
print("Sen Lengths: ", sample["length"])
print(get_text(sample["gt_one_hot"]))

model = BertModelCustom(BertConfig(vocab_size=config.vocab_size_extended,
                                   hidden_size=config.experiment.embedding_dim,
                                   num_attention_heads=config.experiment.attention_heads)).to(config.device)

objective = nn.CrossEntropyLoss()
#objective = nn.NLLLoss(ignore_index=0)

parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"PARAMETERS: {parameters}")

# 51258938
#  2083333

optimizer = AdamW(model.parameters(), lr=config.lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=config.patience, factor=config.decay_factor)

latest = get_latest_file(config.folder_outputs, EXPERIMENT_NAME)
if latest:
    print(f"Loading {latest}")
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

train_dataset.train_mode = "full sequence"

lr = optimizer.param_groups[0]['lr']
print("INITIAL LR: ", lr)

embedding = nn.Linear(config.vocab_size, config.experiment.embedding_dim).to(config.device) if config.experiment.embedding_layer else None

def run_one_batch_default(sample):
    x, y_truth, attention_mask = sample[config.experiment.loader_key].to(config.device), sample["masked_gt"].to(
        config.device), sample["attention_mask"].to(config.device)
    attention_mask = attention_mask.to(config.device)  # 1, sentence_length
    if config.experiment.embedding_layer:
        # Use an additional embedding layer when passing into BERT
        # You still can't use BERT embeddings, since these assume input is discrete
        #output, y_hat = model(input_ids=x, attention_mask=attention_mask)
        x = embedding(x)
    output, y_hat = model(inputs_embeds=x, attention_mask=attention_mask)
    return output, y_hat, y_truth

def run_one_batch_lm_only(sample):
    attention_mask = sample["attention_mask"].to(config.device)
    input_ids = y_truth = sample["gt_idxs"].to(config.device)
    output, y_hat = model(input_ids=input_ids, attention_mask=attention_mask)
    return output, y_hat, y_truth

run_one = run_one_batch_lm_only if "language_only" in config.experiment_type else run_one_batch_default
losses = []
SAVE_PATH = incrementer(MODEL_PATH, EXPERIMENT_NAME + ".pt")

def run_epoch():
    global losses, STEP_GLOBAL
    print("epoch", epoch)
    train_dataset.set_train_mode()
    model.train()
    losses_10 = []
    start_time = datetime.datetime.now()
    for ii, sample in enumerate(train_loader):
        # train_dataset.train_mode = "single character" if random.random() < .5 else "full sequence"
        output, y_hat, y_truth = run_one(sample)
        optimizer.zero_grad()
        loss = objective(output.view(-1, config.vocab_size_extended), y_truth.squeeze(0).view(-1).to(config.device)) # y_hat includes extra tokens?
        loss.backward()
        losses_10.append(loss.item())
        optimizer.step()

        #get_text(output.squeeze())
        #sample["text"]

        STEP_GLOBAL = STEP_GLOBAL + 1
        if STEP_GLOBAL % config.steps_per_lr_update == 0 or STEP_GLOBAL < 10:
            l = np.average(losses_10)
            losses.append(l)
            losses_10 = []
            print("STEP", STEP_GLOBAL, l)
            cer_list.append(cer_calculation(sample,output))
            print("CER", cer_list[-1])
            scheduler.step(l)
            lr = optimizer.param_groups[0]['lr']
            if lr < config["lr"]:
                print("New LR:", lr)
                config["lr"] = lr
            if config.wandb:
                wandb.log({"loss": l, "epoch": epoch, "step": ii}, step=STEP_GLOBAL, commit=False)

        if (datetime.datetime.now() - start_time).seconds / 60 > config.update_freq_time:
            start_time = datetime.datetime.now()
            break

    if epoch % config.save_freq_epoch == 0:
        save_model(SAVE_PATH, model, optimizer, epoch=epoch+1, loss=losses[-1] if losses else 0, scheduler=scheduler)

    # RUN TEST

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
    except Exception as e:
        print(e)
    avg_loss = np.average(losses)
    avg_cer = np.average(cer)
    print(f"TEST CER: {avg_cer:0.3f}")
    print(f"TEST loss: {avg_loss:0.3f}")


for epoch in range(config.starting_epoch if config.starting_epoch else 0, config.epochs):
    run_epoch()
    run_test_set()

total = 0
correct = 0

