from transformers import ReformerConfig, ReformerForMaskedLM, ReformerTokenizer, LineByLineTextDataset,DataCollatorForLanguageModeling
from reformer_utils import encode, decode, CharTokenizer
import torch
from general_tools.utils import get_root
from pathlib import Path
ROOT = get_root("internn")

MSK = 44
config = ReformerConfig.from_pretrained('google/reformer-enwik8')
config.is_decoder = False
model = ReformerForMaskedLM.from_pretrained('google/reformer-enwik8', config=config)

sentence = "The quick brown fox jumps over the lazy dog."

input_ids, attention_masks = encode([sentence])
label_ids, _ = encode([sentence])
for idx in [10,21,26,32,35]:
    input_ids[0,idx] = MSK
    attention_masks[0,idx] = 0

f = model.forward(input_ids=input_ids,
                  position_ids=None,
                  attention_mask=attention_masks,
                  head_mask=None,
                  inputs_embeds=None,
                  num_hashes=None,
                  labels=label_ids)

loss = f.loss
prediction = decode(torch.argmax(f.logits, 2))[0]
print(prediction)

tokenizer = CharTokenizer()
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=ROOT / "data/corpus/enwik8",
    block_size=128)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,
)

trainer.train()
