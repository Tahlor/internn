from reformer import reformer
from transformers import AutoTokenizer, AutoModelWithLMHead
DEVICE="cuda:0"
#tokenizer = AutoTokenizer.from_pretrained("google/reformer-enwik8")
model = AutoModelWithLMHead.from_pretrained("google/reformer-enwik8").to(DEVICE)
model = AutoModelWithLMHead.from_pretrained("google/reformer-enwik8").to(DEVICE)

def predict(prime="In 1965, Brooks left IBM to found the Department of"):
    encoded, attention_masks = reformer.encode([prime])
    globals().update(locals())
    return reformer.decode(model.generate(encoded.to(DEVICE), do_sample=True, max_length=150))

print(predict())

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-small")
# model = AutoModelForSeq2SeqLM.from_pretrained("google/t5-v1_1-small")
import transformers
transformers.models.reformer.modeling_reformer.ReformerModelWithLMHeadOutput