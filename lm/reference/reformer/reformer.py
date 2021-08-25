import torch

class reformer:

    def __init__(self):
        from transformers import AutoTokenizer, AutoModelWithLMHead
        self.model = AutoModelWithLMHead.from_pretrained("google/reformer-enwik8").to("cuda")

    def freeze(self):
        """ Freeze model, but also prevents backprop through network!
        Returns:
        """

        for param in self.model.base_model.parameters():
            param.requires_grad = False

    # Encoding
    @staticmethod
    def encode(list_of_strings, pad_token_id=0):
        max_length = max([len(string) for string in list_of_strings])

        # create emtpy tensors
        attention_masks = torch.zeros((len(list_of_strings), max_length), dtype=torch.long)
        input_ids = torch.full((len(list_of_strings), max_length), pad_token_id, dtype=torch.long)

        for idx, string in enumerate(list_of_strings):
            # make sure string is in byte format
            if not isinstance(string, bytes):
                string = str.encode(string)

            input_ids[idx, :len(string)] = torch.tensor([x + 2 for x in string])
            attention_masks[idx, :len(string)] = 1

        return input_ids, attention_masks

    # Decoding
    @staticmethod
    def decode(outputs_ids):
        decoded_outputs = []
        for output_ids in outputs_ids.tolist():
            # transform id back to char IDs < 2 are simply transformed to ""
            decoded_outputs.append("".join([chr(x - 2) if x > 1 else "" for x in output_ids]))
        return decoded_outputs

if False:
    from transformers import BertForSequenceClassification
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    model.train()

    from transformers import AdamW
    optimizer = AdamW(model.parameters(), lr=1e-5)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text_batch = ["I love Pixar.", "I don't care for Pixar."]
    encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    labels = torch.tensor([1,0]).unsqueeze(0)
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

    # from torch.nn import functional as F
    # labels = torch.tensor([1,0])
    # outputs = model(input_ids, attention_mask=attention_mask)
    # loss = F.cross_entropy(outputs.logits, labels)
    # loss.backward()
    # optimizer.step()

    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)
    loss.backward()
    optimizer.step()
    scheduler.step()