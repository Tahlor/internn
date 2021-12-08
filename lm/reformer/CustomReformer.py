import transformers.generation_logits_process
from transformers import AutoTokenizer, AutoModelWithLMHead
from torch import nn
from torch import tensor
class CustomReformer(AutoModelWithLMHead):
    def __init__(self, config,
                 add_pooling_layer=True,
                 ):
        #super().__init__()
        tokenizer = AutoTokenizer.from_pretrained("google/reformer-enwik8")
        self.model = AutoModelWithLMHead.from_pretrained("google/reformer-enwik8")
        self.config = config

    def forward(self):
        self.lin2 = nn.Linear(self.config.hidden_size, self.config.vocab_size)
        self.model


if __name__ == "__main__":
    #tokenizer = AutoTokenizer.from_pretrained("google/reformer-enwik8")
    #model = AutoModelWithLMHead.from_pretrained("google/reformer-enwik8")
    import torch

    # Encoding
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
    def decode(outputs_ids):
        decoded_outputs = []
        o = outputs_ids.tolist() if torch.is_tensor(outputs_ids) else outputs_ids
        for output_ids in o:
            # transform id back to char IDs < 2 are simply transformed to ""
            decoded_outputs.append("".join([chr(x - 2) if x > 1 else "" for x in output_ids]))
        return decoded_outputs

    from transformers import ReformerModelWithLMHead, ReformerForMaskedLM
    # transformers.ReformerModel - raw hidden states
    # ReformerForMaskedLM - UGH THIS IS WHAT I WANT
    # ReformerModelWithLMHead - next token prediction ONLY
    model = ReformerModelWithLMHead.from_pretrained("google/reformer-enwik8")
    encoded, attention_masks = encode(["In 1965, Brooks left IBM to found the Department of"])
    x = model.generate(encoded, do_sample=True, max_length=150)
    d = decode(x)

    input_ids, attention_masks = encode(["In 1965, Brooks left IBM to found the Department of"])
    #i,a = input_ids.to("cuda"), attention_masks.to("cuda")

    sentence = "The quick brown fox jumps over the lazy dog."
    input_ids, attention_masks = encode([sentence])
    attention_masks[0,37] = attention_masks[0,19] = attention_masks[0,27] = 0
    i,a = input_ids, attention_masks
    f = model.forward(input_ids=i,
                  position_ids=None,
                  attention_mask=a,
                  head_mask=None,
                  inputs_embeds=None,
                  num_hashes=None,
                  past_buckets_states=None,
                  use_cache=None,
                  output_hidden_states=None,
                  output_attentions=None,
                  return_dict=None,
                  labels=i)
    print("loss", f.loss)
    ff = decode(torch.argmax(f.logits, 2))
    print(ff)

    # from transformers import ReformerTokenizer, ReformerModelWithLMHead
    # tokenizer = ReformerTokenizer.from_pretrained('google/reformer-enwik8')
    # model = ReformerModelWithLMHead.from_pretrained('google/reformer-enwik8')
    # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    # outputs = model(**inputs, labels=inputs["input_ids"])
    # loss = outputs.loss
    # logits = outputs.logits

    input()
    transformers.generation_logits_process.MinLengthLogitsProcessor

    for i in range(10,len(sentence)):
        input_ids, attention_masks = encode([sentence[:i]])
        i, a = input_ids, attention_masks
        f = model.forward(input_ids=i,
                          position_ids=None,
                          attention_mask=a,
                          head_mask=None,
                          inputs_embeds=None,
                          num_hashes=None,
                          past_buckets_states=None,
                          use_cache=None,
                          output_hidden_states=None,
                          output_attentions=None,
                          return_dict=None,
                          labels=i)
        prediction = decode(torch.argmax(f.logits[:,-1], 2))
        print(sentence[:i], prediction)

    def masked_mlm():
        from transformers import ReformerConfig, ReformerForMaskedLM
        config = ReformerConfig.from_pretrained('google/reformer-enwik8')
        config.is_decoder = False
        model = ReformerForMaskedLM.from_pretrained('google/reformer-enwik8', config=config)

        sentence = sentence2 = "The quick brown fox jumps over the lazy dog."
        input_ids, attention_masks = encode([sentence])
        if True:
            _input_ids,a = input_ids.clone(), attention_masks.clone()
            for i in [19, 27, 37]:
                a[0, i] = 0
                sentence2 = sentence2[:i] + "%" + sentence2[i + 1:]
        else:
            _input_ids,a = input_ids, attention_masks
        f = model.forward(input_ids=_input_ids,
                          position_ids=None,
                          attention_mask=a,
                          head_mask=None,
                          inputs_embeds=None,
                          num_hashes=None,
                          labels=_input_ids)
        prediction = decode(torch.argmax(f.logits, 2))[0]
        print(sentence2)
        print(prediction)


    def next_word_predictor():
        sentence = sentence2 = "The quick brown fox jumps over the lazy dog."
        for i in range(10, len(sentence)):
            input_ids, attention_masks = encode([sentence[:i]])
            ii, a = input_ids, attention_masks
            f = model.forward(input_ids=ii,
                              position_ids=None,
                              attention_mask=a,
                              head_mask=None,
                              inputs_embeds=None,
                              num_hashes=None,
                              labels=ii)
            prediction = decode(torch.argmax(f.logits[:, -1:], 2))[0]
            print(sentence[:i], prediction)


    def my_generator():
        sentence = sentence2 = "The quick brown fox jumps over the lazy dog."
        for i in range(10, len(sentence)):
            input_ids, attention_masks = encode([sentence[:i]])
            ii, a = input_ids, attention_masks
            f = model.forward(input_ids=ii,
                              position_ids=None,
                              attention_mask=a,
                              head_mask=None,
                              inputs_embeds=None,
                              num_hashes=None,
                              labels=ii)
            prediction = decode(torch.argmax(f.logits[:, -1:], 2))[0]
            sentence = sentence[:i] + prediction
        print(sentence)

    def BERTDEMO():
        """
        inputs = tokenizer("The capital of France is [MASK].")
        {'input_ids': [101, 1996, 3007, 1997, 2605, 2003, 103, 1012, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}

        tokenizer.decode(inputs["input_ids"])
        '[CLS] the capital of france is [MASK]. [SEP]'

        Returns:

        """

        from transformers import BertTokenizer, BertForMaskedLM
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model2 = BertForMaskedLM.from_pretrained('bert-base-uncased')

        inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")
        labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]

        outputs = model2(**inputs, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        preds = torch.argmax(logits, 2).tolist()[0]
        out = tokenizer.decode(preds)
        print(out)

    def attention_test():
        """ # Only changing attention mask yields bad guess -- this is because it wasn't trained to do this!
                                                               it's only trained on predicting "masks" not everything!
            # Only changing to MASK TOKEN (103) yields correct guess
            # Doing both is also correct

        Returns:

        """

        from transformers import BertTokenizer, BertForMaskedLM
        import torch
        from torch import tensor

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model2 = BertForMaskedLM.from_pretrained('bert-base-uncased')

        inputs = tokenizer("The capital of France is Paris.", return_tensors="pt")
        # tokenizer.decode([103]) = '[MASK]'
        inputs['attention_mask'][0, -3] = 0
        inputs['input_ids'][0, -3] = 103

        outputs = model2(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        logits = outputs.logits

        preds = torch.argmax(logits, 2).tolist()[0]
        out = tokenizer.decode(preds)
        print(out)