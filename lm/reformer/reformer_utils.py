import torch
from transformers import LineByLineTextDataset, PreTrainedTokenizer


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


class CharTokenizer(PreTrainedTokenizer):

    def __init__(
        self,
        vocab_file,
        eos_token="|",
        unk_token="*",
        additional_special_tokens=[],
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        super().__init__(
            eos_token=eos_token,
            unk_token=unk_token,
            additional_special_tokens=additional_special_tokens,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

        self.vocab_file = vocab_file
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)

    @staticmethod
    def decode(*args, **kwargs):
        decode(*args, **kwargs)

    @staticmethod
    def encode(*args, **kwargs):
        decode(*args, **kwargs)
