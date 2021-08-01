import string
from reformer import reformer
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForMaskedLM, AutoModelForSeq2SeqLM
import torch.nn as nn
import torch

from general_tools.utils import get_root
ROOT = get_root("internn")

# ReformerForMaskedLM
class CharacterBERT(nn.Module):
    """ Pass a bunch of vectors into it
        Return predictions
    """

    def __init__(self, bidirectional=True):
        super().__init__()
        self.DEVICE = "cuda:0"
        # tokenizer = AutoTokenizer.from_pretrained("google/reformer-enwik8")
        AutoModel = AutoModelForMaskedLM if bidirectional else AutoModelWithLMHead
        self.model = AutoModel.from_pretrained("helboukkouri/character-bert").to(self.DEVICE)
        self.alphabet = list((string.ascii_lowercase + string.ascii_uppercase))

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

    def generate(self, prime):
        encoded, attention_masks = reformer.encode([prime])
        return reformer.decode(self.model.generate(encoded.to(self.DEVICE), do_sample=True, max_length=150))


    def pred(self, cnn_output):
        """

        Returns:
            (tuple) logits :
                    characters : e.g., "This is the prediction"

        """
        cnn_output = cnn_output.to(self.DEVICE)
        logits = self.model(cnn_output).logits # batch_size, num_predict, config.vocab_size
        characters = self.decode(logits.argmax(axis=2))

        return logits, characters

    # def get_loss(self, gt):
    #     logits, characters = self.pred()
    #     return logits - gt

if __name__=='__main__':
    bert = CharacterBERT()
    encoded, attention_masks = bert.encode(["this is a"])
    bert.pred(encoded)
