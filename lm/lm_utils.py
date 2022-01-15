import numpy as np
from general_tools.utils import get_root, get_max_root
ROOT = get_root("internn")
import sys
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "data"))

print(sys.path)
from internn_utils import *
from pytorch_utils import *
from sen_loader import get_text
from error_measures import *
eps = 1e-7

def sample_to_text(sample, output):
    text = [s.lower() for s in sample["text"]]
    out_text = [get_text(o.argmax(-1))[:sample["length"][i]] for i, o in enumerate(output)]
    return text, out_text

def cer_index(sample,output,index, **kwargs):
    """

    Args:
        sample:
        output:
        index: 2D array, Batch x indices of preds

    Returns:

    """
    text, out_text = sample_to_text(sample,output)
    right = wrong = 0
    for i in range(len(text)):
        for ii in index[i]:
            if text[ii] == out_text[ii]:
                right += 1
            else:
                wrong += 1
    if right+wrong==0:
        pass
    return right / (right + wrong + eps)

def cer_calculation(sample,output,verbose=True, **kwargs):
    wtd_sum = 0;wt = 0

    ### CER CALCULATION
    text, out_text = sample_to_text(sample,output)

    wt += np.array(sample["length"]).sum();

    for i in range(len(text)):
        wtd_sum += cer(text[i], out_text[i]) * sample["length"][i]
    if verbose:
        print(out_text[0], ";", text[0])
    _cer = wtd_sum / wt
    return _cer
