import numpy as np
from general_tools.utils import get_root
ROOT = get_root("internn")
import sys
sys.path.append(ROOT)
from internn_utils import *
from pytorch_utils import *
from sen_loader import get_text
from error_measures import *

def cer_calculation(sample,output):
    wtd_sum = 0;wt = 0

    ### CER CALCULATION
    text = [s.lower() for s in sample["text"]]
    wt += np.array(sample["length"]).sum();
    out_text = [get_text(o.argmax(-1))[:sample["length"][i]] for i, o in enumerate(output)];
    for i, t in enumerate(text):
        wtd_sum += cer(text[i], out_text[i]) * sample["length"][i]
    print(out_text[0], ";", text[0])
    _cer = wtd_sum / wt
    print("CER", _cer)
    return _cer
