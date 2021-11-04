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

def cer_calculation(sample,output,verbose=True):
    wtd_sum = 0;wt = 0

    ### CER CALCULATION
    text = [s.lower() for s in sample["text"]]
    wt += np.array(sample["length"]).sum();
    out_text = [get_text(o.argmax(-1))[:sample["length"][i]] for i, o in enumerate(output)];
    for i, t in enumerate(text):
        wtd_sum += cer(text[i], out_text[i]) * sample["length"][i]
    if verbose:
        print(out_text[0], ";", text[0])
    _cer = wtd_sum / wt
    return _cer
