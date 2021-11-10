try:
    import pykaldi2 as pyk2
    import pykaldi  as pyk
    from pykaldi.decoder.fasterdecoder import FasterDecoder
except:
    print("Problem Loading pykaldi")

from kstats import KStats
import signal

import multiprocessing
import traceback
import sys
import string_utils
import time

KALDI_SPECIAL_SYM = {
    "<space>": " ",
    "<unk>": "",
    "<eps>": ""
}
def kaldi2str_single(kaldi_out):
    return u"".join([KALDI_SPECIAL_SYM.get(k, k) for k in kaldi_out])

def run_decode(data):
    global kaldi_decoder_decoder
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        raw_kaldi = kaldi_decoder_decoder.DecodeOne(data.tolist(), kaldi_decoder_decoder)
        kaldi_unicode = kaldi2str_single(raw_kaldi['words'])
        return kaldi_unicode, raw_kaldi['like']
    except:
        traceback.print_exc(file=sys.stdout)
        raise Exception("Error in decode")

COUNTER = 0

class KaldiDecoder(object):

    def __init__(self, idx_to_char, params={}):


        mdl_path = str(params['mdl_path'])
        fst_path = str(params['fst_path'])
        txt_path = str(params['words_path'])

        ac_scale = params.get('acoustic', -1.2)
        allow_partial = params.get('allow_partial', True)
        beam = params.get('beam', 13)
        self.alphaweight = params.get('alphaweight', 0.3)

        self.pool = None
        self.multiprocessing = params.get('multiprocessing', False)
        if params.get('multiprocessing', False):
            self.processes = params['processes']
            print("Using multiprocessing LM decode with {} processes".format(params['processes']))
        else:
            print("Not using multiprocessing for LM decode, add to config if needed")

        self.decoder = FasterDecoder(mdl_path, fst_path, txt_path)
        self.decoder.Init(ac_scale=ac_scale, allow_partial=True, beam=beam)

        self.external_symbols = params['phones_path']
        self.idx_to_char = idx_to_char
        self.char_to_idx = {v: k for k, v in idx_to_char.iteritems()}

        self.reset()

        print("LM Decoder Ready")

    def single_process_decode(self, data):
        reweighted = self.stats.out2Kaldi_rev2(data, self.alphaweight)
        raw_kaldi = self.decoder.DecodeOne(reweighted.tolist(), self.decoder)
        kaldi_unicode = kaldi2str_single(raw_kaldi['words'])
        return kaldi_unicode, raw_kaldi['like']

    def decode(self, data, as_idx=False):
        #COUNTER += 1
        #print(data.shape
        pred, full_pred = t_data = string_utils.naive_decode(data)
        #print("---"
        res = self._decode(data)
        #global COUNTER 
        #if COUNTER % 100 == 0:
        #    print(COUNTER
        #    print("Pr:", string_utils.label2str_single(pred, self.idx_to_char, False).encode("utf-8")
        #    print("Ex:", res[0].encode("utf-8")
        #raw_input()
        if not as_idx:
            return res
        return string_utils.str2label_single(res[0], self.char_to_idx), res[1]

    def _decode(self, data):
        if self.add_stats_phase:
            self.stats.print_kaldi_stats()
            self.add_stats_phase = False

        if len(data.shape) == 3:

            if self.multiprocessing:
                new_data = []
                #This could be moved to the multi processing, but it is
                #already really fast and I'm not sure if it try to copy data
                #to the threads memory, instead of just keeping everything
                #shared
                #http://stackoverflow.com/questions/19366259/multiprocessing-in-python-with-read-only-shared-memory
                for d in data:
                    reweighted = self.stats.out2Kaldi_rev2(d, self.alphaweight)
                    new_data.append(reweighted)

                global kaldi_decoder_decoder
                kaldi_decoder_decoder = self.decoder

                if self.pool is None:
                    self.pool = multiprocessing.Pool(processes=self.processes)

                try:
                    res = self.pool.map(run_decode, new_data)
                except KeyboardInterrupt:
                    self.pool.terminate()
                    self.pool.join()
                    raise KeyboardInterrupt()
                return res
            else:
                return [self.single_process_decode(d) for d in data ]

        else:
            # start = time.time()
            d = self.single_process_decode(data)
            # print(time.time() - start, d
            return d


    def add_stats(self, data):
        if not self.add_stats_phase:
            print("WARNING: Reseting kaldi stats because they are added after decoding")
            self.reset()
        self.stats.add_kaldi_stats(data)

    def reset(self):
        self.stats = KStats(self.idx_to_char, self.external_symbols)
        self.add_stats_phase = True
