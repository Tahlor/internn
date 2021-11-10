#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import numpy as np
import uuid
import codecs
import re
import math
import operator
import os
import uuid
import struct

class KStats:
    def __init__(self, idx_to_char, external_symbols):
        self.idx_to_char = idx_to_char
        self.external_symbols = external_symbols
        self.statscount = 0
        self.stats2count = 0
        self.normtries = 0
        self.maxid =-1
        self.symstats = {}
        self.dictSave = {}
        self.dictId2Phn = {}
        self.dictRes = {}
        self.DEFCOUNT = 1.0E-10
        self.DWEIGHT = -1*math.log(5.0e-16)
        self.dictMorph = {'[': 'LSB', ']': 'RSB', ';': 'SEMI', ':': 'COLON', '_': 'USCORE',
               '!': 'EXCL', '=': 'EQUALS', '~': 'TILDE', '(': 'LRB', ')': 'RRB',
               '\'': 'APOS', '\"': 'QUOTE', '#': 'POUND', '+': 'PLUS', '%': 'PCT', '^': 'CARAT',
               '@': 'ATSIGN', '?': 'QMARK', '>': 'LTSIGN', '<': 'GTSIGN', '$': 'DOLLAR', '&': 'AND',
               '*': 'STAR', ',': 'COMMA', '.': 'PERIOD', '/': 'SLASH', ' ': 'SPACE', '0': 'NUM0',
               '1': 'NUM1', '2': 'NUM2', '3': 'NUM3', '4': 'NUM4', '5': 'NUM5', '6': 'NUM6',
               '7': 'NUM7', '8': 'NUM8', '9': 'NUM9', '-': 'DASH'
           }
        self.create_kaldi_map(self.dictSave, self.dictId2Phn, self.dictRes, self.external_symbols)

        self.optimized_dict = None

    def renamePhone(self, phn):
        return self.dictMorph.get(phn,phn)

    def create_kaldi_map(self, dictSave, dictId2Phn, dictRes, filename):
        dictRes["COUNT"] = 0
        dictId2Phn["COUNT"] = "COUNT"
        #open the kaldi index in self.external_symbols
        with codecs.open(filename,'r',encoding='utf8') as f:
          data = f.read()
        f.close()

        for index, text in enumerate(data.split("\n")):
          entries = re.split('\s', text, 2)
          if (len(entries)<2 or len(entries[1])==0):
            continue
          dictSave[entries[0]] = int(entries[1])
          dictId2Phn[int(entries[1])] = entries[0]
          dictRes[int(entries[1])] = self.DEFCOUNT
          dictRes['COUNT'] = self.DEFCOUNT + dictRes['COUNT']

        dictSave['EPS'] = dictSave['NON']
        for key in self.dictMorph:
          if (dictSave.get(self.dictMorph[key],None) is None):
            continue
          dictSave[key] = dictSave[self.dictMorph[key]]
        #local phones to kaldi index

    #Equivalent of stats-getter
    def print_kaldi_stats(self):
        if self.normtries > 0:
          return
        self.normtries = self.normtries + 1
        for phnpair in sorted(self.dictRes.items(), key=operator.itemgetter(1), reverse=True):
          phnid = phnpair[0]
          phn   = self.dictId2Phn[phnid]
          if (phn == "<eps>" or phn.startswith( '#')):
            continue
          if phnid == "COUNT":
            hmm = phnid
          else:
            hmm = int(phnid)-1
            wt = self.dictRes[phnid]/self.dictRes["COUNT"]
            # print "WT=",wt," for phnid=",phnid," phn=",phn
            #if wt<=0:
            #  wt=self.DEFCOUNT
            self.dictRes[phnid] = math.log(wt)
            if int(phnid) > self.maxid:
              self.maxid=int(phnid)
          #Uncomment if desired:
          #print hmm,"\t",phn,"\t",self.dictRes[phnid]

    def get_optimized_data(self):
        if self.optimized_dict is not None:
            return self.optimized_dict, self.reorder_array_1, self.reorder_array_2

        reorder_array_1 = []
        reorder_array_2 = []
        optimized_dict = np.zeros(len(self.idx_to_char)+1, dtype=np.float64)
        for pyphnid in range(len(self.idx_to_char)+1):
            if (pyphnid == 0):
                a = "EPS"
            else:
                a = self.idx_to_char[pyphnid]
                # if a == ' ' or a == u'º':
                if a == ' ':
                    a="SPACE"
            newa = self.dictSave.get(a,None)
            if newa == None:
                continue

            reorder_array_1.append(newa-1)
            reorder_array_2.append(pyphnid)
            optimized_dict[pyphnid] = self.dictRes[newa]


        self.reorder_array_1 = np.array(reorder_array_1)
        self.reorder_array_2 = np.array(reorder_array_2)
        self.optimized_dict = optimized_dict

        return self.optimized_dict, self.reorder_array_1, self.reorder_array_2


    def out2Kaldi_rev2(self, data, alphaweight):

        optimized_dict, reorder_array_1, reorder_array_2 = self.get_optimized_data()
        out = np.full((data.shape[0],self.maxid), self.DWEIGHT, dtype=np.float32)
        tmp_data = -1 * (data - alphaweight * optimized_dict)
        out[:,reorder_array_1] = tmp_data[:,reorder_array_2]

        return out

    def add_kaldi_stats(self, data):
        self.dictRes["COUNT"] = self.dictRes.get("COUNT",self.DEFCOUNT) + data.shape[0]
        for pyphnid in range(data.shape[1]):
          if (pyphnid == 0):
            a = "EPS"
          else:
            a = self.idx_to_char[pyphnid]
            if a == ' ' or a == u'º':
               a="SPACE"
          newa = self.dictSave.get(a,None)
          if newa == None:
             continue
          for timestep in range(data.shape[0]):
             b = data[timestep][pyphnid]
             self.dictRes[newa] = self.dictRes.get(newa,self.DEFCOUNT) + math.exp(b)
        self.statscount += data.shape[0]
