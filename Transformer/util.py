import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=None,uni=False):
        self.src = src[:,:-1]
        if uni==False:
            self.src_mask = (self.src != pad).unsqueeze(-2)
        else:
            self.src_mask = self.make_std_mask(self.src, pad)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

def greedy_decode(model, src, src_mask, max_len, start_symbol,pad):
    "use decoder for generating"
    with torch.no_grad():
        memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    total=torch.ones(1, 1).fill_(0).type_as(src.data)
    for i in range(max_len-1):
        #global prob
        with torch.no_grad():
            out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        #_, next_word = torch.max(prob, dim = 1)
        next_word =torch.multinomial(torch.exp(prob), 1).item()
        ys = torch.cat([ys,torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)[:,-100:]
        total = torch.cat([total,torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        src=torch.cat([src,torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)[:,-5000:]
        with torch.no_grad():
            src_mask=(src != pad).unsqueeze(-2)
            memory = model.encode(src, src_mask)
    return total[0][1:]
