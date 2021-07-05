#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import argparse
import os
from bs4 import BeautifulSoup
import requests
import json
import time
import random
from torch.utils.data import Dataset, DataLoader

from model import LabelSmoothing,NoamOpt,SimpleLossCompute,make_model,run_epoch
from util import Batch
from util import greedy_decode

parser = argparse.ArgumentParser(description='Transformer Task')
parser.add_argument('--task', type=str, default='phi',choices=['RMSD','phi','psi'],help='choose dataset as phi/psi/RMSD')
parser.add_argument('--batch_size', type=int, default=32)

parser.add_argument('--interval', type=int, default=10,help='saving interval ')

parser.add_argument('--seq_length', type=int, default=100)

parser.add_argument('--unidirection', type=bool, default=False,help='use mask to do unidirection encoding in transformer')

parser.add_argument('--gpu_id', type=str, default='cuda:3')

args = parser.parse_args()
task=args.task
batch_size=args.batch_size
interval=args.interval
device= torch.device(args.gpu_id)
uni=args.unidirection

# load data
if task=='RMSD':
    train,valid=np.loadtxt('data/alanine/train',dtype=int),np.loadtxt('data/alanine/valid',dtype=int)
elif task=='phi':
    train,valid=np.loadtxt('data/phi-psi/train_phi_0.1ps',dtype=int).reshape(-1),np.loadtxt('data/phi-psi/valid_phi_0.1ps',dtype=int).reshape(-1)
elif task=='psi':
    train,valid=np.loadtxt('data/phi-psi/train_psi_0.1ps',dtype=int).reshape(-1),np.loadtxt('data/phi-psi/valid_psi_0.10ps',dtype=int).reshape(-1)

# subsample x to corresponding interval
train=train.reshape(interval,-1).T.flatten().reshape(-1,100)
valid=valid.reshape(interval,-1).T.flatten().reshape(-1,100)
    
# make dirs for saving files

log_dir="Transformer/logs/fit/{}/interval{}_batch{}_uni_{}/".format(task,interval,batch_size,uni)
os.makedirs(log_dir, exist_ok=True)
save_dir = "Transformer/result/{}/interval{}_batch{}_uni_{}/".format(task,interval,batch_size,uni)
os.makedirs(save_dir, exist_ok=True)
ckpt_dir="Transformer/ckpt/training_checkpoints_{}/interval{}_batch{}_uni_{}/".format(task,interval,batch_size,uni)
os.makedirs(ckpt_dir, exist_ok=True)

epoch=200
num_generate=100000
# -------training--------------
def data_generator(fulldata, batch,pad):
    "Generate random data for a src copy task."
    nbatches=int(fulldata.shape[0]//batch)
    choice=[i for i in range(nbatches)]  
    random.shuffle(choice)
    for i in choice:
        data=fulldata[i*batch:(i+1)*batch]      
        src = Variable(torch.from_numpy(data), requires_grad=False)
        yield Batch(src.to(device), src.to(device), pad,uni=uni)

V = len(np.unique(train))
pad=V+1
criterion = LabelSmoothing(size=V, padding_idx=pad, smoothing=0.0)
model = make_model(V, V, N=2)
model.to(device)
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir)

for epoch in range(epoch):
    start = time.time()
    model.train()
    loss_train=run_epoch(data_generator(train, batch_size, pad), model, 
              SimpleLossCompute(model.generator, criterion, model_opt,device))
    elapsed = time.time() - start
    print("Epoch  %d  Time: %f" %
                    (epoch, elapsed))
    model.eval()
    loss_valid=run_epoch(data_generator(valid, batch_size, pad), model, 
                    SimpleLossCompute(model.generator, criterion, None,device))
    if epoch<5 or epoch%5==0:
        torch.save(model.state_dict(), ckpt_dir+'epoch{}.pt'.format(epoch))
    writer.add_scalar('Loss/train', loss_train, epoch)
    writer.add_scalar('Loss/test', loss_valid, epoch)

# ---------generating----------
model.eval()
def single_generation(i):
    seg=int(len(train.reshape(-1))/100)
    text4activation=train.reshape(-1)[i*seg:(i+1)*seg]  
    start0 = time.time()
    src = Variable(torch.from_numpy(text4activation[-5000:]).unsqueeze(0)).to(device)
    src_mask = (src != pad).unsqueeze(-2)
    start_symbol=src[-1][-1]

    prediction=greedy_decode(model, src, src_mask, max_len=num_generate, start_symbol=start_symbol,pad=pad)
    print ('Time taken for total {} sec\n'.format(time.time() - start0))
    save=save_dir+'epoch{}/'.format(epoch)
    os.makedirs(save, exist_ok=True)
    np.savetxt(save+'prediction_'+str(i),prediction.cpu(),fmt='%i')

#for epoch in [2,5,90,100]:
epoch4pre=90
model.load_state_dict(torch.load(ckpt_dir+'epoch{}.pt'.format(epoch4pre)))
for i in range(100):
    single_generation(i)
    
#from joblib import Parallel, delayed
#Parallel(n_jobs=-1, verbose=100)(delayed(single_generation)(i) for i in range(100))





