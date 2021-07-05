from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.compat.v1.enable_eager_execution() 

import numpy as np
import os
import time
from bs4 import BeautifulSoup
import requests
import json
import pandas as pd
import argparse



def split_input_target(chunk):
    """
    split sequences into input and target.
    """
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def data_as_input(data,BATCH_SIZE ,shuffle,seq_length,shift=1,BUFFER_SIZE = 50000):
    char_dataset = tf.data.Dataset.from_tensor_slices(data)
    sequences = char_dataset.batch(seq_length+shift, drop_remainder=True)
    dataset = sequences.map(split_input_target)
    
    examples_per_epoch = len(data)//(seq_length+shift)
    steps_per_epoch = examples_per_epoch//BATCH_SIZE
    if shuffle=='shuffle':
        print('shuffle data')
        dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    else:
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset,steps_per_epoch
def mask_data(data,cut_c=5,cut_l=100):
    fre,infre=pd.DataFrame(),pd.DataFrame()
    for sta in data.state.unique():
        data_t=data[data.state==sta]
        data_t=data_t.value_counts().reset_index(name='counts')

        #set to frequent or infrequent by cutoffs
        infre_t=data_t[(data_t.counts<cut_c) & (data_t.length>cut_l)]  
        fre_t=data_t.drop(infre_t.index)    
        fre=fre.append(fre_t,ignore_index=True)

        infre_t.reset_index(drop=True,inplace=True)  #reset index for slicing


        while len(infre_t.length)>0:    
            min_infre=infre_t.length.min()
            #cluster whose lengths difference are in 10 steps
            to_cluster=infre_t[(infre_t.length-min_infre)<11].copy()#without copy() there would be SettingWithCopyWarning
            #drop these in cluster
            infre_t=infre_t.drop(to_cluster.index)#use inplace=True will cause SettingWithCopyWarning

            to_cluster.reset_index(drop=True,inplace=True)
            #use the first one in cluster to substitute rest and add up their count for validation
            to_cluster.loc[0,'counts']=to_cluster.counts.sum()
            infre=infre.append(to_cluster[:1],ignore_index=True)
            for r in to_cluster.raw:
                data.loc[data.raw==r,'raw']=to_cluster.raw[0]
    return data
def build_model(vocab_size, embedding_dim, rnn_units, batch_size,state):
    if tf.test.is_gpu_available():
        rnn = tf.compat.v1.keras.layers.CuDNNLSTM
    else:
        import functools
        rnn = functools.partial(
        tf.keras.layers.LSTM, recurrent_activation='sigmoid')
    
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    
    rnn(rnn_units, 
        return_sequences=True,
        #recurrent_initializer='glorot_uniform',
        recurrent_initializer='orthogonal',
        kernel_regularizer='l2',
        recurrent_regularizer='l2',
        stateful=state),

    tf.keras.layers.Dense(vocab_size)
    ])

    return model