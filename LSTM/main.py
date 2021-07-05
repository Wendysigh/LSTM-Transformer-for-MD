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
import shutil
from utils import split_input_target,data_as_input,mask_data,build_model

parser = argparse.ArgumentParser(description='LSTM Task')
parser.add_argument('--task', type=str, default='RMSD',choices=['RMSD','phi','psi'],help='dataset to be used')

parser.add_argument('--recrossing_step', type=int, default=1,help='recrossing step for choosing different datasets')

parser.add_argument('--batch_size', type=int, default=64)

parser.add_argument('--seq_length', type=int, default=20)

parser.add_argument('--interval', type=int, default=1, help='saving interval')

parser.add_argument('--learning_rate', type=float, default=0.001)

parser.add_argument('--stateful', type=bool, default=True)

parser.add_argument('--shuffle', type=str, default='shuffle')

parser.add_argument('--gpu_id', type=str, default='1')

args = parser.parse_args()
#args, unknown = parser.parse_known_args()


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

task=args.task
step=args.recrossing_step
interval=args.interval
seq_length=args.seq_length
state=args.stateful
shuffle=args.shuffle
BATCH_SIZE = args.batch_size
lr=args.learning_rate


def generate_text(pmodel, start_string,num_generate):
    """
    genrating next token using training data as start token
    """
    global idx2char
    input_eval = tf.expand_dims(start_string, 0)

    text_generated = np.empty(1)
    temperature = 1

    pmodel.reset_states()   
    for i in range(num_generate):
        start = time.time()
        predictions = pmodel(input_eval)
        
        predictions = tf.squeeze(predictions, 0) 
        # using a multinomial distribution to predict the word returned by the model
        predictions = predictions / temperature  
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy().astype(int)
        # We pass the predicted word as the next input to the model, along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated = np.vstack((text_generated, idx2char[predicted_id.tolist()]))
        
        
    return text_generated


from lossT import sparse_categorical_crossentropy

def loss(labels, logits):
    return sparse_categorical_crossentropy(labels, logits, from_logits=True)

# make dirs to save file
checkpoint_dir = 'LSTM/ckpt/{}/seq{}_lr{}_interval{}/'.format(task,seq_length,lr,interval)
log_dir='LSTM/logs/{}/seq{}_lr{}_interval{}/'.format(task,seq_length,lr,interval)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
save_dir = 'LSTM/result/{}/seq{}_lr{}_interval{}/'.format(task,seq_length,lr,interval)
os.makedirs(save_dir, exist_ok=True)

# load data
if task=='RMSD':
    input_x=np.loadtxt('data/alanine/train',dtype=int)
    valid_x=np.loadtxt('data/alanine/valid',dtype=int)
elif task=='phi':
    input_x,valid_x=np.loadtxt('data/phi-psi/train_phi_0.1ps',dtype=int).reshape(-1),np.loadtxt('data/phi-psi/valid_phi_0.1ps',dtype=int).reshape(-1)
elif tesk=='psi':
     input_x,valid_x=np.loadtxt('data/phi-psi/train_psi_0.1ps',dtype=int).reshape(-1),np.loadtxt('data/phi-psi/valid_psi_0.1ps',dtype=int).reshape(-1)

# subsample x to corresponding interval
input_x=input_x.reshape(interval,-1).T.flatten()
valid_x=valid_x.reshape(interval,-1).T.flatten()       
vocab=np.unique(input_x)
vocab_size=len(vocab)

idx2char={i:u for i, u in enumerate(vocab)}
num_generate=100000
epoch=50


dataset,steps_per_epoch=data_as_input(input_x,BATCH_SIZE,shuffle,seq_length)
vdataset,v_steps_per_epoch=data_as_input(valid_x,BATCH_SIZE,shuffle,seq_length)

embedding_dim = 128 
rnn_units = 1024 

model = build_model(vocab_size,
        embedding_dim,
        rnn_units,
        BATCH_SIZE,
        state)

print(model.summary())

model.compile(
    optimizer = tf.optimizers.Adam(learning_rate=lr),
    loss = loss)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)
# --------training------------
EPOCHS=epoch

history = model.fit(dataset.repeat(EPOCHS), epochs=EPOCHS, steps_per_epoch=steps_per_epoch, validation_data=vdataset.repeat(EPOCHS), validation_steps=v_steps_per_epoch, callbacks=[checkpoint_callback,tensorboard_callback])

# -------generating----------

pmodel = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1,state=state)
pmodel.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
#pmodel.load_weights(checkpoint_dir+'/ckpt_{}'.format(10))
pmodel.build(tf.TensorShape([1, None]))

def single_generation(i):
    seg=int(len(input_x)/100)
    text =input_x[i*seg:(i+1)*seg]
    print('length of seed: {}'.format(len(text[:5000])))
    start0 = time.time()

    prediction=generate_text(pmodel, text[:5000],num_generate)

    print ('Time taken for total {} sec\n'.format(time.time() - start0))

    # Save prediction
    save_path = os.path.join(save_dir, 'prediction_'+str(i))

    np.savetxt(save_path,prediction[1:],fmt='%i')  

    
for i in range(100):
    single_generation(i)
    
# cannot use the parallel session because pmodel cannot be pickled.
#from joblib import Parallel, delayed
#Parallel(n_jobs=-1, verbose=100)(delayed(single_generation)(i) for i in range(100))
  
        



