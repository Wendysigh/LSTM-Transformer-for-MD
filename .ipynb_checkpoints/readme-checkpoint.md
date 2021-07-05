# README #


### What is this repository for? ###

LSTM/Transformer for learning molecular dynamics and generating trajectories on alanine depeptide. 
It is mainly tested on Linux and Python3.6 with GPU while LSTM gets built on tensorflow2.0 and Transformer on Pytorch.


* contact: wzengad@connect.ust.hk

### Basic usage ###

It contains the following files and subfolders:

LSTM/

Transformer/

data/

readme.md

report.pdf


This package consists of 3 major subfolders: LSTM/ for performing molecular dynamics with LSTM model, Transformer/ for Transformer model, data/ for the two dataset used in the report.

After cd $HOME/MD_deep_learning/, you could test LSTM with different sequence length or saving intervals on alanine_RMSD with :
python LSTM/main.py --task='RMSD' --seq_length=50 --interval=10

Or test unidirectional Transformer with different saving intervals on alanine_RMSD with :
python Transformer/main.py --task='RMSD' --unidirection=True --interval=10

The datasets are only available for vanilla alanine_RMSD and alanine_phi_psi with 0.1ps saving interval. The recrossing removal and lumping dataset are not included. Generated trajectories as output will be automatically saved under LSTM/result/ or Transformer/result/.