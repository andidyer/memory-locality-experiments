#!/usr/bin/env python3
import math
import subprocess
#assert False
__file__ = __file__.split("/")[-1]
import random
myID = random.randint(1000,1000000)
import transformers
import glob
import sys

device = 'cuda'

rollout = sys.argv[1]
millionTokensToUseAsTrainingData = int(sys.argv[2])
assert millionTokensToUseAsTrainingData >= 500
assert "SepRel" in rollout or "baseline" in rollout


if len(glob.glob(f"scratch_folder/lm/output/checkpoints/{__file__}_{millionTokensToUseAsTrainingData}_{rollout}_*.txt")):
   assert False, glob.glob(f"scratch_folder/lm/output/checkpoints/{__file__}_{millionTokensToUseAsTrainingData}_{rollout}_*.txt")


print(rollout)
if "Bugfix" not in rollout and "baseline" not in rollout:
  print("SKIPPING", rollout)
  quit()

state = set()
data = []


itos = []
stoi = {}

def registerToken(x):
    if x not in stoi:
        stoi[x] = len(itos)
        itos.append(x)

import os
lines = int(subprocess.check_output(['wc','-l',  f"scratch_folder/worlds/stochastic/{rollout}"]).decode("utf-8").split(" ")[0])
if lines < 1e6 * millionTokensToUseAsTrainingData:
   print("Lines in file", lines / 1e6, "millions")
   quit()
#file_size = os.path.getsize(f"/proj/mhahn.shadow/generalization/worlds/stochastic/{rollout}")
#if file_size < 50000000:
#   print("Small file", file_size)
#   quit()

#file_size_in_mb = file_size / 1000000
#assert file_size_in_mb < 1000
#assert file_size_in_mb > 10
#print("FILE SIZE", file_size, "Bytes", file_size_in_mb, "MB")

tokensRead = 0
with open(f"scratch_folder/worlds/stochastic/{rollout}", "r") as inFile:
  for line in inFile:
     if line.startswith("STAT"):
        state.add(tuple(line.strip().split(" ")))
        continue
     if line.startswith("PROB"):
       continue
     if line.startswith("BASED"):
       continue
     if line.startswith("@"):
        continue
     if line.startswith("%") or line.startswith("#"):
        if len(data) % 10000 == 0:
           if len(data) == 50000 and False:
               break
           print(len(data), tokensRead, len(data)/1e6, tokensRead/1e6, "Million tokens")
        data.append([])
   #  else:
#     print([line])
     line = line.strip()
     data[-1].append(line)
     registerToken(line)
     tokensRead += 1
     if tokensRead/1e6 > millionTokensToUseAsTrainingData:
       break

print("Number of tokens:", tokensRead/1e6, "millions")
#if tokensRead / 1e6 < 100:
#   quit()
print(len(itos))
print(itos[-10:])
#quit()

import random
random.Random(myID).shuffle(data)
data_train = data[:int(0.99*len(data))]
data_dev = data[int(0.99*len(data)):]
data = None # can be garbage-collected

assert millionTokensToUseAsTrainingData * 1e6 <= tokensRead
print("Fraction", millionTokensToUseAsTrainingData * 1e6 / tokensRead)
print(len(data_train))
data_train = data_train[:int(millionTokensToUseAsTrainingData * 1e6 / tokensRead * len(data_train))]
print("After", len(data_train))

#print("measured file size in MB", file_size_in_mb)
def flatten(x):
   r = []
   for y in x:
     for q in y:
       r.append(q)
   return r




import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
#
#tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#print(tokenizer.decode(tokenizer.encode("Hello")))
#quit()

import torch
from torch import nn
from torch import optim

import numpy as np

from transformers import GPT2Config
from transformers import GPT2TokenizerFast
from transformers import GPT2LMHeadModel
from transformers import AdamW

import argparse
import os
from tqdm import tqdm
import copy
import json
import logging
import random
import glob


#from probe_models import (
#    ProbeLinearModel, ProbeConditionalGenerationModel, ProbeLanguageEncoder, encode_target_states,
#    get_probe_model, get_state_encoder, get_lang_model,
#)
from itertools import chain, combinations




#parser = argparse.ArgumentParser()
arch = "t5"
#parser.add_argument('--arch', type=str, default='bart', choices=['bart', 't5'])
#parser.add_argument('--batchsize', type=int, default=4)
#parser.add_argument('--eval_batchsize', type=int, default=32)
#parser.add_argument('--data', type=str, required=True)
#parser.add_argument('--device', type=str, default='cpu')
#parser.add_argument('--epochs', type=int, default=20)
#parser.add_argument('--eval_only', action='store_true', default=False)
#parser.add_argument('--gamefile', type=str, required=False)
#parser.add_argument('--lr', type=float, default=1e-5)
#parser.add_argument('--lm_save_path', type=str, default="SAVE/")
#parser.add_argument('--max_seq_len', type=int, default=512)
#parser.add_argument('--metric', type=str, choices=['em', 'loss'], help='which metric to use on dev set', default='em')
#parser.add_argument('--probe_save_path', type=str, default=None)
#parser.add_argument('--probe_layer', type=int, default=-1, help="which layer of the model to probe")
#parser.add_argument('--num_samples', type=int, default=1)
#parser.add_argument('--no_pretrain', action='store_true', default=False)
#parser.add_argument('--save_path', type=str, default=None)
#parser.add_argument('--probe_type', type=str, choices=['3linear_classify', 'linear_classify', 'linear_retrieve', 'decoder'], default='decoder')
#parser.add_argument('--encode_tgt_state', type=str, default=False, choices=[False, 'NL.bart', 'NL.t5'], help="how to encode the state before probing")
#parser.add_argument('--train_data_size', type=int, default=4000)
#parser.add_argument('--tgt_agg_method', type=str, choices=['sum', 'avg', 'first', 'last', 'lin_attn', 'ffn_attn', 'self_attn'], default='avg', help="how to aggregate across tokens of target, if `encode_tgt_state` is set True")
#parser.add_argument('--probe_agg_method', type=str, choices=[None, 'sum', 'avg', 'first', 'last', 'lin_attn', 'ffn_attn', 'self_attn'], default=None, help="how to aggregate across tokens")
#parser.add_argument('--probe_attn_dim', type=int, default=None, help="what dimensions to compress sequence tokens to")
#parser.add_argument('--patience', type=int, default=10)
#parser.add_argument('--control_input', default=False, action='store_true', help='control inputs to tokenized entity pair')
#parser.add_argument('--local_files_only', action='store_true', default=False, help="use pretrained checkpoints saved in local directories")
#parser.add_argument('--probe_target', type=str, default='final.belief_facts', choices=list(chain(*[[
#    f'{init_final}.full_facts', f'{init_final}.full_belief_facts', f'{init_final}.belief_facts',
#    f'{init_final}.belief_facts_single', f'{init_final}.belief_facts_pair',
#    f'{init_final}.full_belief_facts_single', f'{init_final}.full_belief_facts_pair',
#    f'{init_final}.belief_facts_single.control', f'{init_final}.full_belief_facts_single.control', f'{init_final}.belief_facts_single.control_with_rooms', f'{init_final}.full_belief_facts_single.control_with_rooms',
#    f'{init_final}.belief_facts_pair.control', f'{init_final}.full_belief_facts_pair.control', f'{init_final}.belief_facts_pair.control_with_rooms', f'{init_final}.full_belief_facts_pair.control_with_rooms',
#] for init_final in ['init', 'final']])))
#parser.add_argument('--localizer_type', type=str, default='all',
#    choices=['all'] + [f'belief_facts_{sides}_{agg}' for sides in ['single', 'pair'] for agg in ['all', 'first', 'last']],
#    help="which encoded tokens of the input to probe."
#    "Set to `all`, `belief_facts_{single|pair}_{all|first|last}`")
#parser.add_argument('--seed', type=int, default=42)
#parser.add_argument('--ents_to_states_file', type=str, default=None, help='Filepath to precomputed state vectors')
#args = parser.parse_args()

#arch = args.arch
#pretrain = not args.no_pretrain
#batchsize = 128
#control_input = args.control_input
#eval_batchsize = args.eval_batchsize
#max_seq_len = args.max_seq_len
#lm_save_path = args.lm_save_path
#localizer_type = args.localizer_type
#probe_target = args.probe_target.split('.')
#probe_type = args.probe_type
#retrieve = probe_type.endswith('retrieve')
#classify = probe_type.endswith('classify')
#assert not (retrieve and classify)
#encode_tgt_state = args.encode_tgt_state
#tgt_agg_method = args.tgt_agg_method
#probe_agg_method = args.probe_agg_method
#probe_attn_dim = args.probe_attn_dim
#probe_save_path = args.probe_save_path
#train_data_size = args.train_data_size
#game_kb = None

from transformers import GPT2Config
print("Initializing GPT2 Model")
#model = GPT2LMHeadModel.from_pretrained("gpt2") #config = GPT2Config(vocab_size=len(itos), n_positions=512, n_ctx=512, n_embed=256, n_layer=4, n_head=4))
model = GPT2LMHeadModel(config = GPT2Config(vocab_size=len(itos), n_positions=64, n_ctx=64, n_embed=256, n_layer=6, n_head=8))
print("Moving GPT2 Model to GPU")
model = model.cuda() #.from_pretrained("t5-small").cuda()
print("Initialized GPT2 Model")

# seed everything
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

from transformers import PreTrainedTokenizerBase, BatchEncoding
import torch


def to_device(x):
 if device == "cuda":
  return x.cuda()
 return x

#transformer = to_device(torch.nn.TransformerEncoder(encoder_layer=torch.nn.TransformerEncoderLayer(d_model=64, nhead=8), num_layers=4))
#embedding = to_device(torch.nn.Embedding(num_embeddings=10000, embedding_dim=64))
output = to_device(torch.nn.Linear(64, 1, bias=False))

def parameters():
    for x in [transformer, embedding, output]:
        for y in x.parameters():
            yield y


#model.to(device)

# load optimizer
#all_parameters = [p for p in parameters() if p.requires_grad]
#print(all_parameters)
#quit()
lr = 1e-4 #1e-5

optimizer = AdamW(model.parameters(), lr=lr) # or AdamW with get_linear_schedule_with_warmup
import transformers
lr_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=1000, num_training_steps=40000*50)

print("MODEL TRAINING STATE", model.training)
print("Loaded model")

# Initial eval
print("Initial eval")
avg_val_loss = 0
best_loss_epoch = -1
best_val_loss = avg_val_loss

epochs = 10
loss_running_average = 1

per_epoch = []

num_updates = 0
with open(f"output/{__file__}_{myID}_{millionTokensToUseAsTrainingData}_{rollout}.txt", "w") as outFile:
 for epoch in range(20 if "TRIM" not in rollout else 10):





  # DEV
  model.train(False)
  print("MODEL TRAINING STATE", model.training)
  per_epoch.append(0)
  toProcess = []
  batch_size = 64
  seq_len = 64
  q = 0
  loss_running_average_dev = 1
  for j in range(len(data_dev)):
   toProcess = toProcess + data_dev[j]
   if len(toProcess) >= batch_size * seq_len:
    q += 1
    batch = toProcess[:batch_size * seq_len]
    toProcess = toProcess[batch_size * seq_len:]
    batch = torch.LongTensor([stoi[x] for x in batch]).view(batch_size, seq_len).cuda()
    #print(batch.size(), batch.min(), batch.max())
    #quit()
    output = model(input_ids=batch, labels=batch, return_dict=True)
 #      if j % 100 == 0:
   #        print(tokenizer.decode(tokenized_output[0]))
    #       print(tokenizer.decode(model.generate(input_ids=tokenized_input[:1].cuda(), max_length=300)[0]))
 
    loss = output["loss"]
   
    loss_running_average_dev = 0.95 * loss_running_average_dev + (1-0.95) * float(loss)
         
    if q%1 == 0:
 #       losses = torch.nn.functional.log_softmax(output["logits"], dim=2)
 #       for w in range(10):
    #         print((output["logits"][w,0,1]  > output["logits"][w,0,0]))
      #       quit()
        print(f"DEV epoch {epoch}, batch {q}, loss: {loss.item()}", loss_running_average_dev, flush=True)
        print(f"DEV epoch {epoch}, batch {q}, loss: {loss.item()}", loss_running_average_dev, file=outFile)
    per_epoch[-1] += (float(loss))




  per_epoch[-1] /= q
  if len(per_epoch) > 1 and per_epoch[-1] < min(per_epoch[:-1]):
     torch.save({"state" : model.state_dict(), "itos" : itos, "per_epoch" : per_epoch}, f=f"scratch_folder/lm/output/checkpoints/{__file__}_{millionTokensToUseAsTrainingData}_{rollout}_{myID}_EPOCH_{epoch}.txt")
  elif len(per_epoch) > 1 and per_epoch[-1] >= min(per_epoch[:-1]):
     print("Loss deteriorating")
     pass

  model.train(True)
  print("MODEL TRAINING STATE", model.training)
  assert model.training
 
  tokensDone = 0
  savingThresholds = [math.pow(2,i) * 1e6 for i in range(30)]
  doneSavingFor = [False for _ in range(30)]


  model.train(True)
  assert model.training

  random.Random(epoch).shuffle(data_train)
  toProcess = []
  batch_size = 64
  seq_len = 64
  q = 0
  for j in range(len(data_train)):
   toProcess = toProcess + data_train[j]
   if len(toProcess) >= batch_size * seq_len:
    q += 1
    batch = toProcess[:batch_size * seq_len]
    toProcess = toProcess[batch_size * seq_len:]
    tokensDone += (batch_size * seq_len)
    batch = torch.LongTensor([stoi[x] for x in batch]).view(batch_size, seq_len).cuda()
    #print(batch.size(), batch.min(), batch.max())
    #quit()
    output = model(input_ids=batch, labels=batch, return_dict=True)
 #      if j % 100 == 0:
   #        print(tokenizer.decode(tokenized_output[0]))
    #       print(tokenizer.decode(model.generate(input_ids=tokenized_input[:1].cuda(), max_length=300)[0]))
 
    loss = output["loss"]
   
    #   prediction = torch.sigmoid(output(transformed.mean(dim=1))).view(-1)
   
     #  loss = -torch.where(labels == 1, prediction.log(), (1-prediction).log()).sum()/10
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    num_updates += 1
    loss_running_average = 0.95 * loss_running_average + (1-0.95) * float(loss)
    if len(per_epoch) == 1:
      for w in range(20):
        if tokensDone > savingThresholds[w] and (not doneSavingFor[w]):
           print("Saving checkpoint...", q)
           epoch_ = round(savingThresholds[w] / (millionTokensToUseAsTrainingData * 1e6),4)
           torch.save({"state" : model.state_dict(), "itos" : itos, "per_epoch" : per_epoch}, f=f"scratch_folder/lm/output/checkpoints/{__file__}_{millionTokensToUseAsTrainingData}_{rollout}_{myID}_EPOCH_{epoch_}.txt")
           doneSavingFor[w] = True
        
    if q%10 == 0:
 #       losses = torch.nn.functional.log_softmax(output["logits"], dim=2)
 #       for w in range(10):
    #         print((output["logits"][w,0,1]  > output["logits"][w,0,0]))
      #       quit()
        print(f"epoch {epoch}, batch {q}, loss: {loss.item()}", [round(x,3) for x in per_epoch[-20:]], loss_running_average, __file__, rollout, myID, "Progress", j/len(data_train), tokensDone/1e6, doneSavingFor, flush=True)
        print(f"epoch {epoch}, batch {q}, loss: {loss.item()}", [round(x,3) for x in per_epoch[-20:]], loss_running_average, file=outFile)
#    per_epoch[-1] += (float(loss.mean())) #/len(examples_train)*32)
   #    print(float(loss))
 #      print(per_epoch[-1])
   
