# coding: utf-8
import argparse
import time
import math
import os
import pickle

import torch
import torchtext

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--data', type=str, default='wikitext2',
                    help='data corpus')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')

args = parser.parse_args()
unfinished_set = None

def get_sentences(id_tensor, min_length=2, vocab=None):
    global unfinished_set

    """Converts a sequence of word ids to a sentence"""
    if isinstance(id_tensor, torch.LongTensor):
        ids = id_tensor.transpose(0, 1).contiguous().view(-1)
    elif isinstance(id_tensor, np.ndarray):
        ids = id_tensor.transpose().reshape(-1)

    # Continue from the last partial sentence.
    set_ = unfinished_set or []
    sets = []
    for i in ids:
        set_.append(int(i))
        if int(i) == 9:
            if len(set_) >= min_length:
                sets.append(set_)
            set_ = []

    # Handling a partial sentence on the last of a batch.
    if len(set_) != 0:
        unfinished_set = set_
    return sets

def word_ids_to_sentence(id_tensor, vocab, join=None):
    """Converts a sequence of word ids to a sentence"""
    if isinstance(id_tensor, torch.LongTensor):
        ids = id_tensor.transpose(0, 1).contiguous().view(-1)
    elif isinstance(id_tensor, np.ndarray):
        ids = id_tensor.transpose().reshape(-1)
    batch = [vocab.itos[ind] for ind in ids] # denumericalize
    if join is None:
        return batch
    else:
        return join.join(batch)

###############################################################################
# Load data
###############################################################################

if args.data == "wikitext2":
    train_iters, val_iters, test_iters = torchtext.datasets.WikiText2.iters(batch_size=args.batch_size, bptt_len=args.bptt, device="cpu")
elif args.data == "wikitext103":
    train_iters, val_iters, test_iters = torchtext.datasets.WikiText103.iters(batch_size=args.batch_size, bptt_len=args.bptt, device="cpu")
elif args.data == "ptb":
    train_iters, val_iters, test_iters = torchtext.datasets.PennTreebank.iters(batch_size=args.batch_size, bptt_len=args.bptt, device="cpu")
else:
    raise NotImplementedError()

def compute():
    # Turn on training mode which enables dropout.
    ntokens = len(train_iters.dataset.fields["text"].vocab)

    tf = { t:{} for t in range(ntokens) }
    idf = { t:0 for t in range(ntokens) }
    max_tf = {}
    
    temp = []
    cnt = 0
    for batch, item in enumerate(train_iters):
        data = item.text
        for sentence in get_sentences(data, vocab=train_iters.dataset.fields["text"].vocab):
            for s in sentence:
                if cnt not in tf[s]:
                    tf[s][cnt] = 0
                    idf[s] += 1
                tf[s][cnt] += 1
            max_tf[cnt] = max([tf[s][cnt] for s in sentence])
            cnt += 1

    tf_ = {}
    for t in range(ntokens):
        tf_sum_ = 0
        for d in range(cnt):
            tf__ = tf[t][d] if d in tf[t] else 0
            tf_sum_ += 0.5 + 0.5 * tf__ / max_tf[d]
        tf_[t] = tf_sum_ / cnt

    score = { t:tf_[t] * idf[t] for t in range(ntokens) }

    with open("tf_score_%s.pkl" % args.data, "wb") as f:
        pickle.dump(score, f)
    print(score[0])

compute()
