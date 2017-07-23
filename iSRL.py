#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 11:52:03 2017

@author: egoitz
"""
import sys
import numpy as np
import math
import configparser

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

torch.manual_seed(55555)
np.random.seed(55555)

import conll

config = configparser.ConfigParser()
config.read('iSRL.conf')

batch_size = int(config['MODEL']['batch_size'])
num_epochs = int(config['MODEL']['num_epochs'])
fix_embs = bool(int(config['MODEL']['fix_embs']))
onehot_target = bool(int(config['MODEL']['onehot_target']))
mode = config['MODEL']['mode']


class OnlyEmbs(nn.Module):
    def __init__(self, max_tgt, tgt_dim, max_features, embedding_dim, hidden_size):
        super(OnlyEmbs, self).__init__()
        if onehot_target:
            self.tgt_feats = max_tgt
        else:
            self.tgt_feats = tgt_dim
            self.tgt = nn.Embedding(max_tgt, tgt_dim)
        self.embs = nn.Embedding(max_features, embedding_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.linear1 = nn.Linear(embedding_dim + self.tgt_feats, 200)
        self.sigmoid1 = nn.Sigmoid()
        self.linear2 = nn.Linear(200, 200)
        self.sigmoid2 = nn.Sigmoid()
        self.linear3 = nn.Linear(200, 1)
        self.sigmoid3 = nn.Sigmoid()
        self.hidden = self.init_hidden(hidden_size)

    def init_hidden(self, hidden_size):
        return (Variable(torch.zeros(1, 2, hidden_size // 2)),
                Variable(torch.zeros(1, 2, hidden_size // 2)))
        
    def forward(self, predarg, target_seq, mask, window):
        if onehot_target:
            tgt = predarg
        else:
            tgt = self.tgt(predarg)
        tgt = tgt.view(1, tgt.size()[0],-1)
        w_embeds = self.embs(window)
        wt = Variable(torch.FloatTensor())
        for s in torch.split(w_embeds,1):
            s = torch.cat((tgt,s),2)
            if wt.dim() > 0:
                wt = torch.cat((wt,s))
            else:
                wt = s
        wt = wt.view(wt.size()[0] * wt.size()[1], -1)
        out = self.dropout(wt)
        out = self.sigmoid1(self.linear1(out))
        out = self.dropout(out)
        out = self.sigmoid2(self.linear2(out))
        out = self.dropout(out)
        out = self.sigmoid3(self.linear3(out))
        return out

        
class Recurrent(nn.Module):
    def __init__(self, max_features, embedding_dim, hidden_size):
        super(Recurrent, self).__init__()
        self.embs = nn.Embedding(max_features, embedding_dim)
        self.lstm_t = nn.GRU(embedding_dim, hidden_size // 2, bidirectional=True, num_layers=2)
        self.lstm_w = nn.GRU(embedding_dim, hidden_size // 2, bidirectional=True, num_layers=2)
        self.linear1 = nn.Linear(hidden_size * 2, 200)
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid1 = nn.Sigmoid()
        self.linear2 = nn.Linear(200, 200)
        self.sigmoid2 = nn.Sigmoid()
        self.linear3 = nn.Linear(200, 1)
        self.sigmoid3 = nn.Sigmoid()
        self.hidden = self.init_hidden(hidden_size)

    def init_hidden(self, hidden_size):
        return (Variable(torch.zeros(1, 2, hidden_size // 2)),
                Variable(torch.zeros(1, 2, hidden_size // 2)))
        
    def forward(self, predarg, target_seq, mask, window):
        t_embeds = self.embs(target_seq)
        t_seq, _ = self.lstm_t(t_embeds)
        w_embeds = self.embs(window)
        w_seq, _ = self.lstm_w(w_embeds)
        t_vect = t_seq.mul(mask).sum(0)
        wt = Variable(torch.FloatTensor())
        for s in torch.split(w_seq,1): 
            s = torch.cat((t_vect,s),2)
            if wt.dim() > 0:
                wt = torch.cat((wt,s))
            else:
                wt = s
        wt = wt.view(wt.size()[0] * wt.size()[1], -1)
        out = self.dropout(wt)
        out = self.sigmoid1(self.linear1(out))
        out = self.dropout(out)
        out = self.sigmoid2(self.linear2(out))
        out = self.dropout(out)
        out = self.sigmoid3(self.linear3(out))
        return out

class RecurrentTgt(nn.Module):
    def __init__(self, max_tgt, tgt_dim, max_features, embedding_dim, hidden_size):
        super(RecurrentTgt, self).__init__()
        if onehot_target:
            self.tgt_feats = max_tgt
        else:
            self.tgt_feats = tgt_dim
            self.tgt = nn.Embedding(max_tgt, tgt_dim)
        self.embs = nn.Embedding(max_features, embedding_dim)
        self.lstm_t = nn.GRU(embedding_dim, hidden_size // 2, bidirectional=True, num_layers=1)
        self.lstm_w = nn.GRU(embedding_dim, hidden_size // 2, bidirectional=True, num_layers=1)
        self.linear1 = nn.Linear(hidden_size * 2 + self.tgt_feats, 200)
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid1 = nn.Sigmoid()
        self.linear2 = nn.Linear(200, 200)
        self.sigmoid2 = nn.Sigmoid()
        self.linear3 = nn.Linear(200, 1)
        self.sigmoid3 = nn.Sigmoid()
        self.hidden = self.init_hidden(hidden_size)

    def init_hidden(self, hidden_size):
        return (Variable(torch.zeros(1, 2, hidden_size // 2)),
                Variable(torch.zeros(1, 2, hidden_size // 2)))
        
    def forward(self, predarg, target_seq, mask, window):
        if onehot_target:
            tgt = predarg
        else:
            tgt = self.tgt(predarg)
        tgt = tgt.view(1, tgt.size()[0],-1)
        t_embeds = self.embs(target_seq)
        t_seq, _ = self.lstm_t(t_embeds)
        w_embeds = self.embs(window)
        w_seq, _ = self.lstm_w(w_embeds)
        t_vect = t_seq.mul(mask).sum(0)
        wt = Variable(torch.FloatTensor())
        for s in torch.split(w_seq,1):
            s = torch.cat([tgt,t_vect,s],2)
            if wt.dim() > 0:
                wt = torch.cat((wt,s))
            else:
                wt = s
        wt = wt.view(wt.size()[0] * wt.size()[1], -1)
        out = self.dropout(wt)
        out = self.sigmoid1(self.linear1(out))
        out = self.dropout(out)
        out = self.sigmoid2(self.linear2(out))
        out = self.dropout(out)
        out = self.sigmoid3(self.linear3(out))
        return out
        
def extract_data(batch):
    predargs = list()
    tseq = list()
    tmask = list()
    window = list()
    wonehot = list()
    widx = list()
    for b in batch:
        predarg = b[2]
        seq = b[3]
        mask = b[4]
        if mode == "LSTM":
            mask = np.reshape(np.repeat(mask,50),(len(seq),50))
        if mode == "TGT":
            mask = np.reshape(np.repeat(mask,50),(len(seq),50))
        else:
            mask = np.reshape(np.repeat(mask,100),(len(seq),100))
        predargs.append(predarg)
        tseq.append(seq)
        tmask.append(mask)
        window.append(b[5])
        wonehot.append(b[6])
        widx.append(b[7])
    return predargs, tseq, tmask, window, wonehot, widx
        
def get_answer(prediction):
    answer = list()
    for p in prediction:
        out = np.zeros(len(p))
        answ = np.argmax(p)
        out[answ] = 1
        answer.append(out)
    return answer
    
def evaluate(data, answer):
    match = 0.
    nprec = 0
    nrec = 0
    for d,a in zip(data, answer):
        true = d[6]
        if np.sum(a) > 0:
            nprec += 1
        if np.sum(true) > 0:
            nrec += 1
        match = match + np.sum(true * a)
    prec = match / nprec
    rec = match / nrec
    f1 = 2 * prec * rec / (prec + rec)
    return (prec, rec, f1)
        
def BCELossWithWeights (o, t, w):
    return - (w * ((t * o.log()) + (1 - t) * (1 - o).log())).mean()

def BCELossNegSamp (o, t):
    p = t.clone()
    neg_samp = torch.LongTensor(torch.from_numpy(np.random.choice(range(t.size()[0]),size=20,replace=False)))
    p[neg_samp] = 1
    poss = p.sum() + 1.e-17
    return - (((t * o.log()) + (1 - t) * (1 - o).log()) * p).sum() / poss

    
def train(net, data, nepochs, batch_size, class_weights, vocab, val=None):
    criterion = nn.BCELoss()
    #criterion = BCELossWithWeights
    #criterion = BCELossNegSamp
    parameters = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = optim.Adam(parameters)
    batches = math.ceil(len(data) / batch_size)
    for e in range(nepochs):
        running_loss = 0.0
        for b in range(batches):
            # Get minibatch samples
            batch = data[b*batch_size:b*batch_size+batch_size]
            predarg, tseq, tmask, window, wonehot, widx = extract_data(batch)
            
            if onehot_target:
                tgt_1hot = np.zeros((len(predarg), net.tgt_feats))
                tgt_1hot[np.arange(len(predarg)), predarg] = 1
                predarg = Variable(torch.from_numpy(tgt_1hot).float())
            else:
                predarg = Variable(torch.LongTensor(predarg))
            tseq = Variable(torch.LongTensor(tseq))
            tmask = Variable(torch.FloatTensor(tmask))
            window = Variable(torch.LongTensor(window))
            wonehot = Variable(torch.FloatTensor(wonehot))
            tseq = torch.transpose(tseq,0,1)
            tmask = torch.transpose(tmask,0,1)
            window = torch.transpose(window,0,1)
            
            # Clear gradients
            net.zero_grad()
                
            # Forward pass
            y_pred = net(predarg,tseq,tmask,window)
               
            wonehot = wonehot.view(-1,1) 
            # Compute loss
            weights = wonehot + 0.1 + wonehot * 0.8
            #loss = criterion(y_pred, wonehot, weights)
            loss = criterion(y_pred, wonehot)
            
            # Print loss
            running_loss += loss.data[0]
            sys.stdout.write('\r[epoch: %3d, batch: %3d/%3d] loss: %.3f' % (e + 1, b + 1, batches, running_loss / (b+1)))
            sys.stdout.flush()

            # Backward propagation and update the weights.
            loss.backward()
            optimizer.step()
        if val is not None:
            prediction = predict(net, data_dev)
            answer = get_answer(prediction.data.numpy())
            prec, rec, f1 = evaluate(data_dev, answer)
            sys.stdout.write(' - P: %.3f - R: %.3f - F1: %.3f' % (prec, rec, f1))
        sys.stdout.write('\n')
        sys.stdout.flush()


def predict(net, data):
    predarg, tseq, tmask, window, _, _ = extract_data(data)
    
    if onehot_target:
        tgt_1hot = np.zeros((len(predarg), net.tgt_feats))
        tgt_1hot[np.arange(len(predarg)), predarg] = 1
        predarg = Variable(torch.from_numpy(tgt_1hot).float())
    else:
        predarg = Variable(torch.LongTensor(predarg))
    tseq = Variable(torch.LongTensor(tseq))
    tmask = Variable(torch.FloatTensor(tmask))
    window = Variable(torch.LongTensor(window))
    tseq = torch.transpose(tseq,0,1)
    tmask = torch.transpose(tmask,0,1)
    window = torch.transpose(window,0,1)

    pred = net(predarg,tseq,tmask,window)
    pred = pred.view(window.size())
    pred = torch.transpose(pred,0,1)
    return pred

        
vocab, tg_invent, data_train, data_test, data_dev = conll.get_dataset()
emb_matrix,embs_dim = conll.load_embeddings(vocab)
if mode == "LSTM":
    net = Recurrent(len(vocab),embs_dim,50)
elif mode == "TGT":
    net = RecurrentTgt(len(tg_invent),10,len(vocab),embs_dim,50)
else:
    net = OnlyEmbs(len(tg_invent),50,len(vocab),embs_dim,50)
net.embs.weight.data.copy_(torch.FloatTensor(emb_matrix))
if fix_embs:
    net.embs.weight.requires_grad = False
train(net, data_train, num_epochs, batch_size, [0,1],vocab, val=data_dev)
#prediction = predict(net, data_dev)
#answer = get_answer(prediction.data.numpy())
#prec, rec, f1 = evaluate(data_dev, answer)
#sys.stdout.write(' - P: %.3f - R: %.3f - F1: %.3f' % (prec, rec, f1))
#for d,p in zip(data_dev, prediction.data.numpy()):
#    print(p)
#    answ = np.argmax(p)
#    true = np.argmax(d[5])
#    word = vocab[d[4][answ]]
#    idx = d[6][answ]
#    print (d[0],d[1],answ,true,word,idx)