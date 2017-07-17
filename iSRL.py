#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 11:52:03 2017

@author: egoitz
"""
import sys
import numpy as np
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

torch.manual_seed(55555)

import conll

mode = "LSTM"

def new_outs_lengths(input_lenght, kernel_size, padding=0, dilation=1, stride=1):
    return np.floor((input_lenght + 2*padding - dilation*(kernel_size-1) -1) / stride + 1)

class OnlyEmbs(nn.Module):
    def __init__(self, max_features, embedding_dim, hidden_size):
        super(OnlyEmbs, self).__init__()
        self.embs = nn.Embedding(max_features, embedding_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.linear1 = nn.Linear(embedding_dim * 2, 200)
        self.sigmoid1 = nn.Sigmoid()
        self.linear2 = nn.Linear(200, 200)
        self.sigmoid2 = nn.Sigmoid()
        self.linear3 = nn.Linear(200, 1)
        self.sigmoid3 = nn.Sigmoid()
        self.hidden = self.init_hidden(hidden_size)

    def init_hidden(self, hidden_size):
        return (Variable(torch.zeros(1, 2, hidden_size // 2)),
                Variable(torch.zeros(1, 2, hidden_size // 2)))
        
    def forward(self, target_seq, mask, window):
        t_embeds = self.embs(target_seq)
        w_embeds = self.embs(window)
        t_vect = t_embeds.mul(mask).sum(0)
        wt = Variable(torch.FloatTensor())
        for s in torch.split(w_embeds,1):
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

        
class Recurrent(nn.Module):
    def __init__(self, max_features, embedding_dim, hidden_size):
        super(Recurrent, self).__init__()
        self.embs = nn.Embedding(max_features, embedding_dim)
        self.lstm_t = nn.GRU(embedding_dim, hidden_size // 2, bidirectional=True, num_layers=1)
        self.lstm_w = nn.GRU(embedding_dim, hidden_size // 2, bidirectional=True, num_layers=1)
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
        
    def forward(self, target_seq, mask, window):
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

        
def extract_data(batch):
    tseq = list()
    tmask = list()
    window = list()
    wonehot = list()
    widx = list()
    for b in batch:
        seq = b[2]
        mask = b[3]
        if mode == "LSTM":
            mask = np.reshape(np.repeat(mask,50),(len(seq),50))
        else:
            mask = np.reshape(np.repeat(mask,100),(len(seq),100))
        tseq.append(seq)
        tmask.append(mask)
        window.append(b[4])
        wonehot.append(b[5])
        widx.append(b[6])
    return tseq, tmask, window, wonehot, widx
        

def evaluate(data, prediction):
    match = 0.
    nprec = 0
    nrec = 0
    for d,p in zip(data, prediction):
        true = d[5]
        out = np.zeros(len(true))
        answ = np.argmax(p)     
        out[answ] = 1
        if np.sum(out) > 0:
            nprec += 1
        if np.sum(true) > 0:
            nrec += 1
        match = match + np.sum(true * out)
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
    batch_size = 1
    batches = math.ceil(len(data) / batch_size)
    for e in range(nepochs):
        running_loss = 0.0
        for b in range(batches):
            # Get minibatch samples
            batch = data[b*batch_size:b*batch_size+batch_size]
            tseq, tmask, window, wonehot, widx = extract_data(batch)

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
            y_pred = net(tseq,tmask,window)
               
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
            prec, rec, f1 = evaluate(data_dev, prediction.data.numpy())
            sys.stdout.write(' - P: %.3f - R: %.3f - F1: %.3f' % (prec, rec, f1))
        sys.stdout.write('\n')
        sys.stdout.flush()


def predict(net, data):
    tseq, tmask, window, _, _ = extract_data(data)
    
    tseq = Variable(torch.LongTensor(tseq))
    tmask = Variable(torch.FloatTensor(tmask))
    window = Variable(torch.LongTensor(window))
    tseq = torch.transpose(tseq,0,1)
    tmask = torch.transpose(tmask,0,1)
    window = torch.transpose(window,0,1)

    pred = net(tseq,tmask,window)
    pred = pred.view(window.size())
    pred = torch.transpose(pred,0,1)
    return pred

        
vocab, data_train, data_test, data_dev = conll.get_dataset()
emb_matrix= conll.load_embeddings(vocab)
if mode == "LSTM":
    net = Recurrent(len(vocab),100,50)
else:
    net = OnlyEmbs(len(vocab),100,50)
net.embs.weight.data.copy_(torch.FloatTensor(emb_matrix))
net.embs.weight.requires_grad = False
net.parameters
emb_matrix
train(net, data_train, 5, 1, [0,1],vocab, val=data_dev)
#prediction = predict(net, data_dev)
#for d,p in zip(data_dev, prediction.data.numpy()):
#    print(p)
#    answ = np.argmax(p)
#    true = np.argmax(d[5])
#    word = vocab[d[4][answ]]
#    idx = d[6][answ]
#    print (d[0],d[1],answ,true,word,idx)