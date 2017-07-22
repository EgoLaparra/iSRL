#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 12:30:03 2017

@author: egoitz
"""
import sys
import os
import re
import copy as cp
import numpy as np
import configparser

np.random.seed(55555)

config = configparser.ConfigParser()
config.read('iSRL.conf')

gsfile = config['DATA']['bnGS']
closeddir = config['DATA']['close']
mappingdir = config['DATA']['mappings']
embfile = config['DATA']['embeddings']

def load_embeddings(vocab, nb_words=None, emb_dim=100,
                    w2v=embfile):

    randinit = list(np.random.randn(emb_dim))
    emb_matrix = [randinit for i in range(len(vocab))]
    emb_matrix[0] = list(np.zeros(emb_dim))
                  
    fembs = open(w2v, 'r')
    fembs.readline() # get rid off first line
    for line in fembs:
        f = line.rstrip().split(' ')
        w, e = f[0], f[1:]
        try:
            idx = vocab.index(w)
            e = list(map(float,e))
            emb_matrix[idx] = e
        except ValueError:
            pass
    fembs.close()
    
    return emb_matrix

def padd_data(data, max_pcsent, max_window):
    for i in range(0,len(data)):
        (predid,parg,pargidx,pcsent,masksent,window,windowonehot,windowidx) = data[i]
        pcsent.reverse()
        masksent.reverse()
        for j in range(0,max_pcsent-len(pcsent)):
            pcsent.append(0)
            masksent.append(0)
        pcsent.reverse()
        masksent.reverse()
        window.reverse()
        windowonehot.reverse()
        windowidx.reverse()
        for j in range(0,max_window-len(window)):
            window.append(0)
            windowonehot.append(0)
            windowidx.append(0)
        window.reverse()
        windowonehot.reverse()
        windowidx.reverse()
        data[i] = (predid,parg,pargidx,pcsent,masksent,window,windowonehot,windowidx)
    return data

    
def trunc_data(data, max_pcsent, max_window):
    for i in range(0,len(data)):
        (predid,parg,pargidx,pcsent,masksent,window,windowonehot,windowidx) = data[i]
        pcsent.reverse()
        masksent.reverse()
        pcsent = pcsent[:max_window]
        masksent = masksent[:max_window]
        pcsent.reverse()
        masksent.reverse()
        
        window.reverse()
        windowonehot.reverse()
        windowidx.reverse()
        window = window[:max_window]
        windowonehot = windowonehot[:max_window]
        windowidx = windowidx[:max_window]
        window.reverse()
        windowonehot.reverse()
        windowidx.reverse()
        data[i] = (predid,parg,pargidx,pcsent,masksent,window,windowonehot,windowidx)
    return data    
    
def index(sent, vocab, train):
    sentidx = list()
    for word in sent:
        if word not in vocab:
            if train:
                idx = len(vocab)
                vocab.append(word)
            else:
                idx = vocab.index("UNK")
        else:
            idx = vocab.index(word)
        sentidx.append(idx)
    return sentidx, vocab
        

def get_split(gold, conllclosed, mapp, vocab=list(), tg_invent=list(), max_pcsent=0, max_window=0, train=True):
    if train:
        vocab.append("PADD")
        vocab.append("UNK")
    split = list()
    for predid in sorted(gold):
        doc,pps,ppt,pph = predid.split(':')
        for parg in sorted(gold[predid]):
            try:
                pcs,pct = map(int,mapp[doc][":".join([pps,ppt])].split('-'))
                pcsent = cp.deepcopy(conllclosed[doc][pcs])
                masksent = list(np.zeros(len(pcsent)))
                pcword = pcsent[pct-1]
                masksent[pct-1] = 1
                #pcsent[pct-1] = parg # This line substitutes word by predicate#arg
                argids = gold[predid][parg]
                if parg != "_":
                    if parg not in tg_invent:
                        parg_idx = len(tg_invent)
                        tg_invent.append(parg)
                    else:
                        parg_idx = tg_invent.index(parg)
                    pcsent, vocab = index(pcsent, vocab, train)
                    wstart = pcs-2
                    if wstart < 0:
                        wstart = 0
                    window = list()
                    windowidx = list()
                    windowonehot = list()
                    for i in range(wstart,pcs+1):
                        acsent = cp.deepcopy(conllclosed[doc][i])
                        onehotsent = np.zeros(len(acsent))
                        idxsent = list()
                        for j in range(1,len(acsent)+1):
                            ptbid = mapp[doc]["-".join([str(i),str(j)])]
                            fullptbid = ":".join([doc,ptbid,"0"])
                            idxsent.append(fullptbid)
                            if fullptbid in argids:
                                onehotsent[j-1] = 1
                        acsent, vocab = index(acsent, vocab, train)
                        window.extend(acsent)
                        windowidx.extend(idxsent)
                        windowonehot.extend(onehotsent)
                    if len(pcsent) > max_pcsent:
                        max_pcsent = len(pcsent)
                    if len(window) > max_window:
                        max_window = len(window)
                    split.append((predid,parg,parg_idx,pcsent,masksent,window,windowonehot,windowidx))
            except KeyError:
                print (doc,pps,ppt,"not in mappings.")
    
    return split, vocab, tg_invent, max_pcsent, max_window
  
              
def get_dataset():
    conllclosed = dict()
    for closedfile in os.listdir(closeddir):
        doc = re.sub(r'\.closed',r'',closedfile)
        conllclosed[doc] = list()
        closed = open(os.path.join(closeddir, closedfile),'r')
        sent = list()
        for line in closed:
            if line.rstrip() == "":
                conllclosed[doc].append(sent)
                sent = list()
            else:
                fields = line.rstrip().split('\t')
                word = fields[1]
                sent.append(word)
        if len(sent) > 0:
            conllclosed[doc].append(sent)
        closed.close()
    
    mapp = dict()
    for mappingfile in os.listdir(mappingdir):
        doc = re.sub(r'\.mapping',r'',mappingfile)
        mapp[doc] = dict()
        mappings = open(os.path.join(mappingdir, mappingfile),'r')
        for line in mappings:
            ptb, conll = line.rstrip().split(' ')
            mapp[doc][ptb] = conll
            mapp[doc][conll] = ptb
        mappings.close()
        
    gs = open(gsfile,'r')
    gold = dict()
    for line in gs:
        fields = line.rstrip().split(' ')
        predid = fields[0]
        split = ""
        if predid.startswith('wsj_0'):
            split = "train"
        elif predid.startswith('wsj_23'):
            split = "test"
        elif predid.startswith('wsj_24'):
            split = "dev"
        if split not in gold:
            gold[split] = dict()
        if predid not in gold[split]:
            gold[split][predid] = dict()
        pred = fields[1]
        arg = fields[2]
        if pred + "#" + arg not in gold[split][predid]:
            gold[split][predid][pred + "#" + arg] = list()
        argid = fields[3]
        gold[split][predid][pred + "#" + arg].append(argid)
    gs.close()
    
    vocab = list()
    train, vocab, tg_invent, max_pcsent, max_window = get_split(gold["train"], conllclosed, mapp)
    test, _, _, max_pcsent, max_window = get_split(gold["test"], conllclosed, mapp, vocab=vocab, tg_invent=tg_invent,
                                                max_pcsent=max_pcsent, max_window=max_window, train=False)
    dev, _, _, max_pcsent, max_window = get_split(gold["dev"], conllclosed, mapp, vocab=vocab, tg_invent=tg_invent, 
                                               max_pcsent=max_pcsent, max_window=max_window, train=False)

    train = padd_data(train, max_pcsent, max_window)
    test = padd_data(test, max_pcsent, max_window)        
    dev = padd_data(dev, max_pcsent, max_window)
    train = trunc_data(train, max_pcsent, 100)
    test = trunc_data(test, max_pcsent, 100)        
    dev = trunc_data(dev, max_pcsent, 100)

        
    return vocab, tg_invent, train, test, dev
        