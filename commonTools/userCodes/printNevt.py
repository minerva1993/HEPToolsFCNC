#!/usr/bin/env python
import os, sys
from ROOT import *
from array import array
gROOT.SetBatch(True)
import numpy as np

#basedir = '/data/users/minerva1993/ntuple/V9_6/'
basedir = '/data/users/minerva1993/ntuple/V10_3/'
produce = ['200101', '201215']
dataset = ['TT_powheg_ttbb', 'TT_powheg_ttcc', 'TT_powheg_ttlf',
           'TTLL_powheg_ttbb', 'TTLL_powheg_ttcc', 'TTLL_powheg_ttlf',
           'ST_TH_1L3B_Hct', 'ST_TH_1L3B_Hut',
           'TT_TH_1L3B_aTLep_Hct', 'TT_TH_1L3B_aTLep_Hut', 'TT_TH_1L3B_TLep_Hct', 'TT_TH_1L3B_TLep_Hut']

base = os.path.join(basedir, 'TAG')
base2 = os.path.join(base, 'production')

filelist = []

for data in dataset:
  path_ = os.path.join(base2, data)

  for tag in produce:
    path = path_.replace('TAG', tag)

    for item in os.listdir(path):
      if item.endswith(".root"):
        i = os.path.join(path, item)
        tmp = i.replace(tag, 'TAG')
        filelist.append(tmp)

filelist = list(dict.fromkeys(filelist))
filelist.sort()

for item in filelist:
  for tag in produce:
    filepath = item.replace('TAG', tag)
    data = TFile.Open(item.replace('TAG', tag))
    tree = data.Get('fcncLepJets/tree')
    print filepath, ' ', str(tree.GetEntries())
    filepath = ''
    item.replace(tag, 'TAG')
