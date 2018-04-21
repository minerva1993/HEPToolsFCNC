#!/usr/bin/env python
import os, sys
from ROOT import *
from array import array
gROOT.SetBatch(True)
import numpy as np

path = '/data/users/minerva1993/ntuple_Run2017/V9_2/'
tuples = ['DYJets_10to50.root', 'DYJets.root', 
          'SingleTbar_t.root', 'SingleTbar_tW.root', 'SingleTop_t.root', 'SingleTop_tW.root',
          'ST_TH_1L3B_Hct.root', 'ST_TH_1L3B_Hut.root',
          'TT_powheg_ttbb.root', 'TT_powheg_ttbj.root', 'TT_powheg_ttcc.root',
          'TT_powheg_ttlf.root', 'TT_powheg_ttother.root',
          'W1JetsToLNu_150-250.root', 'W1JetsToLNu_250-400.root', 'W1JetsToLNu_400-inf.root',
          'W1JetsToLNu_50-150.root', 'W2JetsToLNu_250-400.root', 'W2JetsToLNu_400-inf.root',
          'W3JetsToLNu.root', 'W4JetsToLNu.root', 'WW.root', 'WZ.root', 'ZZ.root',]

for filename in tuples:
  data = TFile.Open(path+filename)
  tree = data.Get('/fcncLepJets/tree')

  h1 = TH1F("h1","h1", 150, 0, 150)
  h2 = TH1F("h2","h2", 150, 0, 150)
  hfull = tree.Draw("TruePV>>h1","","")
  hgood = tree.Draw("TruePV>>h2","TruePV != 0","")

  print filename + " : " + str(h1.Integral()/h2.Integral())
  

