#!/usr/bin/env python
import os, sys
from ROOT import *
from array import array
gROOT.SetBatch(True)
import numpy as np

if os.path.isfile('TruePVWeight.txt'):
  os.remove('TruePVWeight.txt')

###########Change two directories!

path = os.listdir('/data/users/minerva1993/ntuple_Run2017/V9_2/180614/')
filelist = []
for item in path:
  if item.endswith(".root"): filelist.append(item)

filelist.sort()

for item in filelist:
  if item.startswith("SingleElectron") or item.startswith("SingleMuon"):
    text_file.write(item + " : " + str(1.0) + "\n")

  else: 
    filepath = os.path.join('/data/users/minerva1993/ntuple_Run2017/V9_2/180614', item)
    print filepath
    data = TFile.Open(filepath)
    tree = data.Get('fcncLepJets/tree')

    h1 = TH1F("h1","h1", 150, 0, 150)
    h2 = TH1F("h2","h2", 150, 0, 150)
    hfull = tree.Draw("TruePV>>h1","","")
    hgood = tree.Draw("TruePV>>h2","TruePV >= 10 && TruePV <= 75","")
    #hgood = tree.Draw("TruePV>>h2","TruePV !=0","")

    text_file = open("TruePVWeight.txt", "a")
    if h2.Integral() == 0 : ratio = 1.0
    else: ratio = float(h1.Integral())/h2.Integral()
    text_file.write(item + " : " + str(ratio) + "\n")
