#!/usr/bin/env python
import os, sys
from ROOT import *
from array import array
gROOT.SetBatch(True)
import numpy as np

if os.path.isfile('TruePVWeight.txt'):
  os.remove('TruePVWeight.txt')

basedir = '/data/users/minerva1993/ntuple_Run2017/V9_2/180703/'

path = os.listdir(basedir)
filelist = []
ext_dataset = []
for item in path:
  if item.endswith(".root"): filelist.append(item)
  if "v2" in item: ext_dataset.append(item)

filelist.sort()

for item in filelist:
  if item.startswith("SingleElectron") or item.startswith("SingleMuon") or item.startswith("TTHad") or item.startswith("TTLL") : continue
    #text_file.write(item + " : " + str(1.0) + "\n")

  else:
    filepath = os.path.join(basedir, item)
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

    if item == filelist[0]: 
      text_file.write('    if      ( option.Contains("' + item.replace("_","")[:-5] + '") ) wrongPVrate = ' + str(ratio) + ";\n")
      if item in ext_dataset:
        text_file.write('    else if ( option.Contains("' + item.replace("_","")[:-7] + '_") or option.Contains("' + item.replace("_","")[:-7] + 'part2") ) wrongPVrate = ' + str(ratio) + ";\n")
    else:
      text_file.write('    else if ( option.Contains("' + item.replace("_","")[:-5] + '") ) wrongPVrate = ' + str(ratio) + ";\n")
      if item in ext_dataset:
        text_file.write('    else if ( option.Contains("' + item.replace("_","")[:-7] + '_") or option.Contains("' + item.replace("_","")[:-7] + 'part2") ) wrongPVrate = ' + str(ratio) + ";\n")

text_file.write('    else    wrongPVrate = 1.0' + ";\n")    
