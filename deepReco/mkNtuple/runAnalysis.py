#!/usr/bin/python

#from ROOT import TChain, TProof, TFile, TH1D, TH1F, TCanvas, gROOT, TTree
from ROOT import *
import os, sys
gROOT.SetBatch(True)
import numpy as np

#version = sys.argv[1]
#inputdir = sys.argv[2]
#tuples = sys.argv[3]
#name = sys.argv[4]

def runAna(dir, file, name):
  chain = TChain("ttbbLepJets/tree","events")
  chain.Add(dir+"/"+file)
  #chain.SetProof();
  chain.Process("makeTuple.C+",name)
  """
  f = TFile("tmva_"+name+".root","update")
  tr = f.Get("tmva_tree")
  totalnevt = np.zeros(1, dtype=float)
  tr.Branch('totnevt', totalnevt, 'totnevt/D')
  nevt = tr.GetEntries()
  for i in xrange(nevt):
    #tr.GetEntry(i)
    totalnevt[0] = nevt
  tr.Fill()
  f.Write()
  f.Close()
  """
#p = TProof.Open("", "workers=8")

#runAna(inputdir+version, tuples, name)
#runAna("/data/users/minerva1993/ntuple_Run2016/v4","TT_powheg_ttbb.root","ttbb")

for bb in range(0, 96):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/TT_powheg_ttbb","Tree_ttbbLepJets_"+ str(bb)+".root","ttbb_"+ str(bb))
"""
for lf in range(0, 96):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/TT_powheg_ttlf","Tree_ttbbLepJets_"+ str(lf) + ".root","ttLF_split_"+ str(lf))
for bj in range(0, 96):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/TT_powheg_ttbj","Tree_ttbbLepJets_"+ str(bj) + ".root","ttbj_split_"+ str(bj))
for cc in range(0, 96):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/TT_powheg_ttcc","Tree_ttbbLepJets_"+ str(cc) + ".root","ttcc_split_"+ str(cc))
for other in range(0, 96):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/TT_powheg_ttother","Tree_ttbbLepJets_"+ str(other) + ".root","tt_split_"+ str(other))
"""
