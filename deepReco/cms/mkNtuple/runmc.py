#!/usr/bin/python

#from ROOT import TChain, TProof, TFile, TH1D, TH1F, TCanvas, gROOT, TTree
from ROOT import *
import os, sys
gROOT.SetBatch(True)

def runAna(dir, file, name):
  chain = TChain("ttbbLepJets/tree","events")
  chain.Add(dir+"/"+file)
  #chain.SetProof();
  chain.Process("makeTuple.C+", name)

#p = TProof.Open("", "workers=8")

#for bb in range(0, 1):
#  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/TT_powheg_ttbb","Tree_ttbbLepJets_"+str(bb)+".root","ttbb_"+ str(bb))
"""
for hct in range(0, 16):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/signalReco/TT_TopLeptonicDecay_TH_1L3B_Eta_Hct","Tree_ttbbLepJets_"+str(hct)+".root","TopHct_"+str(hct))

for ahct in range(0, 15):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/signalReco/TT_AntitopLeptonicDecay_TH_1L3B_Eta_Hct","Tree_ttbbLepJets_"+str(ahct)+".root","AntiTopHct_"+str(ahct))

for hut in range(0, 16):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/signalReco/TT_TopLeptonicDecay_TH_1L3B_Eta_Hut","Tree_ttbbLepJets_"+str(hut)+".root","TopHut_"+str(ahct))

for ahut in range(0, 16):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/signalReco/TT_AntitopLeptonicDecay_TH_1L3B_Eta_Hut","Tree_ttbbLepJets_"+str(ahut)+".root","AntiTopHut_"+str(ahut))
"""
for bb in range(0, 96):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/TT_powheg_ttbb","Tree_ttbbLepJets_"+str(bb)+".root","ttbb_"+str(bb))

for lf in range(0, 96):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/TT_powheg_ttlf","Tree_ttbbLepJets_"+str(lf)+".root","ttLF_"+str(lf))

for bj in range(0, 96):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/TT_powheg_ttbj","Tree_ttbbLepJets_"+str(bj)+".root","ttbj_"+str(bj))

for cc in range(0, 96):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/TT_powheg_ttcc","Tree_ttbbLepJets_"+str(cc)+".root","ttcc_"+str(cc))

for other in range(0, 96):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/TT_powheg_ttother","Tree_ttbbLepJets_"+str(other)+".root","ttother_"+str(other))
