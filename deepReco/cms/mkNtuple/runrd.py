#!/usr/bin/python

#from ROOT import TChain, TProof, TFile, TH1D, TH1F, TCanvas, gROOT, TTree
from ROOT import *
import os, sys
gROOT.SetBatch(True)

def runAna(dir, file, name):
  chain = TChain("ttbbLepJets/tree","events")
  chain.Add(dir+"/"+file)
  #chain.SetProof();
  chain.Process("dataTuple.C+",name)
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

"""
for mub in range(0, 66):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/SingleMuon_Run2016B","Tree_ttbbLepJets_"+str(mub)+".root","DataSingleMuB_"+ str(mub))

for muc in range(0, 22):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/SingleMuon_Run2016C","Tree_ttbbLepJets_"+str(muc)+".root","DataSingleMuC_"+ str(muc))

for mud in range(0, 37):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/SingleMuon_Run2016D","Tree_ttbbLepJets_"+str(mud)+".root","DataSingleMuD_"+ str(mud))

for mue in range(0, 31):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/SingleMuon_Run2016E","Tree_ttbbLepJets_"+str(mue)+".root","DataSingleMuE_"+ str(mue))

for muf in range(0, 23):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/SingleMuon_Run2016F","Tree_ttbbLepJets_"+str(muf)+".root","DataSingleMuF_"+ str(muf))

for mug in range(0, 54):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/SingleMuon_Run2016G","Tree_ttbbLepJets_"+str(mug)+".root","DataSingleMuG_"+ str(mug))

for muh2 in range(0, 58):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/SingleMuon_Run2016H_v2","Tree_ttbbLepJets_"+str(muh2)+".root","DataSingleMuHv2_"+ str(muh2))

for muh3 in range(0, 2):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/SingleMuon_Run2016H_v3","Tree_ttbbLepJets_"+str(muh3)+".root","DataSingleMuHv3_"+ str(muh3))

for elb in range(0, 66):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/SingleElectron_Run2016B","Tree_ttbbLepJets_"+str(elb)+".root","DataSingleEGB_"+ str(elb))

for elc in range(0, 22):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/SingleElectron_Run2016C","Tree_ttbbLepJets_"+str(elc)+".root","DataSingleEGC_"+ str(elc))

for eld in range(0, 37):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/SingleElectron_Run2016D","Tree_ttbbLepJets_"+str(eld)+".root","DataSingleEGD_"+ str(eld))

for ele in range(0, 31):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/SingleElectron_Run2016E","Tree_ttbbLepJets_"+str(ele)+".root","DataSingleEGE_"+ str(ele))

for elf in range(0, 23):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/SingleElectron_Run2016F","Tree_ttbbLepJets_"+str(elf)+".root","DataSingleEGF_"+ str(elf))

for elg in range(0, 54):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/SingleElectron_Run2016G","Tree_ttbbLepJets_"+str(elg)+".root","DataSingleEGG_"+ str(elg))

for elh2 in range(0, 58):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/SingleElectron_Run2016H_v2","Tree_ttbbLepJets_"+str(elh2)+".root","DataSingleEGHv2_"+ str(elh2))

for elh3 in range(0, 2):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/SingleElectron_Run2016H_v3","Tree_ttbbLepJets_"+str(elh3)+".root","DataSingleEGHv3_"+ str(elh3))
"""
for hct in range(0, 16):
  runAna("/data/users/minerva1993/ntuple_Run2016/v4/production/TT_TopLeptonicDecay_TH_1L3B_Eta_Hct","Tree_ttbbLepJets_"+str(hct)+".root","Top_Hct_"+ str(hct))

