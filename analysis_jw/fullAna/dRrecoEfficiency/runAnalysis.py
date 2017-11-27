#!/usr/bin/python

#from ROOT import TChain, TProof, TFile, TH1D, TH1F, TCanvas
from ROOT import *
gROOT.SetBatch(True)

def runAna(dir, file, name):
  chain = TChain("ttbbLepJets/tree","events")
  chain.Add(dir+"/"+file)
  #chain.SetProof();
  chain.Process("higgsRecoEff.C+",name)

  f = TFile(dir+"/"+file,"read")

#p = TProof.Open("", "workers=8")

version = "v4"
inputdir = "/home/minerva1993/fcnc/ntuple_jw/"

runAna(inputdir+version,"TT_TopLeptonicDecay_TH_1L3B_Eta_Hct.root","Top_Hct")
runAna(inputdir+version,"TT_TopLeptonicDecay_TH_1L3B_Eta_Hut.root","Top_Hut")
runAna(inputdir+version,"TT_AntitopLeptonicDecay_TH_1L3B_Eta_Hct.root","AntiTop_Hct")
runAna(inputdir+version,"TT_AntitopLeptonicDecay_TH_1L3B_Eta_Hut.root","AntiTop_Hut")
