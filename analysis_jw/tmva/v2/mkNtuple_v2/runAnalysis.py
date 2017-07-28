#!/usr/bin/python

#from ROOT import TChain, TProof, TFile, TH1D, TH1F, TCanvas
from ROOT import *
gROOT.SetBatch(True)

def runAna(dir, file, name):
  chain = TChain("ttbbLepJets/tree","events")
  chain.Add(dir+"/"+file)
  chain.SetProof();
  chain.Process("MyAnalysis.C+",name)

  f = TFile(dir+"/"+file,"read")

  ## save Event Summary histogram ##
  out = TFile("hist_"+name+".root","update")
  hevt = f.Get("ttbbLepJets/EventInfo")
  hevt.Write()
  out.Write()
  out.Close()


p = TProof.Open("", "workers=8")

version = "v3"
inputdir = "/home/minerva1993/fcnc/ntuple_jw/"
"""
runAna(inputdir+version,"SingleElectron_Run2016.root","DataSingleEG")
runAna(inputdir+version,"SingleMuon_Run2016.root","DataSingleMu")
runAna(inputdir+version,"WJets.root","wjets")
runAna(inputdir+version,"DYJets.root","zjets")
runAna(inputdir+version,"DYJets_10to50.root","zjets10to50")
runAna(inputdir+version,"WW.root","ww")
runAna(inputdir+version,"WZ.root","wz")
runAna(inputdir+version,"ZZ.root","zz")
runAna(inputdir+version,"SingleTop_t.root","tchannel")
runAna(inputdir+version,"SingleTbar_t.root","tbarchannel")
runAna(inputdir+version,"SingleTop_tW.root","tWchannel")
runAna(inputdir+version,"SingleTbar_tW.root","tbarWchannel")
"""
runAna(inputdir+version,"TT_powheg_ttbb.root","ttbb")
"""
runAna(inputdir+version,"TT_powheg_ttbj.root","ttbj")
runAna(inputdir+version,"TT_powheg_ttcc.root","ttcc")
runAna(inputdir+version,"TT_powheg_ttlf.root","ttLF")
runAna(inputdir+version,"TT_powheg_ttother.root","tt")
#runAna(inputdir+version,","ttBkg")
runAna(inputdir+version,"TT_TopLeptonicDecay_TH_1L3B_Eta_Hct.root","Top_Hct")
runAna(inputdir+version,"TT_TopLeptonicDecay_TH_1L3B_Eta_Hut.root","Top_Hut")
runAna(inputdir+version,"TT_AntitopLeptonicDecay_TH_1L3B_Eta_Hct.root","AntiTop_Hct")
runAna(inputdir+version,"TT_AntitopLeptonicDecay_TH_1L3B_Eta_Hut.root","AntiTop_Hut")
"""
