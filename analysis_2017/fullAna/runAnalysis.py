#!/usr/bin/python

#from ROOT import TChain, TProof, TFile, TH1D, TH1F, TCanvas
from ROOT import *
gROOT.SetBatch(True)

def runAna(dir, file, name):
  chain = TChain("fcncLepJets/tree","events")
  chain.Add(dir+"/"+file)
  chain.SetProof();
  chain.Process("MyAnalysis.C+",name)

  f = TFile(dir+"/"+file,"read")

  ## save Event Summary histogram ##
  out = TFile("hist_"+name+".root","update")
  hevt = f.Get("fcncLepJets/EventInfo")
  hevt.Write()
  out.Write()
  out.Close()


p = TProof.Open("", "workers=8")

version = "V9_1"
inputdir = "/data/users/minerva1993/ntuple_Run2017/"

runAna(inputdir+version,"W1JetsToLNu_400-inf.root","w1jets400toinf")
"""
runAna(inputdir+version,"SingleElectron_Run2017.root","DataSingleEG")
runAna(inputdir+version,"SingleMuon_Run2017.root","DataSingleMu")
runAna(inputdir+version,"W1JetsToLNu_50-150.root","w1jets50to150")
runAna(inputdir+version,"W1JetsToLNu_250-400.root","w1jets250to400")
runAna(inputdir+version,"W1JetsToLNu_400-inf.root","w1jets400toinf")
runAna(inputdir+version,"W3JetsToLNu.root","w3jets")
runAna(inputdir+version,"W4JetsToLNu.root","w4jets")
runAna(inputdir+version,"DYJets.root","zjets")
runAna(inputdir+version,"DYJets_10to50.root","zjets10to50")
runAna(inputdir+version,"WW.root","ww")
runAna(inputdir+version,"WZ.root","wz")
runAna(inputdir+version,"ZZ.root","zz")
runAna(inputdir+version,"SingleTop_t.root","tchannel")
runAna(inputdir+version,"SingleTbar_t.root","tbarchannel")
runAna(inputdir+version,"SingleTop_tW.root","tWchannel")
runAna(inputdir+version,"SingleTbar_tW.root","tbarWchannel")
runAna(inputdir+version,"TT_powheg_PSweight.root","tt")
runAna(inputdir+version,"ST_TH_1L3B_Hct.root","singleTop_Hct")
runAna(inputdir+version,"ST_TH_1L3B_Hut.root","singleTop_Hut")
"""
#runAna(inputdir+version,"TT_powheg_ttbb.root","ttbb")
#runAna(inputdir+version,"TT_powheg_ttbj.root","ttbj")
#runAna(inputdir+version,"TT_powheg_ttcc.root","ttcc")
#runAna(inputdir+version,"TT_powheg_ttlf.root","ttLF")
#runAna(inputdir+version,"TT_powheg_ttother.root","tt")
#runAna(inputdir+version,"TT_TopLeptonicDecay_TH_1L3B_Eta_Hct.root","Top_Hct")
#runAna(inputdir+version,"TT_TopLeptonicDecay_TH_1L3B_Eta_Hut.root","Top_Hut")
#runAna(inputdir+version,"TT_AntitopLeptonicDecay_TH_1L3B_Eta_Hct.root","AntiTop_Hct")
#runAna(inputdir+version,"TT_AntitopLeptonicDecay_TH_1L3B_Eta_Hut.root","AntiTop_Hut")
