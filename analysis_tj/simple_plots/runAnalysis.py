#!/usr/bin/python

#from ROOT import TChain, TProof, TFile, TH1D, TH1F, TCanvas
from ROOT import *
gROOT.SetBatch(True)

def runAna(dir, file, name):
  chain = TChain("TopTree/events","events")
  chain.Add(dir+"/"+file)
  chain.SetProof();
  chain.Process("FCNCAnalysis.C+",name)

  f = TFile(dir+"/"+file,"read")

  ## save Event Summary histogram ##
  out = TFile("hist_"+name+".root","update")
  hevt = f.Get("TopTree/EventSummary")
  hevt.Write()
  out.Write()
  out.Close()


p = TProof.Open("", "workers=8")

runAna("/home/minerva1993/fcnc/ntuple_tj","vallot_TT_powheg.root","ttbb")
#runAna("/home/minerva1993/fcnc/ntuple_tj","vallot_TT_TopLeptonicDecay_TH_1L3B_Eta_Hct.root","Top_Hct")
#runAna("/home/minerva1993/fcnc/ntuple_tj","vallot_TT_TopLeptonicDecay_TH_1L3B_Eta_Hut.root","Top_Hut")
#runAna("/home/minerva1993/fcnc/ntuple_tj","vallot_TT_AntitopLeptonicDecay_TH_1L3B_Eta_Hct.root","AntiTop_Hct")
#runAna("/home/minerva1993/fcnc/ntuple_tj","vallot_TT_AntitopLeptonicDecay_TH_1L3B_Eta_Hut.root","AntiTop_Hut")

