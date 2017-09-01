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

  ## save Event Summary histogram ##
  out = TFile("hist_"+name+".root","update")
  hevt = f.Get("ttbbLepJets/EventInfo")
  hevt.Write()
  out.Write()
  out.Close()


#p = TProof.Open("", "workers=2")

inputdir = "/home/minerva1993/fcnc/analysis_jw/fullAna/higgsRecoEfficiency"

runAna(inputdir,"ntuple_tch.root","tch")
runAna(inputdir,"ntuple_tuh.root","tuh")
