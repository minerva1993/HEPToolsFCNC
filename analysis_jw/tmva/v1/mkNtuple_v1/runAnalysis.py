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
  out = TFile("tmva_"+name+".root","update")
  hevt = f.Get("ttbbLepJets/EventInfo")
  hevt.Write()
  out.Write()
  out.Close()


p = TProof.Open("", "workers=8")

version = "v1"
inputdir = "/home/minerva1993/fcnc/ntuple_jw/"

#runAna(inputdir+version,"Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_DataSingleEG.root","DataSingleEG")
#runAna(inputdir+version,"Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_DataSingleMu.root","DataSingleMu")
#runAna(inputdir+version,"Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_WJets_aMCatNLO.root","wjets")
#runAna(inputdir+version,"Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_ZJets_M50_aMCatNLO.root","zjets")
#runAna(inputdir+version,"Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_ZJets_M10to50_aMCatNLO.root","zjets10to50")
#runAna(inputdir+version,"Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_WW_Pythia.root","ww")
#runAna(inputdir+version,"Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_WZ_Pythia.root","wz")
#runAna(inputdir+version,"Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_ZZ_Pythia.root","zz")
#runAna(inputdir+version,"Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_t_tchannel_Powheg.root","tchannel")
#runAna(inputdir+version,"Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_tbar_tchannel_Powheg.root","tbarchannel")
#runAna(inputdir+version,"Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_tW_Powheg.root","tWchannel")
#runAna(inputdir+version,"Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_tbarW_Powheg.root","tbarWchannel")
runAna(inputdir+version,"TT_powheg_ttbb.root","ttbb")
#runAna(inputdir+version,"Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_ttbar_LepJetsPowhegPythiattbj.root","ttbj")
#runAna(inputdir+version,"Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_ttbar_LepJetsPowhegPythiattcc.root","ttcc")
#runAna(inputdir+version,"Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_ttbar_LepJetsPowhegPythiattLF.root","ttLF")
#runAna(inputdir+version,"Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_ttbar_LepJetsPowhegPythiatt.root","tt")
#runAna(inputdir+version,"Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_ttbar_PowhegPythiaBkg.root","ttBkg")
runAna(inputdir+version,"TT_TopLeptonicDecay_TH_1L3B_Eta_Hct.root","Top_Hct")
runAna(inputdir+version,"TT_TopLeptonicDecay_TH_1L3B_Eta_Hut.root","Top_Hut")
runAna(inputdir+version,"TT_AntitopLeptonicDecay_TH_1L3B_Eta_Hct.root","AntiTop_Hct")
runAna(inputdir+version,"TT_AntitopLeptonicDecay_TH_1L3B_Eta_Hut.root","AntiTop_Hut")

