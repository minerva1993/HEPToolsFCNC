#!/usr/bin/python

from ROOT import TChain, TProof, TFile, TH1D, TH1F, TCanvas

def runAna(dir, file, name):
  chain = TChain("ttbbLepJetsQCD/tree","events")
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

#runAna("/xrootd/store/user/brochero/v8-0-6/","Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_DataSingleEG.root","DataSingleEG")
#runAna("/xrootd/store/user/brochero/v8-0-6/","Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_DataSingleMu.root","DataSingleMu")
#runAna("/xrootd/store/user/brochero/v8-0-6/","Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_WJets_aMCatNLO.root","wjets")
#runAna("/xrootd/store/user/brochero/v8-0-6/","Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_ZJets_M50_aMCatNLO.root","zjets")
#runAna("/xrootd/store/user/brochero/v8-0-6/","Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_ZJets_M10to50_aMCatNLO.root","zjets10to50")
#runAna("/xrootd/store/user/brochero/v8-0-6/","Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_WW_Pythia.root","ww")
#runAna("/xrootd/store/user/brochero/v8-0-6/","Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_WZ_Pythia.root","wz")
#runAna("/xrootd/store/user/brochero/v8-0-6/","Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_ZZ_Pythia.root","zz")
#runAna("/xrootd/store/user/brochero/v8-0-6/","Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_t_tchannel_Powheg.root","tchannel")
#runAna("/xrootd/store/user/brochero/v8-0-6/","Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_tbar_tchannel_Powheg.root","tbarchannel")
#runAna("/xrootd/store/user/brochero/v8-0-6/","Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_tW_Powheg.root","tWchannel")
#runAna("/xrootd/store/user/brochero/v8-0-6/","Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_tbarW_Powheg.root","tbarWchannel")
runAna("/home/minerva1993/fcnc/ntuple","Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_ttbar_LepJetsPowhegPythiattbb.root","ttbb")
#runAna("/xrootd/store/user/brochero/v8-0-6/","Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_ttbar_LepJetsPowhegPythiattbj.root","ttbj")
#runAna("/xrootd/store/user/brochero/v8-0-6/","Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_ttbar_LepJetsPowhegPythiattcc.root","ttcc")
#runAna("/xrootd/store/user/brochero/v8-0-6/","Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_ttbar_LepJetsPowhegPythiattLF.root","ttLF")
#runAna("/xrootd/store/user/brochero/v8-0-6/","Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_ttbar_LepJetsPowhegPythiatt.root","tt")
#runAna("/xrootd/store/user/brochero/v8-0-6/","Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_ttbar_PowhegPythiaBkg.root","ttBkg")
##runAna("/xrootd/store/user/brochero/v8-0-6/","Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_TT_TopLeptonicDecay_TH_1L3B_Eta_Hct.root","Top_Hct")
##runAna("/xrootd/store/user/brochero/v8-0-6/","Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_TT_TopLeptonicDecay_TH_1L3B_Eta_Hut.root","Top_Hut")
##runAna("/xrootd/store/user/brochero/v8-0-6/","Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_TT_AntitopLeptonicDecay_TH_1L3B_Eta_Hct.root","AntiTop_Hct")
##runAna("/xrootd/store/user/brochero/v8-0-6/","Tree_LepJets_Final_v8-0-6_Spring16-80X_36814pb-1_TT_AntitopLeptonicDecay_TH_1L3B_Eta_Hut.root","AntiTop_Hut")
