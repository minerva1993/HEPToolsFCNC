#!/usr/bin/python

#from ROOT import TChain, TProof, TFile, TH1D, TH1F, TCanvas
from ROOT import *
gROOT.SetBatch(True)

def runAna(dir, file, name):
  chain = TChain("tree","events")
  chain.Add(dir+"/"+file)
  #chain.SetProof();
  chain.Process("tmva_score.C+",name)

  f = TFile(dir+"/"+file,"read")

#p = TProof.Open("", "workers=8")

#ch = "Hct30"
ch = "Hut30"

version = "v8"
inputdir = "/home/minerva1993/fcnc/analysis_jw/tmva/"+version+"/score_mva/"+ch

runAna(inputdir, "output_"+ch+"_tmva_AntiTop_Hct.root",  ch+"_AntiTop_Hct")
runAna(inputdir, "output_"+ch+"_tmva_AntiTop_Hut.root", ch+"_AntiTop_Hut")
runAna(inputdir, "output_"+ch+"_tmva_Top_Hct.root", ch+"_Top_Hct")
runAna(inputdir, "output_"+ch+"_tmva_Top_Hut.root", ch+"_Top_Hut")
runAna(inputdir, "output_"+ch+"_tmva_SingleLepton_Run2016.root", ch+"_SingleLepton_Run2016")
runAna(inputdir, "output_"+ch+"_tmva_ttbb.root", ch+"_ttbb")
runAna(inputdir, "output_"+ch+"_tmva_ttbj.root", ch+"_ttbj")
runAna(inputdir, "output_"+ch+"_tmva_ttcc.root", ch+"_ttcc")
runAna(inputdir, "output_"+ch+"_tmva_ttLF.root", ch+"_ttLF")
runAna(inputdir, "output_"+ch+"_tmva_ttother.root", ch+"_ttother")
runAna(inputdir, "output_"+ch+"_tmva_tbarchannel.root", ch+"_tbarchannel")
runAna(inputdir, "output_"+ch+"_tmva_tbarWchannel.root", ch+"_tbarWchannel")
runAna(inputdir, "output_"+ch+"_tmva_tchannel.root", ch+"_tchannel")
runAna(inputdir, "output_"+ch+"_tmva_tWchannel.root", ch+"_tWchannel")

runAna(inputdir, "output_"+ch+"_tmva_zjets10to50.root", ch+"_zjets10to50")
runAna(inputdir, "output_"+ch+"_tmva_zjets.root", ch+"_zjets")
runAna(inputdir, "output_"+ch+"_tmva_wjets.root", ch+"_wjets")
"""
runAna(inputdir, "output_"+ch+"_tmva_ww.root", ch+"_ww")
runAna(inputdir, "output_"+ch+"_tmva_wz.root", ch+"_wz")
runAna(inputdir, "output_"+ch+"_tmva_zz.root", ch+"_zz")

runAna(inputdir, "output_"+ch+"_tmva_QCD_EGEnr_120to170.root", ch+"_QCD_EGEnr_120to170")
runAna(inputdir, "output_"+ch+"_tmva_QCD_EGEnr_170to300.root", ch+"_QCD_EGEnr_170to300")
runAna(inputdir, "output_"+ch+"_tmva_QCD_EGEnr_300toInf.root", ch+"_QCD_EGEnr_300toInf")
runAna(inputdir, "output_"+ch+"_tmva_QCD_MuEnr_1000toInf.root", ch+"_QCD_MuEnr_1000toInf")
runAna(inputdir, "output_"+ch+"_tmva_QCD_MuEnr_120to170.root", ch+"_QCD_MuEnr_120to170")
runAna(inputdir, "output_"+ch+"_tmva_QCD_MuEnr_170to300.root", ch+"_QCD_MuEnr_170to300")
runAna(inputdir, "output_"+ch+"_tmva_QCD_MuEnr_300to470.root", ch+"_QCD_MuEnr_300to470")
runAna(inputdir, "output_"+ch+"_tmva_QCD_MuEnr_800to1000.root", ch+"_QCD_MuEnr_800to1000")
runAna(inputdir, "output_"+ch+"_tmva_QCD_MuEnr_80to120.root", ch+"_QCD_MuEnr_80to120")
"""
