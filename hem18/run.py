#!/usr/bin/python
from ROOT import *
import os, sys
gROOT.SetBatch(True)
gROOT.ProcessLine("gErrorIgnoreLevel = kFatal;")

def runAna(file_path, name):
  print 'processing ' + file_path
  chain = TChain("fcncLepJets/tree","events")
  chain.Add(file_path)
  chain.Process("MyAnalysis.C+", name)
  #print chain.GetCurrentFile().GetName()

runAna("/data/users/minerva1993/ntuple_Run2018/V10_1/190318/TTLL_powheg.root", "TTLLpowheg")
runAna("/data/users/minerva1993/ntuple_Run2018/V10_1_HEM/TTLL_HEM.root", "TTLLHEM")
runAna("/data/users/minerva1993/ntuple_Run2018/V10_1/190318/TT_powheg_ttbb.root", "TTpowhegttbb")
runAna("/data/users/minerva1993/ntuple_Run2018/V10_1/190318/TT_powheg_ttbj.root", "TTpowhegttbj")
runAna("/data/users/minerva1993/ntuple_Run2018/V10_1/190318/TT_powheg_ttcc.root", "TTpowhegttcc")
runAna("/data/users/minerva1993/ntuple_Run2018/V10_1/190318/TT_powheg_ttlf.root", "TTpowhegttlf")
runAna("/data/users/minerva1993/ntuple_Run2018/V10_1/190318/TT_powheg_ttother.root", "TTpowhegttother")
runAna("/data/users/minerva1993/ntuple_Run2018/V10_1_HEM/TTLJ_HEM.root", "TTLJHEM")
