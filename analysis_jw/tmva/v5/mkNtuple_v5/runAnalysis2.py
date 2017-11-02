#!/usr/bin/python

from ROOT import TChain, TProof, TFile, TH1D, TH1F, TCanvas, gROOT
#from ROOT import *
import os, sys
gROOT.SetBatch(True)
import numpy as np

def runAna(dir, file, name):
  chain = TChain("ttbbLepJets/tree","events")
  chain.Add(dir+"/"+file)
  chain.Process("tmvaAnalysis.C+",name)

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

version = sys.argv[1]
inputdir = sys.argv[2]
tuples = sys.argv[3]
name = sys.argv[4]

runAna(inputdir+version, tuples, name)
