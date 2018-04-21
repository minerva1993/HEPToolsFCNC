#!/usr/bin/python

#from ROOT import TChain, TProof, TFile, TH1D, TH1F, TCanvas, gROOT, TTree
from ROOT import *
import os, sys
gROOT.SetBatch(True)

inputdir = sys.argv[1]
dataset = sys.argv[2]
tuples = sys.argv[3]
name = sys.argv[4]

def runAna(dir, file, name):
  chain = TChain("ttbbLepJets/tree","events")
  chain.Add(dir+"/"+file)
  chain.Process("dataTuple.C+", name)

runAna(inputdir+'/'+dataset, tuples, name)
