#!/usr/bin/python

#from ROOT import TChain, TProof, TFile, TH1D, TH1F, TCanvas, gROOT, TTree
from ROOT import *
import os, sys
gROOT.SetBatch(True)

input_filename = sys.argv[1]
output_filename = sys.argv[2]

def runAna(input_filename, output_filename):
  chain = TChain("fcncLepJets/tree","events")
  chain.Add(input_filename)
  chain.Process("makeOtherTuple.C+", output_filename)

runAna(input_filename, output_filename)
