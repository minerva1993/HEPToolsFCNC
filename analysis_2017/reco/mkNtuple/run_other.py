#!/usr/bin/python
from ROOT import *
import os, sys
from root_numpy import tree2array
import glob
import pandas as pd
import deepdish.io as io
gROOT.SetBatch(True)

ch = sys.argv[1]
input_filename = sys.argv[2]
output_filename = sys.argv[3]

if not os.path.exists( "root_" + ch ):
  print "No folder"
  sys.exit()
if not os.path.exists( "hdf_" + ch ):
  print "No folder"
  sys.exit()

def runAna(input_filename, output_filename):
  chain = TChain("fcncLepJets/tree","events")
  chain.Add(input_filename)
  if chain.GetEntries() == 0:
    print "The file " + output_filename + ".root has zero entry"
  else: chain.Process("makeOtherTuple.C+", ch + '_' + output_filename)

  if os.path.isfile("root_" + ch + "/deepReco_" + output_filename + ".root"):
    f = TFile.Open("root_" + ch + "/deepReco_" + output_filename + ".root")
    t = f.Get("test_tree")  
    if t.GetEntries() != 0:
      a = tree2array(t)
      df = pd.DataFrame(a)
      io.save("hdf_" + ch + "/deepReco_" + output_filename + ".h5", df)
    else: os.remove("root_" + ch + "/deepReco_" + output_filename + ".root")

runAna(input_filename, output_filename)
