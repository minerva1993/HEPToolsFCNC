#!/usr/bin/python
from ROOT import *
import os, sys
from root_numpy import tree2array
import glob
import pandas as pd
import deepdish.io as io
gROOT.SetBatch(True)

ch = "STFCNC"
#ch = "TTFCNC"
#ch = "TTBKG"

input_filename = sys.argv[1]
output_filename = sys.argv[2]

def runAna(input_filename, output_filename):
  chain = TChain("fcncLepJets/tree","events")
  chain.Add(input_filename)
  if chain.GetEntries() == 0:
    print "The file " + output_filename + ".root has zero entry"
  else: chain.Process("makeTopTuple.C+", ch + '_' + output_filename)

  if os.path.isfile(ch + "_root" + "/deepReco_" + output_filename + ".root"):
    f = TFile.Open(ch + "_root" + "/deepReco_" + output_filename + ".root")
    t = f.Get("test_tree")
    if t.GetEntries() != 0:
      a = tree2array(t)
      df = pd.DataFrame(a)
      io.save(ch + "_hdf" + "/deepReco_" + output_filename + ".h5", df)
    else: os.remove(ch + "_root" + "/deepReco_" + output_filename + ".root")

runAna(input_filename, output_filename)
