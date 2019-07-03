#!/usr/bin/python
from ROOT import *
import os, sys
from root_numpy import tree2array
import glob
import pandas as pd
import deepdish.io as io
gROOT.SetBatch(True)
gROOT.ProcessLine("gErrorIgnoreLevel = kFatal;")

#Channel and version
if len(sys.argv) < 4:
  print("Not enough arguements: Ch, Era, Input, Output")
  sys.exit()
reco_scheme = sys.argv[1]
era = sys.argv[2]
input_filename = sys.argv[3]
output_filename = sys.argv[4]
syst = ["","jecup","jecdown","jerup","jerdown"]
syst2 = ["TuneCP5up","TuneCP5down","hdampup","hdampdown"] #dedecative samples exist

for syst_ext in syst + syst2:
  if not os.path.exists( '/data1/users/minerva1993/work/fcnc_RunII' + era + '/reco/current_ver/root_' + reco_scheme + syst_ext ):
#  if not os.path.exists( era + "/root_" + reco_scheme + syst_ext ):
    print "No folder"
    sys.exit()
  if not os.path.exists( '/data1/users/minerva1993/work/fcnc_RunII' + era + '/reco/current_ver/hdf_' + reco_scheme + syst_ext ):
#  if not os.path.exists( era + "/hdf_" + reco_scheme + syst_ext ):
    print "No folder"
    sys.exit()

def runAna(input_filename, output_filename):
  print 'processing ' + input_filename.split('/')[-2] + '/' + input_filename.split('/')[-1]

  for syst_ext in syst + syst2:
    if   ("Run201" in output_filename) and syst_ext != "": continue
    elif (syst_ext in syst2) and not (syst_ext in output_filename): continue
    elif (syst_ext in syst) and any(tmp in output_filename for tmp in syst2): continue
    else:
      if (syst_ext in syst2): output_filename = output_filename.replace(syst_ext,"")

      chain = TChain("fcncLepJets/tree","events")
      chain.Add(input_filename)
      if chain.GetEntries() == 0:
        print "The file " + output_filename + ".root has zero entry"
      else: chain.Process("makeOtherTuple.C+", era + reco_scheme + syst_ext + '_' + output_filename)

#      if os.path.isfile(era + '/root_' + reco_scheme + syst_ext + '/deepReco_' + output_filename + '.root'):
#        f = TFile.Open(era + '/root_' + reco_scheme + syst_ext + '/deepReco_' + output_filename + '.root')
#        t = f.Get('test_tree')
#        if t.GetEntries() != 0:
#          a = tree2array(t)
#          df = pd.DataFrame(a)
#          io.save(era+ '/hdf_' + reco_scheme + syst_ext + '/deepReco_' + output_filename + '.h5', df)
#        else: os.remove(era + '/root_' + reco_scheme + syst_ext + '/deepReco_' + output_filename + '.root')

      if os.path.isfile('/data1/users/minerva1993/work/fcnc_RunII' + era + '/reco/current_ver/root_' + reco_scheme + syst_ext + '/deepReco_' + output_filename + '.root'):
        f = TFile.Open('/data1/users/minerva1993/work/fcnc_RunII' + era + '/reco/current_ver/root_' + reco_scheme + syst_ext + '/deepReco_' + output_filename + '.root')
        t = f.Get('test_tree')
        if t.GetEntries() != 0:
          a = tree2array(t)
          df = pd.DataFrame(a)
          io.save('/data1/users/minerva1993/work/fcnc_RunII' + era + '/reco/current_ver/hdf_' + reco_scheme + syst_ext + '/deepReco_' + output_filename + '.h5', df)
        else: os.remove('root_' + reco_scheme + syst_ext + '/deepReco_' + output_filename + '.root')

runAna(input_filename, output_filename)

