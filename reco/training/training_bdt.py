#!/usr/bin/env python
import sys, os, shutil
import pandas as pd
import numpy as np
from root_numpy import array2tree
from variables import input_variables, gen_label, train_files
from ROOT import *

TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()

#Channel and version
if len(sys.argv) < 4:
  print("Not enough arguements: Ch, Ver, Era")
  sys.exit()
ch = sys.argv[1]
ver = sys.argv[2]
era = sys.argv[3]
if ch not in ["STFCNC", "TTFCNC", "TTBKG"]:
  print("Check channal again!: STFCNC, TTFCNC, of TTBKG are available")

#directory name
configDir = '../'
weightDir = 'training/' + era + '/reco' + ch
scoreDir = era + '/score' + ch
assignDir = era + '/assign' + ch

#Options for data preparation
input_files = []
input_features = []
signal_label = gen_label(ch)
input_files.extend(train_files(ch, era))
input_features.extend(input_variables(ch))
input_features.append('genMatch')
label_name = 'genMatch'

#Check if the model and files already exist
if not os.path.exists( os.path.join(configDir, weightDir+ver) ):
  os.makedirs( os.path.join(configDir, weightDir+ver) )
#if not os.path.exists( os.path.join(configDir, scoreDir+ver) ):
#  os.makedirs( os.path.join(configDir, scoreDir+ver) )
#if not os.path.exists( os.path.join(configDir, assignDir+ver) ):
#  os.makedirs( os.path.join(configDir, assignDir+ver) )
if not os.path.exists( os.path.join(configDir, weightDir+ver, 'training_bdt.py') ):
  shutil.copy2('training_bdt.py', os.path.join(configDir, weightDir+ver, 'training_bdt.py'))
#for item in os.listdir( os.path.join(configDir, weightDir+ver) ) or os.listdir( os.path.join(configDir, scoreDir+ver) ) or os.listdir( os.path.join(configDir, assignDir+ver) ):
for item in os.listdir( os.path.join(configDir, weightDir+ver) ):
  if item.endswith(".pdf") or item.endswith(".h5") or item.endswith("log"):
    #os.remove(os.path.join(os.path.join(configDir, weightDir+ver), item))
    print("Remove previous files or move on to next version!")
    sys.exit()


fout = TFile("output.root","recreate")

factory = TMVA.Factory("TMVAClassification", fout, "!V:!Silent:Color:DrawProgressBar:AnalysisType=Classification" )

loader = TMVA.DataLoader(era+"/reco"+ch+ver)
for var in input_features:
    if 'gen' in var: continue
    loader.AddVariable(var, "F")

nsig = 0
nbkg = 0
for fName in input_files:
    fileWeight = 1
    data = pd.read_hdf('/data1/users/minerva1993/work/fcnc_RunII' + era + '/reco/current_ver/hdf_' + ch + '/' +fName)
    data = data.filter(input_features, axis=1)
    data = data.loc[data[label_name] == signal_label]
    data = data.drop(label_name, axis=1)
    nsig += data.shape[0]
    data = data.to_records()
    t = array2tree(data, name='tree')
    loader.AddSignalTree(t, fileWeight)
for fName in input_files:
    fileWeight = 1
    data = pd.read_hdf('/data1/users/minerva1993/work/fcnc_RunII' + era + '/reco/current_ver/hdf_' + ch + '/' +fName)
    data = data.filter(input_features, axis=1)
    data = data.loc[data[label_name] != signal_label]
    data = data.drop(label_name, axis=1)
    nbkg += data.shape[0]
    data = data.to_records()
    t = array2tree(data, name='tree')
    loader.AddBackgroundTree(t, fileWeight)

sigCut = TCut("")
bkgCut = TCut("")

options = "nTrain_Signal="+str(nsig*0.8)+":nTrain_Background="+str(nbkg*0.8)+":SplitMode=Random:NormMode=NumEvents:!V"
loader.PrepareTrainingAndTestTree(sigCut, bkgCut, options)
factory.BookMethod(loader, TMVA.Types.kBDT, "BDT", "!H:!V:NTrees=10:MinNodeSize=5%:MaxDepth=8:BoostType=Grad:Shrinkage=0.5:SeparationType=GiniIndex:nCuts=5")

factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()
fout.Close()

if not os.path.exists( os.path.join(configDir, weightDir+ver, 'output.root') ):
  shutil.move('output.root', os.path.join(configDir, weightDir+ver))
