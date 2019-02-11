#!/usr/bin/env python
import sys, os

from ROOT import *

TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()

#Version of classifier
ver = '03'
configDir = '/home/minerva1993/HEPToolsFCNC/analysis_2017/reco/'
weightDir = 'training/recoSTFCNC'
scoreDir = 'scoreSTFCNC'
assignDir = 'assignSTFCNC'

#Options for data preparation
input_files = ['deepReco_STTH1L3BHct_000.root',
               'deepReco_STTH1L3BHut_000.root',
              ]

int_vars = []
float_vars = ['jet0pt', 'jet0eta', 'jet0m', 'jet1pt', 'jet1eta', 'jet1m', 'jet2pt', 'jet2eta', 'jet2m',
                  'jet12pt', 'jet12eta', 'jet12deta', 'jet12dphi', #'jet12dR', 'jet12m',
                  'lepWpt', 'lepWdphi', 'lepWm', 'lepTdphi', 'lepTm',]
options = "nTrain_Signal=51000:nTrain_Background=290000:nTest_Signal=0:nTest_Background=0:SplitMode=Random:NormMode=NumEvents:!V"

#Check if the model and files already exist
if not os.path.exists( os.path.join(configDir, weightDir+ver) ):
  os.makedirs( os.path.join(configDir, weightDir+ver) )
if not os.path.exists( os.path.join(configDir, scoreDir+ver) ):
  os.makedirs( os.path.join(configDir, scoreDir+ver) )
if not os.path.exists( os.path.join(configDir, assignDir+ver) ):
  os.makedirs( os.path.join(configDir, assignDir+ver) )
if not os.path.exists( os.path.join(configDir, weightDir+ver, 'training_bdt.py') ):
  shutil.copy2('training_bdt.py', os.path.join(configDir, weightDir+ver, 'training_bdt.py'))
for item in os.listdir( os.path.join(configDir, weightDir+ver) ) or os.listdir( os.path.join(configDir, scoreDir+ver) ) or os.listdir( os.path.join(configDir, assignDir+ver) ):
  if item.endswith(".pdf") or item.endswith(".h5") or item.endswith("log"):
    #os.remove(os.path.join(os.path.join(configDir, weightDir+ver), item))
    print("Remove previous files or move on to next version!")
    sys.exit()


fout = TFile("output.root","recreate")

factory = TMVA.Factory("TMVAClassification", fout, "!V:!Silent:Color:DrawProgressBar:AnalysisType=Classification" )

loader = TMVA.DataLoader("recoSTFCNC"+ver)
for var in int_vars:
    loader.AddVariable(var, "I")
for var in float_vars:
    loader.AddVariable(var, "F")

loader.SetWeightExpression("EventWeight")
loader.AddSpectator("nevt")
loader.AddSpectator("EventCategory")
loader.AddSpectator("genMatch")
loader.AddSpectator("njets")
loader.AddSpectator("nbjets_m")

trees = []
for fName in input_files:
    #if "Hut" in fName: fileWeight = 1
    #elif "Hct" in fName: fileWeight = 0.15
    fileWeight = 1
    f = TFile("../mkNtuple/dummy/"+fName)
    t = f.Get("sig_tree")
    loader.AddSignalTree(t, fileWeight)
    trees.append([f, t])
for fName in input_files:
    #if "Hut" in fName: fileWeight = 1
    #elif "Hct" in fName: fileWeight = 0.15
    fileWeight = 1
    f = TFile("../mkNtuple/dummy/"+fName)
    t = f.Get("bkg_tree")
    loader.AddBackgroundTree(t, fileWeight)
    trees.append([f, t])

sigCut = TCut("")
bkgCut = TCut("")

loader.PrepareTrainingAndTestTree(sigCut, bkgCut, options)

factory.BookMethod(loader, TMVA.Types.kBDT, "BDT", "!H:!V:NTrees=800:MinNodeSize=5%:MaxDepth=8:BoostType=Grad:Shrinkage=0.5:SeparationType=GiniIndex:nCuts=40")

factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()
fout.Close()

if not os.path.exists( os.path.join(configDir, weightDir+ver, 'output.root') ):
  shutil.move('output.root', os.path.join(configDir, weightDir+ver))
