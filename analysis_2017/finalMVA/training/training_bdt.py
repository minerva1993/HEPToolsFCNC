#!/usr/bin/env python
import sys, os, shutil
from variables import input_variables_bdt, train_files, evalScale
from ROOT import *

TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()

#Version of classifier
ch = sys.argv[1]
jetcat = sys.argv[2]
ver = sys.argv[3]

njets_cut = int(jetcat[1:2]) #Must be jXbX
if njets_cut not in [3,4]:
  print("Check jet category")
  sys.exit()
if len(jetcat) > 3:
  nbjets_cut = int(jetcat[3:4])
  if nbjets_cut not in [2,3,4]:
    print("Check b jet category")
    sys.exit()
else: nbjets_cut = 0

if njets_cut == 3:
  sigCut = TCut("njets == 3 && nbjets_m == " + str(nbjets_cut) + " && EventWeight > 0")
  bkgCut = TCut("njets == 3 && nbjets_m == " + str(nbjets_cut) + " && EventWeight > 0")
elif njets_cut == 4:
  sigCut = TCut("njets >= 4 && nbjets_m == " + str(nbjets_cut) + " && EventWeight > 0")
  bkgCut = TCut("njets >= 4 && nbjets_m == " + str(nbjets_cut) + " && EventWeight > 0")

idx = {}
idx['j3b2'] = 0
idx['j3b3'] = 1
idx['j4b2'] = 2
idx['j4b3'] = 3
idx['j4b4'] = 4

nsig_Hct = ['55000', '23000', '50000', '31000', '2600']
nsig_Hut = ['50000', '17000', '52000', '28000', '1150']
nbkg = ['180000', '7100', '280000', '25000', '1300']
ntree = ['600', '200', '600', '300', '50']
ncut = ['30', '20', '30', '20', '20']

if ch == "Hct":
  options = "nTrain_Signal=" + nsig_Hct[idx[jetcat]] + ":nTrain_Background=" + nbkg[idx[jetcat]] + ":nTest_Signal=0:nTest_Background=0:SplitMode=Random:NormMode=NumEvents:!V"
elif ch == "Hut":
  options = "nTrain_Signal=" + nsig_Hut[idx[jetcat]] + ":nTrain_Background=" + nbkg[idx[jetcat]] + ":nTest_Signal=0:nTest_Background=0:SplitMode=Random:NormMode=NumEvents:!V"

#directory name
#rootDir = '/data/users/minerva1993/work/2018_fcnc_RunII2017/finalMVA/mkNtuple/1101/root_/'
rootDir = '/home/minerva1993/HEPToolsFCNC/analysis_2017/finalMVA/mkNtuple/root_/'
configDir = '/home/minerva1993/HEPToolsFCNC/analysis_2017/finalMVA/'
weightDir = 'training/final' + '_' + ch + '_' +jetcat + '_'
scoreDir = 'scores/' + ch + '_' +jetcat + '_'
njets_cut = int(jetcat[1:2]) #Must be jXbX
if njets_cut not in [3,4]:
  print("Check jet category")
  sys.exit()
if len(jetcat) > 3:
  nbjets_cut = int(jetcat[3:4])
  if nbjets_cut not in [2,3,4]:
    print("Check b jet category")
    sys.exit()
else: nbjets_cut = 0

#Check if the model and files already exist
if not os.path.exists( os.path.join(configDir, weightDir+ver, 'weights') ):
  os.makedirs( os.path.join(configDir, weightDir+ver, 'weights') )
if not os.path.exists( os.path.join(configDir, scoreDir+ver) ):
  os.makedirs( os.path.join(configDir, scoreDir+ver) )
for item in os.listdir( os.path.join(configDir, weightDir+ver, 'weights') ) or os.listdir( os.path.join(configDir, scoreDir+ver) ):
  if item.endswith(".C") or item.endswith(".root") or item.endswith("log"):
    #os.remove(os.path.join(os.path.join(configDir, weightDir+ver), item))
    print("Remove previous files or move on to next version!")
    #sys.exit()
if not os.path.exists( os.path.join(configDir, weightDir+ver, 'training_bdt.py') ):
  shutil.copy2('training_bdt.py', os.path.join(configDir, weightDir+ver, 'training_bdt.py'))

#Options for data preparation
sig_files, bkg_files = train_files(ch)

#int_vars = []
input_features = []
input_features.extend(input_variables_bdt(jetcat))
input_features.remove('STTT')
input_features.remove('channel')

fout = TFile("output_"+ch+"_"+jetcat+".root","recreate")
factory = TMVA.Factory("TMVAClassification", fout, "!V:!Silent:Color:DrawProgressBar:AnalysisType=Classification" )

loader = TMVA.DataLoader((weightDir+ver).split("/")[-1])
#for var in int_vars:
#    loader.AddVariable(var, "I")
for var in input_features:
    loader.AddVariable(var, "F")

loader.SetWeightExpression("EventWeight")
loader.AddSpectator("EventCategory")
loader.AddSpectator("njets")
loader.AddSpectator("nbjets_m")

trees = []
for fName in sig_files:
    fileWeight = 1
    if "TTTH" in fName: fileWeight = 1
    elif "STTH" in fName and "Hct" in fName: fileWeight = 0.04
    elif "STTH" in fName and "Hut" in fName: fileWeight = 0.3
    f = TFile(rootDir+fName.replace("h5","root"))
    t = f.Get("tree")
    loader.AddSignalTree(t, fileWeight)
    trees.append([f, t])
for fName in bkg_files:
    fileWeight = 1
    f = TFile(rootDir+fName.replace("h5","root"))
    t = f.Get("tree")
    loader.AddBackgroundTree(t, fileWeight)
    trees.append([f, t])

loader.PrepareTrainingAndTestTree(sigCut, bkgCut, options)

factory.BookMethod(loader, TMVA.Types.kBDT, "BDT", "!H:!V:NTrees=" + ntree[idx[jetcat]] + ":MinNodeSize=5%:MaxDepth=5:BoostType=Grad:Shrinkage=0.5:SeparationType=GiniIndex:nCuts=" + ncut[idx[jetcat]])

factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()
fout.Close()

if not os.path.exists( os.path.join(configDir, weightDir+ver, "output_"+ch+"_"+jetcat+".root") ):
  shutil.move("output_"+ch+"_"+jetcat+".root", os.path.join(configDir, weightDir+ver))
