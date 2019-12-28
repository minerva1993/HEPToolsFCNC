#!/usr/bin/env python
import sys, os, shutil
from variables import input_variables_bdt, input_selected_bdt, train_files, evalScale
from ROOT import *

TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()

#Version of classifier
if len(sys.argv) < 5:
  print("Not enough arguements: Ch, JetCat, Ver, Era")
  sys.exit()
ch = sys.argv[1]
jetcat = sys.argv[2]
ver = sys.argv[3]
era = sys.argv[4]

all_features = True
feature_sel = False
if len(sys.argv) > 5: #flag for input feature selection
  if sys.argv[5] == 'True':
    feature_sel = True
    ver = '99'

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

if era == "2017":
  nsig_Hct = ['51698', '20553', '54520', '33482', '2930'] #bit too large. (#of file per TT ntuple issue?)
  nsig_Hut = ['51854', '17410', '55746', '30680', '1462'] #bit too large. (#of file per ST ntuple issue?)
  nbkg = ['408366', '16034', '617468', '56877', '2894']
  ntree = ['400', '100', '400', '200', '20']
  ncut = ['20', '20', '20', '20', '10']
elif era == "2018":
  nsig_Hct = ['47200', '19700', '47600', '31200', '2900']
  nsig_Hut = ['40600', '13200', '46000', '26000', '1300']
  nbkg = ['379200', '17400', '400000', '63100', '3200']
  ntree = ['400', '100', '400', '200', '20']
  ncut = ['20', '20', '20', '20', '10']

if ch == "Hct":
  options = "nTrain_Signal=" + nsig_Hct[idx[jetcat]] + ":nTrain_Background=" + nbkg[idx[jetcat]] + ":nTest_Signal=0:nTest_Background=0:SplitMode=Random:NormMode=NumEvents:!V"
elif ch == "Hut":
  options = "nTrain_Signal=" + nsig_Hut[idx[jetcat]] + ":nTrain_Background=" + nbkg[idx[jetcat]] + ":nTest_Signal=0:nTest_Background=0:SplitMode=Random:NormMode=NumEvents:!V"

#directory name
#rootDir = '../mkNtuple/' + era + '/root_/'
rootDir = '/data1/users/minerva1993/work/'
if   era == '2017': rootDir = rootDir + 'fcnc_RunII2017/finalMVA/current_ver/root_/'
elif era == '2018': rootDir = rootDir + 'fcnc_RunII2018/finalMVA/current_ver/root_/'
configDir = '../'
weightDir = 'training/' + era + '/final' + '_' + ch + '_' +jetcat + '_'
scoreDir = 'scores/' + era + '/' + ch + '_' +jetcat + '_'

#Check if the model and files already exist
if not os.path.exists( os.path.join(configDir, weightDir+ver, 'weights') ):
  os.makedirs( os.path.join(configDir, weightDir+ver, 'weights') )
if not os.path.exists( os.path.join(configDir, scoreDir+ver) ):
  os.makedirs( os.path.join(configDir, scoreDir+ver) )
for item in os.listdir( os.path.join(configDir, weightDir+ver, 'weights') ) or os.listdir( os.path.join(configDir, scoreDir+ver) ):
  if item.endswith(".C") or item.endswith(".root") or item.startswith("log"):
    #os.remove(os.path.join(os.path.join(configDir, weightDir+ver), item))
    print("Remove previous files or move on to next version!")
    sys.exit()
if not os.path.exists( os.path.join(configDir, weightDir+ver, 'training_bdt.py') ):
  shutil.copy2('training_bdt.py', os.path.join(configDir, weightDir+ver, 'training_bdt.py'))

#Options for data preparation
sig_files, bkg_files = train_files(ch, era)

#int_vars = []
input_features = []
if feature_sel: input_features.extend(input_variables_bdt(jetcat))
else:
  if all_features: input_features.extend(input_variables_bdt(jetcat))
  else:
    try: input_features.extend(input_selected_bdt(ch, jetcat, era))
    except: input_features.extend(input_variables_bdt(jetcat))
#input_features.remove('STTT')
#input_features.remove('channel')

fout = TFile(era + "/output_"+ch+"_"+jetcat+".root","recreate")
factory = TMVA.Factory("TMVAClassification", fout, "!V:!Silent:DrawProgressBar:AnalysisType=Classification" )

loader = TMVA.DataLoader((weightDir+ver).split("/")[-2] + '/' + (weightDir+ver).split("/")[-1])
#for var in int_vars:
#    loader.AddVariable(var, "I")
for var in input_features:
    loader.AddVariable(var, "F")

loader.SetWeightExpression("EventWeight")
loader.AddSpectator("EventCategory")
loader.AddSpectator("njets")
loader.AddSpectator("nbjets_m")

nst, ntt, nttlj, nttll, frac_sig, frac_bkg = evalScale(ch, era, sig_files, bkg_files)
print("Sum of weights of training samples - ST = " + str(nst) + ", TT = " + str(ntt) + ", TTLJ = " + str(nttlj) + ", TTLL = " + str(nttll)) 
print("Relative file weights - ST/TT = " + str(frac_sig) + ", LL/LJ = " + str(frac_bkg))
#if checkNevt: sys.exit()

trees = []
for fName in sig_files:
    fileWeight = 1
    if "TTTH" in fName: fileWeight = 1
    elif "STTH" in fName and "Hct" in fName: fileWeight = frac_sig#0.044
    elif "STTH" in fName and "Hut" in fName: fileWeight = frac_sig#0.441 #from note, ST/TT
    f = TFile(rootDir+fName.replace("h5","root"))
    t = f.Get("tree")
    loader.AddSignalTree(t, fileWeight)
    trees.append([f, t])
for fName in bkg_files:
    fileWeight = 1
    if "TTLL" in fName: fileWeight = frac_bkg
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

if not os.path.exists( os.path.join(configDir, weightDir+ver, "output_" + ch + "_" + jetcat + ".root") ):
  shutil.move(era + "/output_" + ch + "_" + jetcat + ".root", os.path.join(configDir, weightDir+ver))
