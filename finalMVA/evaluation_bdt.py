#from __future__ import print_function
#!/usr/bin/env python
import os, sys
from ROOT import *
from array import array
import numpy as np
from training.variables import input_variables_bdt, train_files, evalScale, evalFrac

TMVA.Tools.Instance()

#Channel and version
if len(sys.argv) < 7:
  print("Not enough arguements: Ch, JetCat, Ver, Era, Syst. var, Model")
  sys.exit()
ch = sys.argv[1]
jetcat = sys.argv[2]
ver = sys.argv[3]
era = sys.argv[4]
file_path = sys.argv[5]
name = sys.argv[6]

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

#directory name
rootDir = 'mkNtuple/' + era + '/root_'
configDir = './'
weightDir = 'training/' + era + '/final' + '_' + ch + '_' +jetcat + '_'
scoreDir = 'scores/' + era + '/' + ch + '_' +jetcat + '_'

# Load data
syst = ["","jecup","jecdown","jerup","jerdown",]
syst2 = ["TuneCP5up","TuneCP5down","hdampup","hdampdown"] #dedecative samples exist

input_features = []
input_features.extend(input_variables_bdt(jetcat))
input_features.remove('STTT')
input_features.remove('channel')

for syst_ext in syst + syst2:
  if syst_ext != "" and not os.path.exists(os.path.join(configDir, scoreDir + ver + "-" + syst_ext)):
    try: os.makedirs(os.path.join(configDir, scoreDir + ver + "-" + syst_ext))
    except: pass
  elif syst_ext == "" and not os.path.exists(os.path.join(configDir, scoreDir + ver) ):
    try: os.makedirs(os.path.join(configDir, scoreDir + ver))
    except: pass

  if ("Run2017" in name) and syst_ext != "": continue
  elif (syst_ext in syst2) and not (syst_ext in name): continue
  elif (syst_ext in syst) and any(tmp in name for tmp in syst2): continue
  else:
    if (syst_ext in syst2): name = name.replace(syst_ext,"")

  if os.path.exists(os.path.join(configDir, scoreDir + ver + syst_ext, 'score_finalMVA_' + name)):
    print(scoreDir + ver + "/score_finalMVA_"  + name + (' is already exist!').rjust(50-len(name)))
    continue

  reader = TMVA.Reader("Color:!Silent")
  if not os.path.exists(os.path.join(rootDir + syst_ext, 'finalMVA_'+name+'.root')): continue
  data = TFile(os.path.join(rootDir + syst_ext, 'finalMVA_'+name+'.root'))
  data_tree = data.Get('tree')

  if syst_ext != "": syst_ext2 = "-"+syst_ext
  else: syst_ext2 = syst_ext
  outfile = TFile.Open(os.path.join(configDir, scoreDir + ver + syst_ext2, 'score_finalMVA_' + name + '.root'),'RECREATE')
  outtree = TTree("tree","tree")

  branches = {}
  for branch in data_tree.GetListOfBranches():
    branchName = branch.GetName()
    if branchName in input_features:
        branches[branchName] = array('f', [-999])
        reader.AddVariable(branchName, branches[branchName])
        data_tree.SetBranchAddress(branchName, branches[branchName])

    elif branchName in ["EventCategory", "njets", "nbjets_m"]:
      branches[branchName] = array('f', [-999])
      reader.AddSpectator(branchName, branches[branchName])

  reader.BookMVA('BDT', TString(os.path.join(configDir, weightDir+ver, 'weights/TMVAClassification_BDT.weights.xml')))

  totalevt = data_tree.GetEntries()
  #print("this sample contains "+str(totalevt)+" combinations")

  score         = np.zeros(1, dtype=np.float32)
  nevt          = np.zeros(1, dtype=int)
  njets         = np.zeros(1, dtype=int)
  nbjets_m      = np.zeros(1, dtype=int)
  EventCategory = np.zeros(1, dtype=int)
  lepPt         = np.zeros(1, dtype=np.float32)
  missinget     = np.zeros(1, dtype=np.float32)

  outtree.Branch('MLScore'      , score        , 'MLScore/F')
  outtree.Branch('nevt'         , nevt         , 'nevt/I')
  outtree.Branch('njets'        , njets        , 'njets/I')
  outtree.Branch('nbjets_m'     , nbjets_m     , 'nbjets_m/I')
  outtree.Branch('EventCategory', EventCategory, 'EventCategory/I')
  outtree.Branch('lepPt'        , lepPt        , 'lepPt/F')
  outtree.Branch('missinget'    , missinget    , 'missinget/F')

  for i in xrange(totalevt):
    data_tree.GetEntry(i)
    Nevt = data_tree.nevt

    if njets_cut == 3:
      if data_tree.njets != njets_cut: continue
    elif njets_cut == 4:
      if data_tree.njets < njets_cut: continue
    if nbjets_cut != 0:
      if data_tree.nbjets_m != nbjets_cut: continue

    score[0]         = reader.EvaluateMVA('BDT')
    nevt[0]          = data_tree.nevt
    njets[0]         = data_tree.njets
    nbjets_m[0]      = data_tree.nbjets_m
    EventCategory[0] = data_tree.EventCategory
    lepPt[0]         = data_tree.lepton_pt
    missinget[0]     = data_tree.MET
    outtree.Fill()
    #print('processing '+str(Nevt)+'th event', end='\r')

  score[0]         = -1
  nevt[0]          = 0
  njets[0]         = 0
  nbjets_m[0]      = 0
  EventCategory[0] = 0
  lepPt[0]         = 0
  missinget[0]     = 0
  outtree.Fill()

  outfile.Write()
  outfile.Close()
