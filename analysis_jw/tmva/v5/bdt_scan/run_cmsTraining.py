#!/usr/bin/env python
import sys, os
from collections import OrderedDict

if len(sys.argv) < 2:
    print sys.argv[0], "SAMPLETYPE SUFFIX"
    print '  mvaType    = ["BDT", "DNN_TANH", "DNN_ReLU", "Keras_TANH", "Keras_ReLU"]'
    sys.exit(1)

mvaType0 = sys.argv[1]
mvaAlgo = mvaType0.split('_',1)[0]
if mvaAlgo == 'BDT':
    bdtParIndex = mvaType0.split('_')[1]
elif mvaAlgo in ('DNN', 'Keras'):
    ftnName, nX, nY = mvaType0.split('_')[1:]
    nX, nY = int(nX), int(nY)

from ROOT import *
TMVA.Tools.Instance()

fout = TFile("mva_%s.root" % (mvaType0), "recreate")

factory = TMVA.Factory("TMVAClassification", fout,
                       "!V:!Silent:!Color:!DrawProgressBar:Transformations=I;D;P;G:AnalysisType=Classification" )

loader = TMVA.DataLoader("mva")
intVars = [
    "njets", "nbjets_m", "ncjets_m",
]
floatVars = [
    #"lepDPhi", "transverseMass",
    "missingET",
    ##"bjetmDR","bjetmDEta","bjetmDPhi","dibjetsMass","bjetPt_dibjetsm","cjetPt",
    "jet1pt", "jet1eta",#"jet1phi",
    "jet1m","jet1csv","jet1cvsl","jet1cvsb",
    "jet2pt","jet2eta", #"jet2phi",
    "jet2m","jet2csv","jet2cvsl","jet2cvsb",
    "jet3pt","jet3eta",#"jet3phi",
    "jet3m","jet3csv","jet3cvsl","jet3cvsb",
    "jet4pt","jet4eta",#"jet4phi",
    "jet4m","jet4csv","jet4cvsl","jet4cvsb",
    "DRlepWpt","DRlepWeta","DRlepWdeta",#"DRlepWphi",
    "DRlepWdphi","DRlepWm",
    "DRjet0pt","DRjet0eta",#"DRjet0phi",
    "DRjet0m","DRjet0csv","DRjet0cvsl","DRjet0cvsb",
    "DRjet1pt","DRjet1eta",#"DRjet1phi",
    "DRjet1m","DRjet1csv","DRjet1cvsl","DRjet1cvsb",
    "DRjet2pt","DRjet2eta",#"DRjet2phi",
    "DRjet2m","DRjet2csv","DRjet2cvsl","DRjet2cvsb",
    "DRjet3pt","DRjet3eta",#"DRjet3phi",
    "DRjet3m","DRjet3csv","DRjet3cvsl","DRjet3cvsb",
    "DRjet12pt","DRjet12eta","DRjet12deta",#"DRjet12phi",
    "DRjet12dphi","DRjet12m","DRjet12DR",
    "DRjet23pt","DRjet23eta","DRjet23deta",#"DRjet23phi",
    "DRjet23dphi","DRjet23m",
    "DRjet31pt","DRjet31eta","DRjet31deta",#"DRjet31phi",
    "DRjet31dphi","DRjet31m",
    "DRlepTpt","DRlepTeta","DRlepTdeta",#"DRlepTphi",
    "DRlepTdphi","DRlepTm",
    "DRhadTpt","DRhadTeta",#"DRhadTphi",
    "DRhadTm",
    "DRhadTHbdeta","DRhadTWbdeta",
    "DRhadTHbdphi","DRhadTWbdphi",
]
for var in intVars:
    loader.AddVariable(var, "I")
for var in floatVars:
    loader.AddVariable(var, "F")
nVars = len(intVars)+len(floatVars)

## Load input files
sigs = [
    ("../input/tmva_AntiTop_Hct.root", 0.1), 
    ("../input/tmva_Top_Hct.root", 0.1),
]
bkgs = [
    ("../input/tmva_tbarchannel.root", 0.024575262909),
    ("../input/tmva_tbarWchannel.root", 0.193026936331),
    ("../input/tmva_tchannel.root", 0.023844899675),
    ("../input/tmva_tWchannel.root", 0.190335714074),
    ("../input/tmva_ttbb.root", 0.0910581123792),
    ("../input/tmva_ttbj.root", 0.0910581123792),
    ("../input/tmva_ttcc.root", 0.0910581123792),
    ("../input/tmva_ttLF.root", 0.0910581123792),
    ("../input/tmva_tt.root", 0.0910581123792),
]
trees = []
for fName, scale in sigs:
    f = TFile(fName)
    t = f.Get("tmva_tree")
    loader.AddSignalTree(t, scale)
    trees.append([f, t])
for fName, scale in bkgs:
    f = TFile(fName)
    t = f.Get("tmva_tree")
    loader.AddBackgroundTree(t, scale)
    trees.append([f, t])
loader.SetWeightExpression( "EventWeight" );

sigCut = TCut("missingET > 0 && cjetPt > 0 && jet1csv > 0 &&  jet2csv > 0 &&  jet3csv > 0 && jet4csv > 0 && DRlepWpt > 0 && DRjet0csv > 0 && DRjet1csv > 0 && DRjet2csv > 0 && DRjet3csv > 0")
bkgCut = TCut("missingET > 0 && cjetPt > 0 && jet1csv > 0 &&  jet2csv > 0 &&  jet3csv > 0 && jet4csv > 0 && DRlepWpt > 0 && DRjet0csv > 0 && DRjet1csv > 0 && DRjet2csv > 0 && DRjet3csv > 0")

loader.PrepareTrainingAndTestTree(
    sigCut, bkgCut,
    "nTrain_Signal=30000:nTrain_Background=440000:SplitMode=Random:NormMode=NumEvents:!V"
)

if mvaAlgo == "BDT":
    methods = [
    ]
    bdtOption = "!H:!V:BoostType=AdaBoost:AdaBoostBeta=0.5:SeparationType=GiniIndex"
    for nCuts in reversed([20, 30, 40, 50, 60]):
        for maxDepth in [2,3,4]:
            for minNodeSize in [2,2.5,5]:
                for nTree in reversed([200, 400, 500, 600, 850, 1000, 1200]):
                    opt = [bdtOption, "NTrees=%d" % nTree,
                           "nCuts=%d" % nCuts, "MaxDepth=%d" % maxDepth, "MinNodeSize=%g%%" % minNodeSize]
                    suffix = "nCuts%d_maxDepth%d_minNode%g_nTree%d" % (nCuts, maxDepth, minNodeSize, nTree)
                    suffix = suffix.replace('.','p')
                    methods.append([TMVA.Types.kBDT, "BDT_%s" % suffix, ":".join(opt)])
    if bdtParIndex.isdigit():
        bdtIndex = int(bdtParIndex)
        factory.BookMethod(loader, *methods[bdtIndex])
    else:
        print "!!!Number of BDT methods to scan = ", len(methods)
        sys.exit()
    #except:
    #    for m in methods: factory.BookMethod(loader, *m)
    

factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()

fout.Close()


