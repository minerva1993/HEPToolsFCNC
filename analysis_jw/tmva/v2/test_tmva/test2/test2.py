#!/usr/bin/env python
import sys, os
from ROOT import *

TMVA.Tools.Instance()

fout = TFile("output.root","recreate")

factory = TMVA.Factory("TMVAClassification", fout,
                       "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G;D:AnalysisType=Classification" )

loader = TMVA.DataLoader("test2")
loader.AddVariable("njets", "I")
loader.AddVariable("nbjets_m",'I')
loader.AddVariable("ncjets_m",'I')
loader.AddVariable("lepDPhi",'F')
loader.AddVariable("missingET",'F')
loader.AddVariable("bjetmDR",'F')
loader.AddVariable("bjetmDEta",'F')
loader.AddVariable("bjetmDPhi",'F')
loader.AddVariable("dibjetsMass",'F')
loader.AddVariable("bjetPt_dibjetsm",'F')
loader.AddVariable("cjetPt",'F')
loader.AddVariable("jet1pt",'F')
loader.AddVariable("jet2pt",'F')
loader.AddVariable("jet3pt",'F')
loader.AddVariable("jet4pt",'F')
loader.AddVariable("jet1csv",'F')
loader.AddVariable("jet2csv",'F')
loader.AddVariable("jet3csv",'F')
loader.AddVariable("jet4csv",'F')
#loader.AddVariable("DRlepWeta",'F')
#loader.AddVariable("DRlepWphi",'F')
#loader.AddVariable("DRlepWm",'F')
loader.AddVariable("DRjet0pt",'F')
loader.AddVariable("DRjet0eta",'F')
#loader.AddVariable("DRjet0phi",'F')
loader.AddVariable("DRjet0m",'F')
loader.AddVariable("DRjet0csv",'F')
loader.AddVariable("DRjet0cvsl",'F')
loader.AddVariable("DRjet0cvsb",'F')
loader.AddVariable("DRjet1pt",'F')
loader.AddVariable("DRjet1eta",'F')
#loader.AddVariable("DRjet1phi",'F')
loader.AddVariable("DRjet1m",'F')
loader.AddVariable("DRjet1csv",'F')
loader.AddVariable("DRjet1cvsl",'F')
loader.AddVariable("DRjet1cvsb",'F')
loader.AddVariable("DRjet2pt",'F')
loader.AddVariable("DRjet2eta",'F')
#loader.AddVariable("DRjet2phi",'F')
loader.AddVariable("DRjet2m",'F')
loader.AddVariable("DRjet2csv",'F')
loader.AddVariable("DRjet2cvsl",'F')
loader.AddVariable("DRjet2cvsb",'F')
loader.AddVariable("DRjet3pt",'F')
loader.AddVariable("DRjet3eta",'F')
#loader.AddVariable("DRjet3phi",'F')
loader.AddVariable("DRjet3m",'F')
loader.AddVariable("DRjet3csv",'F')
loader.AddVariable("DRjet3cvsl",'F')
loader.AddVariable("DRjet3cvsb",'F')
loader.AddVariable("DRjet12pt",'F')
loader.AddVariable("DRjet12eta",'F')
#loader.AddVariable("DRjet12phi",'F')
loader.AddVariable("DRjet12m",'F')
loader.AddVariable("DRjet12DR",'F')
loader.AddVariable("DRjet23pt",'F')
loader.AddVariable("DRjet23eta",'F')
#loader.AddVariable("DRjet23phi",'F')
loader.AddVariable("DRjet23m",'F')
loader.AddVariable("DRjet31pt",'F')
loader.AddVariable("DRjet31eta",'F')
#loader.AddVariable("DRjet31phi",'F')
loader.AddVariable("DRjet31m",'F')
loader.AddVariable("DRlepTpt",'F')
loader.AddVariable("DRlepTeta",'F')
#loader.AddVariable("DRlepTphi",'F')
loader.AddVariable("DRlepTm",'F')
loader.AddVariable("DRhadTpt",'F')
loader.AddVariable("DRhadTeta",'F')
#loader.AddVariable("DRhadTphi",'F')
loader.AddVariable("DRhadTm",'F')


## Load input files
fsig = TFile("tmva_Top_Hct.root")
fbkg1 = TFile("tmva_ttLF.root")
fbkg2 = TFile("tmva_ttbb.root")

tsig = fsig.Get("tmva_tree")
tbkg1 = fbkg1.Get("tmva_tree")
tbkg2 = fbkg2.Get("tmva_tree")

loader.AddSignalTree(tsig, 0.156137331574)
loader.AddBackgroundTree(tbkg1, 0.0910581123792)
loader.AddBackgroundTree(tbkg2, 0.0910581123792)

sigCut = TCut("njets > 0 && nbjets_m > 0 && missingET > 0 && bjetmDR < 5 && bjetmDEta < 5 && bjetmDPhi < 5 && dibjetsMass < 9999 && bjetPt_dibjetsm < 9999 && cjetPt < 9999 && jet1pt < 9999 && jet2pt < 9999 && jet3pt < 9999 && jet4pt < 9999 && jet1csv > 0 && jet1csv < 1.01 && jet2csv > 0 && jet2csv < 1.01 && jet3csv > 0 && jet3csv < 1.01 && jet4csv > 0 && jet4csv < 1.01 && DRlepWpt > 0 &&DRlepWm > 0")
bkgCut = TCut("njets > 0 && nbjets_m > 0 && missingET > 0 && bjetmDR < 5 && bjetmDEta < 5 && bjetmDPhi < 5 && dibjetsMass < 9999 && bjetPt_dibjetsm < 9999 && cjetPt < 9999 && jet1pt < 9999 && jet2pt < 9999 && jet3pt < 9999 && jet4pt < 9999 && jet1csv > 0 && jet1csv < 1.01 && jet2csv > 0 && jet2csv < 1.01 && jet3csv > 0 && jet3csv < 1.01 && jet4csv > 0 && jet4csv < 1.01 && DRlepWpt > 0 &&DRlepWm > 0")
loader.PrepareTrainingAndTestTree(
    sigCut, bkgCut,
    "nTrain_Signal=30000:nTrain_Background=30000:nTest_Signal=10000:nTest_Background=10000:SplitMode=Random:NormMode=NumEvents:!V"
)

factory.BookMethod(loader, TMVA.Types.kBDT, "BDT", "!H:!V:NTrees=850:MinNodeSize=2.5%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20")

"""
# For the DNN
dnnCommonOpt = "!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=N:WeightInitialization=XAVIERUNIFORM"
trainingCommonOpt = ["Repetitions=1", "ConvergenceSteps=20", "Multithreading=True", "Regularization=L2",
                     "WeightDecay=1e-3", "BatchSize=200", "TestRepetitions=5",]
dnnLayouts = [
    ["DNN", [
        ["TANH|128", trainingCommonOpt+["LearningRate=1e-2","Momentum=0.6","DropConfig=0.0+0.3+0.3+0.5"]]],
    ]
]

for name, dnnLayout in dnnLayouts:
  dnnOpts = [dnnCommonOpt,
      ("Layout="+("|".join([x[0] for x in dnnLayout]))+"|64,LINEAR"),
      ("TrainingStrategy="+("|".join([",".join(x[1]) for x in dnnLayout]))),
  ]
  factory.BookMethod(loader, TMVA.Types.kDNN, name, ":".join(dnnOpts))
"""

factory.BookMethod(loader, TMVA.Types.kDNN, "DNN", '!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=N:WeightInitialization=XAVIERUNIFORM:Layout=ReLU|300,ReLU|300,ReLU|300,ReLU|300,ReLU|300,ReLU|300,ReLU|300,ReLU|300,ReLU|200,ReLU|100,LINEAR:TrainingStrategy=LearningRate=1e-2,Repetitions=1,ConvergenceSteps=20,Multithreading=True,Regularization=L2,WeightDecay=1e-3,BatchSize=200,TestRepetitions=5,DropConfig=0.5+0.5+0.5+0.5+0.5+0.5+0.5+0.5+0.5+0.5+0.0,Momentum=0.6')

factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()
fout.Close()

TMVA.TMVAGui("output.root")



