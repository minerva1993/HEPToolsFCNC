#!/usr/bin/env python
import sys, os
import google.protobuf
from ROOT import *
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.regularizers import l2
from keras import initializations
from keras.optimizers import SGD, Adam

TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()

fout = TFile("output_keras.root","recreate")

factory = TMVA.Factory("TMVAClassification", fout,
                       "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G;D:AnalysisType=Classification" )

loader = TMVA.DataLoader("keras4")
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
loader.AddVariable("jet1eta",'F')
loader.AddVariable("jet2eta",'F')
loader.AddVariable("jet3eta",'F')
loader.AddVariable("jet4eta",'F')
#loader.AddVariable("jet1phi",'F')
#loader.AddVariable("jet2phi",'F')
#loader.AddVariable("jet3phi",'F')
#loader.AddVariable("jet4phi",'F')
loader.AddVariable("jet1m",'F')
loader.AddVariable("jet2m",'F')
loader.AddVariable("jet3m",'F')
loader.AddVariable("jet4m",'F')
loader.AddVariable("jet1csv",'F')
loader.AddVariable("jet2csv",'F')
loader.AddVariable("jet3csv",'F')
loader.AddVariable("jet4csv",'F')
loader.AddVariable("jet1cvsl",'F')
loader.AddVariable("jet2cvsl",'F')
loader.AddVariable("jet3cvsl",'F')
loader.AddVariable("jet4cvsl",'F')
loader.AddVariable("jet1cvsb",'F')
loader.AddVariable("jet2cvsb",'F')
loader.AddVariable("jet3cvsb",'F')
loader.AddVariable("jet4cvsb",'F')
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

sigCut = TCut("njets > 0 && nbjets_m > 0 && missingET > 0 && bjetmDR < 10 && bjetmDEta < 10 && bjetmDPhi < 5 && dibjetsMass > 0 && bjetPt_dibjetsm > 0 && cjetPt > 0 && jet1pt > 0 && jet2pt > 0 && jet3pt > 0 && jet4pt > 0 && jet1eta < 10 && jet2eta < 10 && jet3eta < 10 && jet4eta < 10 && jet1m > 0 && jet2m > 0 && jet3m > 0 && jet4m > 0  && jet1csv < 5 &&  jet2csv < 5 &&  jet3csv < 5 && jet4csv < 5 && jet1csv > 0 &&  jet2csv > 0 &&  jet3csv > 0 && jet4csv > 0 && jet1cvsl < 5 && jet2cvsl < 5 && jet3cvsl < 5 && jet4cvsl < 5 && jet1cvsb < 5 && jet2cvsb < 5 && jet3cvsb < 5 && jet4cvsb < 5 && DRlepWpt > 0 &&DRlepWm > 0 && DRlepWeta < 10 && DRjet0eta < 10 && DRjet1eta < 10 && DRjet2eta < 10 && DRjet3eta < 10 && DRjet0pt > 0 && DRjet1pt > 0 && DRjet2pt > 0 && DRjet3pt > 0 && DRjet0m > 0 && DRjet1m > 0 && DRjet2m > 0 && DRjet3m > 0 && DRjet0csv > 0 && DRjet1csv > 0 && DRjet2csv > 0 && DRjet3csv > 0 && DRjet0csv < 5 && DRjet1csv < 5 && DRjet2csv < 5 && DRjet3csv < 5 && DRjet0cvsl < 5 && DRjet1cvsl < 5 && DRjet2cvsl < 5 && DRjet3cvsl < 5 && DRjet0cvsb < 5 && DRjet1cvsb < 5 && DRjet2cvsb < 5 && DRjet3cvsb < 5 && DRjet12DR > 0 && DRjet12pt > 0 && DRjet23pt > 0 && DRjet31pt > 0 && DRjet12eta < 10  && DRjet23eta < 10 && DRjet31eta < 10 && DRjet12m > 0  && DRjet23m > 0 && DRjet31m > 0 && DRlepTpt > 0 && DRlepTeta < 10 && DRlepTm > 0 && DRhadTpt > 0 && DRhadTeta < 10 && DRhadTm > 0")

bkgCut = TCut("njets > 0 && nbjets_m > 0 && missingET > 0 && bjetmDR < 10 && bjetmDEta < 10 && bjetmDPhi < 5 && dibjetsMass > 0 && bjetPt_dibjetsm > 0 && cjetPt > 0 && jet1pt > 0 && jet2pt > 0 && jet3pt > 0 && jet4pt > 0 && jet1eta < 10 && jet2eta < 10 && jet3eta < 10 && jet4eta < 10 && jet1m > 0 && jet2m > 0 && jet3m > 0 && jet4m > 0  && jet1csv < 5 &&  jet2csv < 5 &&  jet3csv < 5 && jet4csv < 5 && jet1csv > 0 &&  jet2csv > 0 &&  jet3csv > 0 && jet4csv > 0 && jet1cvsl < 5 && jet2cvsl < 5 && jet3cvsl < 5 && jet4cvsl < 5 && jet1cvsb < 5 && jet2cvsb < 5 && jet3cvsb < 5 && jet4cvsb < 5 && DRlepWpt > 0 &&DRlepWm > 0 && DRlepWeta < 10 && DRjet0eta < 10 && DRjet1eta < 10 && DRjet2eta < 10 && DRjet3eta < 10 && DRjet0pt > 0 && DRjet1pt > 0 && DRjet2pt > 0 && DRjet3pt > 0 && DRjet0m > 0 && DRjet1m > 0 && DRjet2m > 0 && DRjet3m > 0 && DRjet0csv > 0 && DRjet1csv > 0 && DRjet2csv > 0 && DRjet3csv > 0 && DRjet0csv < 5 && DRjet1csv < 5 && DRjet2csv < 5 && DRjet3csv < 5 && DRjet0cvsl < 5 && DRjet1cvsl < 5 && DRjet2cvsl < 5 && DRjet3cvsl < 5 && DRjet0cvsb < 5 && DRjet1cvsb < 5 && DRjet2cvsb < 5 && DRjet3cvsb < 5 && DRjet12DR > 0 && DRjet12pt > 0 && DRjet23pt > 0 && DRjet31pt > 0 && DRjet12eta < 10  && DRjet23eta < 10 && DRjet31eta < 10 && DRjet12m > 0  && DRjet23m > 0 && DRjet31m > 0 && DRlepTpt > 0 && DRlepTeta < 10 && DRlepTm > 0 && DRhadTpt > 0 && DRhadTeta < 10 && DRhadTm > 0")

loader.PrepareTrainingAndTestTree(
    sigCut, bkgCut,
    "nTrain_Signal=30000:nTrain_Background=30000:nTest_Signal=10000:nTest_Background=10000:SplitMode=Random:NormMode=NumEvents:!V"
)

#factory.BookMethod(loader, TMVA.Types.kBDT, "BDT", "!H:!V:NTrees=850:MinNodeSize=2.5%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20")

#factory.BookMethod(loader, TMVA.Types.kDNN, "DNN", '!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=N:WeightInitialization=XAVIERUNIFORM:Architecture=CPU:Layout=TANH|300,TANH|300,TANH|300,TANH|300,TANH|300,TANH|300,TANH|300,TANH|300,TANH|200,TANH|100,LINEAR:TrainingStrategy=LearningRate=1e-2,Repetitions=1,ConvergenceSteps=20,Multithreading=True,Regularization=L2,WeightDecay=1e-3,BatchSize=200,TestRepetitions=5,DropConfig=0.0+0.3+0.3+0.3+0.3+0.3+0.3+0.3+0.3+0.3+0.0,Momentum=0.6')

#Keras
#def normal(shape, name=None):
#  return initializations.normal(shape, scale=0.05, name=name)

model = Sequential()
model.add(Dense(300, init='glorot_uniform', activation='relu', W_regularizer=l2(1e-5), input_dim=75))
model.add(Dense(500, init='glorot_uniform', activation='relu'))
model.add(Dense(700, init='glorot_uniform', activation='relu'))
model.add(Dense(700, init='glorot_uniform', activation='relu'))
model.add(Dense(700, init='glorot_uniform', activation='relu'))
model.add(Dense(700, init='glorot_uniform', activation='relu'))
model.add(Dense(700, init='glorot_uniform', activation='relu'))
model.add(Dense(700, init='glorot_uniform', activation='relu'))
model.add(Dense(700, init='glorot_uniform', activation='relu'))
model.add(Dense(500, init='glorot_uniform', activation='relu'))
model.add(Dense(300, init='glorot_uniform', activation='relu'))
model.add(Dense(100, init='glorot_uniform', activation='relu'))
model.add(Dense(2, init='glorot_uniform', activation='softmax'))

#model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy',])
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1E-4), metrics=['accuracy'])
model.save('model.h5')
model.summary()

factory.BookMethod(loader, TMVA.Types.kPyKeras, 'PyKeras',"H:!V:VarTransform=D,G:FilenameModel=model.h5:NumEpochs=20:BatchSize=100")

factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()
fout.Close()

TMVA.TMVAGui("output_keras.root")

