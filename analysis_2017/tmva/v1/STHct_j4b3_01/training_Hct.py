#!/usr/bin/env python
import sys, os
#import google.protobuf

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from ROOT import *
"""
import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Activation, Dropout, add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
"""
TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()

fout = TFile("output.root","recreate")

factory = TMVA.Factory("TMVAClassification", fout, "!V:!Silent:Color:DrawProgressBar:AnalysisType=Classification" )

loader = TMVA.DataLoader("STHct_j4b3_01")
#loader = TMVA.DataLoader("STHct_j4b3_01")
#loader = TMVA.DataLoader("STHct_j4b4_01")

#loader.AddVariable("njets", "I")
#loader.AddVariable("nbjets_m",'I')
loader.AddVariable("lepWpt",'F')
#loader.AddVariable("lepWeta",'F')
loader.AddVariable("lepWdphi",'F')
loader.AddVariable("lepWm",'F')
loader.AddVariable("jet0pt",'F')
loader.AddVariable("jet0eta",'F')
loader.AddVariable("jet0m",'F')
#loader.AddVariable("jet0csv",'F')
#loader.AddVariable("jet0cvsl",'F')
#loader.AddVariable("jet0cvsb",'F')
loader.AddVariable("jet1pt",'F')
loader.AddVariable("jet1eta",'F')
loader.AddVariable("jet1m",'F')
#loader.AddVariable("jet1csv",'F')
#loader.AddVariable("jet1cvsl",'F')
#loader.AddVariable("jet1cvsb",'F')
loader.AddVariable("jet2pt",'F')
loader.AddVariable("jet2eta",'F')
loader.AddVariable("jet2m",'F')
#loader.AddVariable("jet2csv",'F')
#loader.AddVariable("jet2cvsl",'F')
#loader.AddVariable("jet2cvsb",'F')
#loader.AddVariable("jet3pt",'F')
#loader.AddVariable("jet3eta",'F')
#loader.AddVariable("jet3m",'F')
#loader.AddVariable("jet3csv",'F')
#loader.AddVariable("jet3cvsl",'F')
#loader.AddVariable("jet3cvsb",'F')
loader.AddVariable("jet12pt",'F')
loader.AddVariable("jet12eta",'F')
loader.AddVariable("jet12deta",'F')
loader.AddVariable("jet12dphi",'F')
loader.AddVariable("jet12m",'F')
loader.AddVariable("jet12DR",'F')
#loader.AddVariable("jet23pt",'F')
#loader.AddVariable("jet23eta",'F')
#loader.AddVariable("jet23deta",'F')
#loader.AddVariable("jet23dphi",'F')
#loader.AddVariable("jet23m",'F')
#loader.AddVariable("jet31pt",'F')
#loader.AddVariable("jet31eta",'F')
#loader.AddVariable("jet31deta",'F')
#loader.AddVariable("jet31dphi",'F')
#loader.AddVariable("jet31m",'F')
#loader.AddVariable("lepTpt",'F')
#loader.AddVariable("lepTeta",'F')
#loader.AddVariable("lepTdeta",'F')
loader.AddVariable("lepTdphi",'F')
loader.AddVariable("lepTm",'F')
#loader.AddVariable("hadTpt",'F')
#loader.AddVariable("hadTeta",'F')
#loader.AddVariable("hadTHbdeta",'F')
#loader.AddVariable("hadTWbdeta",'F')
#loader.AddVariable("hadTHbdphi",'F')
#loader.AddVariable("hadTWbdphi",'F')
#loader.AddVariable("hadTm",'F')

## Load input files
signalA = TFile("mkNtuple/tmva_singleTopHct.root")
#signalB = TFile("mkNtuple/tmva_singleTopHut.root")
background1 = TFile("mkNtuple/tmva_tchannel.root")
background2 = TFile("mkNtuple/tmva_tbarchannel.root")
background3 = TFile("mkNtuple/tmva_tWchannel.root")
background4 = TFile("mkNtuple/tmva_tbarWchannel.root")
background5 = TFile("mkNtuple/tmva_ttbb.root")
background6 = TFile("mkNtuple/tmva_ttbj.root")
background7 = TFile("mkNtuple/tmva_ttcc.root")
background8 = TFile("mkNtuple/tmva_ttLF.root")
background9 = TFile("mkNtuple/tmva_ttother.root")

sigTreeA = signalA.Get("tmva_tree")
#sigTreeB = signalB.Get("tmva_tree")
backgroundTree1 = background1.Get("tmva_tree")
backgroundTree2 = background2.Get("tmva_tree")
backgroundTree3 = background3.Get("tmva_tree")
backgroundTree4 = background4.Get("tmva_tree")
backgroundTree5 = background5.Get("tmva_tree")
backgroundTree6 = background6.Get("tmva_tree")
backgroundTree7 = background7.Get("tmva_tree")
backgroundTree8 = background8.Get("tmva_tree")
backgroundTree9 = background9.Get("tmva_tree")

loader.AddSignalTree(sigTreeA,0.006)#Hct
#loader.AddSignalTree(sigTreeB,0.008)#Hut
loader.AddBackgroundTree(backgroundTree1,0.962)
loader.AddBackgroundTree(backgroundTree2,0.859)
loader.AddBackgroundTree(backgroundTree3,0.197)
loader.AddBackgroundTree(backgroundTree4,0.192)
loader.AddBackgroundTree(backgroundTree5,0.171)
loader.AddBackgroundTree(backgroundTree6,0.137)
loader.AddBackgroundTree(backgroundTree7,0.137)
loader.AddBackgroundTree(backgroundTree8,0.137)
loader.AddBackgroundTree(backgroundTree9,0.137)

loader.SetWeightExpression("EventWeight")
loader.AddSpectator("GoodPV")
loader.AddSpectator("EventCategory")
loader.AddSpectator("GenMatch")

sigCut = TCut("nevt %3!=0 && njets==4 && nbjets_m==3")
bkgCut = TCut("nevt %3!=0 && njets==4 && nbjets_m==3")

loader.PrepareTrainingAndTestTree(
    sigCut, bkgCut,
    #"nTrain_Signal=45000:nTrain_Background=50000:SplitMode=Random:NormMode=NumEvents:!V" #j3b3
    "nTrain_Signal=26000:nTrain_Background=90000:SplitMode=Random:NormMode=NumEvents:!V"#j4b3
)

factory.BookMethod(loader, TMVA.Types.kBDT, "BDT", "!H:!V:NTrees=800:MinNodeSize=5%:MaxDepth=4:BoostType=AdaBoost:AdaBoostBeta=0.5:SeparationType=GiniIndex:nCuts=40")
"""
#Keras
a = 1000
b = 0.6
init = 'glorot_uniform'

inputs = Input(shape=(58,))
x = Dense(a, kernel_regularizer=l2(5E-3))(inputs)
x = BatchNormalization()(x)

branch_point1 = Dense(a, name='branch_point1')(x)

x = Dense(a, activation='relu', kernel_initializer=init, bias_initializer='zeros')(x)
x = Dropout(b)(x)
x = BatchNormalization()(x)
x = Dense(a, activation='relu', kernel_initializer=init, bias_initializer='zeros')(x)
x = Dropout(b)(x)

x = add([x, branch_point1])

x = BatchNormalization()(x)
branch_point2 = Dense(a, name='branch_point2')(x)

x = Dense(a, activation='relu', kernel_initializer=init, bias_initializer='zeros')(x)
x = Dropout(b)(x)
x = BatchNormalization()(x)
x = Dense(a, activation='relu', kernel_initializer=init, bias_initializer='zeros')(x)
x = Dropout(b)(x)

x = add([x, branch_point2])

x = BatchNormalization()(x)
branch_point3 = Dense(a, name='branch_point3')(x)

x = Dense(a, activation='relu', kernel_initializer=init, bias_initializer='zeros')(x)
x = Dropout(b)(x)
x = BatchNormalization()(x)
x = Dense(a, activation='relu', kernel_initializer=init, bias_initializer='zeros')(x)
x = Dropout(b)(x)

x = add([x, branch_point3])

x = BatchNormalization()(x)
branch_point4 = Dense(a, name='branch_point4')(x)

x = Dense(a, activation='relu', kernel_initializer=init, bias_initializer='zeros')(x)
x = Dropout(b)(x)
x = BatchNormalization()(x)
x = Dense(a, activation='relu', kernel_initializer=init, bias_initializer='zeros')(x)
x = Dropout(b)(x)

x = add([x, branch_point4])

x = BatchNormalization()(x)
branch_point5 = Dense(a, name='branch_point5')(x)

x = Dense(a, activation='relu', kernel_initializer=init, bias_initializer='zeros')(x)
x = Dropout(b)(x)
x = BatchNormalization()(x)
x = Dense(a, activation='relu', kernel_initializer=init, bias_initializer='zeros')(x)
x = Dropout(b)(x)

x = add([x, branch_point5])

x = BatchNormalization()(x)
branch_point6 = Dense(a, name='branch_point6')(x)

x = Dense(a, activation='relu', kernel_initializer=init, bias_initializer='zeros')(x)
x = Dropout(b)(x)
x = BatchNormalization()(x)
x = Dense(a, activation='relu', kernel_initializer=init, bias_initializer='zeros')(x)
x = Dropout(b)(x)

x = add([x, branch_point6])

x = BatchNormalization()(x)
x = Dense(a, activation='relu', kernel_initializer=init, bias_initializer='zeros')(x)
x = Dropout(b)(x)

predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1E-3), metrics=['binary_accuracy'])
model.save('model_Hct.h5')
#model.summary()

factory.BookMethod(loader, TMVA.Types.kPyKeras, 'Keras_TF',"H:!V:VarTransform=G,D,P:FilenameModel=model_Hct.h5:NumEpochs=40:BatchSize=1000")
"""
factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()
fout.Close()

