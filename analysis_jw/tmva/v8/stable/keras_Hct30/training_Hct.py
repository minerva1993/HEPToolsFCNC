#!/usr/bin/env python
import sys, os
import google.protobuf

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from ROOT import *
import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Activation, Dropout, add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam

TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()

fout = TFile("output_keras_Hct.root","recreate")

factory = TMVA.Factory("TMVAClassification", fout, "!V:!Silent:Color:DrawProgressBar:AnalysisType=Classification" )

loader = TMVA.DataLoader("keras_Hct30")
loader.AddVariable("njets", "I")
loader.AddVariable("nbjets_m",'I')
#loader.AddVariable("ncjets_m",'I')
#loader.AddVariable("lepDPhi",'F')
#loader.AddVariable("transverseMass",'F')
#loader.AddVariable("missingET",'F')
##loader.AddVariable("bjetmDR",'F')
##loader.AddVariable("bjetmDEta",'F')
##loader.AddVariable("bjetmDPhi",'F')
##loader.AddVariable("dibjetsMass",'F')
##loader.AddVariable("bjetPt_dibjetsm",'F')
##loader.AddVariable("cjetPt",'F')
loader.AddVariable("DRlepWpt",'F')
loader.AddVariable("DRlepWeta",'F')
#loader.AddVariable("DRlepWdeta",'F')
loader.AddVariable("DRlepWdphi",'F')
loader.AddVariable("DRlepWm",'F')
loader.AddVariable("DRjet0pt",'F')
loader.AddVariable("DRjet0eta",'F')
loader.AddVariable("DRjet0m",'F')
loader.AddVariable("DRjet0csv",'F')
loader.AddVariable("DRjet0cvsl",'F')
loader.AddVariable("DRjet0cvsb",'F')
loader.AddVariable("DRjet1pt",'F')
loader.AddVariable("DRjet1eta",'F')
loader.AddVariable("DRjet1m",'F')
loader.AddVariable("DRjet1csv",'F')
loader.AddVariable("DRjet1cvsl",'F')
loader.AddVariable("DRjet1cvsb",'F')
loader.AddVariable("DRjet2pt",'F')
loader.AddVariable("DRjet2eta",'F')
loader.AddVariable("DRjet2m",'F')
loader.AddVariable("DRjet2csv",'F')
loader.AddVariable("DRjet2cvsl",'F')
loader.AddVariable("DRjet2cvsb",'F')
loader.AddVariable("DRjet3pt",'F')
loader.AddVariable("DRjet3eta",'F')
loader.AddVariable("DRjet3m",'F')
loader.AddVariable("DRjet3csv",'F')
loader.AddVariable("DRjet3cvsl",'F')
loader.AddVariable("DRjet3cvsb",'F')
loader.AddVariable("DRjet12pt",'F')
loader.AddVariable("DRjet12eta",'F')
loader.AddVariable("DRjet12deta",'F')
loader.AddVariable("DRjet12dphi",'F')
loader.AddVariable("DRjet12m",'F')
loader.AddVariable("DRjet12DR",'F')
loader.AddVariable("DRjet23pt",'F')
loader.AddVariable("DRjet23eta",'F')
loader.AddVariable("DRjet23deta",'F')
loader.AddVariable("DRjet23dphi",'F')
loader.AddVariable("DRjet23m",'F')
loader.AddVariable("DRjet31pt",'F')
loader.AddVariable("DRjet31eta",'F')
loader.AddVariable("DRjet31deta",'F')
loader.AddVariable("DRjet31dphi",'F')
loader.AddVariable("DRjet31m",'F')
loader.AddVariable("DRlepTpt",'F')
loader.AddVariable("DRlepTeta",'F')
loader.AddVariable("DRlepTdeta",'F')
loader.AddVariable("DRlepTdphi",'F')
loader.AddVariable("DRlepTm",'F')
loader.AddVariable("DRhadTpt",'F')
loader.AddVariable("DRhadTeta",'F')
loader.AddVariable("DRhadTHbdeta",'F')
loader.AddVariable("DRhadTWbdeta",'F')
loader.AddVariable("DRhadTHbdphi",'F')
loader.AddVariable("DRhadTWbdphi",'F')
loader.AddVariable("DRhadTm",'F')

## Load input files
signalA = TFile("input/tmva_Top_Hct.root")
signalB = TFile("input/tmva_AntiTop_Hct.root")
background1 = TFile("input/tmva_tchannel.root")
background2 = TFile("input/tmva_tbarchannel.root")
background3 = TFile("input/tmva_tWchannel.root")
background4 = TFile("input/tmva_tbarWchannel.root")
background5 = TFile("input/tmva_ttbb.root")
background6 = TFile("input/tmva_ttbj.root")
background7 = TFile("input/tmva_ttcc.root")
background8 = TFile("input/tmva_ttLF.root")
background9 = TFile("input/tmva_ttother.root")

sigTreeA = signalA.Get("tmva_tree")
sigTreeB = signalB.Get("tmva_tree")
backgroundTree1 = background1.Get("tmva_tree")
backgroundTree2 = background2.Get("tmva_tree")
backgroundTree3 = background3.Get("tmva_tree")
backgroundTree4 = background4.Get("tmva_tree")
backgroundTree5 = background5.Get("tmva_tree")
backgroundTree6 = background6.Get("tmva_tree")
backgroundTree7 = background7.Get("tmva_tree")
backgroundTree8 = background8.Get("tmva_tree")
backgroundTree9 = background9.Get("tmva_tree")

loader.AddSignalTree(sigTreeA,0.06316)
loader.AddSignalTree(sigTreeB,0.06317)
loader.AddBackgroundTree(backgroundTree1,0.08782)
loader.AddBackgroundTree(backgroundTree2,0.07553)
loader.AddBackgroundTree(backgroundTree3,0.19063)
loader.AddBackgroundTree(backgroundTree4,0.19332)
loader.AddBackgroundTree(backgroundTree5,0.11397)
loader.AddBackgroundTree(backgroundTree6,0.09117)
loader.AddBackgroundTree(backgroundTree7,0.09117)
loader.AddBackgroundTree(backgroundTree8,0.09117)
loader.AddBackgroundTree(backgroundTree9,0.09117)

#background10 = TFile("input/tmva_Top_Hct.root")
#background11 = TFile("input/tmva_AntiTop_Hct.root")
#backgroundTree10 = background10.Get("tmva_tree")
#backgroundTree11 = background11.Get("tmva_tree")
#loader.AddBackgroundTree(backgroundTree10,0.06316)
#loader.AddBackgroundTree(backgroundTree11,0.06317)

loader.SetWeightExpression("EventWeight")
loader.AddSpectator("GoodPV")
loader.AddSpectator("EventCategory")
loader.AddSpectator("GenMatch")

sigCut = TCut("nevt %5 != 0 && GenMatch == 2")

bkgCut = TCut("nevt %5 != 0")

loader.PrepareTrainingAndTestTree(
    sigCut, bkgCut,
    "nTrain_Signal=15000:nTrain_Background=370000:SplitMode=Random:NormMode=NumEvents:!V"
)

factory.BookMethod(loader, TMVA.Types.kBDT, "BDT", "!H:!V:NTrees=250:MinNodeSize=7.5%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:SeparationType=GiniIndex:nCuts=15")

#Keras
a = 1000
b = 0.65
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
model.summary()

factory.BookMethod(loader, TMVA.Types.kPyKeras, 'Keras_TF',"H:!V:VarTransform=G,D,P:FilenameModel=model_Hct.h5:NumEpochs=30:BatchSize=1000")

factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()
fout.Close()

