#!/usr/bin/env python
import sys, os
import google.protobuf

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from ROOT import *
import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Activation, Dropout, add, concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD, Adam, Adadelta
from keras.utils import plot_model

TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()

fout = TFile("output_keras_Hct.root","recreate")

factory = TMVA.Factory("TMVAClassification", fout,
                       "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G;D:AnalysisType=Classification" )

loader = TMVA.DataLoader("keras6_Hct3")
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
"""
loader.AddVariable("jet1pt",'F')
loader.AddVariable("jet1eta",'F')
#loader.AddVariable("jet1phi",'F')
loader.AddVariable("jet1m",'F')
loader.AddVariable("jet1csv",'F')
loader.AddVariable("jet1cvsl",'F')
loader.AddVariable("jet1cvsb",'F')
loader.AddVariable("jet2pt",'F')
loader.AddVariable("jet2eta",'F')
#loader.AddVariable("jet2phi",'F')
loader.AddVariable("jet2m",'F')
loader.AddVariable("jet2csv",'F')
loader.AddVariable("jet2cvsl",'F')
loader.AddVariable("jet2cvsb",'F')
loader.AddVariable("jet3pt",'F')
loader.AddVariable("jet3eta",'F')
#loader.AddVariable("jet3phi",'F')
loader.AddVariable("jet3m",'F')
loader.AddVariable("jet3csv",'F')
loader.AddVariable("jet3cvsl",'F')
loader.AddVariable("jet3cvsb",'F')
loader.AddVariable("jet4pt",'F')
loader.AddVariable("jet4eta",'F')
#loader.AddVariable("jet4phi",'F')
loader.AddVariable("jet4m",'F')
loader.AddVariable("jet4csv",'F')
loader.AddVariable("jet4cvsl",'F')
loader.AddVariable("jet4cvsb",'F')
"""
loader.AddVariable("DRlepWpt",'F')
loader.AddVariable("DRlepWeta",'F')
loader.AddVariable("DRlepWdeta",'F')
#loader.AddVariable("DRlepWphi",'F')
loader.AddVariable("DRlepWdphi",'F')
loader.AddVariable("DRlepWm",'F')
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
loader.AddVariable("DRjet12deta",'F')
#loader.AddVariable("DRjet12phi",'F')
loader.AddVariable("DRjet12dphi",'F')
loader.AddVariable("DRjet12m",'F')
loader.AddVariable("DRjet12DR",'F')
loader.AddVariable("DRjet23pt",'F')
loader.AddVariable("DRjet23eta",'F')
loader.AddVariable("DRjet23deta",'F')
#loader.AddVariable("DRjet23phi",'F')
loader.AddVariable("DRjet23dphi",'F')
loader.AddVariable("DRjet23m",'F')
loader.AddVariable("DRjet31pt",'F')
loader.AddVariable("DRjet31eta",'F')
loader.AddVariable("DRjet31deta",'F')
#loader.AddVariable("DRjet31phi",'F')
loader.AddVariable("DRjet31dphi",'F')
loader.AddVariable("DRjet31m",'F')
loader.AddVariable("DRlepTpt",'F')
loader.AddVariable("DRlepTeta",'F')
loader.AddVariable("DRlepTdeta",'F')
#loader.AddVariable("DRlepTphi",'F')
loader.AddVariable("DRlepTdphi",'F')
loader.AddVariable("DRlepTm",'F')
loader.AddVariable("DRhadTpt",'F')
loader.AddVariable("DRhadTeta",'F')
loader.AddVariable("DRhadTHbdeta",'F')
loader.AddVariable("DRhadTWbdeta",'F')
#loader.AddVariable("DRhadTphi",'F')
loader.AddVariable("DRhadTHbdphi",'F')
loader.AddVariable("DRhadTWbdphi",'F')
loader.AddVariable("DRhadTm",'F')


## Load input files
#signalA = TFile("input/tmva_AntiTop_Hut.root")
signalB = TFile("input/tmva_AntiTop_Hct.root")
#signalC = TFile("input/tmva_Top_Hut.root")
signalD = TFile("input/tmva_Top_Hct.root")
#sigTreeA = signalA.Get("tmva_tree")
sigTreeB = signalB.Get("tmva_tree")
#sigTreeC = signalC.Get("tmva_tree")
sigTreeD = signalD.Get("tmva_tree")
#loader.AddSignalTree(sigTreeA,0.1)#0.156137331574/2=0.0780687
loader.AddSignalTree(sigTreeB,0.1)#0.113541253338/2=0.0567706
#loader.AddSignalTree(sigTreeC,0.1)
loader.AddSignalTree(sigTreeD,0.1)

background1 = TFile("input/tmva_tchannel.root")
background2 = TFile("input/tmva_tbarchannel.root")
background3 = TFile("input/tmva_tWchannel.root")
background4 = TFile("input/tmva_tbarWchannel.root")
backgroundTree1 = background1.Get("tmva_tree")
backgroundTree2 = background2.Get("tmva_tree")
backgroundTree3 = background3.Get("tmva_tree")
backgroundTree4 = background4.Get("tmva_tree")
loader.AddBackgroundTree(backgroundTree1,0.08782)
loader.AddBackgroundTree(backgroundTree2,0.07553)
loader.AddBackgroundTree(backgroundTree3,0.19063)
loader.AddBackgroundTree(backgroundTree4,0.19332)

background5 = TFile("input/tmva_ttbb.root")
background6 = TFile("input/tmva_ttbj.root")
background7 = TFile("input/tmva_ttcc.root")
background8 = TFile("input/tmva_ttLF.root")
background9 = TFile("input/tmva_ttother.root")
backgroundTree5 = background5.Get("tmva_tree")
backgroundTree6 = background6.Get("tmva_tree")
backgroundTree7 = background7.Get("tmva_tree")
backgroundTree8 = background8.Get("tmva_tree")
backgroundTree9 = background9.Get("tmva_tree")
loader.AddBackgroundTree(backgroundTree5,0.11396)
loader.AddBackgroundTree(backgroundTree6,0.09117)
loader.AddBackgroundTree(backgroundTree7,0.09117)
loader.AddBackgroundTree(backgroundTree8,0.09117)
loader.AddBackgroundTree(backgroundTree9,0.09117)

#background10 = TFile("mkNtuple_v5/tmva_wjets.root")
#background11 = TFile("mkNtuple_v5/tmva_ww.root")
#background12 = TFile("mkNtuple_v5/tmva_wz.root")
#background13 = TFile("mkNtuple_v5/tmva_zz.root")
#background14 = TFile("mkNtuple_v5/tmva_zjets10to50.root")
#background15 = TFile("mkNtuple_v5/tmva_zjets.root")
#backgroundTree10 = background10.->Get("tmva_tree")
#backgroundTree11 = background11.Get("tmva_tree")
#backgroundTree12 = background12.Get("tmva_tree")
#backgroundTree13 = background13.Get("tmva_tree")
#backgroundTree14 = background14.Get("tmva_tree")
#backgroundTree15 = background15.Get("tmva_tree")
#loader.AddBackgroundTree(backgroundTree10,134.276449237)
#loader.AddBackgroundTree(backgroundTree11,4.70407414855)
#loader.AddBackgroundTree(backgroundTree12,1.69041171)
#loader.AddBackgroundTree(backgroundTree13,0.598577911125)
#loader.AddBackgroundTree(backgroundTree14,22.7143748772)
#loader.AddBackgroundTree(backgroundTree15,2.65549294802)

loader.SetWeightExpression("EventWeight")
loader.AddSpectator("GoodPV")
loader.AddSpectator("EventCategory")

sigCut = TCut("nevt/totnevt < 0.8")

bkgCut = TCut("nevt/totnevt < 0.8")

loader.PrepareTrainingAndTestTree(
    sigCut, bkgCut,
    "nTrain_Signal=33000:nTrain_Background=370000:SplitMode=Random:NormMode=NumEvents:!V"
)

factory.BookMethod(loader, TMVA.Types.kBDT, "BDT", "!H:!V:NTrees=100:MinNodeSize=2.5%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:SeparationType=GiniIndex:nCuts=20")

#Keras
a = 1000
b = 0.7
init = 'glorot_uniform'

inputs = Input(shape=(59,))
#inputs = Input(shape=(85,))
x = Dense(a, kernel_regularizer=l2(1E-2))(inputs)
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
x = BatchNormalization()(x)#
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
x = Dense(a, activation='relu', kernel_initializer=init, bias_initializer='zeros')(x)
x = Dropout(b)(x)

predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1E-3), metrics=['binary_accuracy'])
model.save('model_Hct.h5')
model.summary()
#plot_model(model, to_file='model.png')

factory.BookMethod(loader, TMVA.Types.kPyKeras, 'Keras_TF',"H:!V:VarTransform=G,D,P:FilenameModel=model_Hct.h5:NumEpochs=30:BatchSize=1000")

factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()
fout.Close()

