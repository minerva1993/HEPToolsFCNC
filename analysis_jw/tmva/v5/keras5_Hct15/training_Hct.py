#!/usr/bin/env python
import sys, os
import google.protobuf
from ROOT import *
import tensorflow as tf
import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Activation, Dropout, add, concatenate
from keras.layers.normalization import BatchNormalization
#from keras.layers.core import Dropout
from keras.regularizers import l2
from keras.optimizers import SGD, Adam, Adadelta
from keras.utils import plot_model

config = tf.ConfigProto()
config.gpu_options.visible_device_list = "2,3"
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()

fout = TFile("output_keras_Hct.root","recreate")

factory = TMVA.Factory("TMVAClassification", fout,
                       "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G;D:AnalysisType=Classification" )

loader = TMVA.DataLoader("keras5_Hct16")
loader.AddVariable("njets", "I")
loader.AddVariable("nbjets_m",'I')
loader.AddVariable("ncjets_m",'I')
#loader.AddVariable("lepDPhi",'F')
#loader.AddVariable("transverseMass",'F')
loader.AddVariable("missingET",'F')
##loader.AddVariable("bjetmDR",'F')
##loader.AddVariable("bjetmDEta",'F')
##loader.AddVariable("bjetmDPhi",'F')
##loader.AddVariable("dibjetsMass",'F')
##loader.AddVariable("bjetPt_dibjetsm",'F')
##loader.AddVariable("cjetPt",'F')
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

background1 = TFile("input/tmva_tbarchannel.root")
background2 = TFile("input/tmva_tbarWchannel.root")
background3 = TFile("input/tmva_tchannel.root")
background4 = TFile("input/tmva_tWchannel.root")
backgroundTree1 = background1.Get("tmva_tree")
backgroundTree2 = background2.Get("tmva_tree")
backgroundTree3 = background3.Get("tmva_tree")
backgroundTree4 = background4.Get("tmva_tree")

background5 = TFile("input/tmva_ttbb.root")
background6 = TFile("input/tmva_ttbj.root")
background7 = TFile("input/tmva_ttcc.root")
background8 = TFile("input/tmva_ttLF.root")
background9 = TFile("input/tmva_tt.root")
backgroundTree5 = background5.Get("tmva_tree")
backgroundTree6 = background6.Get("tmva_tree")
backgroundTree7 = background7.Get("tmva_tree")
backgroundTree8 = background8.Get("tmva_tree")
backgroundTree9 = background9.Get("tmva_tree")

#background10 = TFile("mkNtuple_v5/tmva_wjets.root")
##background11 = TFile("mkNtuple_v5/tmva_ww.root")
##background12 = TFile("mkNtuple_v5/tmva_wz.root")
##background13 = TFile("mkNtuple_v5/tmva_zz.root")
##background14 = TFile("mkNtuple_v5/tmva_zjets10to50.root")
##background15 = TFile("mkNtuple_v5/tmva_zjets.root")
#backgroundTree10 = background10->Get("tmva_tree")
##backgroundTree11 = background11.Get("tmva_tree")
##backgroundTree12 = background12.Get("tmva_tree")
##backgroundTree13 = background13.Get("tmva_tree")
##backgroundTree14 = background14.Get("tmva_tree")
##backgroundTree15 = background15.Get("tmva_tree")

#loader.AddSignalTree(sigTreeA,0.1)#0.156137331574/2=0.0780687
loader.AddSignalTree(sigTreeB,0.1)#0.113541253338/2=0.0567706
#loader.AddSignalTree(sigTreeC,0.1)
loader.AddSignalTree(sigTreeD,0.1)

loader.AddBackgroundTree(backgroundTree1,0.024575262909)
loader.AddBackgroundTree(backgroundTree2,0.193026936331)
loader.AddBackgroundTree(backgroundTree3,0.023844899675)
loader.AddBackgroundTree(backgroundTree4,0.190335714074)

loader.AddBackgroundTree(backgroundTree5,0.0910581123792)
loader.AddBackgroundTree(backgroundTree6,0.0910581123792)
loader.AddBackgroundTree(backgroundTree7,0.0910581123792)
loader.AddBackgroundTree(backgroundTree8,0.0910581123792)
loader.AddBackgroundTree(backgroundTree9,0.0910581123792)

#loader.AddBackgroundTree(backgroundTree10,134.276449237)
##loader.AddBackgroundTree(backgroundTree11,4.70407414855)
##loader.AddBackgroundTree(backgroundTree12,1.69041171)
##loader.AddBackgroundTree(backgroundTree13,0.598577911125)
##loader.AddBackgroundTree(backgroundTree14,22.7143748772)
##loader.AddBackgroundTree(backgroundTree15,2.65549294802)

loader.SetWeightExpression("EventWeight")

sigCut = TCut("missingET > 0 && cjetPt > 0 && jet1csv > 0 &&  jet2csv > 0 &&  jet3csv > 0 && jet4csv > 0 && DRlepWpt > 0 && DRjet0csv > 0 && DRjet1csv > 0 && DRjet2csv > 0 && DRjet3csv > 0")

bkgCut = TCut("missingET > 0 && cjetPt > 0 && jet1csv > 0 &&  jet2csv > 0 &&  jet3csv > 0 && jet4csv > 0 && DRlepWpt > 0 && DRjet0csv > 0 && DRjet1csv > 0 && DRjet2csv > 0 && DRjet3csv > 0")

loader.PrepareTrainingAndTestTree(
    sigCut, bkgCut,
    "nTrain_Signal=30000:nTrain_Background=440000:SplitMode=Random:NormMode=NumEvents:!V"
)

factory.BookMethod(loader, TMVA.Types.kBDT, "BDT_optimized", "!H:!V:NTrees=850:MinNodeSize=2.5%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:SeparationType=GiniIndex:nCuts=40")

#factory.BookMethod(loader, TMVA.Types.kDNN, "DNN", '!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=N:WeightInitialization=XAVIERUNIFORM:Architecture=CPU:Layout=ReLU|300,ReLU|500,ReLU|700,ReLU|700,ReLU|700,ReLU|700,ReLU|700,ReLU|700,ReLU|700,ReLU|500,ReLU|300,ReLU|100,LINEAR:TrainingStrategy=LearningRate=1e-2,Repetitions=1,ConvergenceSteps=20,Multithreading=True,Regularization=L2,WeightDecay=1e-4,BatchSize=100,TestRepetitions=5,DropConfig=0.2+0.0+0.0+0.0+0.2+0.0+0.0+0.0+0.2+0.0+0.0+0.0+0.0,Momentum=0.7')

#Keras
a = 1000
b = 0.5
init = 'glorot_uniform'

inputs = Input(shape=(85,))
x = Dense(a, kernel_regularizer=l2(1E-2))(inputs)#4E-6
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

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1E-3), metrics=['binary_accuracy'])
model.save('model_Hct.h5')
model.summary()
#plot_model(model, to_file='model.png')

factory.BookMethod(loader, TMVA.Types.kPyKeras, 'Keras+TF',"H:!V:VarTransform=D,G:FilenameModel=model_Hct.h5:NumEpochs=50:BatchSize=1000")#120 epochs

factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()
fout.Close()

#TMVA.TMVAGui("output_keras.root")
