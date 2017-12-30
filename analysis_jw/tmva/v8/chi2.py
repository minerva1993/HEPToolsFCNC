import os, sys
from ROOT import *

ch = sys.argv[1]

if 'Hct' in ch:
  out = TFile.Open('output_keras_Hct.root')
else: out = TFile.Open('output_keras_Hut.root')

bdt = out.GetDirectory('keras_'+ch+'/Method_BDT/BDT')
MVA_BDT_S = bdt.Get('MVA_BDT_S')
MVA_BDT_Train_S = bdt.Get('MVA_BDT_Train_S')
MVA_BDT_B = bdt.Get('MVA_BDT_B')
MVA_BDT_Train_B = bdt.Get('MVA_BDT_Train_B')

bdt_s = MVA_BDT_S.Chi2Test(MVA_BDT_Train_S, "WW CHI2/NDF")
bdt_b = MVA_BDT_B.Chi2Test(MVA_BDT_Train_B, "WW CHI2/NDF")

keras = out.GetDirectory('keras_'+ch+'/Method_Keras_TF/Keras_TF')
MVA_Keras_TF_S = keras.Get('MVA_Keras_TF_S')
MVA_Keras_TF_Train_S = keras.Get('MVA_Keras_TF_Train_S')
MVA_Keras_TF_B = keras.Get('MVA_Keras_TF_B')
MVA_Keras_TF_Train_B = keras.Get('MVA_Keras_TF_Train_B')

keras_s = MVA_Keras_TF_S.Chi2Test(MVA_Keras_TF_Train_S, "WW CHI2/NDF")
keras_b = MVA_Keras_TF_B.Chi2Test(MVA_Keras_TF_Train_B, "WW CHI2/NDF")

print 'BDT signal: ' + str(bdt_s)
print 'BDT background: ' + str(bdt_b)
print 'DL signal: ' + str(keras_s)
print 'DL background: ' + str(keras_b)





