from __future__ import print_function
import sys, os
import google.protobuf

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.utils import shuffle, class_weight
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
from root_numpy import array2tree, tree2array
from ROOT import TFile, TTree

import tensorflow as tf
import keras
from keras.utils import np_utils, multi_gpu_model
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Activation, Dropout, add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam, SGD
from keras.callbacks import Callback

bestModel = sys.argv[1]
#ch = "STFCNC"
#ch = "TTFCNC"
ch = "TTBKG"
ver = '01'
configDir = '/home/minerva1993/HEPToolsFCNC/analysis_2017/reco/'
weightDir = 'training/reco'+ch
scoreDir = 'score'+ch
signal_only = False

input_files = []
input_features = []
if ch == "STFCNC":
  input_features.extend(['jet0pt', 'jet0eta', 'jet0m', 'jet1pt', 'jet1eta', 'jet1m', 'jet2pt', 'jet2eta', 'jet2m',
                         'jet12pt', 'jet12eta', 'jet12deta', 'jet12dphi', 'jet12dR', 'jet12m',
                         'lepWpt', 'lepWdphi', 'lepWm', 'lepTdphi', 'lepTm',])

elif ch == "TTFCNC":
  input_features.extend([])

elif ch == "TTBKG":
  input_features.extend(['jet0pt', 'jet0eta', 'jet0m', 'jet1pt', 'jet1eta', 'jet1m', 'jet2pt', 'jet2eta', 'jet2m', 'jet3pt', 'jet3eta', 'jet3m',
                         'jet12pt', 'jet12eta', 'jet12deta', 'jet12dphi', 'jet12dR', 'jet12m',
                         'jet23pt', 'jet23eta', 'jet23deta', 'jet23dphi', 'jet23dR', 'jet23m',
                         'jet31pt', 'jet31eta', 'jet31deta', 'jet31dphi', 'jet31dR', 'jet31m',
                         'lepWpt', 'lepWdphi', 'lepTdphi', 'lepTm',
                         'hadTpt', 'hadTeta', 'hadT12_3deta', 'hadT23_1deta', 'hadT31_2deta',
                         'hadTphi', 'hadT12_3dphi', 'hadT23_1dphi', 'hadT31_2dphi',
                         'hadT12_3dR', 'hadT23_1dR', 'hadT31_2dR', 'hadTm',])

else: print("Check reco scenario!")

if not os.path.exists(os.path.join(configDir, scoreDir+ver)):
  os.makedirs(os.path.join(configDir, scoreDir+ver))

model_best = load_model(os.path.join(configDir, weightDir+ver, bestModel))
print('Start evaluation on version '+ver+' classifier with the model '+ bestModel)

for filename in os.listdir(os.path.join(configDir, 'mkNtuple', ch + '_hdf')):
  if filename == '.gitkeep': continue
  if os.path.exists(os.path.join(configDir, scoreDir+ver, 'score_' + filename.replace('h5','root'))):
    print('score_' + filename.replace('h5','root') + (' is already exist!').rjust(50-len(filename)))
    continue
  if signal_only:
    if   ch == "STFCNC":
      if "STTH1L3BH" not in filename: continue
    elif ch == "TTFCNC":
      if "TTTH1L3BH" not in filename: continue
    elif ch == "TTBKG":
      if "TTpowheg" not in filename: continue
      if not filename.endswith(('010.h5','011.h5','012.h5','013.h5','014.h5')): continue

  eval_df = pd.read_hdf(os.path.join(configDir, 'mkNtuple', ch + '_hdf' , filename))
  print(filename + ": " + str(eval_df.shape[0]).rjust(60-len(filename)))

  outfile = TFile.Open(os.path.join(scoreDir+ver, 'score_' + filename.replace('h5','root')),'RECREATE')
  outtree = TTree("tree","tree")

  spectator = eval_df.filter(['nevt', 'njets', 'nbjets_m', 'file', 'EventCategory', 'genMatch', 'jet0Idx', 'jet1Idx', 'jet2Idx', 'jet3Idx', 'lepton_pt', 'MET', 'jet12m', 'lepTm', 'hadTm'], axis=1)
  eval_df = eval_df.filter(input_features, axis=1)
  eval_df.astype('float32')

  eval_scaler = StandardScaler()
  eval_scaler.fit(eval_df)
  eval_df_sc = eval_scaler.transform(eval_df)
  X = eval_df_sc
  y = model_best.predict(X, batch_size=4000)
  y.dtype = [('MLScore', np.float32)]
  y = y[:,1]
  array2tree(y, name='tree', tree=outtree)

  for colname, value in spectator.iteritems():
    spect = spectator[colname].values
    if colname == 'lepton_pt': branchname = 'lepPt'
    elif colname == 'MET'    : branchname = 'missinget'
    elif colname == 'jet12m' : branchname = 'whMass'
    elif colname == 'lepTm'  : branchname = 'leptMass'
    elif colname == 'hadTm'  : branchname = 'hadtMass'
    else: branchname = colname

    if branchname in ['nevt', 'njets', 'nbjets_m', 'EventCategory', 'genMatch', 'jet0Idx', 'jet1Idx', 'jet2Idx', 'jet3Idx' ]: spect.dtype = [(branchname, np.int32)]
    else:
      spect.dtype = [(branchname, np.float32)]
    #print(branchname)
    #print(spect.shape)
    array2tree(spect, name='tree', tree=outtree)

  outtree.Fill()
  outfile.Write()
  outfile.Close()
