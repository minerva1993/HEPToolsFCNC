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
from keras.callbacks import Callback, ModelCheckpoint

bestModel = sys.argv[1]
ch = 'Hct42'
configDir = '/home/minerva1993/tmva/test/'

if not os.path.exists(configDir+'score_mva/'+ch):
  os.makedirs(configDir+'score_mva/'+ch)

test = os.listdir(configDir+'score_mva/'+ch)
for item in test:
  if item.endswith(".root"):
    if 'corr' not in item:
      os.remove(os.path.join(configDir+'score_mva/'+ch, item))

print('Start evaluation on '+ch+' samples with the model '+bestModel)

inpath = '/home/minerva1993/tmva/v8/input'
for filename in os.listdir(inpath):
  model_best = load_model(configDir+'keras_'+ch+'/'+bestModel)
  print('model is loaded')
  infile = TFile.Open(inpath+'/'+filename)
  print('processig '+filename)
  intree = infile.Get('tmva_tree')
  inarray = tree2array(intree)
  eval_df = pd.DataFrame(inarray)
  print(eval_df.shape)

  outfile = TFile.Open(configDir+'score_mva/'+ch+'/output_'+ch+'_'+filename,'RECREATE')
  outtree = TTree("tree","tree")

  if filename not in ["tmva_SingleLepton_Run2016.root", "tmva_wjets.root", "tmva_zjets10to50.root", "tmva_zjets.root", "tmva_wjetsV2.root","tmva_zjets10to50V2.root","tmva_ww.root", "tmva_wz.root", "tmva_zz.root"]:
    eval_df = eval_df.drop(eval_df.query('nevt %5 != 0').index)
  spectator = eval_df.filter(["GoodPV", "EventCategory", "GenMatch", "EventWeight"], axis=1)
  eval_df = eval_df.drop(["EventWeight", "totnevt", "nevt", "GoodPV", "GenMatch", "EventCategory",
                          "lepDPhi", "bjetmDR", "bjetmDEta", "bjetmDPhi", "dibjetsMass", "bjetPt_dibjetsm", "cjetPt", "transverseMass",
                          "jet1phi", "jet2phi", "jet3phi", "jet4phi", "DRlepWphi",
                          "DRjet0phi", "DRjet1phi", "DRjet2phi", "DRjet3phi", "DRjet12phi", "DRjet23phi", "DRjet31phi",
                          "DRlepTphi", "DRhadTphi",
                          "ncjets_m", "missingET",
                          "jet1pt", "jet2pt", "jet3pt", "jet4pt",
                          "jet1eta", "jet2eta", "jet3eta", "jet4eta",
                          "jet1m", "jet2m", "jet3m", "jet4m",
                          "jet1csv", "jet2csv", "jet3csv", "jet4csv",
                          "jet1cvsl", "jet2cvsl", "jet3cvsl", "jet4cvsl",
                          "jet1cvsb","jet2cvsb", "jet3cvsb", "jet4cvsb", "DRlepWdeta"
                          ], axis=1)
  print(eval_df.shape)
  eval_df.astype('float32')
  
  eval_scaler = StandardScaler()
  eval_scaler.fit(eval_df)
  eval_df_sc = eval_scaler.transform(eval_df)
  dim = 58
  """
  if(len(eval_df_sc) >= 58):
    eval_pca = PCA(n_components=dim)
    X = eval_pca.fit_transform(eval_df_sc)
  else: X = eval_df_sc
  """
  X = eval_df_sc
  y = model_best.predict(X, batch_size=2000)
  y.dtype = [('KerasScore', np.float32)]
  y = y[:,1]
  array2tree(y, name='tree', tree=outtree)

  for colname, value in spectator.iteritems():
    spect = spectator[colname].values
    if colname == 'GenMatch'       : branchname = 'genMatch'
    elif colname == 'GoodPV'       : branchname = 'PU'
    elif colname == 'EventWeight'  : branchname = 'Event_Weight'
    elif colname == 'EventCategory': branchname = 'category'

    if branchname in ['category', 'genMatch', 'PU']: spect.dtype = [(branchname, np.int32)]
    else:
      spect.dtype = [(branchname, np.float32)]
    #print(branchname)
    #print(spect.shape)
    array2tree(spect, name='tree', tree=outtree)

  print('Prediction on '+filename+' is saved\n  ')

  outtree.Fill()
  outfile.Write()
  outfile.Close()

