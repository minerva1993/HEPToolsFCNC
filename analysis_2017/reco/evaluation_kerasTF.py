from __future__ import print_function
import sys, os
import google.protobuf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pandas as pd
from sklearn.preprocessing import StandardScaler, label_binarize
import numpy as np
from root_numpy import array2tree, tree2array
from ROOT import TFile, TTree

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))
import keras
from keras.models import load_model
from training.variables import input_variables

ch = sys.argv[1] #STFCNC, TTFCNC. TTBKG
ver = sys.argv[2] #01
signal_only = sys.argv[3] == "True"
bestModel = sys.argv[4]

configDir = '/home/minerva1993/HEPToolsFCNC/analysis_2017/reco/'
weightDir = 'training/reco'+ch
scoreDir = 'score'+ch
assignDir = 'assign'+ch

input_files = []
input_features = []
syst = [""]
if not signal_only: syst = syst + ["jecup", "jecdown", "jerup", "jerdown", "hdampup", "hdampdown", "TuneCP5up", "TuneCP5down"]

input_features.extend(input_variables(ch))

for syst_ext in syst:
  if not os.path.exists(os.path.join(configDir, scoreDir + ver + syst_ext)):
    os.makedirs(os.path.join(configDir, scoreDir + ver + syst_ext))
  if not os.path.exists(os.path.join(configDir, assignDir + ver + syst_ext)):
    os.makedirs(os.path.join(configDir, assignDir + ver + syst_ext))

  model_best = load_model(os.path.join(configDir, weightDir+ver, bestModel))
  print('Start evaluation on version '+ ch + ver + syst_ext + ' classifier with the model '+ bestModel)

  for filename in os.listdir(os.path.join(configDir, 'mkNtuple', 'hdf_' + ch + syst_ext)):
    if filename == '.gitkeep': continue
    if os.path.exists(os.path.join(configDir, scoreDir + ver + syst_ext, 'score_' + filename.replace('h5','root'))):
      print('score_' + filename.replace('h5','root') + (' is already exist!').rjust(50-len(filename)))
      continue
    if signal_only:
      if   ch == "STFCNC":
        if "STTH1L3BH" not in filename: continue
      elif ch == "TTFCNC":
        if "TTTH1L3B" not in filename: continue
      elif ch == "TTBKG":
        if "TTpowheg" not in filename: continue
        if not filename.endswith(('012.h5','013.h5','014.h5')): continue

    eval_df = pd.read_hdf(os.path.join(configDir, 'mkNtuple', 'hdf_' + ch + syst_ext, filename))
    print(filename + ": " + str(eval_df.shape[0]).rjust(60-len(filename)))

    outfile = TFile.Open(os.path.join(scoreDir + ver + syst_ext, 'score_' + filename.replace('h5','root')),'RECREATE')
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
