from __future__ import print_function
import sys, os
import google.protobuf

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import pandas as pd
from sklearn.preprocessing import StandardScaler, label_binarize
import numpy as np
from root_numpy import array2tree, tree2array
from ROOT import TFile, TTree
from itertools import groupby
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))
import keras
from keras.models import load_model
from training.variables import input_variables, gen_label

ch = sys.argv[1] #STFCNC, TTFCNC. TTBKG
ver = sys.argv[2] #01

configDir = '/home/minerva1993/HEPToolsFCNC/analysis_2017/reco/'
weightDir = 'training/reco'+ch

input_files = []
input_features = []
signal_label = gen_label(ch)
input_features.extend(input_variables(ch))

model_list = []
file_list = sorted(os.listdir( os.path.join(configDir, weightDir+ver)) )
for files in file_list:
  if files.startswith("model"):

    model_best = load_model(os.path.join(configDir, weightDir+ver, files))
    print('Start evaluation on version '+ ch + ver + ' classifier with the model '+ files)

#    for filename in os.listdir(os.path.join(configDir, 'mkNtuple', 'hdf_' + ch)):
    for filename in os.listdir(os.path.join('/data/users/minerva1993/work/2018_fcnc_RunII2017/reco/current_ver', 'mkNtuple', 'hdf_' + ch)):
      if filename == '.gitkeep': continue

      if   ch == "STFCNC":
        if "STTH1L3BH" not in filename: continue
        if not filename.endswith(('002.h5')): continue
      elif ch == "TTFCNC":
        if "TTTH1L3B" not in filename: continue
        if not filename.endswith(('001.h5')): continue
      elif ch == "TTBKG":
        if "TTpowheg" not in filename: continue
        if not filename.endswith(('013.h5')): continue

#      eval_df = pd.read_hdf(os.path.join(configDir, 'mkNtuple', 'hdf_' + ch, filename))
      eval_df = pd.read_hdf(os.path.join('/data/users/minerva1993/work/2018_fcnc_RunII2017/reco/current_ver', 'mkNtuple', 'hdf_' + ch, filename))
      ncombi = eval_df.shape[0]

      matchable = len(eval_df.query('genMatch == '+ str(signal_label)).index)

      df1 = eval_df.filter(['nevt','genMatch'])
      eval_df = eval_df.filter(input_features, axis=1)
      eval_df.astype('float32')

      eval_scaler = StandardScaler()
      eval_scaler.fit(eval_df)
      eval_df_sc = eval_scaler.transform(eval_df)
      X = eval_df_sc
      y = model_best.predict(X, batch_size=4000)
      y = y[:,1]

      df2 = pd.DataFrame(data=y, index=df1.index)
      df2 = df2.rename(columns = {0: 'score'})
      df3 = pd.concat([df1,df2], axis=1)
      best_idx = df3.groupby('nevt')['score'].transform('max') == df3['score']
      selected = len(df3[best_idx].query('genMatch ==' + str(signal_label)))

      print(filename + ": " + str(eval_df.shape[0]) + " " + " eff: = " + str(selected) + " / " + str(matchable) + " = " + str(selected*1.0/matchable))
