from __future__ import print_function
import sys, os
import google.protobuf

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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

ch = sys.argv[1]
ver = sys.argv[2]
jetcat = sys.argv[3]
syst_cat = sys.argv[4]
bestModel = sys.argv[5]

configDir = '/home/minerva1993/HEPToolsFCNC/analysis_2017/finalMVA/'
weightDir = 'training/final' + '_' + ch + '_' +jetcat + '_'
scoreDir = 'scores/' + ch + '_' +jetcat + '_'

input_files = []
input_features = []
input_features.extend(input_variables(jetcat))

if   int(syst_cat) < 2: syst = [""]
elif int(syst_cat) == 2: syst = ["jecup"]
elif int(syst_cat) == 3: syst = ["jecdown"]
elif int(syst_cat) == 4: syst = ["jerup"]
elif int(syst_cat) == 5: syst = ["jerdown"]
elif int(syst_cat) == 6: syst = ["hdampup", "hdampdown", "TuneCP5up", "TuneCP5down"]
else: print("Wrong systematic category number: 0(TT), 1(other), 2,3(jec up/down), 4,5(jer up/down), 6(hdamp and tune)")


for syst_ext in syst:
  if  int(syst_cat) >=2 and not os.path.exists(os.path.join(configDir, scoreDir + ver + "-" + syst_ext)):
    os.makedirs(os.path.join(configDir, scoreDir + ver + "-" + syst_ext))
  elif int(syst_cat) < 2  and not os.path.exists(os.path.join(configDir, scoreDir + ver)):
    os.makedirs(os.path.join(configDir, scoreDir + ver))

  model_best = load_model(os.path.join(configDir, weightDir+ver, bestModel))
  print('Start evaluation on version '+ ch + ver + syst_ext + ' classifier with the model '+ bestModel)

  for filename in os.listdir(os.path.join(configDir, 'mkNtuple', 'hdf_' + syst_ext)):
    if filename == '.gitkeep': continue
    if int(syst_cat) == 0 and all(x not in filename for x in ["TTpowheg", "TTLL"]): continue
    if int(syst_cat) == 1 and any(x in filename for x in ["TTpowheg", "TTLL"]): continue

    if os.path.exists(os.path.join(configDir, scoreDir + ver + syst_ext, 'score_' + filename.replace('h5','root'))):
      print(scoreDir + ver + "/"  + filename.replace('h5','root') + (' is already exist!').rjust(50-len(filename)))
      continue

    eval_df = pd.read_hdf(os.path.join(configDir, 'mkNtuple', 'hdf_' + syst_ext, filename))
    print(filename + ": " + str(eval_df.shape[0]).rjust(60-len(filename)))

    outfile = TFile.Open(os.path.join(scoreDir + ver + syst_ext, 'score_' + filename.replace('h5','root')),'RECREATE')
    outtree = TTree("tree","tree")

    spectator = eval_df.filter(['nevt', 'njets', 'nbjets_m', 'file', 'EventCategory', 'lepton_pt', 'MET',], axis=1)
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
      else: branchname = colname

      if branchname in ['nevt', 'njets', 'nbjets_m', 'EventCategory']: spect.dtype = [(branchname, np.int32)]
      else:
        spect.dtype = [(branchname, np.float32)]
      #print(branchname)
      #print(spect.shape)
      array2tree(spect, name='tree', tree=outtree)

    outtree.Fill()
    outfile.Write()
    outfile.Close()
