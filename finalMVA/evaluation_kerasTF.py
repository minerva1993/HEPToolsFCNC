from __future__ import print_function
import sys, os
import google.protobuf

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import pandas as pd
from sklearn.preprocessing import StandardScaler, label_binarize
import numpy as np
from root_numpy import array2tree, tree2array
from ROOT import TFile, TTree
from training.variables import input_variables, train_files, evalScale, evalFrac

#Channel and version
if len(sys.argv) < 7:
  print("Not enough arguements: Ch, JetCat, Ver, Era, Syst. var, Model")
  sys.exit()
ch = sys.argv[1]
jetcat = sys.argv[2]
ver = sys.argv[3]
era = sys.argv[4]
syst_cat = sys.argv[5]
bestModel = sys.argv[6]

rootDir = '/data1/users/minerva1993/work/'
if   era == '2017': rootDir = rootDir + 'fcnc_RunII2017/finalMVA/current_version/hdf_'
elif era == '2018': rootDir = rootDir + 'fcnc_RunII2018/finalMVA/current_version/hdf_' #no /
configDir = '.'
weightDir = 'training/' + era + '/final' + '_' + ch + '_' +jetcat + '_'
scoreDir = 'scores/' + era + '/' + ch + '_' +jetcat + '_'

input_files = []
input_features = []
input_features.extend(input_variables(jetcat))
sig_files, bkg_files = train_files(ch, era)
#input_features.remove('STTT')
#input_features.remove('channel')

njets_cut = int(jetcat[1:2]) #Must be jXbX
if njets_cut not in [3,4]:
  print("Check jet category")
  sys.exit()
if len(jetcat) > 3:
  nbjets_cut = int(jetcat[3:4])
  if nbjets_cut not in [2,3,4]:
    print("Check b jet category")
    sys.exit()
else: nbjets_cut = 0

frac_list = evalFrac(ch, era, sig_files, njets_cut, nbjets_cut)

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))
import keras
from keras.models import load_model

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

  for filename in os.listdir(os.path.join(rootDir + syst_ext)):
    if filename == '.gitkeep': continue
    if int(syst_cat) == 0 and all(x not in filename for x in ["TTpowheg", "TTLL"]): continue
    if int(syst_cat) == 1 and any(x in filename for x in ["TTpowheg", "TTLL"]): continue

    if os.path.exists(os.path.join(configDir, scoreDir + ver + syst_ext, 'score_' + filename.replace('h5','root'))):
      print(scoreDir + ver + "/"  + filename.replace('h5','root') + (' is already exist!').rjust(50-len(filename)))
      continue

    eval_df = pd.read_hdf(os.path.join(rootDir + syst_ext, filename))
    print(filename + ": " + str(eval_df.shape[0]).rjust(60-len(filename)))

    eval_df = eval_df[eval_df['njets'] ==  njets_cut]
    if nbjets_cut != 0:
      eval_df = eval_df[eval_df['nbjets_m'] == nbjets_cut]
    if eval_df.shape[0] < 1: continue

    tmp_path = scoreDir + ver + '-' + syst_ext
    if syst_ext == "": tmp_path = tmp_path.replace("-","")
    outfile = TFile.Open(os.path.join(tmp_path, 'score_' + filename.replace('h5','root')),'RECREATE')
    outtree = TTree("tree","tree")

    eval_df['STTT'] = np.random.choice(2, size=eval_df.shape[0], p=frac_list)
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
