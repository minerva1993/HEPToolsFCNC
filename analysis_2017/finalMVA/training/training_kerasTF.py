from __future__ import print_function
import sys, os, shutil
import google.protobuf

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils import shuffle, class_weight
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from root_numpy import array2tree, tree2array
from variables import input_variables, train_files, evalScale
from variables import input_variables_bdt

#Version of classifier
ch = sys.argv[1]
jetcat = sys.argv[2]
ver = sys.argv[3]

#Options
multiGPU = True
if os.environ["CUDA_VISIBLE_DEVICES"] in ["0", "1","2","3"]:
  multiGPU = False
bkg_drop_rate = 0.0
train_test_rate = 0.8
plot_figures = True
include_eventWeight = True
sklearn_based_overtraining_check = False #If it set to false, directly plot DNN scores

#directory name
configDir = '/home/minerva1993/HEPToolsFCNC/analysis_2017/finalMVA/'
weightDir = 'training/final' + '_' + ch + '_' +jetcat + '_'
scoreDir = 'scores/' + ch + '_' +jetcat + '_'
label_name = 'label'
weight_name = 'EventWeight'
weight_name_modi = 'SampleWeight'
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

input_features = []
input_features.extend(input_variables(jetcat))
#input_features.extend(input_variables_bdt(jetcat))
input_features.append(label_name)
sig_files, bkg_files = train_files(ch)
scaleST, scaleTT, scaleTTLJ, scaleTTLL, frac_sig, frac_bkg = evalScale(ch, sig_files, bkg_files)
input_features.remove('STTT')
input_features.remove('channel')

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))

import keras
from keras.utils import np_utils, multi_gpu_model
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Activation, Dropout, add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import Callback, ModelCheckpoint

#Check if the model and files already exist
if not os.path.exists( os.path.join(configDir, weightDir+ver) ):
  os.makedirs( os.path.join(configDir, weightDir+ver) )
if not os.path.exists( os.path.join(configDir, scoreDir+ver) ):
  os.makedirs( os.path.join(configDir, scoreDir+ver) )
if not os.path.exists( os.path.join(configDir, weightDir+ver, 'training_kerasTF.py') ):
  shutil.copy2('training_kerasTF.py', os.path.join(configDir, weightDir+ver, 'training_kerasTF.py'))
for item in os.listdir( os.path.join(configDir, weightDir+ver) ) or os.listdir( os.path.join(configDir, scoreDir+ver) ):
  if item.endswith(".pdf") or item.endswith(".h5") or item.endswith("log"):
    #os.remove(os.path.join(os.path.join(configDir, weightDir+ver), item))
    print("Remove previous files or move on to next version!")
    sys.exit()

sumWST = 0
sumWTT = 0
sumWTTLJ = 0
sumWTTLL = 0

#list for scores for plotting
f1_list = []
val_f1_list = []
auc_list = []
val_auc_list = []
recall_list = []
val_recall_list = []
precision_list = []
val_precision_list = []

#######################
#Plot correlaton matrix
#######################
def correlations(data, name, **kwds):
    """Calculate pairwise correlation between features.
    Extra arguments are passed on to DataFrame.corr()
    """
    # simply call df.corr() to get a table of
    # correlation values if you do not need
    # the fancy plotting
    corrmat = data.corr(**kwds)

    fig, ax1 = plt.subplots(ncols=1, figsize=(6,5))

    opts = {'cmap': plt.get_cmap("RdBu"),
            'vmin': -1, 'vmax': +1}
    heatmap1 = ax1.pcolor(corrmat, **opts)
    plt.colorbar(heatmap1, ax=ax1)

    ax1.set_title("Correlations")

    labels = corrmat.columns.values
    for ax in (ax1,):
        ax.tick_params(labelsize=6)
        # shift location of ticks to center of the bins
        ax.set_xticks(np.arange(len(labels))+0.5, minor=False)
        ax.set_yticks(np.arange(len(labels))+0.5, minor=False)
        ax.set_xticklabels(labels, minor=False, ha='right', rotation=90)
        ax.set_yticklabels(labels, minor=False)

    plt.tight_layout()
    #plt.show()
    if name == 'sig':
      plt.savefig(os.path.join(configDir, weightDir+ver, 'fig_corr_s.pdf'))
      print('Correlation matrix for signal is saved!')
      plt.gcf().clear()
    elif name == 'bkg':
      plt.savefig(os.path.join(configDir, weightDir+ver, 'fig_corr_b.pdf'))
      plt.gcf().clear()
      print('Correlation matrix for background is saved!')
    else: print('Wrong class name!')


#####################
#Plot input variables
#####################
def inputvars(sigdata, bkgdata, signame, bkgname, **kwds):
    print('Plotting input variables')
    bins = 40
    for colname in sigdata:
      dataset = [sigdata, bkgdata]
      low = min(np.min(d[colname].values) for d in dataset)
      high = max(np.max(d[colname].values) for d in dataset)
      if high > 500: low_high = (low,500)
      else: low_high = (low,high)
      if any(name in colname for name in ['jet0m','jet1m','jet2m','jet3m']):
        low_high = (low,50)

      plt.figure()
      sigdata[colname].plot.hist(color='b', density=True, range=low_high, bins=bins, histtype='step', label='signal')
      bkgdata[colname].plot.hist(color='r', density=True, range=low_high, bins=bins, histtype='step', label='background')
      plt.xlabel(colname)
      plt.ylabel('A.U.')
      plt.title('Intput variables')
      plt.legend(loc='upper right')
      plt.savefig(os.path.join(configDir, weightDir+ver, 'fig_input_'+colname+'.pdf'))
      plt.gcf().clear()
      plt.close()


########################################
#Compute AUC after training and plot ROC
########################################
class roc_callback(Callback):
  def __init__(self, training_data, validation_data, model, event_weight_train, event_weight_test):
      self.x = training_data[0]
      self.y = training_data[1]
      self.x_val = validation_data[0]
      self.y_val = validation_data[1]
      self.model_to_save = model

  def on_train_begin(self, logs={}):
      return

  def on_train_end(self, logs={}):
      return

  def on_epoch_begin(self, epoch, logs={}):
      return

  def on_epoch_end(self, epoch, logs={}):
      ############
      #compute AUC
      ############
      print('Calculating AUC of epoch '+str(epoch+1))
      y_pred = self.model.predict(self.x, batch_size=2000)
      roc = roc_auc_score(self.y, y_pred, sample_weight=weights_train)
      y_pred_val = self.model.predict(self.x_val, batch_size=2000)
      roc_val = roc_auc_score(self.y_val, y_pred_val, sample_weight=weights_test)
      print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)), str(round(roc_val,4))),end=100*' '+'\n')
      auc_list.append(roc)
      val_auc_list.append(roc_val)

      ###################
      #Calculate f1 score
      ###################
      predict = (y_pred[:,1]).round()
      targ = self.y[:,1]
      val_predict = (y_pred_val[:,1]).round()
      val_targ = self.y_val[:,1]
      f1 = f1_score(targ, predict)
      val_f1 = f1_score(val_targ, val_predict)
      recall = recall_score(targ, predict)
      val_recall = recall_score(val_targ, val_predict)
      precision = precision_score(targ, predict)
      val_precision = precision_score(val_targ, val_predict)
      print('val_f1: %.4f, val_precision: %.4f, val_recall %.4f' %(val_f1, val_precision, val_recall))
      f1_list.append(f1)
      val_f1_list.append(val_f1)
      recall_list.append(recall)
      val_recall_list.append(val_recall)
      precision_list.append(precision)
      val_precision_list.append(val_precision)

      ###############
      #Plot ROC curve
      ###############
      fpr = dict()
      tpr = dict()
      roc_auc = dict()
      #fpr[0], tpr[0], thresholds0 = roc_curve(self.y_val[:,0], y_pred_val[:,0], pos_label=1)#w.r.t bkg is truth in val set
      fpr[1], tpr[1], thresholds1 = roc_curve(self.y_val[:,1], y_pred_val[:,1], pos_label=1, sample_weight=weights_test)#w.r.t sig is truth in val set
      fpr[2], tpr[2], thresholds2 = roc_curve(self.y[:,1], y_pred[:,1], pos_label=1, sample_weight=weights_train)#w.r.t sig is truth in training set, for overtraining check
      #plt.plot(1-fpr[0], 1-(1-tpr[0]), 'b')#same as [1]
      plt.plot(tpr[1], 1-fpr[1])#HEP style ROC
      plt.plot(tpr[2], 1-fpr[2])#training ROC
      plt.xlabel('Signal Efficiency')
      plt.ylabel('Background Rejection')
      plt.title('ROC Curve')
      plt.legend(['Test', 'Train'], loc='lower left')
      plt.savefig(os.path.join(configDir, weightDir+ver, 'fig_roc_%d_%.4f.pdf' %(epoch+1,round(roc_val,4))))
      plt.gcf().clear()

      ########################################################
      #Overtraining Check, as well as bkg & sig discrimination
      ########################################################
      bins = 40

      if not sklearn_based_overtraining_check:
        ltpr1 = []; ltpr2 = []; lfpr1 = []; lfpr2 = []
        for i in range(len(self.y_val[:,1])):
          if self.y_val[:,1][i] > 0.5: ltpr1.append(y_pred_val[:,1][i])
          else: lfpr1.append(y_pred_val[:,1][i])
        for i in range(len(self.y[:,1])):
          if self.y[:,1][i] > 0.5: ltpr2.append(y_pred[:,1][i])
          else: lfpr2.append(y_pred[:,1][i])

        tpr[1] = ltpr1
        tpr[2] = ltpr2
        fpr[1] = lfpr1
        fpr[2] = lfpr2

      scores = [tpr[1], fpr[1], tpr[2], fpr[2]]
      low = min(np.min(d) for d in scores)
      high = max(np.max(d) for d in scores)
      low_high = (low,high)

      #test is filled
      plt.hist(tpr[2],
               color='b', alpha=0.5, range=low_high, bins=bins,
               histtype='stepfilled', density=True, label='S (train)')
      plt.hist(fpr[2],
               color='r', alpha=0.5, range=low_high, bins=bins,
               histtype='stepfilled', density=True, label='B (train)')

      #training is dotted
      hist, bins = np.histogram(tpr[1], bins=bins, range=low_high, density=True)
      scale = len(tpr[1]) / sum(hist)
      err = np.sqrt(hist * scale) / scale
      width = (bins[1] - bins[0])
      center = (bins[:-1] + bins[1:]) / 2
      plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='S (test)')
      hist, bins = np.histogram(fpr[1], bins=bins, range=low_high, density=True)
      scale = len(tpr[1]) / sum(hist)
      err = np.sqrt(hist * scale) / scale
      plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='B (test)')

      plt.xlim(0,1)
      plt.ylim(bottom=0)
      plt.xlabel("Deep Learning Score")
      plt.ylabel("Arbitrary units")
      plt.legend(loc='best')
      plt.savefig(os.path.join(configDir, weightDir+ver, 'fig_overtraining_%d_%.4f.pdf' %(epoch+1,round(roc_val,4))))

      min_, max_ = plt.gca().get_ylim()
      plt.yscale("log")
      if min_ < 0.01: plt.ylim(bottom=0.01)
      else: plt.ylim(bottom=min_)
      plt.savefig(os.path.join(configDir, weightDir+ver, 'fig_overtraining_log_%d_%.4f.pdf' %(epoch+1,round(roc_val,4))))
      plt.gcf().clear()
      print('ROC curve and overtraining check plots are saved!')

      del y_pred, y_pred_val, fpr, tpr, roc_auc

      ###############################
      #Save single gpu model manually
      ###############################
      modelfile = 'model_%d_%.4f.h5' %(epoch+1,round(roc_val,4))
      self.model_to_save.save(os.path.join(configDir, weightDir+ver, modelfile))
      print('Current model is saved')

      return

  def on_batch_begin(self, batch, logs={}):
      return

  def on_batch_end(self, batch, logs={}):
      return


####################
#read input and skim
####################
nST, nTT = (0,0)
#Signal first
for files in sig_files:
  data_temp = pd.read_hdf('../mkNtuple/hdf_/' + files)
  if njets_cut == 3:
    data_temp = data_temp[data_temp['njets'] ==  njets_cut]
  elif njets_cut == 4:
    data_temp = data_temp[data_temp['njets'] >=  njets_cut]
  if nbjets_cut != 0:
    data_temp = data_temp[data_temp['nbjets_m'] == nbjets_cut]
  elif nbjets_cut == 0:
    data_temp = data_temp[data_temp['nbjets_m'] <= 4]
  data_temp['label'] = 1

  data_temp[weight_name_modi] = data_temp[weight_name] #for sample_weight
  if not include_eventWeight:
    data_temp[weight_name_modi] = 1.0

  if "STTH" in files:
    data_temp['STTT'] = 0
    nST += len(data_temp.index)
    data_temp[weight_name_modi] = data_temp[weight_name_modi] * frac_sig #define TTTH->frac == 1
    sumWST += np.sum(data_temp[weight_name_modi])
  elif "TTTH" in files:
    data_temp['STTT'] = 1
    nTT += len(data_temp.index)
    sumWTT += np.sum(data_temp[weight_name_modi])
  else: print("Wrong signal sample!")

  if files == sig_files[0]: data_sig = data_temp
  else: data_sig = pd.concat([data_sig,data_temp], ignore_index=True)

frac_list = [(nST * scaleST)/(nST * scaleST + nTT * scaleTT), (nTT * scaleTT)/(nST * scaleST + nTT * scaleTT)]

#Next, background
for files in bkg_files:
  data_temp = pd.read_hdf('../mkNtuple/hdf_/' + files)
  if njets_cut == 3:
    data_temp = data_temp[data_temp['njets'] ==  njets_cut]
  elif njets_cut == 4:
    data_temp = data_temp[data_temp['njets'] >=  njets_cut]
  if nbjets_cut != 0:
    data_temp = data_temp[data_temp['nbjets_m'] == nbjets_cut]
  elif nbjets_cut == 0:
    data_temp = data_temp[data_temp['nbjets_m'] <= 4]
  data_temp['label'] = 0

  data_temp[weight_name_modi] = data_temp[weight_name] #for sample_weight
  if not include_eventWeight:
    data_temp[weight_name_modi] = 1.0

  #STTT ratio
  data_temp['STTT'] = np.random.choice(2, size=data_temp.shape[0], p=frac_list)

  if "TTLL" in files:
    data_temp[weight_name_modi] = data_temp[weight_name_modi] * frac_bkg #define TTLJ->frac == 1
    sumWTTLL += np.sum(data_temp[weight_name_modi])
  else: sumWTTLJ += np.sum(data_temp[weight_name_modi])

  if files == bkg_files[0]: data_bkg = data_temp
  else: data_bkg = pd.concat([data_bkg,data_temp], ignore_index=True)


print("ST: ", sumWST, ", TT: ", sumWTT, ", TTLJ: ", sumWTTLJ, ", TTLL: ", sumWTTLL)

data = pd.concat([data_sig,data_bkg], ignore_index=True)
#print(daaxis=data.index.is_unique)#check if indices are duplicated
#data[label_name] = (data[label_name] == signal_label).astype(int) #don't need here
data = shuffle(data)
NumEvt = data[label_name].value_counts(sort=True, ascending=True)
print(NumEvt)
#print('bkg/sig events : '+ str(NumEvt.tolist()))
data = data.drop(data.query(label_name + ' < 1').sample(frac=bkg_drop_rate, axis=0).index)
NumEvt2 = data[label_name].value_counts(sort=True, ascending=True)
print(NumEvt2)
#print('bkg/sig events after bkg skim : '+ str(NumEvt2.tolist()))


##########################################
#drop phi and label features, correlations
##########################################
#col_names = list(data_train)
weights = data.filter([weight_name], axis=1) #native event weights, for roc/auc
weights_modi = data.filter([weight_name_modi], axis=1) #for sample weight
labels = data.filter([label_name], axis=1)
data = data.filter(input_features, axis=1)
data.astype('float32')
#print list(data_train)

if plot_figures:
  correlations(data.loc[data[label_name] == 0].drop(label_name, axis=1), 'bkg')
  correlations(data.loc[data[label_name] == 1].drop(label_name, axis=1), 'sig')
  inputvars(data.loc[data[label_name] == 1].drop(label_name, axis=1), data.loc[data[label_name] == 0].drop(label_name, axis=1), 'sig', 'bkg')

data = data.drop(label_name, axis=1) #then drop label

sys.exit() #for input draw

###############
#split datasets
###############
train_sig = labels.loc[labels[label_name] == 1].sample(frac=train_test_rate, random_state=200)
train_bkg = labels.loc[labels[label_name] == 0].sample(frac=train_test_rate, random_state=200)
test_sig = labels.loc[labels[label_name] == 1].drop(train_sig.index)
test_bkg = labels.loc[labels[label_name] == 0].drop(train_bkg.index)

train_idx = pd.concat([train_sig, train_bkg]).index
test_idx = pd.concat([test_sig, test_bkg]).index

data_train = data.loc[train_idx,:].copy()
data_test = data.loc[test_idx,:].copy()
labels_train = labels.loc[train_idx,:].copy()
labels_test = labels.loc[test_idx,:].copy()
weights_train = weights.loc[train_idx,:].copy()
weights_test = weights.loc[test_idx,:].copy()
weights_modi_train = weights_modi.loc[train_idx,:].copy()
weights_modi_test = weights_modi.loc[test_idx,:].copy()

print('Training signal: '+str(len(train_sig))+' / testing signal: '+str(len(test_sig))+' / training background: '+str(len(train_bkg))+' / testing background: '+str(len(test_bkg)))
#print(str(len(X_train)) +' '+ str(len(Y_train)) +' ' + str(len(X_test)) +' '+ str(len(Y_test)))
#print(labels)

labels_train = labels_train.values
Y_train = np_utils.to_categorical(labels_train)
labels_test = labels_test.values
Y_test = np_utils.to_categorical(labels_test)
weights_train = weights_train.values.flatten()
weights_test = weights_test.values.flatten()
weights_modi_train = weights_modi_train.values.flatten()
weights_modi_test = weights_modi_test.values.flatten()


################
#Standardization
################
scaler = StandardScaler()
data_train_sc = scaler.fit_transform(data_train)
data_test_sc = scaler.fit_transform(data_test)
X_train = data_train_sc
X_test = data_test_sc


#######################################
#Class weight for automatic computation
#######################################
#class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train.flatten())
class_weights = [(sumWST+sumWTT)/(sumWST+sumWTT+sumWTTLJ+sumWTTLL), (sumWTTLJ+sumWTTLL)/(sumWST+sumWTT+sumWTTLJ+sumWTTLL)]
print(class_weights)


#################################
#Keras model compile and training
#################################
a = 300
b = 0.2
init = 'glorot_uniform'

with tf.device("/cpu:0"):
  inputs = Input(shape=(data.shape[1],))
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
  x = Dense(a, activation='relu', kernel_initializer=init, bias_initializer='zeros')(x)
  x = Dropout(b)(x)

  predictions = Dense(2, activation='softmax')(x)
  model = Model(inputs=inputs, outputs=predictions)

if multiGPU: train_model = multi_gpu_model(model, gpus=2)
else: train_model = model

train_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=1E-3), metrics=['binary_accuracy'])
#model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1E-3), metrics=['binary_accuracy'])
#parallel_model.summary()

modelfile = 'model_{epoch:02d}_{val_binary_accuracy:.4f}.h5'
checkpoint = ModelCheckpoint(os.path.join(configDir, weightDir+ver, modelfile),
                            monitor='val_binary_accuracy', verbose=1, save_best_only=False)#, mode='max')
history = train_model.fit(X_train, Y_train,
                          epochs=50, batch_size=1000,
                          validation_data=(X_test, Y_test),
                          class_weight = class_weights, #sample_weight=weights_modi_train,
                          callbacks=[roc_callback(training_data=(X_train, Y_train), validation_data=(X_test, Y_test),
                                                  model=model, event_weight_train=weights_train, event_weight_test=weights_test)]
                          )
model.save(os.path.join(configDir, weightDir+ver, 'model.h5'))#save template model, rather than the model returned by multi_gpu_model.

print("Plotting scores")
#print(history.history.keys())
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.savefig(os.path.join(configDir, weightDir+ver, 'fig_score_accuracy.pdf'))
plt.gcf().clear()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Binary Crossentropy')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.savefig(os.path.join(configDir, weightDir+ver, 'fig_score_loss.pdf'))
plt.gcf().clear()

plt.plot(auc_list)
plt.plot(val_auc_list)
plt.title('Area Under Curve')
plt.ylabel('AUC')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.savefig(os.path.join(configDir, weightDir+ver, 'fig_score_auc.pdf'))
plt.gcf().clear()

plt.plot(f1_list)
plt.plot(val_f1_list)
plt.title('F1 Score')
plt.ylabel('F1 Score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.savefig(os.path.join(configDir, weightDir+ver, 'fig_score_f1.pdf'))
plt.gcf().clear()

plt.plot(recall_list)
plt.plot(val_recall_list)
plt.title('Recall')
plt.ylabel('Recall')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.savefig(os.path.join(configDir, weightDir+ver, 'fig_score_recall.pdf'))
plt.gcf().clear()

plt.plot(precision_list)
plt.plot(val_precision_list)
plt.title('Precision')
plt.ylabel('Precision')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.savefig(os.path.join(configDir, weightDir+ver, 'fig_score_precision.pdf'))
plt.gcf().clear()
print("Training ended!")
