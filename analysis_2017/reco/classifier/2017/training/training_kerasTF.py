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
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
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

#MultiGPU option
multiGPU = True
if os.environ["CUDA_VISIBLE_DEVICES"] in ["1","2","3","4"]:
  multiGPU = False

#Version of classifier
ver = '01'
configDir = '/home/minerva1993/HEPToolsFCNC/analysis_2017/reco/classifier/2017/'
weightDir = 'training/recoSTFCNC'
scoreDir = 'scoreSTFCNC'

#Options for data preparation
input_files = ['deepReco_STTH1L3BHct_000.h5',
              ]
label_name = 'genMatch'
signal_label = 1011
bkg_drop_rate = 0.8
train_test_rate = 0.8

#Check if the model and files already exist
if not os.path.exists( os.path.join(configDir, weightDir+ver) ):
  os.makedirs( os.path.join(configDir, weightDir+ver) )
if not os.path.exists( os.path.join(configDir, scoreDir+ver) ):
  os.makedirs(configDir+scoreDir+ver)
test = os.listdir( os.path.join(configDir, weightDir+ver) )
for item in test:
  if item.endswith(".pdf") or item.endswith(".h5") or item.endswith("log"):
    #os.remove(os.path.join(os.path.join(configDir, weightDir+ver), item))
    print("Remove previous files or move on to next version!")
    sys.exit()

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
  def __init__(self, training_data, validation_data, model):
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
      roc = roc_auc_score(self.y, y_pred)
      y_pred_val = self.model.predict(self.x_val, batch_size=2000)
      roc_val = roc_auc_score(self.y_val, y_pred_val)
      print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)), str(round(roc_val,4))),end=100*' '+'\n')

      ###################
      #Calculate f1 score
      ###################
      val_predict = (y_pred_val[:,1]).round()
      val_targ = self.y_val[:,1]
      val_f1 = f1_score(val_targ, val_predict)
      val_recall = recall_score(val_targ, val_predict)
      val_precision = precision_score(val_targ, val_predict)
      print('val_f1: %.4f, val_precision: %.4f, val_recall %.4f' %(val_f1, val_precision, val_recall))

      ###############
      #Plot ROC curve
      ###############
      fpr = dict()
      tpr = dict()
      roc_auc = dict()
      #fpr[0], tpr[0], thresholds0 = roc_curve(self.y_val[:,0], y_pred_val[:,0], pos_label=1)#w.r.t bkg is truth in val set
      fpr[1], tpr[1], thresholds1 = roc_curve(self.y_val[:,1], y_pred_val[:,1], pos_label=1)#w.r.t sig is truth in val set
      fpr[2], tpr[2], thresholds2 = roc_curve(self.y[:,1], y_pred[:,1], pos_label=1)#w.r.t sig is truth in training set, for overtraining check
      #plt.plot(1-fpr[0], 1-(1-tpr[0]), 'b')#same as [1]
      plt.plot(tpr[1], 1-fpr[1], 'r')#HEP style ROC
      #plt.plot([0,1], [0,1], 'r--')
      #plt.legend(['class 1'], loc = 'lower right')
      plt.xlabel('Signal Efficiency')
      plt.ylabel('Background Rejection')
      plt.title('ROC Curve')
      plt.savefig(os.path.join(configDir, weightDir+ver, 'fig_roc_%d_%.4f.pdf' %(epoch+1,round(roc_val,4)) ))
      plt.gcf().clear()

      ########################################################
      #Overtraining Check, as well as bkg & sig discrimination
      ########################################################
      bins = 40
      scores = [tpr[1], fpr[1], tpr[2], fpr[2]]
      low = min(np.min(d) for d in scores)
      high = max(np.max(d) for d in scores)
      low_high = (low,high)

      #test is filled
      plt.hist(tpr[1],
               color='b', alpha=0.5, range=low_high, bins=bins,
               histtype='stepfilled', density=True, label='S (test)')
      plt.hist(fpr[1],
               color='r', alpha=0.5, range=low_high, bins=bins,
               histtype='stepfilled', density=True, label='B (test)')

      #training is dotted
      hist, bins = np.histogram(tpr[2], bins=bins, range=low_high, density=True)
      scale = len(tpr[2]) / sum(hist)
      err = np.sqrt(hist * scale) / scale
      width = (bins[1] - bins[0])
      center = (bins[:-1] + bins[1:]) / 2
      plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='S (training)')
      hist, bins = np.histogram(fpr[2], bins=bins, range=low_high, density=True)
      scale = len(tpr[2]) / sum(hist)
      err = np.sqrt(hist * scale) / scale
      plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='B (training)')

      plt.xlabel("Deep Learning Score")
      plt.ylabel("Arbitrary units")
      plt.legend(loc='best')
      plt.savefig(os.path.join(configDir, weightDir+ver, 'fig_overtraining_%d_%.4f.pdf' %(epoch+1,round(roc_val,4)) ))
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
for files in input_files:
  data_temp = pd.read_hdf('../mkNtuple/hdf/' + files)
  if files == input_files[0]: data = data_temp
  else: data = pd.concat([data,data_temp], ignore_index=True)
#print(daaxis=data.index.is_unique)#check if indices are duplicated
data[label_name] = (data[label_name] == signal_label).astype(int)
data = shuffle(data)
NumEvt = data[label_name].value_counts(sort=True, ascending=True)
#print(NumEvt)
print('bkg/sig events : '+ str(NumEvt.tolist()))
data = data.drop(data.query(label_name + ' < 1').sample(frac=bkg_drop_rate, axis=0).index)
NumEvt2 = data[label_name].value_counts(sort=True, ascending=True)
#print(NumEvt2)
print('bkg/sig events after bkg skim : '+ str(NumEvt2.tolist()))


##########################################
#drop phi and label features, correlations
##########################################
#col_names = list(data_train)
labels = data.filter([label_name], axis=1)
data = data.filter(['jet0pt', 'jet0eta', 'jet0m', 'jet1pt', 'jet1eta', 'jet1m', 'jet2pt', 'jet2eta', 'jet2m',
                    'jet12pt', 'jet12eta', 'jet12deta', 'jet12dphi', 'jet12dR', 'jet12m', 
                    'lepWpt', 'lepWdphi', 'lepWm', 'lepTdphi', 'lepTm',
                    'genMatch',
                  ], axis=1)
data.astype('float32')
#print list(data_train)

#correlations(data.loc[data[label_name] == 0].drop(label_name, axis=1), 'bkg')
#correlations(data.loc[data[label_name] == 1].drop(label_name, axis=1), 'sig')
#inputvars(data.loc[data[label_name] == 1].drop(label_name, axis=1), data.loc[data[label_name] == 0].drop(label_name, axis=1), 'sig', 'bkg')

data = data.drop(label_name, axis=1) #then drop label


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

print('Training signal: '+str(len(train_sig))+' / testing signal: '+str(len(test_sig))+' / training background: '+str(len(train_bkg))+' / testing background: '+str(len(test_bkg)))
#print(str(len(X_train)) +' '+ str(len(Y_train)) +' ' + str(len(X_test)) +' '+ str(len(Y_test)))
#print(labels)

labels_train = labels_train.values
Y_train = np_utils.to_categorical(labels_train)
labels_test = labels_test.values
Y_test = np_utils.to_categorical(labels_test)


########################
#Standardization and PCA
########################
scaler = StandardScaler()
data_train_sc = scaler.fit_transform(data_train)
data_test_sc = scaler.fit_transform(data_test)
X_train = data_train_sc
X_test = data_test_sc


#################################
#Keras model compile and training
#################################
a = 300
b = 0.2
init = 'glorot_uniform'

with tf.device("/cpu:0"):
  inputs = Input(shape=(20,))
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

if multiGPU: train_model = multi_gpu_model(model, gpus=4)
else: train_model = model

train_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=1E-3), metrics=['binary_accuracy'])
#model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1E-3), metrics=['binary_accuracy'])
#parallel_model.summary()

modelfile = 'model_{epoch:02d}_{val_binary_accuracy:.4f}.h5'
checkpoint = ModelCheckpoint(os.path.join(configDir, weightDir+ver, modelfile), monitor='val_binary_accuracy', verbose=1, save_best_only=False)#, mode='max')
history = train_model.fit(X_train, Y_train, 
                          epochs=30, batch_size=4000, 
                          validation_data=(X_test, Y_test), 
                          #class_weight={ 0: 14, 1: 1 }, 
                          callbacks=[roc_callback(training_data=(X_train, Y_train), validation_data=(X_test, Y_test), model=model)]
                          )
model.save(os.path.join(configDir, weightDir+ver, 'model.h5'))#save template model, rather than the model returned by multi_gpu_model.


#print(history.history.keys())
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.savefig(os.path.join(configDir, weightDir+ver, 'fig_accuracy.pdf'))
plt.gcf().clear() 

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('binary crossentropy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig(os.path.join(configDir, weightDir+ver, 'fig_loss.pdf'))
plt.gcf().clear() 
