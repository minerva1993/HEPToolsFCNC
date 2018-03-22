from __future__ import print_function
import sys, os
import google.protobuf

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
      plt.savefig('fig_corr_s.pdf')
      print('Correlation matrix for signal is saved!')
      plt.gcf().clear() 
    elif name == 'bkg':
      plt.savefig('fig_corr_b.pdf')
      plt.gcf().clear() 
      print('Correlation matrix for background is saved!')
    else: print('Wrong class name!')


########################################
#Compute AUC after training and plot ROC
########################################
class roc_callback(Callback):
  def __init__(self, training_data, validation_data):
      self.x = training_data[0]
      self.y = training_data[1]
      self.x_val = validation_data[0]
      self.y_val = validation_data[1]

  def on_train_begin(self, logs={}):
      return

  def on_train_end(self, logs={}):
      ############
      #compute AUC
      ############
      print('Calculating AUC')
      y_pred = self.model.predict(self.x)
      roc = roc_auc_score(self.y, y_pred)
      y_pred_val = self.model.predict(self.x_val)
      roc_val = roc_auc_score(self.y_val, y_pred_val)
      print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)), str(round(roc_val,4))),end=100*' '+'\n')

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
      plt.savefig('fig_roc.pdf')
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
      plt.savefig('fig_overtraining.pdf')
      plt.gcf().clear()
      return

  def on_epoch_begin(self, epoch, logs={}):
      return

  """
  def on_epoch_end(self, epoch, logs={}):
      y_pred = self.model.predict(self.x)
      roc = roc_auc_score(self.y, y_pred)
      y_pred_val = self.model.predict(self.x_val)
      roc_val = roc_auc_score(self.y_val, y_pred_val)
      print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)), str(round(roc_val,4))),end=100*' '+'\n')
      return
  """

  def on_epoch_end(self, epoch, logs={}):
      return

  def on_batch_begin(self, batch, logs={}):
      return

  def on_batch_end(self, batch, logs={}):
      return

####################
#read input and skim
####################
data = pd.read_hdf('fcnc.h5')
#print(daaxis=1ta.index.is_unique)#check if indices are duplicated
data["EventCategory"] = (data['EventCategory'] == 0).astype(int)
data = shuffle(data)
NumEvt = data['EventCategory'].value_counts()
#print(NumEvt)
print('bkg/sig events : '+ str(NumEvt.tolist()))
data = data.drop(data.query('nevt %5 == 0').index)
NumEvt2 = data['EventCategory'].value_counts()
##print(NumEvt2)
print('bkg/sig events after bkg skim : '+ str(NumEvt2.tolist()))


##########################################
#drop phi and label features, correlations
##########################################
#col_names = list(data_train)
labels = data.filter(['EventCategory'], axis=1)
labels = labels.values
labels = np_utils.to_categorical(labels)
nevt = data.filter(['nevt'], axis=1)
nevt = nevt.values
data = data.drop(["EventWeight", "totnevt", "nevt", "GoodPV", "GenMatch",
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
data.astype('float32')
#print list(data_train)

correlations(data.loc[data['EventCategory'] == 0].drop('EventCategory', axis=1), 'bkg')
correlations(data.loc[data['EventCategory'] == 1].drop('EventCategory', axis=1), 'sig')

data = data.drop('EventCategory', axis=1) #then drop label


########################
#Standardization and PCA
########################
scaler = StandardScaler()
scaler.fit(data)
data_sc = scaler.transform(data)

NCOMPONENTS = 58
pca = PCA(n_components=NCOMPONENTS)
X = pca.fit_transform(data_sc)
#pca_std = np.std(data_pca)
#print(data_pca_train.shape)


###############
#split datasets
###############
totcombi = len(data)
numTrain = int(round(totcombi*0.739))
numTest = int(round(totcombi*0.261))#the last 5 percent of data
X_train = data_sc[:numTrain]
X_test = data_sc[numTest:]
Y_train = labels[:numTrain]
Y_test = labels[numTest:]
#print(str(totcombi))
#print(str(len(X_train)) +' '+ str(len(Y_train)) +' ' + str(len(X_test)) +' '+ str(len(Y_test)))
#print(labels)


#################################
#Keras model compile and training
#################################
a = 1000
b = 0.6
init = 'glorot_uniform'

inputs = Input(shape=(58,))
x = Dense(a, kernel_regularizer=l2(5E-3))(inputs)
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
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1E-3), metrics=['binary_accuracy'])
model.save('model.h5')

checkpoint = ModelCheckpoint('/home/minerva1993/tmva/test/model_best.h5', monitor='val_binary_accuracy', verbose=1, save_best_only=True, mode='max')
history = model.fit(X_train, Y_train, 
                             epochs=40, batch_size=1000, 
                             validation_data=(X_test, Y_test), 
                             class_weight={ 1: 10, 0: 1 }, 
                             callbacks=[roc_callback(training_data=(X_train, Y_train), validation_data=(X_test, Y_test)), checkpoint]
                             )
del model

#print(history.history.keys())
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.savefig('fig_accuracy.pdf')
plt.gcf().clear() 

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('binary crossentropy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig('fig_loss.pdf')
plt.gcf().clear() 


ch = 'Hct31'
for filename in os.listdir("/home/minerva1993/tmva/v8/input"):
  model = load_model('model_best.h5')
  print('model is loaded')
  infile = TFile.Open('/home/minerva1993/tmva/v8/input/'+filename)
  print('processig '+filename)
  intree = infile.Get('tmva_tree')
  inarray = tree2array(intree)
  eval_df = pd.DataFrame(inarray)
  print(eval_df.shape)

  outfile = TFile.Open(ch+'/output_'+ch+'_'+filename,'RECREATE')
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
  if(len(eval_df_sc) >= 58):
    eval_pca = PCA(n_components=dim)
    X = eval_pca.fit_transform(eval_df_sc)
  else: X = eval_df_sc
  y = model.predict(X)
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

  outtree.Fill()
  outfile.Write()
  outfile.Close()

