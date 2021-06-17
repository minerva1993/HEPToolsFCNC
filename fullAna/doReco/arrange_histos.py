from ROOT import *
import ROOT
import sys, os
import multiprocessing
import pandas as pd
import numpy as np
from root_numpy import array2tree, tree2array

#Channel and version
if len(sys.argv) < 5:
  print("Not enough arguements: Reco, Ver(b2j3 b2j4 b3j3 b3j4 b4j4), Era, Output ver")
  sys.exit()
ch = sys.argv[1]
ver = sys.argv[2]
era = sys.argv[3]
out_ver = sys.argv[4]
if len(ver) != 10: print 'Wrong version, should be 10 digits'

def arrange(fname):

  print fname

  ver_b2j3 = ver[0:2]
  ver_b2j4 = ver[2:4]
  ver_b3j3 = ver[4:6]
  ver_b3j4 = ver[6:8]
  ver_b4j4 = ver[8:10]
  f_b2j3 = TFile.Open(fname.replace('VER', ver_b2j3), 'READ')
  f_b2j4 = TFile.Open(fname.replace('VER', ver_b2j4), 'READ')
  f_b3j3 = TFile.Open(fname.replace('VER', ver_b3j3), 'READ')
  f_b3j4 = TFile.Open(fname.replace('VER', ver_b3j4), 'READ')
  f_b4j4 = TFile.Open(fname.replace('VER', ver_b4j4), 'READ')
  fout   = TFile.Open(fname.replace('VER', out_ver), 'RECREATE')

  hnames = [x.GetName() for x in f_b2j3.GetListOfKeys()]
  infos = [x for x in hnames if any(s in x for s in ['bSFInfo', 'EventInfo', 'Weights'])]
  #print len(hnames), len(infos)
  h_out = []

  h_S2 = [x for x in hnames if any(s in x for s in ['S2'])]
  h_S3 = [x for x in hnames if any(s in x for s in ['S3'])]
  h_S4 = [x for x in hnames if any(s in x for s in ['S4'])]
  h_S6 = [x for x in hnames if any(s in x for s in ['S6'])]
  h_S7 = [x for x in hnames if any(s in x for s in ['S7'])]
  h_S8 = [x for x in hnames if any(s in x for s in ['S8'])]
  #print len(h_S2) + len(h_S3) + len(h_S4) +len(h_S6) + len(h_S7) + len(h_S8) + len(infos)

  for i in infos:
    htmp = f_b2j3.Get(i)
    htmp.Write()

  for i in h_S2:
    htmp = f_b2j3.Get(i)
    htmp.Write()

  for i in h_S3:
    htmp = f_b3j3.Get(i)
    htmp.Write()

  for i in h_S4: #Take S4 from b2j3 file
    htmp = f_b2j3.Get(i)
    htmp.Write()

  for i in h_S6:
    htmp = f_b2j4.Get(i)
    htmp.Write()

  for i in h_S7:
    htmp = f_b3j4.Get(i)
    htmp.Write()

  for i in h_S8:
    htmp = f_b4j4.Get(i)
    htmp.Write()

  fout.Write()
  fout.Close()


if __name__ == '__main__':

  base_path = './' + era

  file_list = []

  for files in os.listdir(era + '/' + ch + ver[0:2] + '/post_process'):
    if files.endswith(".root"): file_list.append(era + '/' + ch + 'VER/post_process/' + files)

  out_path = os.path.join(base_path, ch + out_ver) + '/post_process'
  if not os.path.exists( out_path ):
    os.makedirs( out_path )

  file_list_sorted = sorted(file_list, key=lambda x:(x!='tag', x))

  pool = multiprocessing.Pool(54)
  pool.map(arrange, file_list)
  pool.close()
  pool.join()
