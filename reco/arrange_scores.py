from ROOT import *
import ROOT
import sys, os
import multiprocessing
import pandas as pd
import numpy as np
from root_numpy import array2tree, tree2array

#Channel and version
if len(sys.argv) < 5:
  print("Not enough arguements: Ch, Ver(b2j3 b2j4 b3j3 b3j4 b4j4), Era, Output ver")
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

  dfs = []

  if os.path.exists(fname.replace('VER', ver_b2j3)):
    f_b2j3 = TFile.Open(fname.replace('VER', ver_b2j3), 'READ')
    t_b2j3 = f_b2j3.Get('tree')
    a_b2j3 = tree2array(t_b2j3)
    d_b2j3 = pd.DataFrame(a_b2j3)
    d_b2j3 = d_b2j3.query('(njets == 3) and (nbjets_m == 2)')
    dfs.append(d_b2j3)

  if os.path.exists(fname.replace('VER', ver_b2j4)):
    f_b2j4 = TFile.Open(fname.replace('VER', ver_b2j4), 'READ')
    t_b2j4 = f_b2j4.Get('tree')
    a_b2j4 = tree2array(t_b2j4)
    d_b2j4 = pd.DataFrame(a_b2j4)
    d_b2j4 = d_b2j4.query('(njets >= 4) and (nbjets_m == 2)')
    dfs.append(d_b2j4)

  if os.path.exists(fname.replace('VER', ver_b3j3)):
    f_b3j3 = TFile.Open(fname.replace('VER', ver_b3j3), 'READ')
    t_b3j3 = f_b3j3.Get('tree')
    a_b3j3 = tree2array(t_b3j3)
    d_b3j3 = pd.DataFrame(a_b3j3)
    d_b3j3 = d_b3j3.query('(njets == 3) and (nbjets_m == 3)')
    dfs.append(d_b3j3)

  if os.path.exists(fname.replace('VER', ver_b3j4)):
    f_b3j4 = TFile.Open(fname.replace('VER', ver_b3j4), 'READ')
    t_b3j4 = f_b3j4.Get('tree')
    a_b3j4 = tree2array(t_b3j4)
    d_b3j4 = pd.DataFrame(a_b3j4)
    d_b3j4 = d_b3j4.query('(njets >= 4) and (nbjets_m == 3)')
    dfs.append(d_b3j4)

  if os.path.exists(fname.replace('VER', ver_b4j4)):
    f_b4j4 = TFile.Open(fname.replace('VER', ver_b4j4), 'READ')
    t_b4j4 = f_b4j4.Get('tree')
    a_b4j4 = tree2array(t_b4j4)
    d_b4j4 = pd.DataFrame(a_b4j4)
    d_b4j4 = d_b4j4.query('(njets >= 4) and (nbjets_m >= 4)')
    dfs.append(d_b4j4)

  if len(dfs) > 0:
    df = pd.concat(dfs)

    fout = TFile.Open(fname.replace('VER', out_ver), 'RECREATE')
    tout = TTree("tree","tree")

    for colname, value in df.iteritems():
      leaf = df[colname].values
      branchname = colname
      if branchname in ['match', 'njets', 'nbjets_m', 'idx0', 'idx1', 'idx2', 'idx3']: leaf.dtype = [(branchname, np.int32)]
      else: leaf.dtype = [(branchname, np.float32)]

      array2tree(leaf, name='tree', tree=tout)

    tout.Fill()
    fout.Write()
    fout.Close()


if __name__ == '__main__':

  syst = ["","jecAbsoluteup","jecAbsolutedown", "jecAbsolute"+era+"up", "jecAbsolute"+era+"down",
          "jecBBEC1up", "jecBBEC1down", "jecBBEC1"+era+"up", "jecBBEC1"+era+"down",
          "jecFlavorQCDup", "jecFlavorQCDdown", "jecRelativeBalup", "jecRelativeBaldown",
          "jecRelativeSample"+era+"up", "jecRelativeSample"+era+"down",
          "jerup","jerdown"]
  syst2 = ["TuneCP5up","TuneCP5down","hdampup","hdampdown"] #dedecative samples exist
  #syst = [""]
  #syst2 = []

  base_path = './' + era

  file_list = []

  with open('../commonTools/file_'+era+'_all.txt') as f:
    lines = f.readlines()
    for line in lines:
      name = line.split(' ')[1].rstrip('\n')
      for syst_ext in syst + syst2:
        #if any(s in line for s in ['hdamp','Tune']): continue
        if ("Run201" in name) and syst_ext != "": continue
        elif (syst_ext in syst2) and not (syst_ext in name): continue
        elif (syst_ext in syst) and any(tmp in name for tmp in syst2): continue
        elif (syst_ext in syst2): name = name.replace(syst_ext,"")
        file_list.append( os.path.join(base_path, 'assign' + ch + 'VER' + syst_ext + '/assign_deepReco_' + name + '.root') )

  for syst_ext in syst + syst2:
    out_path = os.path.join(base_path, 'assign' + ch + out_ver + syst_ext)
    if not os.path.exists( out_path ):
      os.makedirs( out_path )

  pool = multiprocessing.Pool(54)
  pool.map(arrange, file_list)
  pool.close()
  pool.join()
