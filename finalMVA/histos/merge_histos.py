#!/usr/bin/python
import os, sys
from subprocess import call
from ROOT import *

if len(sys.argv) < 8:
  print("Not enough arguements: Ver, Input, Output")
  sys.exit()
ch = sys.argv[1]
era = sys.argv[2]
v_j3b2 = sys.argv[3]
v_j3b3 = sys.argv[4]
v_j4b2 = sys.argv[5]
v_j4b3 = sys.argv[6]
v_j4b4 = sys.argv[7]
v_out = sys.argv[8]

base_path = "./" + era
if not os.path.exists(os.path.join( base_path + ch + "_merged_" + v_out )):
  os.makedirs(os.path.join( base_path, ch + "_merged_" + v_out, "hist_j3" ))
  os.makedirs(os.path.join( base_path, ch + "_merged_" + v_out, "hist_j4" ))
  os.makedirs(os.path.join( base_path, ch + "_merged_" + v_out, "merged" ))

#First check hist contents
for jetcat in ['j3b2', 'j3b3', 'j4b2', 'j4b3', 'j4b4']:

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

  if   jetcat == 'j3b2': ver = v_j3b2
  elif jetcat == 'j3b3': ver = v_j3b3
  elif jetcat == 'j4b2': ver = v_j4b2
  elif jetcat == 'j4b3': ver = v_j4b3
  elif jetcat == 'j4b4': ver = v_j4b4
  else:
    print "Check jetcat and version"
    sys.exit(0)

  for hist_file in os.listdir(os.path.join( base_path, ch + "_" + jetcat + "_" + ver, "post_process" )):
    #print hist_file
    if not hist_file.endswith('root'): continue

    f_tmp = TFile.Open(os.path.join( base_path, ch + "_" + jetcat + "_" + ver, "post_process", hist_file ), "READ")
    h2_tmp = f_tmp.Get("h_DNN_b2_Ch2")
    h3_tmp = f_tmp.Get("h_DNN_b3_Ch2")
    h4_tmp = f_tmp.Get("h_DNN_b4_Ch2")
    n2_tmp = h2_tmp.Integral()
    n3_tmp = h3_tmp.Integral()
    n4_tmp = h4_tmp.Integral()

    if nbjets_cut == 2 and (n3_tmp != 0 or n4_tmp != 0):
      print "Non-zero entries in " + hist_file + " " + "h_DNN_b3_Ch2" + " or " + "h_DNN_b4_Ch2"
    elif nbjets_cut == 3 and (n2_tmp != 0 or n4_tmp != 0):
      print "Non-zero entries in " + hist_file + " " + "h_DNN_b2_Ch2" + " or " + "h_DNN_b4_Ch2"
    elif nbjets_cut == 4 and (n2_tmp != 0 or n3_tmp != 0):
      print "Non-zero entries in " + hist_file + " " + "h_DNN_b2_Ch2" + " or " + "h_DNN_b3_Ch2"


#Then hadd histos for j3 and j4, respectively
for hist_file in os.listdir(os.path.join( base_path, ch + "_j3b2_" + v_j3b2, "post_process" )):
  #print hist_file
  if not hist_file.endswith('root'): continue

  call(["hadd", os.path.join( base_path, ch + "_merged_" + v_out, "hist_j3", hist_file ),
      os.path.join( base_path, ch + "_j3b2_" + v_j3b2, "post_process", hist_file ),
      os.path.join( base_path, ch + "_j3b3_" + v_j3b3, "post_process", hist_file )], shell=False)

  call(["hadd", os.path.join( base_path, ch + "_merged_" + v_out, "hist_j4", hist_file ),
      os.path.join( base_path, ch + "_j4b2_" + v_j4b2, "post_process", hist_file ),
      os.path.join( base_path, ch + "_j4b3_" + v_j4b3, "post_process", hist_file ),
      os.path.join( base_path, ch + "_j4b4_" + v_j4b4, "post_process", hist_file )], shell=False)


for hist_file in os.listdir(os.path.join( base_path, ch + "_merged_" + v_out, "hist_j3" )):
  #print hist_file
  if not hist_file.endswith('root'): continue

  f3_tmp = TFile.Open(os.path.join( base_path, ch + "_merged_" + v_out, "hist_j3", hist_file ), "READ")
  f4_tmp = TFile.Open(os.path.join( base_path, ch + "_merged_" + v_out, "hist_j4", hist_file ), "READ")
  #Rename all hists
  f_out = TFile.Open(os.path.join( base_path, ch + "_merged_" + v_out, "merged", hist_file ), "RECREATE")
  hname_tmp = [x.GetName() for x in f3_tmp.GetListOfKeys()]
  hname_tmp.remove("EventInfo")
  hname_tmp.remove("ScaleWeights")
  hname_tmp.remove("PSWeights")
  hname_tmp.remove("PDFWeights")

  for hname in hname_tmp:
    f_out.cd()
    h3_tmp = f3_tmp.Get(hname)
    new3_name = hname[:6] + "j3" + hname[6:]
    h3_tmp.SetName(new3_name)
    h3_tmp.Write()

    h4_tmp = f4_tmp.Get(hname)
    new4_name = hname[:6] + "j4" + hname[6:]
    h4_tmp.SetName(new4_name)
    h4_tmp.Write()
  f_out.Close()
  f3_tmp.Close()
  f4_tmp.Close()
