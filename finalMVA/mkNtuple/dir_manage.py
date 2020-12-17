#!/usr/bin/python
import os, sys

print "Usuage: python nfiles <year> (default: print all)"
print17 = True; print18 = True;
if len(sys.argv) > 1:
  if   sys.argv[1] == "2017": print18 = False
  elif sys.argv[1] == "2018": print17 = False

#base_path17 = '2017'
#base_path18 = '2018'
base_path17 = '/data1/users/minerva1993/work/fcnc_RunII2017/finalMVA/current_ver'
base_path18 = '/data1/users/minerva1993/work/fcnc_RunII2018/finalMVA/current_ver'

#syst = ["","jecup","jecdown","jerup","jerdown"]
syst = ["","jecAbsoluteup","jecAbsolutedown", "jecAbsoluteERAup", "jecAbsoluteERAdown",
        "jecBBEC1up", "jecBBEC1down", "jecBBEC1ERAup", "jecBBEC1ERAdown",
        "jecFlavorQCDup", "jecFlavorQCDdown", "jecRelativeBalup", "jecRelativeBaldown",
        "jecRelativeSampleERAup", "jecRelativeSampleERAdown",
        "jerup","jerdown"]
syst2 = ["TuneCP5up","TuneCP5down","hdampup","hdampdown"] #dedecative samples exist
ch17 = []
ch18 = []

for sy in syst+syst2:
  ch17.append((sy).replace('ERA', '2017'))
  ch18.append((sy).replace('ERA', '2018'))

for chs in ch17:
  if not os.path.exists( base_path17 ):
    os.makedirs( base_path17 )
  if not os.path.exists( base_path17 + "/root_" + chs ):
    os.makedirs( base_path17 + "/root_" + chs )
  if not os.path.exists( base_path17 + "/hdf_" + chs ):
    os.makedirs( base_path17 + "/hdf_" + chs )
for chs in ch18:
  if not os.path.exists( base_path18 ):
    os.makedirs( base_path18 )
  if not os.path.exists( base_path18 + "/root_" + chs ):
    os.makedirs( base_path18 + "/root_" + chs )
  if not os.path.exists( base_path18 + "/hdf_" + chs ):
    os.makedirs( base_path18 + "/hdf_" + chs )


def nfile(dirname):
  tmp = []
  allfiles = os.listdir(dirname)
  for files in allfiles:
    if files.endswith(".root") or files.endswith(".h5"): tmp.append(files)
  print dirname.split('/')[-1] + ": " + str(len(tmp)).rjust(40-len(dirname.split('/')[-1]))

if print17:
  print "2017 Data/MC"
  for chs in ch17:
    nfile(base_path17 + "/root_" + chs)
  for chs in ch17:
    nfile(base_path17 + "/hdf_" + chs)
if print18:
  print "2018 Data/MC"
  for chs in ch18:
    nfile(base_path18 + "/root_" + chs)
  for chs in ch18:
    nfile(base_path18 + "/hdf_" + chs)
