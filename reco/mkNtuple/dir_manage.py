#!/usr/bin/python
import os, sys

print "Usuage: python nfiles <year> (default: print all)"
print17 = True; print18 = True;
if len(sys.argv) > 1:
  if   sys.argv[1] == "2017": print18 = False
  elif sys.argv[1] == "2018": print17 = False

#base_path17 = '2017'
#base_path18 = '2018'
base_path17 = '/data1/users/minerva1993/work/fcnc_RunII2017/reco/current_ver'
base_path18 = '/data1/users/minerva1993/work/fcnc_RunII2018/reco/current_ver'

era = 'ERA'
eras = ['2017', '2018']
scheme = ['STFCNC', 'TTFCNC', 'TTBKG']
#syst = ["","jecup","jecdown","jerup","jerdown"]
syst = ["","jecAbsoluteup","jecAbsolutedown", "jecAbsolute"+era+"up", "jecAbsolute"+era+"down",
        "jecBBEC1up", "jecBBEC1down", "jecBBEC1"+era+"up", "jecBBEC1"+era+"down",
        "jecFlavorQCDup", "jecFlavorQCDdown", "jecRelativeBalup", "jecRelativeBaldown",
        "jecRelativeSample"+era+"up", "jecRelativeSample"+era+"down",
        "jerup","jerdown"]
syst2 = ["TuneCP5up","TuneCP5down","hdampup","hdampdown"] #dedecative samples exist
ch = []

if   print17: eras = ['2017']
elif print18: eras = ['2018']
for er in eras:
  for sc in scheme:
    for sy in syst+syst2:
      ch.append((sc+sy).replace('ERA', er))

for chs in ch:
  if not '2018' in chs:
    if not os.path.exists( base_path17 ):
      os.makedirs( base_path17 )
    if not os.path.exists( base_path17 + "/root_" + chs ):
      os.makedirs( base_path17 + "/root_" + chs )
    if not os.path.exists( base_path17 + "/hdf_" + chs ):
      os.makedirs( base_path17 + "/hdf_" + chs )
  if not '2017' in chs:
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
  for chs in ch:
    nfile(base_path17 + "/root_" + chs)
  for chs in ch:
    nfile(base_path17 + "/hdf_" + chs)
if print18:
  print "2018 Data/MC"
  for chs in ch:
    nfile(base_path18 + "/root_" + chs)
  for chs in ch:
    nfile(base_path18 + "/hdf_" + chs)
