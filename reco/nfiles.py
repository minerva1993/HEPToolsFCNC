#!/usr/bin/python
import os, sys

print "Usuage: python nfiles <year> <ver> (default: print all)"
print17 = True; print18 = True; ver = '00'
if len(sys.argv) > 1:
  if   sys.argv[1] == "2017": print18 = False
  elif sys.argv[1] == "2018": print17 = False
  elif sys.argv[1] == "1718": pass
  if len(sys.argv) > 2:
    ver = sys.argv[2]

if print17:
  folder17 = sorted(os.listdir("./2017"))
  print "Scores for 2017 Reco"
  print 'Looking at version ' + ver + ' files'
  for folder in folder17:
    if ver != '00' and ver not in folder: continue
    if 'assign' in folder:
      tmp = []
      for files in os.listdir("./2017/" + folder):
        if files.endswith(".root"): tmp.append(files)
      print folder + ": " + str(len(tmp)).rjust(30-len(folder))
  for folder in folder17:
    if ver != '00' and ver not in folder: continue
    if folder.startswith("score"):
      tmp = []
      for files in os.listdir("./2017/" + folder):
        if files.endswith(".root"): tmp.append(files)
      print folder + ": " + str(len(tmp)).rjust(30-len(folder))

if print18:
  folder18 = sorted(os.listdir("./2018"))
  print "Scores for 2018 Reco"
  print 'Looking at version ' + ver + ' files'
  for folder in folder18:
    if ver != '00' and ver not in folder: continue
    if 'assign' in folder:
      tmp = []
      for files in os.listdir("./2018/" + folder):
        if files.endswith(".root"): tmp.append(files)
      print folder + ": " + str(len(tmp)).rjust(30-len(folder))
  for folder in folder18:
    if ver != '00' and ver not in folder: continue
    if folder.startswith("score"):
      tmp = []
      for files in os.listdir("./2018/" + folder):
        if files.endswith(".root"): tmp.append(files)
      print folder + ": " + str(len(tmp)).rjust(30-len(folder))
