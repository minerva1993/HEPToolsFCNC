#!/usr/bin/python
import os, sys

ver = -1
if len(sys.argv) > 1: #flag for version selection
  ver = sys.argv[1]

folder17 = sorted(os.listdir("./2017"))
print "Scores for 2017 Reco"
for folder in folder17:
  if ver > 0 and '_' + ver not in folder: continue
  if 'H' in folder:
    tmp = []
    for files in os.listdir("./2017/" + folder):
      if files.endswith(".root"): tmp.append(files)
    print folder + ": " + str(len(tmp)).rjust(30-len(folder))

folder18 = sorted(os.listdir("./2018"))
print "Scores for 2018 Reco"
for folder in folder18:
  if ver > 0 and '_' + ver not in folder: continue
  if 'H' in folder:
    tmp = []
    for files in os.listdir("./2018/" + folder):
      if files.endswith(".root"): tmp.append(files)
    print folder + ": " + str(len(tmp)).rjust(30-len(folder))
