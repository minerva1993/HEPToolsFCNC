#!/usr/bin/python
import os, sys

folder17 = sorted(os.listdir("./2017"))
print "Scores for 2017 Reco"
for folder in folder17:
  if 'H' in folder:
    tmp = []
    for files in os.listdir("./2017/" + folder):
      if files.endswith(".root"): tmp.append(files)
    print folder + ": " + str(len(tmp)).rjust(30-len(folder))

folder18 = sorted(os.listdir("./2018"))
print "Scores for 2018 Reco"
for folder in folder18:
  if 'H' in folder:
    tmp = []
    for files in os.listdir("./2018/" + folder):
      if files.endswith(".root"): tmp.append(files)
    print folder + ": " + str(len(tmp)).rjust(30-len(folder))
