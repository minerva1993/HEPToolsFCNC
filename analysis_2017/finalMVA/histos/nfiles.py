#!/usr/bin/python
import os, sys


allfolders = sorted(os.listdir("."))

for folder in allfolders:
  if folder.startswith("H"):
    tmp = []
    for files in os.listdir("./" + folder):
      if files.endswith(".root"): tmp.append(files)
    print folder + ": " + str(len(tmp)).rjust(30-len(folder))
