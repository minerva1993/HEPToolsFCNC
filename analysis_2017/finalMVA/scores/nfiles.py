#!/usr/bin/python
import os, sys


allfolders = sorted(os.listdir("."))

for folder in allfolders:
  if folder.startswith("Hct"):
    tmp = []
    for files in os.listdir("./" + folder):
      if files.endswith(".root"): tmp.append(files)
    print folder + ": " + str(len(tmp)).rjust(30-len(folder))
for folder in allfolders:
  if folder.startswith("Hut"):
    tmp = []
    for files in os.listdir("./" + folder):
      if files.endswith(".root"): tmp.append(files)
    print folder + ": " + str(len(tmp)).rjust(30-len(folder))
