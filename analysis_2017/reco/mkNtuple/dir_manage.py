#!/usr/bin/python
import os, sys

ch = ['STFCNC', 'STFCNCjecup', 'STFCNCjecdown', 'STFCNCjerup', 'STFCNCjerdown',
      'TTFCNC', 'TTFCNCjecup', 'TTFCNCjecdown', 'TTFCNCjerup', 'TTFCNCjerdown',
      'TTBKG',  'TTBKGjecup',  'TTBKGjecdown',  'TTBKGjerup',  'TTBKGjerdown']

for chs in ch:
  if not os.path.exists( "root_" + chs ):
    os.makedirs( "root_" + chs )
  if not os.path.exists( "hdf_" + chs ):
    os.makedirs( "hdf_" + chs )

def nfile(dirname):
  tmp = []
  allfiles = os.listdir(dirname)
  for files in allfiles:
    if files.endswith(".root") or files.endswith(".h5"): tmp.append(files)
  print dirname + ": " + str(len(tmp)).rjust(30-len(dirname))

for chs in ch:
  nfile("root_" + chs)
for chs in ch:
  nfile("hdf_" + chs)
