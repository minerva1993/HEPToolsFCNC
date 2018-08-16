from subprocess import call
import sys

sig_only = sys.argv[1]
ch = sys.argv[2]
ver = sys.argv[3]
file_path = sys.argv[4]
file_name = sys.argv[5]

if ch not in ["STFCNC", "TTFCNC", "TTBKG"]: 
  print "Checak channel again!"
  sys.exit(0)

if sig_only == "True":
  if ch == "STFCNC":
    if "STTH1L3B" not in file_name: sys.exit(0)
  elif ch == "TTFCNC":
    if "TTTH1L3B" not in file_name: sys.exit(0)
  elif ch == "TTBKG":
    if "TTpowheg"  not in file_name: sys.exit(0)

call(["root", "-l", 'run.C("'+ ch + '","' + ver + '","' + file_path + ' ' + file_name + '")'], shell=False)
