from subprocess import call
import sys

sig_only = sys.argv[1]
ch = sys.argv[2]
ver = sys.argv[3]
filepath = sys.argv[4]
filename = sys.argv[5]

syst = [""]
syst1 = ["jecup", "jecdown", "jerup", "jerdown"]
syst2 = ["TuneCP5up", "TuneCP5down", "hdampup", "hdampdown"]

if sig_only == "True":
  syst = [""]
  syst1 = []
  syst2 = []
  if   ch == "STFCNC":
    if "STTH1L3BH" not in filename: sys.exit(0)
    if not filename.endswith(('004','005')): sys.exit(0)
  elif ch == "TTFCNC":
    if "TTTH1L3B" not in filename: sys.exit(0)
    if not filename.endswith(('004','005')): sys.exit(0)
  elif ch == "TTBKG":
    if "TTpowheg" not in filename: sys.exit(0)
    if not (any(x in filename for x in ["bb","bj","cc"]) and filename.endswith(('020','021'))) and not (any(x in filename for x in ["lf","other"]) and filename.endswith(('080','081','082','083','084','085','086','087'))): sys.exit(0)

for syst_ext in syst + syst1 + syst2:
  if   ("Run2017" in filename) and syst_ext != "": continue
  elif (syst_ext in syst2) and not (syst_ext in filename): continue
  elif (syst_ext in syst1) and any(tmp in filename for tmp in syst2): continue
  else:
    if (syst_ext in syst2): filename = filename.replace(syst_ext,"")

  call(["root", "-b", "-l", 'run.C("'+ ch + '","' + ver + '","' + syst_ext + '","' + filepath + ' ' + filename + '")'], shell=False)
