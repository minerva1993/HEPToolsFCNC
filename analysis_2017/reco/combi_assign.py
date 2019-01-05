from subprocess import call
import sys

sig_only = sys.argv[1]
ch = sys.argv[2]
ver = sys.argv[3]
file_path = sys.argv[4]
file_name = sys.argv[5]

syst = [""]
syst1 = ["jecup", "jecdown", "jerup", "jerdown"]
syst2 = ["TuneCP5up", "TuneCP5down", "hdampup", "hdampdown"]

if sig_only == "True":
  syst = [""]
  syst1 = []
  syst2 = []
  if ch == "STFCNC":
    if "STTH1L3B" not in file_name: sys.exit(0)
    if not file_name.endswith(('003','004')): sys.exit(0)
  elif ch == "TTFCNC":
    if "TTTH1L3B" not in file_name: sys.exit(0)
    if not file_name.endswith(('003','004')): sys.exit(0)
  elif ch == "TTBKG":
    if "TTpowheg"  not in file_name: sys.exit(0)
    if not file_name.endswith(('020','021')): sys.exit(0)

for syst_ext in syst + syst1 + syst2:
  if   ("Run2017" in file_name) and syst_ext != "": continue
  elif (syst_ext in syst2) and not (syst_ext in file_name): continue
  elif (syst_ext in syst1) and any(tmp in file_name for tmp in syst2): continue
  else:
    if (syst_ext in syst2): file_name = file_name.replace(syst_ext,"")

  call(["root", "-b", "-l", 'run.C("'+ ch + '","' + ver + '","' + syst_ext + '","' + file_path + ' ' + file_name + '")'], shell=False)
