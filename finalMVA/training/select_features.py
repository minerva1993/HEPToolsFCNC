#!/usr/bin/env python
import sys, os, shutil
import subprocess
import signal
import time
from variables import input_variables_bdt

#Version of classifier
if len(sys.argv) < 6:
  print("Not enough arguements: Ch, JetCat, Ver, Era, Recompute")
  sys.exit()
ch = sys.argv[1]
jetcat = sys.argv[2] #will be overidden as 99
ver = sys.argv[3]
era = sys.argv[4]
do_recompute = sys.argv[5] == "True"

nfeat = {'j3b2':10, 'j3b3':10, 'j4b2':10, 'j4b3':10, 'j4b4':5}
#nfeat = {'j3b2':28, 'j3b3':28, 'j4b2':28, 'j4b3':28, 'j4b4':28}
print '####Extracting top N features for ' + ch + '_' + jetcat

#Run training for check separation
if do_recompute: proc = subprocess.Popen('python training_bdt.py ' + ch + ' ' + jetcat + ' ' + ver + ' '  + era + ' True > ' + 'log_' + ch + '_' + jetcat + '_99 &', shell=True, preexec_fn=os.setsid)

#wait until log file created: for double check!!
while True:
  if do_recompute:
    try:
      log_file = open('./log_' + ch + '_' + jetcat + '_' + ver, 'r')
      break
    except: continue
    else: break
  else:
    try:
      log_file = open('Var_logs/' + era + '/log_' + ch + '_' + jetcat + '_' + ver, 'r')
      break
    except:
      log_file = open('Var_logs/' + era + '/log_' + ch + '_' + jetcat, 'r')
      break

if do_recompute:
  while True:
    logline = log_file.readline()
    #if logline.find('number of events passed') > 0: print logline
    #if logline.find('Separation') > 0: print logline
    if logline.find('Train method') > 0:
      os.killpg(proc.pid, signal.SIGTERM)
      break

keep_line = False
rank_list = []
num_evt = []

#going to the first line of log file, check NEvents and ranking
#if do_recompute: log_file = open('./log_' + ch + '_' + jetcat + '_' + ver, 'r')
#else: log_file = open('Var_logs/' + era + '/log_' + ch + '_' + jetcat + '_' + ver, 'r')
if do_recompute:
  try:
    log_file = open('./log_' + ch + '_' + jetcat + '_' + ver, 'r')
  except: print "Wrong log file name for re-computation"
else:
  try:
    log_file = open('Var_logs/' + era + '/log_' + ch + '_' + jetcat + '_' + ver, 'r')
  except:
    log_file = open('Var_logs/' + era + '/log_' + ch + '_' + jetcat, 'r')

loglines = log_file.readlines()

for line in loglines:
  if any(i in line for i in ['number of events passed', '-- training events', '-- testing events']):
    #print line
    pass
  if 'Train method' in line: keep_line = False
  if 'Separation' in line: keep_line = True
  if 'number of events passed' in line: num_evt.append(line.split(':')[3].split('/')[0].strip())
  if keep_line and line.count(':') == 3: rank_list.append(line.split(':')[2].strip())

#Check number of events for training
print num_evt
print 'Signal * 0.8 =', str(int(round(int(num_evt[0]) * 0.8))) #Signal first in TMVA
print 'Background * 0.8 =', str(int(round(int(num_evt[1]) * 0.8)))

#Select top N = nfeat[jetcat] variables and sort by the order of features in root file
rank_list = rank_list[1:]
rank_list = rank_list[:nfeat[jetcat]]
#print rank_list

all_vars = input_variables_bdt(jetcat)
sorted_rank_list = []
for var in all_vars:
  if var in rank_list: sorted_rank_list.append(var)

print "  selected['" + ch + '_' + jetcat + '_' + era + "'] = " + str(sorted_rank_list)

try: #only do for 99
  shutil.rmtree( os.path.join(era, 'final_' + ch + '_' + jetcat + '_99') )
  os.remove( os.path.join(era, 'output_' + ch + '_' + jetcat + '.root') )
except: print "No folder or output file!"
print " "
