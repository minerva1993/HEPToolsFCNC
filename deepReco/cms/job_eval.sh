#!/bin/bash
declare -a arr=("ttLF" "TopHct" "AntiTopHct")

cd /cms/ldap_home/minerva1993/catTools/CMSSW_9_4_0_pre3
eval `scram runtime -sh`
cd -

MAX=96
NPERJOB=1

BEGIN=$(($1*$NPERJOB))
for j in "${arr[@]}"; do
  for i in `seq $BEGIN $(($BEGIN+$NPERJOB-1))`; do
      [ $i -ge $MAX ] && break
      python evaluation.py 04 mc deepReco_${j}_${i}.root
  done
done
