#!/bin/bash

cd /cms/ldap_home/minerva1993/catTools/CMSSW_9_4_0_pre3
eval `scram runtime -sh`
cd -

MAX=96
NPERJOB=1

BEGIN=$(($1*$NPERJOB))
for i in `seq $BEGIN $(($BEGIN+$NPERJOB-1))`; do
    [ $i -ge $MAX ] && break
    python evaluation.py 01 mc deepReco_ttbb_${i}.root
done
