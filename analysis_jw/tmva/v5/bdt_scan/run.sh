#!/bin/bash

cd /cms/ldap_home/minerva1993/CMSSW_9_2_13
eval `scram runtime -sh`
cd -

cd scanBDT
MAX=315
NPERJOB=1

BEGIN=$(($1*$NPERJOB))
for i in `seq $BEGIN $(($BEGIN+$NPERJOB-1))`; do
    [ $i -ge $MAX ] && break
    ../run_cmsTraining.py BDT_${i}
done
