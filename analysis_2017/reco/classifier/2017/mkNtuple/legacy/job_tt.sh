#!/bin/sh

declare -A arr
arr["ttbb"]="TT_powheg_ttbb" arr["ttbj"]="TT_powheg_ttbj" arr["ttcc"]="TT_powheg_ttcc"
arr["ttLF"]="TT_powheg_ttlf"
arr["ttother"]="TT_powheg_ttother"
arr["TopHct"]="signalReco/TT_TopLeptonicDecay_TH_1L3B_Eta_Hct" arr["AntiTopHct"]="signalReco/TT_AntitopLeptonicDecay_TH_1L3B_Eta_Hct" arr["TopHut"]="signalReco/TT_TopLeptonicDecay_TH_1L3B_Eta_Hut" arr["AntiTopHut"]="signalReco/TT_AntitopLeptonicDecay_TH_1L3B_Eta_Hut"

cd /cms/ldap_home/minerva1993/catTools/CMSSW_9_4_0_pre3
eval `scram runtime -sh`
cd -

MAX=96
NPERJOB=1

#INPUTDIR="root://cms-xrdr.sdfarm.kr:1094//xrd/store/user/minerva1993/ntuple_jw/2016/v4/production"
INPUTDIR="/data/users/minerva1993/ntuple_Run2016/v4/production"

BEGIN=$(($1*$NPERJOB))
for key in "${!arr[@]}"; do
  for i in `seq $BEGIN $(($BEGIN+$NPERJOB-1))`; do
  #for i in $(seq 0 95); do
      #[ $i -ge $MAX ] && break
      filename='Tree_ttbbLepJets_'${i}'.root'
      outname=${key}'_'${i}
      python runtt.py ${INPUTDIR} ${arr[${key}]} ${filename} ${outname}
      #echo ${INPUTDIR} ${arr[${key}]} ${filename} ${outname} >> file_tt.txt
  done
done
