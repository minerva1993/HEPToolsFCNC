#!/bin/sh

declare -A arr
arr["singleTopHct"]="ST_TH_1L3B_Hct" arr["singleTopHut"]="ST_TH_1L3B_Hut" 

INPUTDIR="/data/users/minerva1993/ntuple_Run2017/V9_2/production"

for key in "${!arr[@]}"; do
  for filepath in ${INPUTDIR}/${arr[${key}]}/*; do
      files=${filepath##/*/}
      idx3=${files%.root}
      idx2=${idx3##T*_}
      idx1=${idx2/#0/}
      idx=${idx1/#0/}
      outname=${key}'_'${idx}
      echo ${INPUTDIR} ${arr[${key}]} ${files} ${outname} >> file_st.txt
  done
done
