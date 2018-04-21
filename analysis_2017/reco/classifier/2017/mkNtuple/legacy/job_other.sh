#!/bin/sh

declare -A arr
arr["ttbb"]="TT_powheg_ttbb" arr["ttbj"]="TT_powheg_ttbj" arr["ttcc"]="TT_powheg_ttcc" arr["ttLF"]="TT_powheg_ttlf" arr["ttother"]="TT_powheg_ttother"
arr["tchannel"]="SingleTop_t" arr["tbarchannel"]="SingleTbar_t" arr["tWchannel"]="SingleTop_tW" arr["tbarWchannel"]="SingleTbar_tW"
arr["zjets"]="DYJets" arr["zjets10to50"]="DYJets_10to50" arr["ww"]="WW" arr["wz"]="WZ" arr["zz"]="ZZ"
arr["w1jets50to150"]="W1JetsToLNu_50-150" arr["w1jets150to250"]="W1JetsToLNu_150-250" arr["w1jets250to400"]="W1JetsToLNu_250-400"
arr["w1jets400toinf"]="W1JetsToLNu_400-inf" arr["w2jets250to400"]="W2JetsToLNu_250-400" arr["w2jets400toinf"]="W2JetsToLNu_400-inf"
arr["w3jets"]="W3JetsToLNu" arr["w4jets"]="W4JetsToLNu"
arr["DataSingleMuB"]="SingleMuon_Run2017B" arr["DataSingleMuC"]="SingleMuon_Run2017C" arr["DataSingleMuD"]="SingleMuon_Run2017D" 
arr["DataSingleMuE"]="SingleMuon_Run2017E" arr["DataSingleMuF"]="SingleMuon_Run2017F" 
arr["DataSingleEGB"]="SingleElectron_Run2017B" arr["DataSingleEGC"]="SingleElectron_Run2017C" arr["DataSingleEGD"]="SingleElectron_Run2017D"
arr["DataSingleEGE"]="SingleElectron_Run2017E" arr["DataSingleEGF"]="SingleElectron_Run2017F"

INPUTDIR="/data/users/minerva1993/ntuple_Run2017/V9_2/production"

for key in "${!arr[@]}"; do
  for filepath in ${INPUTDIR}/${arr[${key}]}/*; do
      files=${filepath##/*/}
      idx3=${files%.root}
      idx2=${idx3##T*_}
      idx1=${idx2/#0/}
      idx=${idx1/#0/}
      outname=${key}'_'${idx}
      echo ${INPUTDIR} ${arr[${key}]} ${files} ${outname} >> file_other.txt
  done
done
