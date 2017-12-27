version="v4"
inputdir="/home/minerva1993/fcnc/ntuple_jw/"

#python runAnalysis.py $version $inputdir "TT_TopLeptonicDecay_TH_1L3B_Eta_Hct.root" "Top_Hct"&
python runAnalysis.py $version $inputdir "TT_powheg_ttbb.root" "ttbb"&
#python runAnalysis.py $version $inputdir "DYJets.root" "zjets"&
#python runAnalysis.py $version $inputdir "WJets_part2.root" "wjets"&
