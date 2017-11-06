version="v3"
inputdir="/home/minerva1993/fcnc/ntuple_jw/"

python runAnalysis2.py $version $inputdir "TT_powheg_ttbb.root" "ttbb"&
python runAnalysis2.py $version $inputdir "TT_powheg_ttlf.root" "ttLF"&
python runAnalysis2.py $version $inputdir "TT_powheg_ttbj.root" "ttbj"&
python runAnalysis2.py $version $inputdir "TT_powheg_ttcc.root" "ttcc"&
python runAnalysis2.py $version $inputdir "TT_powheg_ttother.root" "tt"&
python runAnalysis2.py $version $inputdir "DYJets.root" "zjets"&
python runAnalysis2.py $version $inputdir "WJets.root" "wjets"&
python runAnalysis2.py $version $inputdir "SingleTbar_t.root" "tbarchannel"&
python runAnalysis2.py $version $inputdir "SingleTop_t.root" "tchannel"&
python runAnalysis2.py $version $inputdir "DYJets_10to50.root" "zjets10to50"&
python runAnalysis2.py $version $inputdir "WW.root" "ww"&
python runAnalysis2.py $version $inputdir "WZ.root" "wz"&
python runAnalysis2.py $version $inputdir "ZZ.root" "zz"&
python runAnalysis2.py $version $inputdir "SingleTop_tW.root" "tWchannel"&
python runAnalysis2.py $version $inputdir "SingleTbar_tW.root" "tbarWchannel"&
python runAnalysis2.py $version $inputdir "TT_TopLeptonicDecay_TH_1L3B_Eta_Hct.root" "Top_Hct"&
python runAnalysis2.py $version $inputdir "TT_TopLeptonicDecay_TH_1L3B_Eta_Hut.root" "Top_Hut"&
python runAnalysis2.py $version $inputdir "TT_AntitopLeptonicDecay_TH_1L3B_Eta_Hct.root" "AntiTop_Hct"&
python runAnalysis2.py $version $inputdir "TT_AntitopLeptonicDecay_TH_1L3B_Eta_Hut.root" "AntiTop_Hut"&
