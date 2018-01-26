version="v4"
inputdir="/home/minerva1993/fcnc/ntuple_jw/"

python runAnalysis.py $version $inputdir "TT_powheg_ttbb.root" "ttbb"&
python runAnalysis.py $version $inputdir "TT_powheg_ttlf.root" "ttLF"&
python runAnalysis.py $version $inputdir "TT_powheg_ttbj.root" "ttbj"&
python runAnalysis.py $version $inputdir "TT_powheg_ttcc.root" "ttcc"&
python runAnalysis.py $version $inputdir "TT_powheg_ttother.root" "ttother"&
python runAnalysis.py $version $inputdir "DYJets.root" "zjets"&
python runAnalysis.py $version $inputdir "WJets_v2.root" "wjetsV2"&
python runAnalysis.py $version $inputdir "SingleTbar_t.root" "tbarchannel"&
python runAnalysis.py $version $inputdir "SingleTop_t.root" "tchannel"&
python runAnalysis.py $version $inputdir "DYJets_10to50_v2.root" "zjets10to50V2"&
python runAnalysis.py $version $inputdir "WW.root" "ww"&
python runAnalysis.py $version $inputdir "WZ.root" "wz"&
python runAnalysis.py $version $inputdir "ZZ.root" "zz"&
python runAnalysis.py $version $inputdir "SingleTop_tW.root" "tWchannel"&
python runAnalysis.py $version $inputdir "SingleTbar_tW.root" "tbarWchannel"&
python runAnalysis.py $version $inputdir "TT_TopLeptonicDecay_TH_1L3B_Eta_Hct.root" "Top_Hct"
python runAnalysis.py $version $inputdir "TT_TopLeptonicDecay_TH_1L3B_Eta_Hut.root" "Top_Hut"&
python runAnalysis.py $version $inputdir "TT_AntitopLeptonicDecay_TH_1L3B_Eta_Hct.root" "AntiTop_Hct"&
python runAnalysis.py $version $inputdir "TT_AntitopLeptonicDecay_TH_1L3B_Eta_Hut.root" "AntiTop_Hut"&
python runAnalysis.py $version $inputdir "SingleMuon_Run2016.root" "DataSingleMu"&
python runAnalysis.py $version $inputdir "SingleElectron_Run2016.root" "DataSingleEG"&
