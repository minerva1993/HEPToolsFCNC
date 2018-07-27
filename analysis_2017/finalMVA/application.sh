#source ~/python27.sh

declare -a arr=("STHct_j3b3_01" "STHct_j4b2_01" "STHct_j4b3_01" "STHut_j3b3_01" "STHut_j4b2_01" "STHut_j4b3_01")

for j in "${arr[@]}"; do
  #python evaluation.py 'tmva_AntiTop_Hct.root' ${j}&
  #python evaluation.py 'tmva_Top_Hct.root' ${j}&
  #python evaluation.py 'tmva_AntiTop_Hut.root' ${j}&
  #python evaluation.py 'tmva_Top_Hut.root' ${j}&

  python evaluation.py 'tmva_singleTopHct.root' ${j}&
  python evaluation.py 'tmva_singleTopHut.root' ${j}&

  python evaluation.py 'tmva_SingleLepton_Run2017.root' ${j}&
  python evaluation.py 'tmva_tbarchannel.root' ${j}&
  python evaluation.py 'tmva_tbarWchannel.root' ${j}&
  python evaluation.py 'tmva_tchannel.root' ${j}&
  python evaluation.py 'tmva_tWchannel.root' ${j}&
  python evaluation.py 'tmva_ttbb.root' ${j}&
  python evaluation.py 'tmva_ttbj.root' ${j}&
  python evaluation.py 'tmva_ttcc.root' ${j}&
  python evaluation.py 'tmva_ttLF.root' ${j}&
  python evaluation.py 'tmva_ttother.root' ${j}&

  python evaluation.py 'tmva_w1jets50to150.root' ${j}&
  python evaluation.py 'tmva_w1jets150to250.root' ${j}&
  python evaluation.py 'tmva_w1jets250to400.root' ${j}&
  python evaluation.py 'tmva_w1jets400toinf.root' ${j}&
  python evaluation.py 'tmva_w2jets250to400.root' ${j}&
  python evaluation.py 'tmva_w2jets400toinf.root' ${j}&
  python evaluation.py 'tmva_w3jets.root' ${j}&
  python evaluation.py 'tmva_w4jets.root' ${j}&
  python evaluation.py 'tmva_zjets10to50.root' ${j}&
  python evaluation.py 'tmva_zjets.root' ${j}&

  python evaluation.py 'tmva_ww.root' ${j}&
  python evaluation.py 'tmva_wz.root' ${j}&
  python evaluation.py 'tmva_zz.root' ${j}&

  #python evaluation.py 'tmva_QCD_EGEnr_120to170.root'
  #python evaluation.py 'tmva_QCD_EGEnr_170to300.root'
  #python evaluation.py 'tmva_QCD_EGEnr_300toInf.root'
  #python evaluation.py 'tmva_QCD_MuEnr_1000toInf.root'
  #python evaluation.py 'tmva_QCD_MuEnr_120to170.root'
  #python evaluation.py 'tmva_QCD_MuEnr_170to300.root'
  #python evaluation.py 'tmva_QCD_MuEnr_300to470.root'
  #python evaluation.py 'tmva_QCD_MuEnr_800to1000.root'
  #python evaluation.py 'tmva_QCD_MuEnr_80to120.root'
done
