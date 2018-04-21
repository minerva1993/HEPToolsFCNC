#!/bin/sh

rm hist_*.root
#hadd hist_TopHct.root temp/hist_TopHct_*.root
#hadd hist_AntiTopHct.root temp/hist_AntiTopHct_*.root
#hadd hist_TopHut.root temp/hist_TopHut_*.root
#hadd hist_AntiTopHut.root temp/hist_AntiTopHut_*.root
hadd hist_DataSingleEG.root temp/hist_DataSingleEG*.root
hadd hist_DataSingleMu.root temp/hist_DataSingleMu*.root
hadd hist_ttbb.root temp/hist_ttbb_*.root
hadd hist_ttbj.root temp/hist_ttbj_*.root
hadd hist_ttcc.root temp/hist_ttcc_*.root
hadd hist_ttLF.root temp/hist_ttLF_*.root
hadd hist_ttother.root temp/hist_ttother_*.root
hadd hist_tchannel.root temp/hist_tchannel_*.root
hadd hist_tWchannel.root temp/hist_tWchannel_*.root
hadd hist_tbarchannel.root temp/hist_tbarchannel_*.root
hadd hist_tbarWchannel.root temp/hist_tbarWchannel_*.root
hadd hist_w1jets50to150.root temp/hist_w1jets50to150_*.root
hadd hist_w1jets150to250.root temp/hist_w1jets150to250_*.root
hadd hist_w1jets250to400.root temp/hist_w1jets250to400_*.root
hadd hist_w1jets400toinf.root temp/hist_w1jets400toinf_*.root
hadd hist_w2jets250to400.root temp/hist_w2jets250to400_*.root
hadd hist_w2jets400toinf.root temp/hist_w2jets400toinf_*.root
hadd hist_w3jets.root temp/hist_w3jets_*.root
hadd hist_w4jets.root temp/hist_w4jets_*.root
hadd hist_zjets.root temp/hist_zjets_*.root
hadd hist_zjets10to50.root temp/hist_zjets10to50_*.root
hadd hist_ww.root temp/hist_ww_*.root
hadd hist_wz.root temp/hist_wz_*.root
hadd hist_zz.root temp/hist_zz_*.root
hadd hist_singleTopHct.root temp/hist_singleTopHct_*.root
hadd hist_singleTopHut.root temp/hist_singleTopHut_*.root
