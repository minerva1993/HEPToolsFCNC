cd /cms/ldap_home/minerva1993/catTools/CMSSW_9_4_0_pre3
eval `scram runtime -sh`
cd -

VER=04

hadd ref_Top_Hct.root assign$VER/assign_deepReco_TopHct_*
hadd ref_Top_Hut.root assign$VER/assign_deepReco_TopHut_*
hadd ref_AntiTop_Hct.root assign$VER/assign_deepReco_AntiTopHct_*
hadd ref_AntiTop_Hut.root assign$VER/assign_deepReco_AntiTopHut_*
hadd ref_ttbb.root assign$VER/assign_deepReco_ttbb_*
hadd ref_ttbj.root assign$VER/assign_deepReco_ttbj_*
hadd ref_ttcc.root assign$VER/assign_deepReco_ttcc_*
hadd ref_ttLF.root assign$VER/assign_deepReco_ttLF_*
hadd ref_ttother.root assign$VER/assign_deepReco_ttother_*
