# Analysis for 2017 TT and ST FCNC (H to bb)

Before start, make sure the nuples are located in correct place and update the path.

  * Making file lists, ntuple merge script,  and overall PU weight
```{.Bash}
cd HEPToolsFCNC/analysis_2017/commonTools
python create_input_file_list.py
python countZeroPU.py
source merge_ntuples.sh
```
From TruePVWeight.txt, you can find the overall weights for MC events, which compensate the effect of clean up with respect to TruePV in MC. Copy and paste the lines into MyAnalysis.C, and reco ntuplizers.

  * Control plots without reconstruction
You can make control plots without signal reconstruction to save time and check Data/MC agreement. You need to compile the code, before launch parallel jobs!
```{.Bash}
cd ../fullAna
python create_script.py
source compile.sh
python runNoReco.py
cp doReco/*.root ./
python ratioEMuCombine.py
```
  *Reconstruction
This is for ST FCNC reconstruction using Keras+TF. For TT FCNC, some options in flat ntuplizer must be changes (eg. event selection, b tagging requirements). The flat ntuples for jer assignment is stored in both root and hdf format. root output is kept for BDT test. Default training code uses 0th ST Hct ntuple with classifier version '01'. score and assign folders will be made automatically.
```{.Bash}
#First you make flat ntuples.
../reco/mkNtuple/
python dir_manage.py #make directories
source compile.sh
source job_ntuple.sh
python dir_manage.py #check number of generated files
#Launch training
cd ../training
python training_kerasTF.py STFCNC 01
#With classifier, run prediction. In evlauation, you can assign sytematic from 0 to 6
cd ..
python select_model.py STFCNC 01
python evaluation_kerasTF.py STFCNC 01 True 1 model.h5
#python evaluation_kerasTF.py TTFCNC 01 True 1 model.h5
#python evaluation_kerasTF.py TTBKG 01 True 0 model.h5
cat ../commonTools/file_top.txt | xargs -i -P1 -n2 python combi_assign.py True STFCNC 01 #for signal efficiency
cat ../commonTools/file_all.txt | xargs -i -P$(nproc) -n2 python combi_assign.py False STFCNC 01
#cat ../commonTools/file_all.txt | xargs -i -P$(nproc) -n2 python nohup combi_assign.py False STFCNC 01 > log &
#Plot histograms with reconstruction
cd ../fullAna/
cat ../commonTools/file_all.txt | xargs -i -P$(nproc) -n2 python runReco.py STFCNC01
#cat ../commonTools/file_all.txt | xargs -i -P$(nproc) -n2 nohup python runReco.py STFCNC01 > log &
cd doReco
source job_merge.sh
python do_post_process.py
mkdir STFCNC01
mkdir figures
mv post_process pre_process temp STFCNC01
#python ratioEMuCombine.py
cd STFCNC01/post_process
../../../../plotIt/plotIt -o ../figures/ ../../../../plotIt/configs/config.yml -y
```
  *Final MVA
```{.Bash}
cd /HEPToolsFCNC/finalMVA/mkNtuple
python dir_manage.py
cat ../../commonTools/file_all.txt | xargs -i -P$(nproc) -n2 python run.py
#cat ../../commonTools/file_all.txt | xargs -i -P$(nproc) -n2 nohup python run.py > log &
cd ../training
python training_kerasTF.py Hct j3b2 01
cd ..
python evaluation_kerasTF.py Hct j4 01 model.h5 0
cat ../commonTools/file_all.txt | xargs -i -P$(nproc) -n2 python run.py Hct_j3b2_01
#cat ../commonTools/file_all.txt | xargs -i -P$(nproc) -n2 nohup python run.py Hct_j3b2_01 > log &
cd histos
source job_merge.sh
python do_post_process.py
mkdir Hct_j4_01
mv post_process pre_process temp Hct_j3b2_01
...
python merge_histos.py Hct 01 01 01 01 01 01
../../../../plotIt/plotIt -o ../ ../../../../plotIt/configs/config.yml -y
```
If you use BDT,
```{.Bash}
python training_bdt.py Hct j3b2 01
#nohup python training_bdt.py Hct j3b2 01 > log_Hct_j3b2 &
cat ../commonTools/file_all.txt | xargs -i -P$(nproc) -n2 python evaluation_bdt.py Hct j3b2 01
#cat ../commonTools/file_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hct j3b2 01 > log &
```


  *All rights for plotIt: https://github.com/cp3-llbb/plotIt/
