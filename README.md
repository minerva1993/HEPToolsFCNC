# Analysis for 2017 TT and ST FCNC (H to bb)

Before start, make sure the nuples are located in correct place and update the path.

  * Making file lists, ntuple merge script,  and overall PU weight
```{.Bash}
cd HEPToolsFCNC/commonTools
python preprocessSmallFiles.py 2017 
source arrange_2017_ntuples.sh #merge small ntuples first
python create_input_file_list.py 2017
python countZeroPU.py
source merge_ntuples.sh
```
From TruePVWeight.txt, you can find the overall weights for MC events, which compensate the effect of clean up with respect to TruePV in MC. Copy and paste the lines into MyAnalysis.C, and reco ntuplizers.

  * Control plots without reconstruction
You can make control plots without signal reconstruction to save time and check Data/MC agreement. You need to compile the code, before launch parallel jobs!
```{.Bash}
cd ../fullAna
python create_script.py 2017
python runNoReco.py
cd doNoReco
mv hist*.root 2017
cd 2017
python ../../ratioPlot.py
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
python training_kerasTF.py STFCNC 01 2017

#With classifier, run prediction. In evlauation, you can assign sytematic from 0 to 6
cd ..
python select_model.py STFCNC 01 2017
source job_eval.sh
cat ../commonTools/file_2017_top.txt | xargs -i -P1 -n2 python combi_assign.py STFCNC 01 2017 True #for signal efficiency
cat ../commonTools/file_2017_all.txt | xargs -i -P$(nproc) -n2 python combi_assign.py STFCNC 01 2017 False

#Plot histograms with reconstruction
cd ../fullAna/
source compile.sh
cat ../commonTools/file_2017_all.txt | xargs -i -P$(nproc) -n2 python runReco.py STFCNC01 2017
cd doReco
#source job_merge_2017.sh
source job_prep_2017.sh
cat job_hadd_2017.sh | xargs -i -P$(nporc) -n3 python multiHadd.py 
#python do_post_process.py
python do_post_process_mult.py
mkdir STFCNC01
mkdir figures
mv post_process pre_process temp STFCNC01
mv STFCNC01 2017
cd 2017/STFCNC01/post_process
../../../../../plotIt/plotIt -o ../figures/ ../../../../../plotIt/configs/config_2017.yml -y
#17+18 plot
cd ../../../
python stack_signal.py 01 01
python plot_1718.py 01 01
```
  *Final MVA
```{.Bash}
cd /HEPToolsFCNC/finalMVA/mkNtuple
python dir_manage.py
cat ../../commonTools/file_2017_all.txt | xargs -i -P$(nproc) -n2 python run.py 2017
#cat ../../commonTools/file_2017_all.txt | xargs -i -P$(nproc) -n2 nohup python run.py 2017 > log &
# Currently, due to location of reco assign files, ntuples should be generated in t630 and then move to htop

#Select features if needed, and train
cd ../training
source job_select_features.sh #Change flag 'False' to 'True' if you want recompute rankings
source job_select_common_features.sh #print out common features between 17 and 18

source job_train_bdt.sh #If you want to train with all features, change flag in the code
source job_eval_bdt.sh
cd scores
python nfiles 01 #print out the number of files with specified version

#Histogram
cat ../commonTools/file_2017_all.txt | xargs -i -P$(nproc) -n2 python run.py Hct_0101010101 2017
cd histos
source job_merge.sh
python do_post_process.py
mkdir Hct_0101010101
mkdir prefit
mv post_process pre_process prefit temp Hct_0101010101
cd Hct_0101010101/post_process
../../../../../plotIt/plotIt -o ../prefit ../../../../../plotIt/configs/config_2017.yml -y
python print_syst_table.py 2017 0101010101 #Systematic table
########################################################
#DON'T FORGET TO UPDATE YML FILE IN LIMIT SETTING REPO!!
########################################################
```


  *All rights for plotIt: https://github.com/cp3-llbb/plotIt/
