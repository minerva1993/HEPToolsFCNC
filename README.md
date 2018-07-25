# Analysis for 2017 TT and ST FCNC (H to bb)

Before start, make sure the nuples are located in correct place and update the path.

  * Making file lists, ntuple merge script,  and overall PU weight
```{.Bash}
cd HEPToolsFCNC/analysis_2017/commonTools
python create_input_file_list.py
python countZeroPU.py
source merge_ntuples.sh
```
From TruePVWeight.txt, you can find the overall weights for MC events, which compensate the effect of clean up with respect to TruePV in MC. Copy and paste the lines into MyAnalysis.C

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
  *Recnstruction
This is for ST FCNC reconstruction using Keras+TF. For TT FCNC, some options in flat ntuplizer must be changes (eg. event selection, b tagging requirements). The flat ntuples for jer assignment is stored in both root and hdf format. root output is kept for BDT test. Default training code uses 0th ST Hct ntuple with classifier version '01'. score and assign folders will be made automatically.
```{.Bash}
#First you make flat ntuples.
../reco/classifier/2017/mkNtuple/
source compile.sh
cat ../../../../commonTools/file_signal.txt | xargs -i -P$(nproc) -n2 python run_signal.py
cat ../../../../commonTools/file_other.txt | xargs -i -P$(nproc) -n2 python run_other.py
#Launch training
cd ../training
py27 #activate your venv
python training_kerasTF.py
#With classifier, run prediction.
source compile.sh
cat ../../../commonTools/file_signal.txt | xargs -i -P$(nproc) root -l -b run.C'("01","{}")'
cat ../../../commonTools/file_other.txt | xargs -i -P$(nproc) root -l -b run.C'("01","{}")'
#Plot histograms with reconstruction
cd ../../../fullAna/
cat ../commonTools/file_signal.txt | xargs -i -P$(nproc) -n2 python runReco.py
cat ../commonTools/file_other.txt | xargs -i -P$(nproc) -n2 python runReco.py
source job_merge.sh
python ratioEMuCombine.py
```

  *Todo list
1. Automtically generate cutflow-friendly output from plotting code
2. Rearrange BDT codes for reco
3. Final MVA
4. Systematic-ready
