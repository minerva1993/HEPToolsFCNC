cat ../commonTools/file_2017_all.txt | xargs -i -P$(nproc) -n2 python runReco.py STFCNC01 2017
