head -1 ./../../commonTools/file_2017_top.txt | xargs -i -n2 python run_top.py STFCNC 2017
head -1 ./../../commonTools/file_2017_syst.txt | xargs -i -n2 python run_top.py STFCNC 2017
head -1 ./../../commonTools/file_2017_other.txt | xargs -i -n2 python run_other.py STFCNC 2017
