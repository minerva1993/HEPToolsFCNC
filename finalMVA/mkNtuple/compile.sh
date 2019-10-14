head -1 ../../commonTools/file_2017_top.txt | xargs -i -P$(nproc) -n2 python run.py 2017
#head -1 ../../commonTools/file_2018_top.txt | xargs -i -P$(nproc) -n2 python run.py 2018
#head -1 ../../commonTools/file_2017_other.txt | xargs -i -P$(nproc) -n2 python run.py 2017
#head -1 ../../commonTools/file_2017_syst.txt | xargs -i -P$(nproc) -n2 python run.py 2017
