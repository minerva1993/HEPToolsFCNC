head -1 ../../commonTools/file_top.txt | xargs -i -P$(nproc) -n2 python run.py
#head -1 ../../commonTools/file_other.txt | xargs -i -P$(nproc) -n2 python run.py
#head -1 ../../commonTools/file_syst.txt | xargs -i -P$(nproc) -n2 python run.py
