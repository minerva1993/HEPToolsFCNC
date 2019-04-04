#head -1 ./../commonTools/file_2017_top.txt | xargs -i -n2 python runReco.py STFCNC01 2017
#head -1 ./../commonTools/file_2017_other.txt | xargs -i -n2 python runReco.py TTFCNC01 2017
head -1 ./../commonTools/file_2018_other.txt | xargs -i -n2 python runReco.py TTFCNC01 2018
