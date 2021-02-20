ERA=2017
VER=01
cat ../commonTools/file_${ERA}_all.txt | xargs -i -P$(nproc) -n2 python combi_assign.py STFCNC $VER $ERA False
cat ../commonTools/file_${ERA}_all.txt | xargs -i -P$(nproc) -n2 python combi_assign.py TTFCNC $VER $ERA False
cat ../commonTools/file_${ERA}_all.txt | xargs -i -P$(nproc) -n2 python combi_assign.py TTBKG $VER $ERA False
