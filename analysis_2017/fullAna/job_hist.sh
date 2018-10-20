#cat ../commonTools/file_top.txt | xargs -i -P$(nproc) -n2 python runReco.py TTBKG01
#cat ../commonTools/file_other.txt | xargs -i -P$(nproc) -n2 python runReco.py TTBKG01
#cat ../commonTools/file_syst.txt | xargs -i -P$(nproc) -n2 python runReco.py TTBKG01
cat ../commonTools/file_all.txt | xargs -i -P$(nproc) -n2 python runReco.py TTFCNC01
