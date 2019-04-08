ERA=2018
cat ../../commonTools/file_${ERA}_top.txt | xargs -i -P$(nproc) -n2 python run_top.py STFCNC ${ERA}
cat ../../commonTools/file_${ERA}_syst.txt | xargs -i -P$(nproc) -n2 python run_top.py STFCNC ${ERA}
cat ../../commonTools/file_${ERA}_other.txt | xargs -i -P$(nproc) -n2 python run_other.py STFCNC ${ERA}
cat ../../commonTools/file_${ERA}_top.txt | xargs -i -P$(nproc) -n2 python run_top.py TTFCNC ${ERA}
cat ../../commonTools/file_${ERA}_syst.txt | xargs -i -P$(nproc) -n2 python run_top.py TTFCNC ${ERA}
cat ../../commonTools/file_${ERA}_other.txt | xargs -i -P$(nproc) -n2 python run_other.py TTFCNC ${ERA}
cat ../../commonTools/file_${ERA}_top.txt | xargs -i -P$(nproc) -n2 python run_top.py TTBKG ${ERA}
cat ../../commonTools/file_${ERA}_syst.txt | xargs -i -P$(nproc) -n2 python run_top.py TTBKG ${ERA}
cat ../../commonTools/file_${ERA}_other.txt | xargs -i -P$(nproc) -n2 python run_other.py TTBKG ${ERA}
