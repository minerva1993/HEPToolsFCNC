YEAR=2017
head -1 ./../../commonTools/file_${YEAR}_top.txt | xargs -i -n2 python run_top.py STFCNC ${YEAR}
#head -1 ./../../commonTools/file_${YEAR}_syst.txt | xargs -i -n2 python run_top.py STFCNC ${YEAR}
#head -1 ./../../commonTools/file_${YEAR}_other.txt | xargs -i -n2 python run_other.py STFCNC ${YEAR}
