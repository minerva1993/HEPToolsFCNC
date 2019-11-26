VER=01
ERA=2017
#head -1 ../commonTools/file_${ERA}_all.txt | xargs -i -P$(nproc) -n2 python evaluation_bdt.py Hct j3b2 ${VER} ${ERA}
cat ../commonTools/file_${ERA}_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hct j3b2 ${VER} ${ERA} > log0
cat ../commonTools/file_${ERA}_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hct j3b3 ${VER} ${ERA} > log1
cat ../commonTools/file_${ERA}_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hct j4b2 ${VER} ${ERA} > log2
cat ../commonTools/file_${ERA}_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hct j4b3 ${VER} ${ERA} > log3
cat ../commonTools/file_${ERA}_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hct j4b4 ${VER} ${ERA} > log4
cat ../commonTools/file_${ERA}_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hut j3b2 ${VER} ${ERA} > log5
cat ../commonTools/file_${ERA}_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hut j3b3 ${VER} ${ERA} > log6
cat ../commonTools/file_${ERA}_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hut j4b2 ${VER} ${ERA} > log7
cat ../commonTools/file_${ERA}_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hut j4b3 ${VER} ${ERA} > log8
cat ../commonTools/file_${ERA}_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hut j4b4 ${VER} ${ERA} > log9
