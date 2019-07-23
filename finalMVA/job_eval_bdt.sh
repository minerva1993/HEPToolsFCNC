VER=01
ERA=2018
#head -1 ../commonTools/file_2018_all.txt | xargs -i -P$(nproc) -n2 python evaluation_bdt.py Hct j3b2 ${VER} ${ERA}
cat ../commonTools/file_${ERA}_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hct j3b2 ${VER} ${ERA} > log 
cat ../commonTools/file_${ERA}_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hct j3b3 ${VER} ${ERA} >> log
cat ../commonTools/file_${ERA}_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hct j4b2 ${VER} ${ERA} >> log
cat ../commonTools/file_${ERA}_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hct j4b3 ${VER} ${ERA} >> log
cat ../commonTools/file_${ERA}_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hct j4b4 ${VER} ${ERA} >> log
cat ../commonTools/file_${ERA}_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hut j3b2 ${VER} ${ERA} >> log
cat ../commonTools/file_${ERA}_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hut j3b3 ${VER} ${ERA} >> log
cat ../commonTools/file_${ERA}_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hut j4b2 ${VER} ${ERA} >> log
cat ../commonTools/file_${ERA}_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hut j4b3 ${VER} ${ERA} >> log
cat ../commonTools/file_${ERA}_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hut j4b4 ${VER} ${ERA} >> log

