VER=02
cat ../commonTools/file_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hct j3b2 ${VER} > log 
cat ../commonTools/file_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hct j3b3 ${VER} > log
cat ../commonTools/file_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hct j4b2 ${VER} > log
cat ../commonTools/file_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hct j4b3 ${VER} > log
cat ../commonTools/file_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hct j4b4 ${VER} > log
cat ../commonTools/file_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hut j3b2 ${VER} > log
cat ../commonTools/file_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hut j3b3 ${VER} > log
cat ../commonTools/file_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hut j4b2 ${VER} > log
cat ../commonTools/file_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hut j4b3 ${VER} > log
cat ../commonTools/file_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hut j4b4 ${VER} > log

