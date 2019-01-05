cat ../commonTools/file_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hct j3b2 01 > log 
cat ../commonTools/file_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hct j3b3 01 > log
cat ../commonTools/file_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hct j4b2 01 > log
cat ../commonTools/file_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hct j4b3 01 > log
cat ../commonTools/file_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hct j4b4 01 > log
cat ../commonTools/file_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hut j3b2 01 > log
cat ../commonTools/file_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hut j3b3 01 > log
cat ../commonTools/file_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hut j4b2 01 > log
cat ../commonTools/file_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hut j4b3 01 > log
cat ../commonTools/file_all.txt | xargs -i -P$(nproc) -n2 nohup python evaluation_bdt.py Hut j4b4 01 > log

