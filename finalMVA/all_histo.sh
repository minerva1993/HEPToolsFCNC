ERA=2018
VER=01
CH=Hct
JETCAT=( "j3b2" "j3b3" "j4b2" "j4b3" "j4b4" )

for i in "${JETCAT[@]}"; do
  cd histos
  mkdir temp
  cd ..
  cat ../commonTools/file_2018_all.txt | xargs -i -P$(nproc) -n2 python run.py ${CH}_${i}_${VER} ${ERA}
  cd histos
  source job_merge_${ERA}.sh
  python do_post_process.py
  mkdir ${CH}_${i}_${VER}
  mv pre_process post_process temp ${CH}_${i}_${VER}
  mv ${CH}_${i}_${VER} ${ERA}
  cd ..
done
