SCENARIO=TTBKG
ERA=2017
#ERA=2018
VER=01
MODEL=model_42_0.9501.h5
#MODEL=model_38_0.9508.h5
nohup python evaluation_kerasTF.py ${SCENARIO} ${VER} ${ERA} False 0 ${MODEL} 0 > log_${SCENARIO}_${VER}_${ERA}_0 &
nohup python evaluation_kerasTF.py ${SCENARIO} ${VER} ${ERA} False 1 ${MODEL} 0 > log_${SCENARIO}_${VER}_${ERA}_1 &
nohup python evaluation_kerasTF.py ${SCENARIO} ${VER} ${ERA} False 2 ${MODEL} 0 > log_${SCENARIO}_${VER}_${ERA}_2 &
nohup python evaluation_kerasTF.py ${SCENARIO} ${VER} ${ERA} False 3 ${MODEL} 0 > log_${SCENARIO}_${VER}_${ERA}_3 &
nohup python evaluation_kerasTF.py ${SCENARIO} ${VER} ${ERA} False 4 ${MODEL} 0 > log_${SCENARIO}_${VER}_${ERA}_4 &
nohup python evaluation_kerasTF.py ${SCENARIO} ${VER} ${ERA} False 5 ${MODEL} 0 > log_${SCENARIO}_${VER}_${ERA}_5 &
nohup python evaluation_kerasTF.py ${SCENARIO} ${VER} ${ERA} False 6 ${MODEL} 1 > log_${SCENARIO}_${VER}_${ERA}_6 &
nohup python evaluation_kerasTF.py ${SCENARIO} ${VER} ${ERA} False 7 ${MODEL} 1 > log_${SCENARIO}_${VER}_${ERA}_7 &
nohup python evaluation_kerasTF.py ${SCENARIO} ${VER} ${ERA} False 8 ${MODEL} 1 > log_${SCENARIO}_${VER}_${ERA}_8 &
nohup python evaluation_kerasTF.py ${SCENARIO} ${VER} ${ERA} False 9 ${MODEL} 1 > log_${SCENARIO}_${VER}_${ERA}_9 &
nohup python evaluation_kerasTF.py ${SCENARIO} ${VER} ${ERA} False 10 ${MODEL} 2 > log_${SCENARIO}_${VER}_${ERA}_10 &
nohup python evaluation_kerasTF.py ${SCENARIO} ${VER} ${ERA} False 11 ${MODEL} 2 > log_${SCENARIO}_${VER}_${ERA}_11 &
nohup python evaluation_kerasTF.py ${SCENARIO} ${VER} ${ERA} False 12 ${MODEL} 2 > log_${SCENARIO}_${VER}_${ERA}_12 &
nohup python evaluation_kerasTF.py ${SCENARIO} ${VER} ${ERA} False 13 ${MODEL} 2 > log_${SCENARIO}_${VER}_${ERA}_13 &
nohup python evaluation_kerasTF.py ${SCENARIO} ${VER} ${ERA} False 14 ${MODEL} 3 > log_${SCENARIO}_${VER}_${ERA}_14 &
nohup python evaluation_kerasTF.py ${SCENARIO} ${VER} ${ERA} False 15 ${MODEL} 3 > log_${SCENARIO}_${VER}_${ERA}_15 &
nohup python evaluation_kerasTF.py ${SCENARIO} ${VER} ${ERA} False 16 ${MODEL} 3 > log_${SCENARIO}_${VER}_${ERA}_16 &
nohup python evaluation_kerasTF.py ${SCENARIO} ${VER} ${ERA} False 17 ${MODEL} 3 > log_${SCENARIO}_${VER}_${ERA}_17 &
nohup python evaluation_kerasTF.py ${SCENARIO} ${VER} ${ERA} False 18 ${MODEL} 3 > log_${SCENARIO}_${VER}_${ERA}_18 &
