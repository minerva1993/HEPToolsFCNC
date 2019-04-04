SCENARIO=TTBKG
ERA=2018
VER=01
MODEL=model_47_0.9503.h5
nohup python evaluation_kerasTF.py ${SCENARIO} ${VER} ${ERA} False 0 ${MODEL} > log1 &
nohup python evaluation_kerasTF.py ${SCENARIO} ${VER} ${ERA} False 1 ${MODEL} > log2 &
nohup python evaluation_kerasTF.py ${SCENARIO} ${VER} ${ERA} False 2 ${MODEL} > log3 &
nohup python evaluation_kerasTF.py ${SCENARIO} ${VER} ${ERA} False 3 ${MODEL} > log4 &
nohup python evaluation_kerasTF.py ${SCENARIO} ${VER} ${ERA} False 4 ${MODEL} > log5 &
nohup python evaluation_kerasTF.py ${SCENARIO} ${VER} ${ERA} False 5 ${MODEL} > log6 &
nohup python evaluation_kerasTF.py ${SCENARIO} ${VER} ${ERA} False 6 ${MODEL} > log7 &
#python evaluation_kerasTF.py ${SCENARIO} ${VER} ${ERA} False 0 ${MODEL} &
#python evaluation_kerasTF.py ${SCENARIO} ${VER} ${ERA} False 1 ${MODEL} &
#python evaluation_kerasTF.py ${SCENARIO} ${VER} ${ERA} False 2 ${MODEL} &
#python evaluation_kerasTF.py ${SCENARIO} ${VER} ${ERA} False 3 ${MODEL} &
#python evaluation_kerasTF.py ${SCENARIO} ${VER} ${ERA} False 4 ${MODEL} &
#python evaluation_kerasTF.py ${SCENARIO} ${VER} ${ERA} False 5 ${MODEL} &
#python evaluation_kerasTF.py ${SCENARIO} ${VER} ${ERA} False 6 ${MODEL} &
