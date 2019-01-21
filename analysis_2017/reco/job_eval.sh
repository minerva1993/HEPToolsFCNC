SCENARIO=TTFCNC
VER=01
MODEL=model_47_0.9858.h5
#nohup python evaluation_kerasTF.py ${SCENARIO} ${VER} False 0 ${MODEL} > log1 &
#nohup python evaluation_kerasTF.py ${SCENARIO} ${VER} False 1 ${MODEL} > log2 &
#nohup python evaluation_kerasTF.py ${SCENARIO} ${VER} False 2 ${MODEL} > log3 &
#nohup python evaluation_kerasTF.py ${SCENARIO} ${VER} False 3 ${MODEL} > log4 &
#nohup python evaluation_kerasTF.py ${SCENARIO} ${VER} False 4 ${MODEL} > log5 &
#nohup python evaluation_kerasTF.py ${SCENARIO} ${VER} False 5 ${MODEL} > log6 &
#nohup python evaluation_kerasTF.py ${SCENARIO} ${VER} False 6 ${MODEL} > log7 &
python evaluation_kerasTF.py ${SCENARIO} ${VER} False 0 ${MODEL} &
python evaluation_kerasTF.py ${SCENARIO} ${VER} False 1 ${MODEL} &
python evaluation_kerasTF.py ${SCENARIO} ${VER} False 2 ${MODEL} &
python evaluation_kerasTF.py ${SCENARIO} ${VER} False 3 ${MODEL} &
python evaluation_kerasTF.py ${SCENARIO} ${VER} False 4 ${MODEL} &
python evaluation_kerasTF.py ${SCENARIO} ${VER} False 5 ${MODEL} &
python evaluation_kerasTF.py ${SCENARIO} ${VER} False 6 ${MODEL} &
