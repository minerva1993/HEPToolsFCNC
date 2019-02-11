ERA=2017
CH=Hct
JETCAT=j3b2
VER=06
MODEL=model_41_0.9084.h5
python evaluation_kerasTF.py ${CH} ${JETCAT} ${VER} ${ERA} 0 ${MODEL} &
python evaluation_kerasTF.py ${CH} ${JETCAT} ${VER} ${ERA} 1 ${MODEL} &
python evaluation_kerasTF.py ${CH} ${JETCAT} ${VER} ${ERA} 2 ${MODEL} &
python evaluation_kerasTF.py ${CH} ${JETCAT} ${VER} ${ERA} 3 ${MODEL} &
python evaluation_kerasTF.py ${CH} ${JETCAT} ${VER} ${ERA} 4 ${MODEL} &
python evaluation_kerasTF.py ${CH} ${JETCAT} ${VER} ${ERA} 5 ${MODEL} &
python evaluation_kerasTF.py ${CH} ${JETCAT} ${VER} ${ERA} 6 ${MODEL} &
