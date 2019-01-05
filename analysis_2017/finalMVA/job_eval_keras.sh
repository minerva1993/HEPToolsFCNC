CH=Hct
JETCAT=j3b2
VER=06
MODEL=model_41_0.9084.h5
python evaluation_kerasTF.py ${CH} ${JETCAT} ${VER} ${MODEL} 0 &
python evaluation_kerasTF.py ${CH} ${JETCAT} ${VER} ${MODEL} 1 &
python evaluation_kerasTF.py ${CH} ${JETCAT} ${VER} ${MODEL} 2 &
python evaluation_kerasTF.py ${CH} ${JETCAT} ${VER} ${MODEL} 3 &
python evaluation_kerasTF.py ${CH} ${JETCAT} ${VER} ${MODEL} 4 &
python evaluation_kerasTF.py ${CH} ${JETCAT} ${VER} ${MODEL} 5 &
python evaluation_kerasTF.py ${CH} ${JETCAT} ${VER} ${MODEL} 6 &
