CH=Hct
JETCAT=j3b2
VER=02
MODEL=model_32_0.7468.h5
python evaluation_kerasTF.py ${CH} ${JETCAT} ${VER} ${MODEL} 0 &
python evaluation_kerasTF.py ${CH} ${JETCAT} ${VER} ${MODEL} 1 &
python evaluation_kerasTF.py ${CH} ${JETCAT} ${VER} ${MODEL} 2 &
python evaluation_kerasTF.py ${CH} ${JETCAT} ${VER} ${MODEL} 3 &
python evaluation_kerasTF.py ${CH} ${JETCAT} ${VER} ${MODEL} 4 &
python evaluation_kerasTF.py ${CH} ${JETCAT} ${VER} ${MODEL} 5 &
python evaluation_kerasTF.py ${CH} ${JETCAT} ${VER} ${MODEL} 6 &
