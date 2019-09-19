import os
from variables import input_selected_bdt

for ch in ['Hut', 'Hct']:
  for jetcat in ['j3b2', 'j4b2', 'j3b3', 'j4b3', 'j4b4']:
    print "Common (17 & 18) top features in " + ch + '_' + jetcat
    print set(input_selected_bdt(ch, jetcat, '2017')).intersection(set(input_selected_bdt(ch, jetcat, '2018')))
    print ' '
