import os

def getFolderSize(folder):
    total_size = os.path.getsize(folder)
    for item in os.listdir(folder):
      itempath = os.path.join(folder, item)

      if os.path.isfile(itempath):
        total_size += os.path.getsize(itempath)
      elif os.path.isdir(itempath):
        total_size += getFolderSize(itempath)
    return round(total_size/(1024*1024.0),1)

#base_path = "/data/users/minerva1993/ntuple_Run2017/V9_5/190117/production/"
base_path = "/data1/users/minerva1993/work/2018_fcnc_RunII2017/reco/current_ver/"
folder_list = sorted(os.listdir(base_path), key=str.lower)

for split_path in folder_list:
  if not os.path.isdir(base_path + split_path): continue
  print "Size: " + split_path + " " + str(getFolderSize(base_path + split_path))
