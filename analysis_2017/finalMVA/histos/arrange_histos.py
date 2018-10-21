import sys, os
from shutil import copy2

if not os.path.exists("./dnn_score"):
  os.makedirs("./dnn_score")

histo_folders = []
for tmp_path in os.listdir("./"):
  if tmp_path.startswith("H"):
    histo_folders.append(tmp_path)

for folders in histo_folders:
  if not os.path.exists(os.path.join("./dnn_score", folders)):
    os.makedirs(os.path.join("./dnn_score", folders))
  file_to_copy = []
  for tmp_files in os.listdir("./" + folders):
    if tmp_files.endswith(".pdf") or tmp_files.endswith(".tex"):
      file_to_copy.append(tmp_files)
  for files in file_to_copy:
    copy2(os.path.join("./" + folders, files), os.path.join("./dnn_score",folders))
