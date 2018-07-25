head -1 ./../../../commonTools/file_signal.txt | xargs -i -P$(nproc) root -l -b run.C'("01","{}")'
