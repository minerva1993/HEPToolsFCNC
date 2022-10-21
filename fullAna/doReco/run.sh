source job_prep_2017.sh
#cat job_hadd_2017.sh | xargs -i -P$(nproc) -l1 python multiHadd.py
cat temp_2017.sh | xargs -i -P$(nproc) -l1 python multiHadd.py
python do_post_process_mult.py
