#!/bin/sh

#SBATCH -J test
#SBATCH -p cpu,gpu
#SBATCH -N 1
#SBATCH --open-mode=append
#SBATCH -o %x.out
#SBATCH -e %x.error
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1gb 
#SBATCH --comment python
#SBATCH --time 00:30:00
#SBATCH --hint=compute_bound

opt1=$1
opt2=$2

python runReco.py STFCNC01 2017 $opt1 $opt2
