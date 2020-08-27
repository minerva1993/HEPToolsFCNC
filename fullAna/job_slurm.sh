#!/bin/sh

#SBATCH -J slurm_job
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

scenario=$1
era=$2
path=$3
name=$4

python runReco.py $scenario $era $path $name
