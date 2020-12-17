#!/bin/sh

#SBATCH -J test
#SBATCH -p gpu,cpu -x compute-0-1,gpu-0-2
#SBATCH -N 1
#SBATCH --open-mode=append
#SBATCH -o %x.out
#SBATCH -e %x.error
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1gb 
#SBATCH --comment python
#SBATCH --time 02:00:00
#SBATCH --hint=compute_bound

scenario=$1
era=$2
path=$3
name=$4

python run_top.py $scenario $era $path $name
