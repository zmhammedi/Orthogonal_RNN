#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --mem=4GB
#SBATCH --ntasks-per-node=1
# SBATCH --gres=gpu:1


module load python/3.5.1
export THEANO_FLAGS='floatX=float64'
python train.py $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10}

#rm *.out



