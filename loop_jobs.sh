#!/bin/bash

data_path='./data/'
file_out='logs_3'  # Output log file


act='leaky_relu'
for task in 'pMNIST'	# Choose among the tasks ['MNIST','pMNIST','PTB','PTB_5','copying','copyingVariable']
do
    for seq_len in 784	# For MNIST set seq_len to 784 
    do
        for m in 32	# Number of reflections
        do
            for hidden_units in 32    
            do
                for model_type in 'oRNN' 
                do
                    for lr in 0.001	# Learning rate 
                    do
                        for seed in 1	# Random seed
                        do
                            for bs in 5	# Batch size
                            do
				python train.py $seed $task $seq_len $model_type $hidden_units $m $lr $bs $act $data_path &> $file_out & 
			        # python test.py $seed $task $seq_len $model_type $hidden_units $m $lr $bs $act $data_path &> $file_out & 
                                # sbatch job.sh $seed $task $seq_len $model_type $hidden_units $m $lr $bs $act $data_path
                            done
                        done
		    done
                done
            done
        done
    done
done




