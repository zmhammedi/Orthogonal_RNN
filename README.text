# Orthogonal_RNN
# Mhammedi, Zakaria, et al. "Efficient Orthogonal Parametrisation of Recurrent Neural Networks Using Householder Reflections." arXiv preprint arXiv:1612.00188 (2016).


#  Usage  #
oRNN
============

Requirements
------------

- Python 3
- Theano 0.9.0

Installation
------------

- Copy the folder "oRNN/" to your local machine::
	cp -r oRNN/ DESTINATION_FOLDER

- Install C_fun.c (C-implementation of the forward and backward propagation)::
	cd oRNN/
	rm C_fun.cpython-35m-darwin.so
	python setup.py build_ext --inplace

- Install mnist::
	cd ./data/MNIST/python-mnist/
	python setup.py install

Usage
-----

- Edit the file "loop_jobs.sh" to set the name of the output file and hyperparameters  
- Run a job::
	./loop_job.sh
- Stopping a job:
	ps -l  # Displays the job list
	kill JOBID  # JOBID selected from the job list





