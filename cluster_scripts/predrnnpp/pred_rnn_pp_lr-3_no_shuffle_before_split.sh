#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=LongJobs
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-23:59:00

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}


export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

mkdir -p ${TMP}/datasets/
export DATASET_DIR=${TMP}/datasets/
# Activate the relevant virtual environment:

rsync -ua --progress /home/${STUDENT_ID}/dissertation/Mobile-Data-Forecasting-With-Spatio-Temporal-Networks/data/*.npz /disk/scratch/${STUDENT_ID}/data

source /home/${STUDENT_ID}/miniconda3/bin/activate mlp
cd /home/${STUDENT_ID}/dissertation/Mobile-Data-Forecasting-With-Spatio-Temporal-Networks


python train.py --use_gpu True --gpu_id "0" --num_epochs 100\
                --toy False --learning_rate 0.001\
                --weight_decay_coefficient 0\
                --experiment_name pred_rnn_pp_lr_-3_in12_out10_no_shuffle_before_split\
                --model predrnnpp\
                --seq_start 12 --seq_length 22\
                --batch_size 3