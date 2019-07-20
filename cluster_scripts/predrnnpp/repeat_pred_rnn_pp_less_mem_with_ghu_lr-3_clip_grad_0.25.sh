/#!/bin/sh


for i in {0..88..8}
    do
    sbatch --export=start_epoch=$i cluster_scripts/predrnnpp/pred_rnn_pp_less_mem_with_ghu_lr-3_clip_grad_0.25.sh
    sleep 2h
    done






